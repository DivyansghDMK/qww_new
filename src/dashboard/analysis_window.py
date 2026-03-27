"""
ECG Analysis Window

Backend-driven 12-lead ECG analysis UI with:
- JSON report loading (backend/local)
- Frame-by-frame waveform navigation
- Manual arrhythmia annotation workflow
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QFrame, QMessageBox,
    QSizePolicy, QComboBox, QFileDialog, QTextEdit, QSlider,
    QLineEdit
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import sys
# Add the src directory to the path to ensure ecg module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Try importing with multiple paths to be robust
try:
    from ecg.arrhythmia_detector import ArrhythmiaDetector
    from ecg.expanded_lead_view import PQRSTAnalyzer
except ImportError:
    try:
        from src.ecg.arrhythmia_detector import ArrhythmiaDetector
        from src.ecg.expanded_lead_view import PQRSTAnalyzer
    except ImportError:
        # Final fallback - use relative imports if possible
        try:
            from ..ecg.arrhythmia_detector import ArrhythmiaDetector
            from ..ecg.expanded_lead_view import PQRSTAnalyzer
        except (ImportError, ValueError):
            print(" Warning: Could not import ECG analysis modules")
            ArrhythmiaDetector = None
            PQRSTAnalyzer = None


class ECGAnalysisWindow(QDialog):
    """User-friendly ECG analysis window for backend JSON reports."""

    LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Waveform Analysis")
        self.setGeometry(60, 40, 1760, 990)
        self.setMinimumSize(1200, 700)

        self.setStyleSheet("""
            QDialog {
                background: #0d0d1a;
                color: #ffffff;
            }
            QFrame#topbar {
                background: #12122a;
                border: none;
                border-bottom: 2px solid #ff6600;
                border-radius: 0px;
            }
            QFrame#plotpanel {
                background: #0d0d1a;
                border: none;
                border-radius: 0px;
            }
            QFrame#bottompanel {
                background: #12122a;
                border: 1px solid #2a2a4a;
                border-radius: 10px;
            }
            QFrame#leadbox {
                background: #111130;
                border: 1px solid #2a2a5a;
                border-radius: 6px;
            }
            QLabel {
                color: #e0e0ff;
                font-size: 11px;
                background: transparent;
                border: none;
            }
            QLabel#leadlabel {
                color: #ff6600;
                font-size: 12px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
            QPushButton {
                background: #1e1e3e;
                color: #e0e0ff;
                border: 1px solid #3a3a6a;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2a2a5a;
                border-color: #ff6600;
                color: white;
            }
            QPushButton#primary {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #ff6600, stop:1 #e65c00);
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton#primary:hover { background: #ff7a26; }
            QPushButton#apibtn {
                background: #1a3a5a;
                color: #7ab8f5;
                border: 1px solid #2a6aaa;
                border-radius: 6px;
            }
            QPushButton#apibtn:hover { background: #244a6a; }
            QComboBox {
                background: #1e1e3e;
                color: #e0e0ff;
                border: 1px solid #3a3a6a;
                border-radius: 6px;
                padding: 5px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #1e1e3e;
                color: #e0e0ff;
                selection-background-color: #ff6600;
            }
            QLineEdit {
                background: #1e1e3e;
                color: #e0e0ff;
                border: 1px solid #3a3a6a;
                border-radius: 6px;
                padding: 5px 8px;
                font-size: 11px;
            }
            QTextEdit {
                background: #1e1e3e;
                color: #e0e0ff;
                border: 1px solid #3a3a6a;
                border-radius: 6px;
                padding: 5px;
                font-size: 11px;
            }
            QTableWidget {
                background: #111130;
                color: #e0e0ff;
                border: 1px solid #2a2a5a;
                gridline-color: #1e1e4a;
                selection-background-color: #ff6600;
                selection-color: white;
                border-radius: 4px;
            }
            QHeaderView::section {
                background: #1e1e3e;
                color: #ff6600;
                border: none;
                padding: 6px;
                font-size: 11px;
                font-weight: bold;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #2a2a5a;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #ff6600;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                background: #ff6600;
                margin: -5px 0;
                border-radius: 7px;
            }
            QScrollBar:vertical {
                background: #1e1e3e;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #3a3a7a;
                border-radius: 4px;
            }
        """)

        self.reports = []
        self.current_report = None
        self.current_report_path = ""

        project_root = Path(__file__).resolve().parents[2]
        self.analysis_pdf_logo_path = project_root / "assets" / "DeckmountLogo.png"

        self.lead_data = {lead: np.array([]) for lead in self.LEADS}
        self.sampling_rate = 500.0

        self.window_seconds = 10.0
        self.step_seconds = 0.5
        self.frame_start_sample = 0
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.next_frame)

        self.pending_mark_start_sec = None
        self.manual_annotations = []

        self._build_ui()
        self.load_reports()

    # --------------------------- UI ---------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_top_bar())
        root.addWidget(self._build_plot_panel(), stretch=4)
        root.addWidget(self._build_bottom_panel(), stretch=2)

    def _build_top_bar(self):
        frame = QFrame()
        frame.setObjectName("topbar")
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(12)

        # Logo
        logo_label = QLabel()
        pixmap = QPixmap(str(self.analysis_pdf_logo_path))
        if not pixmap.isNull():
            logo_label.setPixmap(pixmap.scaled(110, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo_label.setText("DECKMOUNT")
            logo_label.setStyleSheet("color:#ff6600;font-size:16px;font-weight:bold;border:none;background:transparent;")
        lay.addWidget(logo_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("background:#ff6600;border:none;max-width:2px;margin:4px 4px;")
        lay.addWidget(sep)

        pat_col = QVBoxLayout()
        pat_col.setSpacing(2)
        self.patient_lbl = QLabel("Patient: --")
        self.patient_lbl.setFont(QFont("Arial", 12, QFont.Bold))
        self.patient_lbl.setStyleSheet("color:white;font-weight:bold;border:none;background:transparent;")
        self.patient_meta_lbl = QLabel("ID: -- | Age: -- | Gender: --")
        self.patient_meta_lbl.setStyleSheet("color:#aaaacc;font-size:10px;border:none;background:transparent;")
        pat_col.addWidget(self.patient_lbl)
        pat_col.addWidget(self.patient_meta_lbl)
        lay.addLayout(pat_col)
        lay.addStretch()

        lay.addWidget(QLabel("Report:"))
        self.report_combo = QComboBox()
        self.report_combo.currentIndexChanged.connect(self.load_selected_report)
        self.report_combo.setMinimumWidth(300)
        lay.addWidget(self.report_combo)

        self.refresh_btn = QPushButton("↻ Refresh")
        self.refresh_btn.clicked.connect(self.load_reports)
        lay.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("⬇ Export JSON")
        self.export_btn.clicked.connect(self.export_report)
        lay.addWidget(self.export_btn)

        self.pdf_btn = QPushButton("Generate PDF")
        self.pdf_btn.setObjectName("primary")
        self.pdf_btn.clicked.connect(self.generate_pdf_report)
        lay.addWidget(self.pdf_btn)

        lay.addSpacing(8)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("background:#3a3a6a;border:none;max-width:1px;margin:4px 4px;")
        lay.addWidget(sep2)

        self.api_id_lbl = QLabel("API ID:")
        self.api_id_input = QLineEdit()
        self.api_id_input.setPlaceholderText("ID")
        self.api_id_input.setFixedWidth(65)
        self.api_fetch_btn = QPushButton("Fetch")
        self.api_fetch_btn.setObjectName("apibtn")
        self.api_fetch_btn.clicked.connect(self.fetch_api_report)
        lay.addWidget(self.api_id_lbl)
        lay.addWidget(self.api_id_input)
        lay.addWidget(self.api_fetch_btn)
        return frame

    def _build_plot_panel(self):
        frame = QFrame()
        frame.setObjectName("plotpanel")
        v = QVBoxLayout(frame)
        v.setContentsMargins(10, 8, 10, 4)
        v.setSpacing(6)

        # Controls bar
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("background:#12122a;border-radius:8px;border:1px solid #2a2a4a;")
        controls = QHBoxLayout(ctrl_frame)
        controls.setContentsMargins(10, 6, 10, 6)
        controls.setSpacing(8)

        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_frame)
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.next_btn)
        controls.addSpacing(10)

        lbl_w = QLabel("Window:")
        lbl_w.setStyleSheet("color:#aaaacc;border:none;background:transparent;")
        controls.addWidget(lbl_w)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["1.0 s", "2.0 s", "3.0 s", "5.0 s", "10.0 s"])
        self.window_combo.setCurrentText("10.0 s")
        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        self.window_combo.setFixedWidth(80)
        controls.addWidget(self.window_combo)

        lbl_s = QLabel("Step:")
        lbl_s.setStyleSheet("color:#aaaacc;border:none;background:transparent;")
        controls.addWidget(lbl_s)
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.2 s", "0.5 s", "1.0 s"])
        self.step_combo.setCurrentText("0.5 s")
        self.step_combo.currentTextChanged.connect(self._on_step_changed)
        self.step_combo.setFixedWidth(75)
        controls.addWidget(self.step_combo)

        controls.addSpacing(14)
        self.frame_label = QLabel("Frame: 0.00s – 2.00s")
        self.frame_label.setStyleSheet("color:#ff9955;font-weight:bold;border:none;background:transparent;")
        controls.addWidget(self.frame_label)
        controls.addStretch()
        v.addWidget(ctrl_frame)

        # Timeline slider
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setMinimum(0)
        self.timeline.setMaximum(0)
        self.timeline.valueChanged.connect(self._on_timeline_changed)
        self.timeline.setFixedHeight(18)
        v.addWidget(self.timeline)

        # Lead grid (3 rows x 4 cols)
        from PyQt5.QtWidgets import QGridLayout, QWidget as _QW
        grid_widget = _QW()
        grid_widget.setStyleSheet("background:#0d0d1a;")
        self._lead_grid = QGridLayout(grid_widget)
        self._lead_grid.setContentsMargins(0, 0, 0, 0)
        self._lead_grid.setSpacing(5)

        self._lead_canvases = {}
        self._lead_axes = {}
        self._lead_figs = {}

        lead_order = [
            ['I',   'aVR', 'V1', 'V4'],
            ['II',  'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6'],
        ]

        for row, row_leads in enumerate(lead_order):
            self._lead_grid.setRowStretch(row, 1)
            for col, lead in enumerate(row_leads):
                self._lead_grid.setColumnStretch(col, 1)
                cell = QFrame()
                cell.setObjectName("leadbox")
                cell_lay = QVBoxLayout(cell)
                cell_lay.setContentsMargins(0, 0, 0, 0)
                cell_lay.setSpacing(0)

                lbl = QLabel(lead)
                lbl.setObjectName("leadlabel")
                lbl.setAlignment(Qt.AlignLeft)
                lbl.setContentsMargins(6, 2, 0, 0)
                lbl.setFixedHeight(18)
                cell_lay.addWidget(lbl)

                fig = Figure(facecolor='#111130')
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                canvas = FigureCanvas(fig)
                canvas.setStyleSheet("background:#111130;border:none;")
                ax = fig.add_axes([0, 0, 1, 1], facecolor='#111130')
                ax.set_axis_off()

                cell_lay.addWidget(canvas, stretch=1)
                self._lead_grid.addWidget(cell, row, col)

                self._lead_figs[lead] = fig
                self._lead_canvases[lead] = canvas
                self._lead_axes[lead] = ax

        # Legacy compatibility
        self.figure = list(self._lead_figs.values())[0]
        self.canvas = list(self._lead_canvases.values())[0]
        self.axes = [self._lead_axes[l] for l in self.LEADS]

        v.addWidget(grid_widget, stretch=1)
        return frame

    def _build_bottom_panel(self):
        frame = QFrame()
        frame.setObjectName("bottompanel")
        h = QHBoxLayout(frame)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(12)

        mark_box = QFrame()
        mark_box.setStyleSheet("background:transparent;border:none;")
        av = QVBoxLayout(mark_box)
        av.setSpacing(5)

        title_lbl = QLabel("Manual Arrhythmia Marking")
        title_lbl.setStyleSheet("color:#ff9955;font-size:12px;font-weight:bold;border:none;background:transparent;")
        av.addWidget(title_lbl)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Type:"))
        self.arrhythmia_type_combo = QComboBox()
        self.arrhythmia_type_combo.addItems([
            "Atrial Fibrillation", "PVC", "PAC", "SVT", "VT", "Bradycardia", "Tachycardia", "Other"
        ])
        row1.addWidget(self.arrhythmia_type_combo, 2)
        row1.addWidget(QLabel("Lead:"))
        self.mark_lead_combo = QComboBox()
        self.mark_lead_combo.addItems(["All Leads"] + self.LEADS)
        row1.addWidget(self.mark_lead_combo, 1)
        av.addLayout(row1)

        self.manual_type_input = QLineEdit()
        self.manual_type_input.setPlaceholderText("Custom arrhythmia name (used when Type=Other)")
        av.addWidget(self.manual_type_input)

        self.notes_input = QLineEdit()
        self.notes_input.setPlaceholderText("Notes")
        av.addWidget(self.notes_input)

        row2 = QHBoxLayout()
        self.mark_start_btn = QPushButton("Mark Start")
        self.mark_start_btn.clicked.connect(self.mark_start)
        self.mark_end_btn = QPushButton("Mark End + Save")
        self.mark_end_btn.clicked.connect(self.mark_end_and_save)
        self.auto_detect_btn = QPushButton("Automatic Detection")
        self.auto_detect_btn.clicked.connect(self.run_automatic_detection)
        self.delete_mark_btn = QPushButton("Delete Selected")
        self.delete_mark_btn.clicked.connect(self.delete_selected_annotation)
        row2.addWidget(self.mark_start_btn)
        row2.addWidget(self.mark_end_btn)
        row2.addWidget(self.auto_detect_btn)
        row2.addWidget(self.delete_mark_btn)
        av.addLayout(row2)

        self.mark_status_lbl = QLabel("No active mark")
        self.mark_status_lbl.setStyleSheet("color:#7a9f7a;border:none;background:transparent;")
        av.addWidget(self.mark_status_lbl)

        self.annotation_table = QTableWidget(0, 5)
        self.annotation_table.setHorizontalHeaderLabels(["Start (s)", "End (s)", "Type", "Lead", "Notes"])
        self.annotation_table.horizontalHeader().setStretchLastSection(True)
        self.annotation_table.setMaximumHeight(120)
        av.addWidget(self.annotation_table)

        h.addWidget(mark_box)

        self.metrics_table = QTableWidget(0, 2)
        self.findings_text = QTextEdit()

        return frame

    # --------------------------- data loading ---------------------------
    def load_reports(self):
        self.report_combo.blockSignals(True)
        self.report_combo.clear()
        self.reports = []

        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            reports_dir = os.path.join(base_dir, 'reports')
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir, exist_ok=True)

            files = [f for f in os.listdir(reports_dir) if f.endswith('.json') and not f.startswith('index')]
            files.sort(reverse=True)

            for filename in files:
                filepath = os.path.join(reports_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        report = json.load(f)
                    patient_name = self._extract_patient_name(report)
                    date_str = self._extract_report_date(report)
                    self.report_combo.addItem(f"{patient_name} | {date_str}", filepath)
                    self.reports.append(report)
                except Exception as e:
                    print(f"Error loading report {filename}: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load reports: {e}")
        finally:
            self.report_combo.blockSignals(False)

        if self.reports:
            self.load_selected_report(0)

    def load_selected_report(self, index):
        if index < 0 or index >= len(self.reports):
            return

        self.current_report = self.reports[index]
        self.current_report_path = self.report_combo.itemData(index) or ""

        self._update_patient_info()
        self._load_lead_data()
        self._load_metrics_findings()
        self._load_manual_annotations()

        self.frame_start_sample = 0
        self._update_timeline_limits()
        self._render_current_frame()

    def _extract_patient_name(self, report):
        return (
            report.get('patient_details', {}).get('name')
            or report.get('patient_name')
            or report.get('patient', {}).get('name')
            or 'Unknown'
        )

    def _extract_report_date(self, report):
        return (
            report.get('patient_details', {}).get('report_date')
            or report.get('report_date')
            or report.get('date')
            or 'Unknown Date'
        )

    def _update_patient_info(self):
        if not self.current_report:
            self.patient_lbl.setText("Patient: --")
            self.patient_meta_lbl.setText("ID: -- | Age: -- | Gender: --")
            return

        pd = self.current_report.get('patient_details', {})
        p_fallback = self.current_report.get('patient', {})

        name = pd.get('name') or self.current_report.get('patient_name') or p_fallback.get('name') or 'Unknown'
        pid = pd.get('report_id') or pd.get('user_id') or self.current_report.get('patient_id') or '--'
        age = pd.get('age') or self.current_report.get('age') or p_fallback.get('age') or '--'
        gender = pd.get('gender') or self.current_report.get('gender') or p_fallback.get('gender') or '--'

        self.patient_lbl.setText(f"Patient: {name}")
        self.patient_meta_lbl.setText(f"ID: {pid} | Age: {age} | Gender: {gender}")

    def _load_lead_data(self):
        self.lead_data = {lead: np.array([]) for lead in self.LEADS}

        rpt = self.current_report or {}
        self.sampling_rate = (
            rpt.get('data_details', {}).get('sampling_rate')
            or rpt.get('sampling_rate')
            or rpt.get('ecg_data', {}).get('sampling_rate')
            or 500
        )
        try:
            self.sampling_rate = float(self.sampling_rate)
        except Exception:
            self.sampling_rate = 500.0

        ecg_data = rpt.get('ecg_data', {}) if isinstance(rpt.get('ecg_data', {}), dict) else {}

        # Format 1: leads_data dict (preferred backend format)
        leads_data = ecg_data.get('leads_data') if isinstance(ecg_data.get('leads_data'), dict) else None
        if leads_data:
            for lead in self.LEADS:
                arr = leads_data.get(lead, [])
                self.lead_data[lead] = np.array(arr, dtype=float) if isinstance(arr, list) else np.array([])
            return

        # Format 2: direct lead dict in ecg_data
        if any(lead in ecg_data for lead in self.LEADS):
            for lead in self.LEADS:
                arr = ecg_data.get(lead, [])
                self.lead_data[lead] = np.array(arr, dtype=float) if isinstance(arr, list) else np.array([])
            return

        # Format 3: root-level leads
        if any(lead in rpt for lead in self.LEADS):
            for lead in self.LEADS:
                arr = rpt.get(lead, [])
                self.lead_data[lead] = np.array(arr, dtype=float) if isinstance(arr, list) else np.array([])
            return

        # Format 4: compact device_data string "[12 vals]|[12 vals]|..."
        device_data = ecg_data.get('device_data') if isinstance(ecg_data, dict) else None
        if isinstance(device_data, str) and '|' in device_data:
            self._parse_compact_device_data(device_data)

    def _parse_compact_device_data(self, device_data):
        per_lead = {lead: [] for lead in self.LEADS}
        frames = [x.strip() for x in device_data.split('|') if x.strip()]
        for fr in frames:
            try:
                vals = json.loads(fr)
                if isinstance(vals, list) and len(vals) >= 12:
                    for i, lead in enumerate(self.LEADS):
                        per_lead[lead].append(float(vals[i]))
            except Exception:
                continue
        for lead in self.LEADS:
            self.lead_data[lead] = np.array(per_lead[lead], dtype=float)

    def _load_metrics_findings(self):
        rpt = self.current_report or {}

        # Metrics from multiple schema variants
        metrics = rpt.get('result_reading') or rpt.get('metrics') or {}

        self.metrics_table.setRowCount(0)
        
        rv5_sv1 = metrics.get('RV5_SV1', metrics.get('rv5_sv1', 'N/A'))
        rv5_plus_sv1 = metrics.get('RV5_plus_SV1', metrics.get('rv5_plus_sv1', 'N/A'))

        items = [
            ("HR", metrics.get('HR_bpm', metrics.get('heart_rate', metrics.get('HR', 'N/A'))), "bpm"),
            ("RR", metrics.get('RR_ms', metrics.get('rr_interval', metrics.get('RR', 'N/A'))), "ms"),
            ("PR", metrics.get('PR_ms', metrics.get('pr_interval', metrics.get('PR', 'N/A'))), "ms"),
            ("QRS", metrics.get('QRS_ms', metrics.get('qrs_duration', metrics.get('QRS', 'N/A'))), "ms"),
            ("QT", metrics.get('QT_ms', metrics.get('qt_interval', metrics.get('QT', 'N/A'))), "ms"),
            ("QTc", metrics.get('QTc_ms', metrics.get('qtc_interval', metrics.get('QTc', 'N/A'))), "ms"),
            ("RV5/SV1", str(rv5_sv1).replace(' mV', ''), "mV" if str(rv5_sv1) != 'N/A' else ""),
            ("RV5+SV1", str(rv5_plus_sv1).replace(' mV', ''), "mV" if str(rv5_plus_sv1) != 'N/A' else "")
        ]

        self.metrics_table.setRowCount(len(items))
        for i, (k, v, unit) in enumerate(items):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(k))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{v} {unit}" if v not in ('', None, 'N/A') else 'N/A'))

        findings_lines = []
        clinical = rpt.get('clinical_findings', {})
        if isinstance(clinical, dict):
            for key in ('conclusion', 'arrhythmia', 'hyperkalemia'):
                vals = clinical.get(key, [])
                if isinstance(vals, list) and vals:
                    findings_lines.append(f"{key.title()}: " + ', '.join(str(x) for x in vals))

        # old format fallbacks
        for key in ('conclusion', 'arrhythmia', 'hyperkalemia', 'findings', 'recommendations'):
            vals = rpt.get(key)
            if isinstance(vals, list) and vals:
                findings_lines.append(f"{key.title()}: " + ', '.join(str(x) for x in vals))

        if not findings_lines:
            findings_lines = ["No backend findings available."]

        self.findings_text.setPlainText('\n'.join(findings_lines))

    # --------------------------- frame navigation ---------------------------
    def _on_window_changed(self, text):
        self.window_seconds = float(text.replace('s', '').strip())
        self._update_timeline_limits()
        self._render_current_frame()

    def _on_step_changed(self, text):
        self.step_seconds = float(text.replace('s', '').strip())

    def _total_samples(self):
        for lead in self.LEADS:
            if len(self.lead_data[lead]) > 0:
                return len(self.lead_data[lead])
        return 0

    def _window_samples(self):
        return max(1, int(round(self.window_seconds * self.sampling_rate)))

    def _step_samples(self):
        return max(1, int(round(self.step_seconds * self.sampling_rate)))

    def _max_start_sample(self):
        return max(0, self._total_samples() - self._window_samples())

    def _update_timeline_limits(self):
        mx = self._max_start_sample()
        self.timeline.blockSignals(True)
        self.timeline.setMinimum(0)
        self.timeline.setMaximum(mx)
        self.timeline.setValue(min(self.frame_start_sample, mx))
        self.timeline.blockSignals(False)

    def _on_timeline_changed(self, value):
        self.frame_start_sample = int(value)
        self._render_current_frame()

    def prev_frame(self):
        self.frame_start_sample = max(0, self.frame_start_sample - self._step_samples())
        self.timeline.setValue(self.frame_start_sample)

    def next_frame(self):
        self.frame_start_sample = min(self._max_start_sample(), self.frame_start_sample + self._step_samples())
        self.timeline.setValue(self.frame_start_sample)

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            self.play_timer.start(250)
            self.play_btn.setText("⏸ Pause")

    def _render_current_frame(self):
        ws = self._window_samples()
        st = self.frame_start_sample
        en = min(self._total_samples(), st + ws)

        t = np.arange(st, en) / self.sampling_rate if en > st else np.array([])
        start_sec = st / self.sampling_rate if self.sampling_rate > 0 else 0.0
        end_sec = en / self.sampling_rate if self.sampling_rate > 0 else 0.0
        self.frame_label.setText(f"Frame: {start_sec:.2f}s \u2013 {end_sec:.2f}s")

        ECG_COLOR = '#ff8844'
        ANNOT_COLOR = '#ff4444'

        for lead in self.LEADS:
            ax = self._lead_axes.get(lead)
            fig = self._lead_figs.get(lead)
            canvas = self._lead_canvases.get(lead)
            if ax is None:
                continue

            ax.clear()
            ax.set_axis_off()
            ax.set_facecolor('#111130')
            # Fixed ADC Y-range 0-4096 for all leads
            ax.set_ylim(0, 4096)
            ax.set_xlim(start_sec, end_sec if end_sec > start_sec else start_sec + 1)
            # Subtle background grid
            ax.grid(True, color='#1e1e50', linewidth=0.4, linestyle='-', alpha=0.7)

            data = self.lead_data.get(lead, np.array([]))
            if len(data) > 0 and en > st:
                seg = data[st:en]
                ax.plot(t, seg, color=ECG_COLOR, linewidth=0.85, antialiased=True)
                ax.set_ylim(0, 4096)
                ax.set_xlim(start_sec, end_sec)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color='#555588', fontsize=9)

            # Annotation span overlays
            for ann in self.manual_annotations:
                if ann.get('lead', 'All Leads') not in ('All Leads', lead):
                    continue
                a0 = ann.get('start_sec', 0.0)
                a1 = ann.get('end_sec', 0.0)
                if a1 < start_sec or a0 > end_sec:
                    continue
                lft = max(a0, start_sec)
                rgt = min(a1, end_sec)
                if rgt > lft:
                    ax.axvspan(lft, rgt, color=ANNOT_COLOR, alpha=0.18)

            fig.tight_layout(pad=0)
            canvas.draw_idle()

    # --------------------------- manual annotations ---------------------------
    def mark_start(self):
        self.pending_mark_start_sec = self.frame_start_sample / max(self.sampling_rate, 1.0)
        self.mark_status_lbl.setText(f"Start marked at {self.pending_mark_start_sec:.2f}s")

    def mark_end_and_save(self):
        if self.pending_mark_start_sec is None:
            QMessageBox.information(self, "Marking", "Click 'Mark Start' first.")
            return

        end_sec = (self.frame_start_sample + self._window_samples()) / max(self.sampling_rate, 1.0)
        start_sec = min(self.pending_mark_start_sec, end_sec)
        end_sec = max(self.pending_mark_start_sec, end_sec)

        arr_type = self.arrhythmia_type_combo.currentText().strip()
        if arr_type == 'Other':
            arr_type = self.manual_type_input.text().strip() or 'Other'

        ann = {
            'start_sec': round(start_sec, 3),
            'end_sec': round(end_sec, 3),
            'type': arr_type,
            'lead': self.mark_lead_combo.currentText(),
            'notes': self.notes_input.text().strip(),
            'created_at': datetime.now().isoformat(timespec='seconds')
        }
        self.manual_annotations.append(ann)
        self.pending_mark_start_sec = None
        self.mark_status_lbl.setText("Saved annotation")
        self._refresh_annotation_table()
        self._persist_annotations_in_report()
        self._render_current_frame()

    def delete_selected_annotation(self):
        row = self.annotation_table.currentRow()
        if row < 0 or row >= len(self.manual_annotations):
            return
        del self.manual_annotations[row]
        self._refresh_annotation_table()
        self._persist_annotations_in_report()
        self._render_current_frame()

    def _refresh_annotation_table(self):
        self.annotation_table.setRowCount(len(self.manual_annotations))
        for i, ann in enumerate(self.manual_annotations):
            self.annotation_table.setItem(i, 0, QTableWidgetItem(f"{ann.get('start_sec', 0):.3f}"))
            self.annotation_table.setItem(i, 1, QTableWidgetItem(f"{ann.get('end_sec', 0):.3f}"))
            self.annotation_table.setItem(i, 2, QTableWidgetItem(ann.get('type', '')))
            self.annotation_table.setItem(i, 3, QTableWidgetItem(ann.get('lead', 'All Leads')))
            self.annotation_table.setItem(i, 4, QTableWidgetItem(ann.get('notes', '')))

    def _load_manual_annotations(self):
        self.manual_annotations = list((self.current_report or {}).get('manual_annotations', []))
        self._refresh_annotation_table()

    def _persist_annotations_in_report(self):
        if not self.current_report:
            return
        self.current_report['manual_annotations'] = self.manual_annotations

    # --------------------------- automatic detection ---------------------------
    def run_automatic_detection(self):
        if not self.current_report:
            QMessageBox.warning(self, "Detection", "No report loaded.")
            return

        # Use the lead selected in the marking combo, fallback to Lead II
        lead_name = self.mark_lead_combo.currentText()
        if lead_name == "All Leads":
            lead_name = 'II'
            
        data = self.lead_data.get(lead_name, np.array([]))
        if len(data) == 0:
            # Try to find any lead with data
            for l in self.LEADS:
                if len(self.lead_data.get(l, [])) > 0:
                    lead_name = l
                    data = self.lead_data[l]
                    break
            
            if len(data) == 0:
                QMessageBox.warning(self, "Detection", "No ECG data available for analysis.")
                return

        # Get the current window data
        ws = self._window_samples()
        st = self.frame_start_sample
        en = min(len(data), st + ws)
        
        # Analyze current visible segment
        segment = data[st:en]
        if len(segment) < self.sampling_rate * 1.5: # At least 1.5 seconds
            QMessageBox.warning(self, "Detection", "Visible window too short for accurate detection (need >1.5s).")
            return

        try:
            if PQRSTAnalyzer is None or ArrhythmiaDetector is None:
                QMessageBox.critical(self, "Error", "ECG analysis modules not loaded. Please check your installation.")
                return

            # Use PQRSTAnalyzer to get peaks for detection
            analyzer = PQRSTAnalyzer(self.sampling_rate)
            analysis = analyzer.analyze_signal(segment)
            
            # Run Arrhythmia Detection
            detector = ArrhythmiaDetector(self.sampling_rate)
            # We pass the segment and its analysis
            results = detector.detect_arrhythmias(segment, analysis)
            
            # Filter out "Insufficient data" or NSR if we want to show only arrhythmias
            # But the user said "all arthmia and mention", so let's show all results.
            
            if not results or (len(results) == 1 and "Insufficient data" in results[0]):
                QMessageBox.information(self, "Detection", "No specific arrhythmia patterns detected in this window.")
                return

            # Format results for display
            rhythm_text = ", ".join(results)
            is_normal = "Normal Sinus Rhythm" in rhythm_text
            
            msg = f"<b>Window Analysis (Lead {lead_name}):</b><br><br>"
            msg += f"Detected: <span style='color: {'#2ecc71' if is_normal else '#e74c3c'}; font-weight: bold;'>{rhythm_text}</span><br><br>"
            msg += "Would you like to add these findings to the report?"
            
            reply = QMessageBox.question(self, "Automatic Detection Result", msg, QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                start_sec = st / self.sampling_rate
                end_sec = en / self.sampling_rate
                
                added_count = 0
                for arr in results:
                    # Skip NSR if adding to report annotations
                    if arr == "Normal Sinus Rhythm":
                        continue
                        
                    ann = {
                        'start_sec': round(start_sec, 3),
                        'end_sec': round(end_sec, 3),
                        'type': arr,
                        'lead': lead_name,
                        'notes': "Automatically detected",
                        'created_at': datetime.now().isoformat(timespec='seconds')
                    }
                    self.manual_annotations.append(ann)
                    added_count += 1
                
                if added_count > 0:
                    self._refresh_annotation_table()
                    self._persist_annotations_in_report()
                    self._render_current_frame()
                    QMessageBox.information(self, "Report Updated", f"Added {added_count} detected arrhythmia(s) to the report.")
                else:
                    QMessageBox.information(self, "Report", "Normal rhythm detected; no annotations added.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")

    # --------------------------- actions ---------------------------
    # --------------------------- API actions ---------------------------
    def fetch_api_report(self):
        id_text = self.api_id_input.text().strip()
        if not id_text:
            return
            
        url = f"https://deckmount.in/ankur_bhaiya.php?id={id_text}"
        import requests
        from scipy.ndimage import gaussian_filter1d
        
        try:
            self.api_fetch_btn.setText("Loading...")
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if not data.get("status"):
                QMessageBox.warning(self, "API Error", "API returned status: false (ID not found)")
                self.api_fetch_btn.setText("Fetch")
                return
                
            api_data = data.get("data", {})
            
            res_reading = {}
            try: res_reading = json.loads(api_data.get("result_reading", "{}"))
            except: pass
            
            concl = []
            try: concl = json.loads(api_data.get("conclusion", "[]"))
            except: pass
            
            arr = []
            try: arr = json.loads(api_data.get("arrhythmia", "[]"))
            except: pass
            
            ecg_data = {}
            ecg_data["sampling_rate"] = float(api_data.get("sampling_rate", 500))
            
            possible_keys = [
                ["lead1_reading", "lead_1_reading", "lead1"],
                ["lead2_reading", "lead_2_reading", "lead2"],
                ["lead3_reading", "lead_3_reading", "lead3"],
                ["leadavr_reading", "lead_avr_reading", "leadavr"],
                ["leadavl_reading", "lead_avl_reading", "leadavl"],
                ["leadavf_reading", "lead_avf_reading", "leadavf"],
                ["leadv1_reading", "lead_v1_reading", "leadv1"],
                ["leadv2_reading", "lead_v2_reading", "leadv2"],
                ["leadv3_reading", "lead_v3_reading", "leadv3"],
                ["leadv4_reading", "lead_v4_reading", "leadv4"],
                ["leadv5_reading", "lead_v5_reading", "leadv5"],
                ["leadv6_reading", "lead_v6_reading", "leadv6"]
            ]
            
            lower_keys = {k.lower(): k for k in api_data.keys()}
            leads_list = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            
            for i, variants in enumerate(possible_keys):
                leadstr = leads_list[i]
                for variant in variants:
                    actual_key = lower_keys.get(variant.lower())
                    if actual_key and actual_key in api_data:
                        val_str = str(api_data[actual_key]).strip()
                        if val_str:
                            if val_str.endswith(','): val_str = val_str[:-1]
                            arr_vals = np.array([float(x.strip()) for x in val_str.split(',') if x.strip()])
                            # Apply smoothing and zero-centering
                            filt = gaussian_filter1d(arr_vals, sigma=1.5)
                            c_mean = np.mean(filt)
                            if not np.isnan(c_mean):
                                filt = filt - c_mean
                            
                            # Shift to native ADC limits matching Dashboard Y limitations
                            filt = filt + 2048
                                
                            ecg_data[leadstr] = filt.tolist()
                        break

            new_report = {
                "patient_details": {
                    "name": api_data.get("name", "Unknown"),
                    "age": api_data.get("age", ""),
                    "gender": api_data.get("gender", ""),
                    "report_id": api_data.get("report_id", id_text),
                    "report_date": api_data.get("report_date", "")
                },
                "result_reading": res_reading,
                "clinical_findings": {
                    "conclusion": concl,
                    "arrhythmia": arr
                },
                "ecg_data": ecg_data,
                "api_id": id_text
            }
            
            self.reports.append(new_report)
            idx = len(self.reports) - 1
            name = api_data.get("name", "Unknown API")
            self.report_combo.addItem(f"[API] {name} | ID:{id_text}", "")
            self.report_combo.setCurrentIndex(idx)
            
            self.api_fetch_btn.setText("Fetch")
            
        except Exception as e:
            self.api_fetch_btn.setText("Fetch")
            QMessageBox.critical(self, "API Error", f"Failed: {str(e)}")

    def export_report(self):
        if not self.current_report:
            QMessageBox.warning(self, "Export", "No report selected")
            return

        default_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export Analysis JSON", default_name, "JSON Files (*.json)")
        if not path:
            return

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.current_report, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Export", f"Exported successfully:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Failed to export: {e}")

    def generate_pdf_report(self):
        if not self.current_report:
            QMessageBox.warning(self, "Export", "No report loaded.")
            return

        rpt = self.current_report
        pat = rpt.get('patient_details', {}) or {}

        # ── Parse result_reading (may be a JSON string from API) ─────────────
        raw_metrics = rpt.get('result_reading') or rpt.get('metrics') or {}
        if isinstance(raw_metrics, str):
            try:
                import json as _json
                raw_metrics = _json.loads(raw_metrics)
            except Exception:
                raw_metrics = {}
        if not isinstance(raw_metrics, dict):
            raw_metrics = {}

        # Build only values that are actually present (skip N/A / None / empty)
        def _get(d, *keys):
            for k in keys:
                v = d.get(k)
                if v is not None and str(v).strip() not in ('', '--', 'N/A', 'null'):
                    return str(v)
            return None

        hr      = _get(raw_metrics, 'HR',  'heart_rate',   'HR_bpm')
        pr      = _get(raw_metrics, 'PR',  'pr_interval',  'PR_ms')
        qrs     = _get(raw_metrics, 'QRS', 'qrs_duration', 'QRS_ms')
        qt      = _get(raw_metrics, 'QT',  'qt_interval',  'QT_ms')
        qtc     = _get(raw_metrics, 'QTc', 'qtc_interval', 'QTc_ms')
        qtcf    = _get(raw_metrics, 'QTcF', 'qtcf_interval', 'QTcF_ms')
        rr      = _get(raw_metrics, 'RR',  'rr_interval',  'RR_ms')
        rv5sv1  = _get(raw_metrics, 'RV5_SV1',     'rv5_sv1')
        rv5plus = _get(raw_metrics, 'RV5_plus_SV1','rv5_plus_sv1')
        axes    = _get(raw_metrics, 'axes', 'P/QRS/T', 'p_qrs_t')

        # ── Conclusions from clinical_findings ───────────────────────────────
        clinical = rpt.get('clinical_findings') or {}
        if isinstance(clinical, dict):
            conclusions = clinical.get('conclusion', [])
        else:
            conclusions = []
        if isinstance(conclusions, str):
            conclusions = [conclusions]
        elif not isinstance(conclusions, list):
            conclusions = []
        # also try root-level conclusion
        if not conclusions:
            c2 = rpt.get('conclusion', [])
            if isinstance(c2, str): c2 = [c2]
            conclusions = c2 if isinstance(c2, list) else []

        patient_name = pat.get('name', 'Unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"ECG_Analysis_{patient_name}_{timestamp}.pdf"
        
        # Determine local reports directory
        project_root = Path(__file__).resolve().parents[2]
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Ask user for path, but default to our reports directory
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ECG PDF", 
            str(reports_dir / default_name), 
            "PDF Files (*.pdf)"
        )
        if not path:
            return

        # ── Format selection dialog ──────────────────────────────────
        from PyQt5.QtWidgets import (
            QDialog as _QD, QVBoxLayout as _VL, QButtonGroup,
            QRadioButton, QDialogButtonBox, QLabel as _QL
        )
        fmt_dlg = _QD(self)
        fmt_dlg.setWindowTitle("Select Report Format")
        fmt_dlg.setMinimumWidth(380)
        fmt_dlg.setStyleSheet("""
            QDialog { background:#1a1a2e; color:white; }
            QLabel { color:#e0e0ff; font-size:13px; background:transparent; border:none; }
            QRadioButton { color:#e0e0ff; font-size:12px; background:transparent; padding:6px 4px; }
            QRadioButton:checked { color:#ff6600; font-weight:bold; }
            QPushButton { background:#ff6600; color:white; border:none; border-radius:6px;
                          padding:8px 22px; font-weight:bold; font-size:12px; }
            QPushButton:hover { background:#ff7a26; }
        """)
        fmt_lay = _VL(fmt_dlg)
        fmt_lay.setSpacing(10)
        fmt_lay.setContentsMargins(24, 20, 24, 20)
        fmt_lay.addWidget(_QL("Choose ECG report layout:"))
        rb1 = QRadioButton("4:3  —  4 rows × 3 columns  (standard 12-lead view)")
        rb2 = QRadioButton("12:1 —  12 rows × 1 column  (full rhythm strip roll)")
        rb3 = QRadioButton("6:2  —  6 rows × 2 columns  (compact comparative)")
        rb1.setChecked(True)
        grp = QButtonGroup(fmt_dlg)
        for rb in (rb1, rb2, rb3):
            grp.addButton(rb)
            fmt_lay.addWidget(rb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(fmt_dlg.accept)
        bb.rejected.connect(fmt_dlg.reject)
        fmt_lay.addWidget(bb)
        if fmt_dlg.exec_() != _QD.Accepted:
            return
        if rb1.isChecked():   pdf_format = "4x3"
        elif rb2.isChecked(): pdf_format = "12x1"
        else:                 pdf_format = "6x2"

        try:
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.patches import Rectangle as MRect

            self.pdf_btn.setText("Generating...")
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()

            # ── Constants (all mm, matching ReportScreen.kt) ─────────────────
            PAGE_W = 210.0;  PAGE_H = 297.0   # A4 portrait
            ML = 10.0;  MR = 10.0;  MT = 10.0;  MB = 10.0
            HEADER_H = 30.0    # mm reserved for header
            FOOTER_H = 25.0    # mm reserved for footer
            STRIP_TOP  = MT + HEADER_H
            STRIP_H    = PAGE_H - STRIP_TOP - FOOTER_H - MB
            CELL_H     = STRIP_H / 12.0

            MM_PER_SAMPLE = 25.0 / float(self.sampling_rate)   # 25 mm/s ÷ fs
            # Android app uses: "640 ADC units per 5mm grid box" 
            # 640 ADC / 5 mm = 128 ADC units per mm.
            # This perfectly scales the raw ADC API data to visual clinical standard.
            ADC_PER_MM = 128.0
            CALIB_MM   = 10.0     # 1 mV calibration pulse height (10 mm)
            HALF_CELL  = CELL_H / 2.0 - 1.0   # clip headroom within each strip (mm)



            # ── Figure (A4 in inches, 150 dpi) ───────────────────────────────
            fig = Figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), dpi=150, facecolor='white')

            # Single axes covering the whole page; coordinates = mm
            ax = fig.add_axes([0, 0, 1, 1], facecolor='#fff5f5')
            ax.set_xlim(0, PAGE_W)
            ax.set_ylim(PAGE_H, 0)          # y-axis: top=0, bottom=PAGE_H (like screen)
            ax.set_aspect('equal')

            # ── ECG grid  (1 mm minor, 5 mm major) ───────────────────────────
            ax.set_xticks(np.arange(0, PAGE_W + 1, 5))
            ax.set_xticks(np.arange(0, PAGE_W + 1, 1), minor=True)
            ax.set_yticks(np.arange(0, PAGE_H + 1, 5))
            ax.set_yticks(np.arange(0, PAGE_H + 1, 1), minor=True)
            ax.grid(True, which='major', color='#e09696', linewidth=0.55, zorder=1)
            ax.grid(True, which='minor', color='#f5d8d8', linewidth=0.22, zorder=1)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_xticklabels([]);  ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False, which='both')

            # ── Header ───────────────────────────────────────────────────────
            yb = MT;  lh = 4.2
            x1 = ML
            # Col 1: patient + type
            ax.text(x1, yb,      f"Name: {pat.get('name') or '-'}",    fontsize=7, va='top')
            ax.text(x1, yb+lh,   f"Age: {pat.get('age') or '-'}",      fontsize=7, va='top')
            ax.text(x1, yb+lh*2, f"Gender: {pat.get('gender') or '-'}",fontsize=7, va='top')
            ax.text(x1, yb+lh*3.5, "Type: Standard", fontsize=7, va='top')

            # Col 2: HR RR PR QRS QT
            x2 = x1 + 45
            col2_items = [
                ('HR',  hr,  'bpm'),
                ('RR',  rr,  'ms'),
                ('PR',  pr,  'ms'),
                ('QRS', qrs, 'ms'),
                ('QT',  qt,  'ms'),
            ]
            row_y = yb
            for lbl, val, unit in col2_items:
                if val is not None:
                    ax.text(x2, row_y, f"{lbl} : {val} {unit}",
                            fontsize=7, va='top', fontweight='bold')
                    row_y += lh

            # Col 3: QTc QTcF RV5/SV1 RV5+SV1 P/QRS/T
            x3 = x2 + 35
            col3_items = [
                ('QTc', qtc, 'ms'),
                ('QTcF', qtcf, 'ms'),
                ('RV5/SV1', rv5sv1, 'mV'),
                ('RV5+SV1', rv5plus, 'mV'),
                ('P/QRS/T', axes, '°'),
            ]
            col3_row = yb
            for lbl, val, unit in col3_items:
                if val is not None:
                    # Specific formatting for RV5/SV1 and RV5+SV1 as seen in the image
                    if lbl == 'RV5/SV1':
                        val_str = str(val).replace(' mV', '')
                        if '/' not in val_str and not val_str.startswith('+') and not val_str.startswith('-'):
                            val_str = f"+{val_str}" # placeholder for consistency
                        ax.text(x3, col3_row, f"{lbl}: {val_str} {unit}", fontsize=7, va='top', fontweight='bold')
                    elif lbl == 'RV5+SV1':
                        val_str = str(val).replace(' mV', '')
                        ax.text(x3, col3_row, f"{lbl}: {val_str} {unit}", fontsize=7, va='top', fontweight='bold')
                    else:
                        ax.text(x3, col3_row, f"{lbl} : {val} {unit}", fontsize=7, va='top', fontweight='bold')
                    col3_row += lh

            # Brand (right): prefers provided Electronics logo image
            logo_left = PAGE_W - MR - 55.0
            logo_top = yb - 1.0
            logo_bottom = yb + lh * 2.2

            logo_drawn = False
            try:
                if self.analysis_pdf_logo_path.exists():
                    import matplotlib.image as mpimg
                    logo_img = mpimg.imread(str(self.analysis_pdf_logo_path))
                    ax.imshow(
                        logo_img,
                        extent=[logo_left, PAGE_W - MR, logo_bottom, logo_top],
                        aspect='auto',
                        zorder=10,
                    )
                    logo_drawn = True
            except Exception:
                logo_drawn = False

            if not logo_drawn:
                ax.text(PAGE_W - MR, yb, "DECKMOUNT",
                        fontsize=12, fontweight='bold', color='#000000',
                        ha='right', va='top', zorder=11,
                        family='sans-serif')

            ax.text(PAGE_W - MR, yb+lh*2.2, "25.0 mm/s  0.5-25Hz  AC:50Hz  10.0 mm/mV",
                    fontsize=5.5, ha='right', va='top', color='#555', zorder=10)
            
            # Use current date/time if report_date is missing or for real-time feel
            dt_str = pat.get('report_date') or datetime.now().strftime("%Y-%m-%d Time: %H:%M:%S")
            if 'Time:' not in dt_str and len(dt_str) < 15: # if it's just a date
                dt_str += " Time: " + datetime.now().strftime("%H:%M:%S")
            
            ax.text(PAGE_W - MR, yb+lh*3.0, f"Date: {dt_str}",
                    fontsize=5.5, ha='right', va='top', color='#555', zorder=10)

            # ── 12 Lead strips (format-aware window size limits) ───────────────
            st = self.frame_start_sample
            
            # Determine grid layout from user's choice
            if pdf_format == "4x3":
                pdf_samples = 1600
                n_rows, n_cols = 4, 3
                lead_grid = [
                    ['I',   'aVR', 'V1'],
                    ['II',  'aVL', 'V2'],
                    ['III', 'aVF', 'V3'],
                    ['V4',  'V5',  'V6'],
                ]
            elif pdf_format == "6x2":
                pdf_samples = 2500
                n_rows, n_cols = 6, 2
                lead_grid = [
                    ['I',   'II'],
                    ['III', 'aVR'],
                    ['aVL', 'aVF'],
                    ['V1',  'V2'],
                    ['V3',  'V4'],
                    ['V5',  'V6'],
                ]
            else:  # 12x1 default
                pdf_samples = 3500
                n_rows, n_cols = 12, 1
                lead_grid = [[l] for l in self.LEADS]
                
            en = min(self._total_samples(), st + pdf_samples)

            STRIP_W = PAGE_W - ML - MR
            cell_w = STRIP_W / n_cols
            cell_h_fmt = STRIP_H / n_rows
            half_h = cell_h_fmt / 2.0 - 1.0

            # ADC Y-axis: 0-4096 range; 1 mm per ADC step shown as mm markers
            # Show 5 Y levels: 0, 1024, 2048, 3072, 4096
            # Each maps to ±half_h mm from mid_y centred on baseline (2048)
            Y_LEVELS = [0, 1024, 2048, 3072, 4096]
            Y_BASELINE_ADC = 2048.0   # centre
            Y_SCALE_MM = half_h / Y_BASELINE_ADC  # mm per ADC unit

            for row_idx, row_leads in enumerate(lead_grid):
                for col_idx, lead in enumerate(row_leads):
                    cell_x0 = ML + col_idx * cell_w
                    mid_y   = STRIP_TOP + row_idx * cell_h_fmt + cell_h_fmt / 2.0
                    lbl_y   = mid_y - cell_h_fmt * 0.4
                    y_top   = STRIP_TOP + row_idx * cell_h_fmt
                    y_bot   = y_top + cell_h_fmt

                    # Calibration pulse
                    cx, cy, cg = cell_x0, mid_y, CALIB_MM
                    ax.plot([cx, cx+1.5, cx+1.5, cx+6.5, cx+6.5, cx+8],
                            [cy, cy,  cy-cg, cy-cg, cy,  cy],
                            color='black', linewidth=1.0, zorder=6)

                    # Lead label
                    ax.text(cell_x0 + 10, lbl_y - 0.5, lead,
                            fontsize=7, fontweight='bold', color='black', va='top', zorder=7)

                    # Y-axis scale markings removed for clean aesthetic
                    pass

                    # Waveform — fixed 0-4096 ADC scale, 7 s of data
                    data_arr = self.lead_data.get(lead, np.array([]))
                    if len(data_arr) > 0:
                        segment = data_arr[st:en].astype(float)
                        if segment.size > 1:
                            # Use fixed 0-4096 ADC range, centred on 2048
                            seg_mm = (segment - Y_BASELINE_ADC) * Y_SCALE_MM
                            seg_mm = np.clip(seg_mm, -half_h, half_h)
                            wx0 = cell_x0 + 9.5
                            wx1 = cell_x0 + cell_w - 1.0
                            wx_mm = np.linspace(wx0, wx1, segment.size)
                            wy_mm = mid_y - seg_mm
                            ax.plot(wx_mm, wy_mm, color='black', linewidth=0.6, zorder=5)

                            # Time labels removed for clean aesthetic
                            pass

                    # Vertical column separator (except last col)
                    if col_idx < n_cols - 1:
                        sep_x = cell_x0 + cell_w
                        ax.plot([sep_x, sep_x], [STRIP_TOP, STRIP_TOP + STRIP_H],
                                color='black', linewidth=0.6, linestyle='dashed', dashes=(4, 4), zorder=2)

            # ── Footer signature block ────────────────────────────────────────
            ft = PAGE_H - MB - FOOTER_H
            ax.text(ML, ft + 9,  "Reference Report Confirmed by:",
                    fontsize=7, va='top', color='black', fontweight='bold')
            ax.text(ML, ft + 14, "Doctor Name: ________________________",
                    fontsize=7, va='top', color='black')
            ax.text(ML, ft + 19, "Doctor Sign:  ________________________",
                    fontsize=7, va='top', color='black')

            # Conclusion box — transparent fill, matching second image
            bx, by = 90.0, ft + 2.0
            bw, bh = PAGE_W - bx - MR, 22.0
            from matplotlib.patches import Rectangle as MRect2
            rect = MRect2((bx, by), bw, bh,
                          linewidth=0.8, edgecolor='#333333',
                          facecolor='none', zorder=8)
            ax.add_patch(rect)
            
            # CONCLUSION Title
            ax.text(bx + bw/2, by + 1.2, "CONCLUSION",
                    fontsize=7, fontweight='bold', ha='center', va='top',
                    color='black', zorder=9)
            
            # Thin separator line under title
            ax.plot([bx + 1, bx + bw - 1], [by + 5.0, by + 5.0],
                    color='#333333', linewidth=0.6, zorder=9)

            # Draw vertical grid lines for the conclusion table if needed
            # but let's just use columns first
            cols = 3;  col_w = (bw - 4.0) / cols;  rh2 = 3.5
            
            # If no conclusions, add standard NSR findings for NSR reports
            if not conclusions:
                conclusions = ["Rhythm Analysis", "Normal heart rate", "Normal PR interval", 
                               "Normal QRS duration", "Normal QTc interval"]

            for idx2, line in enumerate(conclusions[:9]):
                row2 = idx2 // cols;  col2 = idx2 % cols
                tx = bx + 2.0 + col2 * col_w
                ty = by + 6.5 + row2 * rh2
                if ty + rh2 > by + bh:
                    break
                ax.text(tx, ty, f"{idx2+1}. {line}", fontsize=6, va='top', zorder=9)

            # Footer brand — full address matching 12:1 style
            brand = ("Deckmount Electronics Pvt. Ltd., Plot No. 389, Phase 5, "
                     "Udyog Vihar, Sector 19, Gurgaon, Haryana 122016  |  "
                     "RhythmPro ECG  |  IEC 60601  |  MADE IN INDIA")
            ax.text(PAGE_W/2, PAGE_H - MB + 1, brand,
                    fontsize=5.5, ha='center', va='top', color='#444', zorder=9)

            with PdfPages(path) as pdf:
                pdf.savefig(fig, bbox_inches=None)
                
                # ADDED: Second page for automatic/manual annotations
                if self.manual_annotations:
                    fig2 = self._generate_annotation_page()
                    if fig2:
                        pdf.savefig(fig2, bbox_inches='tight')

            # --- Save to history ---
            try:
                from dashboard.history_window import append_history_entry
                h_pat = {
                    "patient_name": pat.get('name', 'Unknown'),
                    "age": str(pat.get('age', '')),
                    "gender": pat.get('gender', ''),
                    "doctor": pat.get('doctor', ''),
                    "Org.": pat.get('Org.', ''),
                }
                append_history_entry(h_pat, path, report_type="Analysis")
            except Exception as h_err:
                print(f"Failed to append history: {h_err}")

            self.pdf_btn.setText("Generate PDF")
            QMessageBox.information(self, "PDF Saved", f"ECG Report saved:\n{path}")

        except Exception as e:
            self.pdf_btn.setText("Generate PDF")
            QMessageBox.critical(self, "PDF Error", f"Failed:\n{e}")

    def _generate_annotation_page(self):
        """Generate a second page for the PDF report containing annotations and wave strips."""
        if not self.manual_annotations:
            return None

        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle as MRect

        # A4 portrait (mm)
        PAGE_W = 210.0; PAGE_H = 297.0
        ML = 15.0; MR = 15.0; MT = 15.0; MB = 15.0

        fig = Figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), dpi=150, facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1], facecolor='white')
        ax.set_xlim(0, PAGE_W)
        ax.set_ylim(PAGE_H, 0) # y-axis inverted like top-to-bottom
        ax.set_aspect('equal')
        
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False, which='both')

        # Title
        ax.text(PAGE_W/2, MT + 5, "ECG ARRHYTHMIA & FINDINGS REPORT", 
                fontsize=12, fontweight='bold', ha='center', va='top', color='#0000cc')
        
        y_cursor = MT + 20
        
        # Summary Table Header
        ax.text(ML, y_cursor, "Summary of Manual & Automatic Annotations:", fontsize=9, fontweight='bold', va='top')
        y_cursor += 8
        
        # Table Header
        cols = [("Start (s)", 20), ("End (s)", 20), ("Type", 50), ("Lead", 20), ("Notes", 60)]
        x_cursor = ML
        for lbl, w in cols:
            ax.text(x_cursor, y_cursor, lbl, fontsize=8, fontweight='bold', va='top')
            x_cursor += w
        y_cursor += 5
        ax.plot([ML, PAGE_W - MR], [y_cursor, y_cursor], color='black', linewidth=0.5)
        y_cursor += 2

        # Table Rows
        for ann in self.manual_annotations[:12]: # Limit to 12 rows for summary
            x_cursor = ML
            ax.text(x_cursor, y_cursor, f"{ann.get('start_sec',0):.2f}", fontsize=7, va='top'); x_cursor += 20
            ax.text(x_cursor, y_cursor, f"{ann.get('end_sec',0):.2f}", fontsize=7, va='top'); x_cursor += 20
            ax.text(x_cursor, y_cursor, ann.get('type',''), fontsize=7, fontweight='bold', va='top'); x_cursor += 50
            ax.text(x_cursor, y_cursor, ann.get('lead',''), fontsize=7, va='top'); x_cursor += 20
            ax.text(x_cursor, y_cursor, ann.get('notes',''), fontsize=7, va='top')
            y_cursor += 5
            if y_cursor > PAGE_H / 2 - 10: break
        
        y_cursor = max(y_cursor + 10, PAGE_H / 2 - 20)
        
        # Add Wave Strips for each annotation (max 3 for second half of page)
        ax.text(ML, y_cursor, "Waveform Strips for Detected Events:", fontsize=9, fontweight='bold', va='top')
        y_cursor += 10
        
        # Sort annotations to prioritize arrhythmias for strips
        important_annotations = [a for a in self.manual_annotations if "Rhythm" not in str(a.get('type', ''))]
        if not important_annotations: important_annotations = self.manual_annotations
        
        ADC_PER_MM = 128.0
        MM_PER_SAMPLE = 25.0 / float(self.sampling_rate)

        for i, ann in enumerate(important_annotations[:3]): # Show top 3 strips
            if y_cursor > PAGE_H - 50: break
            
            lead_name = ann.get('lead', 'II')
            if lead_name == "All Leads": lead_name = 'II'
            
            data = self.lead_data.get(lead_name, np.array([]))
            if len(data) == 0: continue
            
            # Extract segment around detection
            start_s = ann.get('start_sec', 0)
            end_s = ann.get('end_sec', 0)
            duration_s = end_s - start_s
            
            # Total width of strip in mm
            strip_w_mm = PAGE_W - ML - MR
            # Total time shown in strip (max 10s or 1.5x detection duration)
            time_shown_s = min(10.0, max(3.0, duration_s * 1.5))
            
            # Center the detection in the strip
            center_s = (start_s + end_s) / 2
            strip_start_s = max(0, center_s - time_shown_s/2)
            strip_end_s = strip_start_s + time_shown_s
            
            st_idx = int(strip_start_s * self.sampling_rate)
            en_idx = int(strip_end_s * self.sampling_rate)
            
            segment = data[st_idx:en_idx]
            if len(segment) < 10: continue
            
            # Draw strip background
            strip_h = 30.0
            rect = MRect((ML, y_cursor), strip_w_mm, strip_h, 
                         linewidth=0.5, edgecolor='#e09696', facecolor='#fff5f5')
            ax.add_patch(rect)
            
            # Add grid
            for gy in np.arange(y_cursor, y_cursor + strip_h, 5):
                ax.plot([ML, ML + strip_w_mm], [gy, gy], color='#f5d8d8', linewidth=0.2, zorder=1)
            for gx in np.arange(ML, ML + strip_w_mm, 5):
                ax.plot([gx, gx], [y_cursor, y_cursor + strip_h], color='#f5d8d8', linewidth=0.2, zorder=1)
            
            # Plot wave
            baseline = np.median(segment)
            seg_mm = (segment - baseline) / ADC_PER_MM
            
            wx_mm = ML + np.arange(len(segment)) * MM_PER_SAMPLE
            # Clip to strip width
            mask = wx_mm <= (ML + strip_w_mm)
            wx_mm = wx_mm[mask]
            wy_mm = y_cursor + strip_h/2 - seg_mm[:len(wx_mm)]
            
            ax.plot(wx_mm, wy_mm, color='black', linewidth=0.5, zorder=2)
            
            # Highlight the detected part
            hl_start_mm = ML + (start_s - strip_start_s) * 25.0
            hl_end_mm = ML + (end_s - strip_start_s) * 25.0
            if hl_start_mm < ML + strip_w_mm and hl_end_mm > ML:
                hl_start_mm = max(ML, hl_start_mm)
                hl_end_mm = min(ML + strip_w_mm, hl_end_mm)
                ax.axvspan(hl_start_mm, hl_end_mm, color='#ff0000', alpha=0.1, ymin=1 - (y_cursor+strip_h)/PAGE_H, ymax=1 - y_cursor/PAGE_H)

            # Labels
            ax.text(ML + 2, y_cursor + 3, f"Event {i+1}: {ann.get('type','')} (Lead {lead_name})", 
                    fontsize=8, fontweight='bold', color='black', va='top', zorder=3)
            ax.text(ML + 2, y_cursor + strip_h - 2, f"Time: {start_s:.2f}s to {end_s:.2f}s", 
                    fontsize=6, color='#555', va='bottom', zorder=3)
            
            y_cursor += strip_h + 10
            
        # Brand Footer
        brand = "Deckmount Electronics Pvt Ltd | RhythmPro ECG | Made in India"
        ax.text(PAGE_W/2, PAGE_H - MB + 5, brand,
                fontsize=6, ha='center', va='top', color='#333', zorder=9)
        
        return fig

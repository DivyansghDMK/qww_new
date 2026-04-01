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
                background: #f5f6fa;
                color: #2c3e50;
            }
            QFrame#topbar {
                background: #ffffff;
                border: none;
                border-bottom: 1px solid #dcdde1;
                border-radius: 0px;
            }
            QFrame#plotpanel {
                background: #ffffff;
                border: none;
                border-radius: 0px;
            }
            QFrame#bottompanel {
                background: #ffffff;
                border: 1px solid #dcdde1;
                border-radius: 10px;
            }
            QFrame#leadbox {
                background: #ffffff;
                border: 1px solid #dcdde1;
                border-radius: 6px;
            }
            QLabel {
                color: #2c3e50;
                font-size: 11px;
                background: transparent;
                border: none;
            }
            QLabel#leadlabel {
                color: #e74c3c;
                font-size: 12px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
            QPushButton {
                background: #ffffff;
                color: #2c3e50;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f1f2f6;
                border-color: #0097e6;
                color: #0097e6;
            }
            QPushButton#primary {
                background: #0097e6;
                color: #ffffff;
                border: 1px solid #0097e6;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton#primary:hover { background: #0080bc; }
            QPushButton#apibtn {
                background: #0097e6;
                color: #ffffff;
                border: 1px solid #0097e6;
                border-radius: 6px;
            }
            QPushButton#apibtn:hover { background: #0080bc; }
            QComboBox {
                background: #ffffff;
                color: #2c3e50;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                padding: 5px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #2c3e50;
                selection-background-color: #0097e6;
            }
            QLineEdit {
                background: #ffffff;
                color: #2c3e50;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                padding: 5px 8px;
                font-size: 11px;
            }
            QTextEdit {
                background: #ffffff;
                color: #2c3e50;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                padding: 5px;
                font-size: 11px;
            }
            QTableWidget {
                background: #ffffff;
                color: #2c3e50;
                border: 1px solid #dcdde1;
                gridline-color: #f1f2f6;
                selection-background-color: #0097e6;
                selection-color: #0097e6;
                border-radius: 4px;
            }
            QHeaderView::section {
                background: #ffffff;
                color: #2c3e50;
                border: none;
                padding: 6px;
                font-size: 11px;
                font-weight: bold;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #dcdde1;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #0097e6;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                background: #0097e6;
                margin: -5px 0;
                border-radius: 7px;
            }
            QScrollBar:vertical {
                background: #ffffff;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #dcdde1;
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
            logo_label.setStyleSheet("color:#2c3e50;font-size:16px;font-weight:bold;border:none;background:transparent;")
        lay.addWidget(logo_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("background:#ff6600;border:none;max-width:2px;margin:4px 4px;")
        lay.addWidget(sep)

        pat_col = QVBoxLayout()
        pat_col.setSpacing(2)
        self.patient_lbl = QLabel("Patient: --")
        self.patient_lbl.setFont(QFont("Arial", 12, QFont.Bold))
        self.patient_lbl.setStyleSheet("color:#2c3e50;font-weight:bold;border:none;background:transparent;")
        self.patient_meta_lbl = QLabel("ID: -- | Age: -- | Gender: --")
        self.patient_meta_lbl.setStyleSheet("color:#7f8c8d;font-size:10px;border:none;background:transparent;")
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
        ctrl_frame.setStyleSheet("background:#ffffff;border-radius:8px;border:1px solid #dcdde1;")
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
        lbl_w.setStyleSheet("color:#7f8c8d;border:none;background:transparent;")
        controls.addWidget(lbl_w)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["1.0 s", "2.0 s", "3.0 s", "5.0 s", "10.0 s"])
        self.window_combo.setCurrentText("10.0 s")
        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        self.window_combo.setFixedWidth(80)
        controls.addWidget(self.window_combo)

        lbl_s = QLabel("Step:")
        lbl_s.setStyleSheet("color:#7f8c8d;border:none;background:transparent;")
        controls.addWidget(lbl_s)
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.2 s", "0.5 s", "1.0 s"])
        self.step_combo.setCurrentText("0.5 s")
        self.step_combo.currentTextChanged.connect(self._on_step_changed)
        self.step_combo.setFixedWidth(75)
        controls.addWidget(self.step_combo)

        controls.addSpacing(14)
        self.frame_label = QLabel("Frame: 0.00s – 2.00s")
        self.frame_label.setStyleSheet("color:#2c3e50;font-weight:bold;border:none;background:transparent;")
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

        hint = QLabel(
            "💡 First time? Load a report from dropdown, or enter API ID and click Fetch. "
            "Use Prev/Next to move through ECG. Mark ① Start then ② End + Save."
        )
        hint.setStyleSheet(
            "color:#1a66a6;font-size:10px;background:#e8f4fd;"
            "border:1px solid #b3d7ff;border-radius:4px;padding:4px 10px;"
        )
        hint.setWordWrap(True)
        hint.setFixedHeight(32)
        v.addWidget(hint)

        # Lead grid (3 rows x 4 cols)
        from PyQt5.QtWidgets import QGridLayout, QWidget as _QW
        grid_widget = _QW()
        grid_widget.setStyleSheet("background:#ffffff;")
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

                fig = Figure(facecolor='#ffffff')
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                canvas = FigureCanvas(fig)
                canvas.setStyleSheet("background:#ffffff;border:none;")
                ax = fig.add_axes([0, 0, 1, 1], facecolor='#ffffff')
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
        title_lbl.setStyleSheet("color:#2c3e50;font-size:12px;font-weight:bold;border:none;background:transparent;")
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
        self.mark_start_btn = QPushButton("① Mark Start")
        self.mark_start_btn.clicked.connect(self.mark_start)
        self.mark_end_btn = QPushButton("② Mark End + Save")
        self.mark_end_btn.clicked.connect(self.mark_end_and_save)
        self.mark_end_btn.setEnabled(False)
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

        h.addWidget(mark_box, stretch=2)

        right_col = QVBoxLayout()
        right_col.setSpacing(4)

        metrics_lbl = QLabel("ECG Metrics")
        metrics_lbl.setStyleSheet("color:#2c3e50;font-size:12px;font-weight:bold;border:none;background:transparent;")
        right_col.addWidget(metrics_lbl)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setMaximumHeight(130)
        right_col.addWidget(self.metrics_table)

        findings_lbl = QLabel("Clinical Findings")
        findings_lbl.setStyleSheet("color:#2c3e50;font-size:12px;font-weight:bold;border:none;background:transparent;")
        right_col.addWidget(findings_lbl)
        self.findings_text = QTextEdit()
        self.findings_text.setReadOnly(True)
        self.findings_text.setMaximumHeight(80)
        right_col.addWidget(self.findings_text)


        # Hide metrics per user instruction
        metrics_lbl.setVisible(False)
        self.metrics_table.setVisible(False)
        findings_lbl.setVisible(False)
        self.findings_text.setVisible(False)
        h.addLayout(right_col, stretch=1)


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
        self.pending_mark_start_sec = None
        self.mark_end_btn.setEnabled(False)
        self.mark_status_lbl.setText("No active mark")
        self.mark_status_lbl.setStyleSheet("color:#7a9f7a;border:none;background:transparent;")
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

        ECG_COLOR = '#000000'
        ANNOT_COLOR = '#ff0000'

        for lead in self.LEADS:
            ax = self._lead_axes.get(lead)
            fig = self._lead_figs.get(lead)
            canvas = self._lead_canvases.get(lead)
            if ax is None:
                continue

            ax.clear()
            ax.set_axis_off()
            ax.set_facecolor('#080808')
            # Fixed ADC Y-range 0-4096 for all leads
            ax.set_ylim(0, 4096)
            ax.set_xlim(start_sec, end_sec if end_sec > start_sec else start_sec + 1)
            # ECG paper style green grid
            ax.grid(True, color='#ffd1d1', linewidth=0.4, linestyle='-', alpha=0.9)

            data = self.lead_data.get(lead, np.array([]))
            if len(data) > 0 and en > st:
                seg = data[st:en]
                ax.plot(t, seg, color=ECG_COLOR, linewidth=0.85, antialiased=True)
                ax.set_ylim(0, 4096)
                ax.set_xlim(start_sec, end_sec)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color='#666666', fontsize=9)

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
        self.mark_end_btn.setEnabled(True)
        self.mark_status_lbl.setText(
            f"✅ Start marked at {self.pending_mark_start_sec:.2f}s  →  Now navigate to end, then click ② Mark End + Save"
        )
        self.mark_status_lbl.setStyleSheet("color:#f5c518;border:none;background:transparent;")

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
        self.mark_end_btn.setEnabled(False)
        self.mark_status_lbl.setText(f"✅ Annotation saved: {arr_type}")
        self.mark_status_lbl.setStyleSheet("color:#7a9f7a;border:none;background:transparent;")
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
        self.auto_detect_btn.setText("⏳ Analyzing...")
        self.auto_detect_btn.setEnabled(False)
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

        try:
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
        finally:
            self.auto_detect_btn.setText("Automatic Detection")
            self.auto_detect_btn.setEnabled(True)

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
            QRadioButton:checked { color:#2c3e50; font-weight:bold; }
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
        if rb1.isChecked():   pdf_format = "4_3"
        elif rb2.isChecked(): pdf_format = "12_1"
        else:                 pdf_format = "6_2"

        try:
            self.pdf_btn.setText("Generating...")
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()

            # ── 1. Build snap_raw (List of 12 numpy arrays) ──
            st = self.frame_start_sample
            leads_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            snap_raw = []
            for l in leads_order:
                data_arr = self.lead_data.get(l, np.array([]))
                if len(data_arr) > st:
                    snap_raw.append(data_arr[st:])
                else:
                    snap_raw.append(np.array([]))

            # ── 2. Build frozen (Metrics dictionary) ──
            frozen = {
                'HR':       int(float(hr) if hr else 0),
                'RR':       int(float(rr) if rr else 0),
                'PR':       int(float(pr) if pr else 0),
                'QRS':      int(float(qrs) if qrs else 0),
                'QT':       int(float(qt) if qt else 0),
                'QTc':      int(float(qtc) if qtc else 0),
                'QTcF':     int(float(qtcf) if qtcf else 0),
                'rv5':      0.0,
                'sv1':      0.0,
                'p_axis':   '--',
                'QRS_axis': '--',
                't_axis':   '--',
                'lead_seq': 'Standard',
                'logo_path': str(self.analysis_pdf_logo_path) if hasattr(self, 'analysis_pdf_logo_path') else '',
            }
            # Attempt to parse rv5, sv1, and axes if they exist in raw metrics
            try:
                if rv5sv1:
                    frozen['rv5'] = float(str(rv5sv1).split('/')[0].strip(' mV').strip('+'))
                    if len(str(rv5sv1).split('/')) > 1:
                        frozen['sv1'] = float(str(rv5sv1).split('/')[1].strip(' mV').strip('+'))
            except Exception: pass
            
            try:
                if axes and len(str(axes).split('/')) == 3:
                    frozen['p_axis'] = str(axes).split('/')[0].strip()
                    frozen['QRS_axis'] = str(axes).split('/')[1].strip()
                    frozen['t_axis'] = str(axes).split('/')[2].strip()
            except Exception: pass

            # ── 3. Build patient (Details dictionary) ──
            pat_mapped = {
                'first_name': pat.get('name', 'Unknown'),
                'last_name': '',
                'age': pat.get('age', ''),
                'gender': pat.get('gender', ''),
                'date_time': pat.get('report_date') or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # ── 4. Generate optional Annotation Page ──
            extra_figs = []
            if self.manual_annotations:
                fig2 = self._generate_annotation_page()
                if fig2:
                    extra_figs.append(fig2)

            # ── 5. Generate Report using Standard Android Generator ──
            from ecg.ecg_report_android import generate_report as _gen
            
            # Use fixed 7-second bounds like before? Actually we want full data.
            # We already bounded the views in android generator.
            _gen(snap_raw=snap_raw, 
                 frozen=frozen, 
                 patient=pat_mapped,
                 filename=path, 
                 fmt=pdf_format, 
                 conc_list=conclusions, 
                 fs=float(self.sampling_rate),
                 extra_figs=extra_figs)

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

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

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QFrame, QMessageBox,
    QSizePolicy, QComboBox, QFileDialog, QTextEdit, QSlider,
    QLineEdit
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ECGAnalysisWindow(QDialog):
    """User-friendly ECG analysis window for backend JSON reports."""

    LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ECG Analysis")
        self.setGeometry(80, 60, 1700, 980)

        self.setStyleSheet("""
            QDialog { background: #ffffff; color: #111111; }
            QFrame { background: #ffffff; border: 1px solid #e2e2e2; border-radius: 8px; }
            QLabel { color: #111111; font-size: 11px; }
            QPushButton {
                background: #111111; color: #ffffff; border: 1px solid #111111;
                border-radius: 6px; padding: 6px 12px; font-size: 11px; font-weight: 600;
            }
            QPushButton:hover { background: #000000; }
            QPushButton:pressed { background: #2a2a2a; }
            QPushButton#secondary {
                background: #ffffff; color: #111111; border: 1px solid #111111;
            }
            QPushButton#secondary:hover { background: #f3f3f3; }
            QComboBox, QLineEdit, QTextEdit {
                background: #ffffff; color: #111111; border: 1px solid #cfcfcf;
                border-radius: 6px; padding: 5px 8px; font-size: 11px;
            }
            QTableWidget {
                background: #ffffff; color: #111111; border: 1px solid #d9d9d9;
                gridline-color: #ededed; selection-background-color: #111111; selection-color: #ffffff;
            }
            QHeaderView::section {
                background: #111111; color: #ffffff; border: none;
                padding: 6px; font-size: 11px; font-weight: bold;
            }
            QSlider::groove:horizontal { height: 6px; background: #dddddd; border-radius: 3px; }
            QSlider::handle:horizontal { width: 14px; background: #111111; margin: -5px 0; border-radius: 7px; }
        """)

        self.reports = []
        self.current_report = None
        self.current_report_path = ""

        self.lead_data = {lead: np.array([]) for lead in self.LEADS}
        self.sampling_rate = 500.0

        self.window_seconds = 2.0
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
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(10, 8, 10, 8)

        self.patient_lbl = QLabel("Patient: --")
        self.patient_lbl.setFont(QFont("Arial", 11, QFont.Bold))
        self.patient_meta_lbl = QLabel("ID: -- | Age: -- | Gender: --")

        left = QVBoxLayout()
        left.addWidget(self.patient_lbl)
        left.addWidget(self.patient_meta_lbl)

        right = QHBoxLayout()
        right.addWidget(QLabel("Report:"))
        self.report_combo = QComboBox()
        self.report_combo.currentIndexChanged.connect(self.load_selected_report)
        self.report_combo.setMinimumWidth(360)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setObjectName("secondary")
        self.refresh_btn.clicked.connect(self.load_reports)

        self.export_btn = QPushButton("Export JSON")
        self.export_btn.setObjectName("secondary")
        self.export_btn.clicked.connect(self.export_report)

        right.addWidget(self.report_combo)
        right.addWidget(self.refresh_btn)
        right.addWidget(self.export_btn)

        lay.addLayout(left)
        lay.addStretch()
        lay.addLayout(right)
        return frame

    def _build_plot_panel(self):
        frame = QFrame()
        v = QVBoxLayout(frame)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        controls = QHBoxLayout()
        self.prev_btn = QPushButton("Prev Frame")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.next_btn = QPushButton("Next Frame")
        self.next_btn.clicked.connect(self.next_frame)

        controls.addWidget(self.prev_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.next_btn)

        controls.addSpacing(12)
        controls.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["1.0 s", "2.0 s", "3.0 s", "5.0 s"])
        self.window_combo.setCurrentText("2.0 s")
        self.window_combo.currentTextChanged.connect(self._on_window_changed)

        controls.addWidget(self.window_combo)
        controls.addWidget(QLabel("Step:"))
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.2 s", "0.5 s", "1.0 s"])
        self.step_combo.setCurrentText("0.5 s")
        self.step_combo.currentTextChanged.connect(self._on_step_changed)
        controls.addWidget(self.step_combo)

        controls.addSpacing(12)
        self.frame_label = QLabel("Frame: 0.00s - 2.00s")
        controls.addWidget(self.frame_label)
        controls.addStretch()

        v.addLayout(controls)

        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setMinimum(0)
        self.timeline.setMaximum(0)
        self.timeline.valueChanged.connect(self._on_timeline_changed)
        v.addWidget(self.timeline)

        self.figure = Figure(figsize=(14, 9), dpi=80, facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.axes = [self.figure.add_subplot(4, 3, i + 1) for i in range(12)]
        self.figure.tight_layout(pad=1.3)
        v.addWidget(self.canvas, stretch=1)

        return frame

    def _build_bottom_panel(self):
        frame = QFrame()
        h = QHBoxLayout(frame)
        h.setContentsMargins(8, 8, 8, 8)
        h.setSpacing(8)

        # Metrics
        metrics_box = QFrame()
        mv = QVBoxLayout(metrics_box)
        mv.addWidget(QLabel("Metrics"))
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        mv.addWidget(self.metrics_table)

        # Findings
        findings_box = QFrame()
        fv = QVBoxLayout(findings_box)
        fv.addWidget(QLabel("Backend Analysis"))
        self.findings_text = QTextEdit()
        self.findings_text.setReadOnly(True)
        fv.addWidget(self.findings_text)

        # Manual arrhythmia marking
        mark_box = QFrame()
        av = QVBoxLayout(mark_box)
        av.addWidget(QLabel("Manual Arrhythmia Marking"))

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Type:"))
        self.arrhythmia_type_combo = QComboBox()
        self.arrhythmia_type_combo.addItems([
            "Atrial Fibrillation", "PVC", "PAC", "SVT", "VT", "Bradycardia", "Tachycardia", "Other"
        ])
        row1.addWidget(self.arrhythmia_type_combo)
        row1.addWidget(QLabel("Lead:"))
        self.mark_lead_combo = QComboBox()
        self.mark_lead_combo.addItems(["All Leads"] + self.LEADS)
        row1.addWidget(self.mark_lead_combo)
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
        self.delete_mark_btn = QPushButton("Delete Selected")
        self.delete_mark_btn.setObjectName("secondary")
        self.delete_mark_btn.clicked.connect(self.delete_selected_annotation)
        row2.addWidget(self.mark_start_btn)
        row2.addWidget(self.mark_end_btn)
        row2.addWidget(self.delete_mark_btn)
        av.addLayout(row2)

        self.mark_status_lbl = QLabel("No active mark")
        av.addWidget(self.mark_status_lbl)

        self.annotation_table = QTableWidget(0, 5)
        self.annotation_table.setHorizontalHeaderLabels(["Start (s)", "End (s)", "Type", "Lead", "Notes"])
        self.annotation_table.horizontalHeader().setStretchLastSection(True)
        av.addWidget(self.annotation_table)

        h.addWidget(metrics_box, 1)
        h.addWidget(findings_box, 1)
        h.addWidget(mark_box, 2)

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
        hrv_metrics = rpt.get('hrv_result_reading') or {}

        self.metrics_table.setRowCount(0)
        items = [
            ("HR", metrics.get('HR_bpm', metrics.get('heart_rate', metrics.get('HR', 'N/A'))), "bpm"),
            ("RR", metrics.get('RR_ms', metrics.get('rr_interval', metrics.get('RR', 'N/A'))), "ms"),
            ("PR", metrics.get('PR_ms', metrics.get('pr_interval', metrics.get('PR', 'N/A'))), "ms"),
            ("QRS", metrics.get('QRS_ms', metrics.get('qrs_duration', metrics.get('QRS', 'N/A'))), "ms"),
            ("QT", metrics.get('QT_ms', metrics.get('qt_interval', metrics.get('QT', 'N/A'))), "ms"),
            ("QTc", metrics.get('QTc_ms', metrics.get('qtc_interval', metrics.get('QTc', 'N/A'))), "ms"),
            ("HRV HR", hrv_metrics.get('HR_bpm', hrv_metrics.get('HR', 'N/A')), "bpm"),
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
            self.play_btn.setText("Play")
        else:
            self.play_timer.start(250)
            self.play_btn.setText("Pause")

    def _render_current_frame(self):
        ws = self._window_samples()
        st = self.frame_start_sample
        en = min(self._total_samples(), st + ws)

        t = np.arange(st, en) / self.sampling_rate if en > st else np.array([])
        start_sec = st / self.sampling_rate if self.sampling_rate > 0 else 0.0
        end_sec = en / self.sampling_rate if self.sampling_rate > 0 else 0.0
        self.frame_label.setText(f"Frame: {start_sec:.2f}s - {end_sec:.2f}s")

        for i, lead in enumerate(self.LEADS):
            ax = self.axes[i]
            ax.clear()
            ax.set_facecolor('#ffffff')
            ax.grid(True, alpha=0.25, color='#dcdcdc', linestyle='-', linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_color('#cccccc')
            ax.set_title(f"Lead {lead}", fontsize=10, fontweight='bold', color='#111111')
            ax.tick_params(labelsize=7, colors='#444444')

            data = self.lead_data.get(lead, np.array([]))
            if len(data) > 0 and en > st:
                seg = data[st:en]
                ax.plot(t, seg, color='#111111', linewidth=0.9)
                ax.set_xlim(start_sec, end_sec)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='#777777')

            # Show manual annotation overlays that intersect this frame
            for ann in self.manual_annotations:
                lead_ok = ann.get('lead', 'All Leads') in ('All Leads', lead)
                if not lead_ok:
                    continue
                a0 = ann.get('start_sec', 0.0)
                a1 = ann.get('end_sec', 0.0)
                if a1 < start_sec or a0 > end_sec:
                    continue
                left = max(a0, start_sec)
                right = min(a1, end_sec)
                if right > left:
                    ax.axvspan(left, right, color='#999999', alpha=0.25)

        self.figure.tight_layout(pad=1.2)
        self.canvas.draw_idle()

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

    # --------------------------- actions ---------------------------
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

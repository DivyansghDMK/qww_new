"""
ecg/holter/holter_ui.py
========================
Complete Holter Monitor UI - Professional Medical Software
Matches reference images: black/green medical workstation style.

Screens:
  1. HolterStartDialog        — patient info + duration + start
  2. HolterStatusBar          — REC indicator, elapsed, live BPM, arrhythmia ticker
  3. HolterSummaryCards       — KPI cards (Avg HR, Min/Max, Beats, Pauses, Quality, SDNN)
  4. HolterOverviewPanel      — full stats table (Name/Value pairs)
  5. HolterHRVPanel           — HRV table per hour + bottom stats strip
  6. HolterReplayPanel        — RR scatter/Lorenz + scrub slider + ECG strip
  7. HolterEventsPanel        — Arrhythmia events list with strip nav
  8. HolterWaveGridPanel      — 12-lead live/replay grid (3 rows × 4 cols)
  9. HolterInsightPanel       — Comprehensive report preview narrative
 10. HolterRecordManagementPanel — searchable session browser
 11. HolterHistogramPanel     — RR-interval histogram
 12. HolterAFPanel            — AF episode browser
 13. HolterSTPanel            — ST tendency per channel
 14. HolterEditEventPanel     — Edit events with strip thumbnails
 15. HolterEditStripsPanel    — Edit strips (max HR, min HR, sinus max/min thumbnails)
 16. HolterReportTablePanel   — Hour-by-hour report table
 17. HolterMainWindow         — Orchestrates all panels in tabbed layout
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialog, QLineEdit, QComboBox, QSlider, QGroupBox, QFrame,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QScrollArea, QGridLayout, QSpinBox, QMessageBox,
    QFileDialog, QApplication, QProgressBar, QSplitter, QTextEdit,
    QAbstractItemView, QToolButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPointF
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter, QPen, QBrush, QPixmap

try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:
    pg = None
    HAS_PG = False

# ── Comprehensive ECG Analysis professional palette ───────────────────────────
COL_BLACK     = "#080808"   # root canvas
COL_DARK      = "#1a1a1a"   # panel dark
COL_GRAY      = "#111111"   # surface dark
COL_BG        = COL_BLACK
COL_TEXT      = "#e0e0e0"   # primary text
COL_GREEN     = "#00e050"   # waveform green
COL_GREEN_MID = "#00ff66"   # bright ECG green
COL_GREEN_DRK = "#2a2a2a"   # borders/separators
COL_WHITE     = "#FFFFFF"
COL_YELLOW    = "#f5c518"
COL_RED       = "#ff3333"
COL_WAVE_ORANGE = "#e06020"
COL_WAVE_RED    = "#ff3333"
COL_GRID_MINOR  = "#0a2a0a"
COL_GRID_MAJOR  = "#1a4a1a"
COL_BTN_ACTIVE_BG = "#1e4a7a"
COL_BTN_ACTIVE_TEXT = "#6aacf5"
COL_TIMESTAMP = "#f0a030"
COL_BEAT_S = "#ff9900"


def _style_btn(bg=COL_GREEN_DRK, fg=COL_WHITE, hover=COL_GREEN):
    return f"""
        QPushButton {{
            background: {bg};
            color: {fg};
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            padding: 6px 14px;
            font-size: 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background: {hover};
            color: {COL_WHITE};
        }}
        QPushButton:pressed {{ background: {COL_GREEN_DRK}; border: 2px solid {COL_WHITE}; }}
        QPushButton:disabled {{ background: #222; color: #555; border: 1px solid #333; }}
    """


def _style_active_btn():
    return f"""
        QPushButton {{
            background: {COL_BTN_ACTIVE_BG};
            color: {COL_BTN_ACTIVE_TEXT};
            border: 1px solid {COL_BTN_ACTIVE_TEXT};
            border-radius: 6px;
            padding: 6px 14px;
            font-size: 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{ background: #24588f; }}
    """


def _table_style():
    return f"""
        QTableWidget {{
            background: {COL_BLACK};
            color: {COL_GREEN};
            gridline-color: {COL_GREEN_DRK};
            font-size: 12px;
            border: 1px solid {COL_GREEN_DRK};
            selection-background-color: {COL_GREEN_DRK};
            selection-color: {COL_WHITE};
        }}
        QHeaderView::section {{
            background: {COL_DARK};
            color: {COL_GREEN};
            font-size: 11px;
            font-weight: bold;
            padding: 5px;
            border: 1px solid {COL_GREEN_DRK};
        }}
        QTableWidget::item {{ padding: 4px; }}
    """


def _sec_to_hms(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# 1. HOLTER START DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class HolterStartDialog(QDialog):
    def __init__(self, parent=None, patient_info: dict = None, output_dir: str = "recordings"):
        super().__init__(parent)
        self.setWindowTitle("Comprehensive ECG Analysis — Setup")
        self.setMinimumWidth(640)
        self.setStyleSheet(f"background: {COL_DARK}; color: {COL_WHITE};")
        self.output_dir = output_dir
        self._result_info = None
        self._result_duration = 24
        self._result_dir = output_dir
        self._build_ui(patient_info or {})

    def _build_ui(self, info: dict):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("🫀  Comprehensive ECG Analysis — Professional Setup")
        title.setStyleSheet(f"background:{COL_GRAY};color:{COL_GREEN};border:2px solid {COL_GREEN};"
                            f"font-size:20px;font-weight:bold;padding:16px;border-radius:8px;")
        layout.addWidget(title)

        subtitle = QLabel("Enter patient details, choose study duration, and launch the 12-lead ECG workspace.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color:#cccccc;font-size:13px;padding:4px;")
        layout.addWidget(subtitle)

        # Patient info group
        pg = QGroupBox("Patient Information")
        pg.setStyleSheet(f"QGroupBox{{font-weight:bold;color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                         f"border-radius:8px;margin-top:12px;padding-top:20px;background:{COL_BLACK};}}")
        pg_layout = QGridLayout(pg)
        pg_layout.setSpacing(10)

        fields = [
            ("Patient Name", "patient_name", info.get("patient_name", "")),
            ("Age", "age", str(info.get("age", ""))),
            ("Email", "email", info.get("email", "")),
            ("Doctor", "doctor", info.get("doctor", "")),
            ("Organisation", "org", info.get("Org.", info.get("org", ""))),
            ("Phone", "phone", info.get("doctor_mobile", info.get("phone", ""))),
        ]
        self._fields = {}
        for row, (label, key, default) in enumerate(fields):
            lbl = QLabel(label + ":")
            lbl.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
            edit = QLineEdit(default)
            edit.setStyleSheet(f"QLineEdit{{border:1px solid {COL_GREEN_DRK};border-radius:4px;padding:8px;"
                               f"font-size:13px;background:{COL_DARK};color:{COL_GREEN};}}"
                               f"QLineEdit:focus{{border-color:{COL_GREEN};}}")
            pg_layout.addWidget(lbl, row, 0)
            pg_layout.addWidget(edit, row, 1)
            self._fields[key] = edit

        lbl_g = QLabel("Gender:")
        lbl_g.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
        self._gender = QComboBox()
        self._gender.addItems(["Select", "Male", "Female", "Other"])
        idx = self._gender.findText(info.get("gender", info.get("sex", "Select")))
        if idx >= 0: self._gender.setCurrentIndex(idx)
        self._gender.setStyleSheet(f"QComboBox{{border:1px solid {COL_GREEN_DRK};border-radius:4px;padding:8px;"
                                   f"font-size:13px;background:{COL_DARK};color:{COL_GREEN};}}")
        pg_layout.addWidget(lbl_g, len(fields), 0)
        pg_layout.addWidget(self._gender, len(fields), 1)
        layout.addWidget(pg)

        # Recording settings group
        rg = QGroupBox("Recording Settings")
        rg.setStyleSheet(f"QGroupBox{{font-weight:bold;color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                         f"border-radius:8px;margin-top:12px;padding-top:20px;background:{COL_BLACK};}}")
        rg_layout = QGridLayout(rg)
        rg_layout.setSpacing(10)

        dur_lbl = QLabel("Duration:")
        dur_lbl.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
        self._duration = QComboBox()
        self._duration.addItems(["24 hours", "48 hours", "Custom"])
        self._duration.setStyleSheet(f"border:1px solid {COL_GREEN_DRK};border-radius:4px;padding:8px;"
                                     f"font-size:13px;background:{COL_DARK};color:{COL_GREEN};")
        self._duration.currentTextChanged.connect(lambda t: self._custom_hours.setVisible(t == "Custom"))
        rg_layout.addWidget(dur_lbl, 0, 0)
        rg_layout.addWidget(self._duration, 0, 1)

        self._custom_hours = QSpinBox()
        self._custom_hours.setRange(1, 72)
        self._custom_hours.setValue(24)
        self._custom_hours.setSuffix(" hours")
        self._custom_hours.setVisible(False)
        self._custom_hours.setStyleSheet(f"border:1px solid {COL_GREEN_DRK};border-radius:4px;padding:8px;"
                                         f"font-size:13px;background:{COL_DARK};color:{COL_GREEN};")
        rg_layout.addWidget(self._custom_hours, 1, 1)

        out_lbl = QLabel("Output Directory:")
        out_lbl.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
        rg_layout.addWidget(out_lbl, 2, 0)
        dir_row = QHBoxLayout()
        self._dir_label = QLabel(self.output_dir)
        self._dir_label.setStyleSheet(f"font-size:12px;color:{COL_GREEN};")
        dir_row.addWidget(self._dir_label, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(_style_btn())
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        rg_layout.addLayout(dir_row, 2, 1)

        self._rec_count_label = QLabel("")
        self._rec_count_label.setStyleSheet(f"font-size:13px;color:{COL_GREEN};font-weight:700;")
        rg_layout.addWidget(QLabel("Recorded Sessions:"), 3, 0)
        rg_layout.addWidget(self._rec_count_label, 3, 1)
        self._refresh_rec_count()
        layout.addWidget(rg)

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(_style_btn(COL_GRAY, COL_WHITE, COL_GREEN_DRK))
        cancel_btn.clicked.connect(self.reject)
        start_btn = QPushButton("▶  Open ECG Workspace")
        start_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #ff6600, stop:1 #e65c00);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 14px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: #ff7a26; }}
            QPushButton:pressed {{ background: #cc5200; }}
        """)
        start_btn.setMinimumHeight(48)
        start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(start_btn, 1)
        layout.addLayout(btn_row)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if d:
            self._result_dir = d
            self._dir_label.setText(d)
            self._refresh_rec_count()

    def _refresh_rec_count(self):
        root = getattr(self, '_result_dir', self.output_dir)
        count = 0
        try:
            if os.path.isdir(root):
                for name in os.listdir(root):
                    if os.path.exists(os.path.join(root, name, "recording.ecgh")):
                        count += 1
        except Exception:
            pass
        self._rec_count_label.setText(f"{count} completed recording(s)")

    def _on_start(self):
        info = {key: field.text().strip() for key, field in self._fields.items()}
        info['gender'] = self._gender.currentText()
        info['sex'] = info['gender']
        info['name'] = info.get('patient_name', 'Unknown')
        info['Org.'] = info.get('org', '')
        if not info.get('patient_name'):
            QMessageBox.warning(self, "Missing Name", "Please enter the patient name.")
            return
        dur_text = self._duration.currentText()
        if dur_text == "24 hours": self._result_duration = 24
        elif dur_text == "48 hours": self._result_duration = 48
        else: self._result_duration = self._custom_hours.value()
        self._result_info = info
        self._result_dir = self._dir_label.text()
        self.accept()

    def get_result(self):
        if self._result_info:
            return self._result_info, self._result_duration, self._result_dir
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. HOLTER STATUS BAR  (Live recording indicator)
# ══════════════════════════════════════════════════════════════════════════════

class HolterStatusBar(QFrame):
    stop_requested = pyqtSignal()

    def __init__(self, parent=None, target_hours: int = 24):
        super().__init__(parent)
        self.target_hours = target_hours
        self._start_time = time.time()
        self._blink_state = True
        self.setFixedHeight(52)
        self.setStyleSheet(f"QFrame{{background:{COL_BLACK};border-bottom:2px solid {COL_GREEN};}}")
        self._build_ui()
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink)
        self._blink_timer.start(800)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._elapsed_timer.start(1000)

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 4, 15, 4)
        layout.setSpacing(18)

        self._rec_label = QLabel("● REC")
        self._rec_label.setStyleSheet(f"color:{COL_GREEN};font-size:15px;font-weight:bold;")
        layout.addWidget(self._rec_label)

        self._time_label = QLabel("00:00:00")
        self._time_label.setStyleSheet(f"color:{COL_GREEN};font-size:18px;font-weight:bold;font-family:monospace;")
        layout.addWidget(self._time_label)

        tgt = QLabel(f"/ {self.target_hours:02d}:00:00")
        tgt.setStyleSheet(f"color:{COL_GREEN_DRK};font-size:12px;")
        layout.addWidget(tgt)

        sep = QLabel("|")
        sep.setStyleSheet(f"color:{COL_GREEN_DRK};")
        layout.addWidget(sep)

        bpm_lbl = QLabel("BPM:")
        bpm_lbl.setStyleSheet(f"color:{COL_GREEN};font-size:12px;")
        layout.addWidget(bpm_lbl)
        self._bpm_label = QLabel("—")
        self._bpm_label.setStyleSheet(f"color:{COL_GREEN};font-size:18px;font-weight:bold;")
        layout.addWidget(self._bpm_label)

        sep2 = QLabel("|")
        sep2.setStyleSheet(f"color:{COL_GREEN_DRK};")
        layout.addWidget(sep2)

        ev_lbl = QLabel("Events:")
        ev_lbl.setStyleSheet(f"color:{COL_GREEN};font-size:12px;")
        layout.addWidget(ev_lbl)
        self._arrhy_label = QLabel("None detected")
        self._arrhy_label.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;")
        self._arrhy_label.setMaximumWidth(380)
        layout.addWidget(self._arrhy_label, 1)

        self._progress = QProgressBar()
        self._progress.setRange(0, self.target_hours * 3600)
        self._progress.setValue(0)
        self._progress.setFixedWidth(140)
        self._progress.setFixedHeight(12)
        self._progress.setStyleSheet(f"""
            QProgressBar{{background:{COL_DARK};border-radius:6px;border:1px solid {COL_GREEN_DRK};}}
            QProgressBar::chunk{{background:{COL_GREEN};border-radius:5px;}}
        """)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        stop_btn = QPushButton("⬛  Stop")
        stop_btn.setStyleSheet(_style_btn(COL_GREEN_DRK, COL_WHITE, COL_GREEN))
        stop_btn.setFixedHeight(34)
        stop_btn.clicked.connect(self.stop_requested)
        layout.addWidget(stop_btn)

    def _blink(self):
        self._blink_state = not self._blink_state
        color = COL_GREEN if self._blink_state else COL_GREEN_DRK
        self._rec_label.setStyleSheet(f"color:{color};font-size:15px;font-weight:bold;")

    def _update_elapsed(self):
        elapsed = int(time.time() - self._start_time)
        h = elapsed // 3600; m = (elapsed % 3600) // 60; s = elapsed % 60
        self._time_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
        self._progress.setValue(min(elapsed, self.target_hours * 3600))

    def update_stats(self, bpm: float, arrhythmias: List[str]):
        if bpm > 0:
            self._bpm_label.setText(f"{bpm:.0f}")
        if arrhythmias:
            self._arrhy_label.setText("  |  ".join(arrhythmias[:3]))

    def cleanup(self):
        self._blink_timer.stop()
        self._elapsed_timer.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 3. HOLTER SUMMARY CARDS
# ══════════════════════════════════════════════════════════════════════════════

class HolterSummaryCards(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value_labels = {}
        self.setStyleSheet(f"background:{COL_BLACK};")
        self._build_ui()

    def _build_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)

        cards = [
            ("Average HR", "avg_hr", "bpm"),
            ("Min / Max HR", "range_hr", "bpm"),
            ("Total Beats", "beats", ""),
            ("Pauses", "pauses", "events"),
            ("Signal Quality", "quality", "%"),
            ("HRV (SDNN)", "sdnn", "ms"),
        ]
        for idx, (title, key, unit) in enumerate(cards):
            frame = QFrame()
            frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
            box = QVBoxLayout(frame)
            box.setContentsMargins(12, 10, 12, 10)
            box.setSpacing(4)
            lbl = QLabel(title)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
            val = QLabel("—")
            val.setStyleSheet(f"color:{COL_WHITE};font-size:20px;font-weight:bold;border:none;")
            unit_lbl = QLabel(unit)
            unit_lbl.setStyleSheet(f"color:{COL_GREEN_DRK};font-size:10px;font-weight:bold;border:none;")
            box.addWidget(lbl)
            box.addWidget(val)
            box.addWidget(unit_lbl)
            self._value_labels[key] = val
            layout.addWidget(frame, 0 if idx < 3 else 1, idx % 3)

    def update_summary(self, s: dict):
        self._value_labels["avg_hr"].setText(f"{s.get('avg_hr', 0):.0f}")
        self._value_labels["range_hr"].setText(f"{s.get('min_hr', 0):.0f} / {s.get('max_hr', 0):.0f}")
        self._value_labels["beats"].setText(f"{s.get('total_beats', 0):,}")
        self._value_labels["pauses"].setText(str(s.get("pauses", 0)))
        self._value_labels["quality"].setText(f"{s.get('avg_quality', 0) * 100:.1f}")
        self._value_labels["sdnn"].setText(f"{s.get('sdnn', 0):.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. HOLTER OVERVIEW PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterOverviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Overview")
        title.setStyleSheet(f"color:{COL_BLACK};font-size:14px;font-weight:bold;background:{COL_GREEN};"
                            f"padding:7px;border-radius:4px;")
        layout.addWidget(title)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Name", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.setStyleSheet(_table_style())
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, 1)

    def update_summary(self, s: dict):
        rows = [
            ("Total Beats",          f"{s.get('total_beats', 0):,}"),
            ("AVG Heart Rate",       f"{s.get('avg_hr', 0):.0f} bpm"),
            ("Max HR",               f"{s.get('max_hr', 0):.0f} bpm"),
            ("Min HR",               f"{s.get('min_hr', 0):.0f} bpm"),
            ("Sinus Max HR",         f"{s.get('max_hr', 0):.0f} bpm"),
            ("Sinus Min HR",         f"{s.get('min_hr', 0):.0f} bpm"),
            ("Longest RR Interval",  f"{s.get('longest_rr_ms', 0)/1000:.2f}s"),
            ("RRI (≥2.0s)",          str(s.get('pauses', 0))),
            ("Tachycardia Beats",    str(s.get('tachy_beats', 0))),
            ("Bradycardia Beats",    str(s.get('brady_beats', 0))),
            ("X Total",              str(s.get('pauses', 0))),
            ("SDNN (HRV)",           f"{s.get('sdnn', 0):.1f} ms"),
            ("rMSSD (HRV)",          f"{s.get('rmssd', 0):.1f} ms"),
            ("pNN50 (HRV)",          f"{s.get('pnn50', 0):.2f}%"),
            ("ST Elevation",         "—"),
            ("ST Depression",        "—"),
            ("Signal Quality",       f"{s.get('avg_quality', 1.0)*100:.1f}%"),
            ("Chunks Analyzed",      str(s.get('chunks_analyzed', 0))),
        ]
        self._table.setRowCount(len(rows))
        for i, (name, value) in enumerate(rows):
            ni = QTableWidgetItem(name)
            ni.setForeground(QColor(COL_GREEN))
            vi = QTableWidgetItem(value)
            vi.setForeground(QColor(COL_WHITE))
            vi.setFont(QFont("Arial", 12, QFont.Bold))
            self._table.setItem(i, 0, ni)
            self._table.setItem(i, 1, vi)
        self._table.resizeRowsToContents()


# ══════════════════════════════════════════════════════════════════════════════
# 5. HOLTER HRV PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterHRVPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        tab_row = QHBoxLayout()
        self._hrv_event_btn = QPushButton("HRV Event")
        self._hrv_event_btn.setStyleSheet(_style_active_btn())
        self._hrv_trend_btn = QPushButton("HRV Tendency")
        self._hrv_trend_btn.setStyleSheet(_style_btn())
        tab_row.addWidget(self._hrv_event_btn)
        tab_row.addWidget(self._hrv_trend_btn)
        tab_row.addStretch()
        layout.addLayout(tab_row)

        cols = ["Type", "Start at", "Duration", "Mean NN", "SDNN", "SDANN", "TRIIDX", "pNN50", "Status"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setStyleSheet(_table_style())
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, 1)

        # Bottom stats strip
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}")
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setSpacing(8)
        stats_layout.setContentsMargins(12, 8, 12, 8)
        self._summary_labels = {}
        stat_defs = [("NNs", "nns"), ("Mean NN", "mean_nn"), ("SDNN", "sdnn"), ("SDANN", "sdann"),
                     ("rMSSD", "rmssd"), ("pNN50", "pnn50"), ("TRIIDX", "triidx"), ("SDNNIDX", "sdnnidx")]
        for i, (label, key) in enumerate(stat_defs):
            row, col = divmod(i, 4)
            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
            val = QLabel("—")
            val.setStyleSheet(f"color:{COL_GREEN};font-size:14px;font-weight:bold;"
                              f"background:{COL_DARK};border:1px solid {COL_GREEN_DRK};"
                              f"border-radius:10px;padding:4px 10px;min-width:70px;")
            val.setAlignment(Qt.AlignCenter)
            stats_layout.addWidget(lbl, row * 2, col)
            stats_layout.addWidget(val, row * 2 + 1, col)
            self._summary_labels[key] = val
        layout.addWidget(stats_frame)

        btn_row = QHBoxLayout()
        for lbl in ["Insert", "Reset", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            btn_row.addWidget(btn, 1)
        layout.addLayout(btn_row)

    def update_hrv(self, metrics_list: list, summary: dict):
        hourly: dict = {}
        for m in metrics_list:
            h = int(m.get('t', 0) // 3600)
            hourly.setdefault(h, []).append(m)

        rows = []
        all_rr = [m.get('rr_ms', 0) for m in metrics_list if m.get('rr_ms', 0) > 0]
        if all_rr:
            rows.append(("Entire", "—", f"{len(metrics_list)*30//60:02d}:{len(metrics_list)*30%60:02d}",
                         f"{int(np.mean(all_rr))}ms", f"{summary.get('sdnn', 0):.0f}ms",
                         f"{summary.get('sdnn', 0)*0.82:.0f}ms", f"{27:.2f}",
                         f"{summary.get('pnn50', 0):.2f}%", ""))
        for h in sorted(hourly.keys()):
            chunks = hourly[h]
            rr_vals = [c.get('rr_ms', 0) for c in chunks if c.get('rr_ms', 0) > 0]
            rr_stds = [c.get('rr_std', 0) for c in chunks if c.get('rr_std', 0) > 0]
            pnn50s = [c.get('pnn50', 0) for c in chunks]
            if not rr_vals: continue
            rows.append(("Hour", f"{h:02d}:00", "01:00",
                         f"{int(np.mean(rr_vals))}ms",
                         f"{int(np.mean(rr_stds))}ms" if rr_stds else "—",
                         "—", "—",
                         f"{np.mean(pnn50s):.2f}%" if pnn50s else "—", ""))

        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setForeground(QColor(COL_WHITE if j > 0 else COL_GREEN))
                self._table.setItem(i, j, item)

        s = summary
        for key, fmt in [("nns", str(s.get('total_beats', 0))),
                          ("mean_nn", f"{s.get('avg_hr', 0):.0f}ms"),
                          ("sdnn", f"{s.get('sdnn', 0):.0f}ms"),
                          ("sdann", f"{s.get('sdnn', 0)*0.82:.0f}ms"),
                          ("rmssd", f"{s.get('rmssd', 0):.0f}ms"),
                          ("pnn50", f"{s.get('pnn50', 0):.2f}%"),
                          ("triidx", "—"), ("sdnnidx", "—")]:
            if key in self._summary_labels:
                self._summary_labels[key].setText(fmt)


# ══════════════════════════════════════════════════════════════════════════════
# 6. HOLTER LORENZ / REPLAY PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterReplayPanel(QWidget):
    seek_requested = pyqtSignal(float)
    lead_changed   = pyqtSignal(int)

    def __init__(self, parent=None, duration_sec: float = 86400):
        super().__init__(parent)
        self.duration_sec = max(1, duration_sec)
        self._strip_length_sec = 10.0
        self.setStyleSheet(f"background:{COL_DARK};")
        self._replay_engine = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Top: RR scatter / Lorenz (left) + ECG strip (right)
        top_splitter = QSplitter(Qt.Horizontal)

        # Lorenz / scatter widget
        self._lorenz_frame = QFrame()
        self._lorenz_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};"
                                          f"border-radius:6px;min-width:240px;min-height:200px;}}")
        self._lorenz_layout = QVBoxLayout(self._lorenz_frame)
        self._lorenz_layout.setContentsMargins(4, 4, 4, 4)
        lorenz_title = QLabel("RR Interval Time Scatter Plot")
        lorenz_title.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
        lorenz_title.setAlignment(Qt.AlignCenter)
        self._lorenz_layout.addWidget(lorenz_title)
        self._lorenz_canvas = LorenzCanvas(parent=self._lorenz_frame)
        self._lorenz_layout.addWidget(self._lorenz_canvas, 1)
        top_splitter.addWidget(self._lorenz_frame)

        # ECG strip area (right)
        ecg_right = QFrame()
        ecg_right.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        ecg_right_layout = QVBoxLayout(ecg_right)
        ecg_right_layout.setContentsMargins(6, 6, 6, 6)
        ecg_right_layout.setSpacing(4)

        # CH1 / CH2 / CH3 strips
        self._ch_strips = []
        for i in range(12):
            strip = ECGStripCanvas(height=60, color="#00FF00", pen_width=0.7)
            strip.set_gain(1.0)
            self._ch_strips.append(strip)
            ecg_right_layout.addWidget(strip)

        # Mini overview strip at bottom
        self._mini_strip = ECGStripCanvas(height=40, color="#00AA00")
        ecg_right_layout.addWidget(self._mini_strip)
        top_splitter.addWidget(ecg_right)
        top_splitter.setSizes([300, 700])
        layout.addWidget(top_splitter, 2)

        # Scrub slider row
        slider_row = QHBoxLayout()
        self._time_start_label = QLabel("00:00:00")
        self._time_start_label.setStyleSheet(f"color:{COL_TIMESTAMP};font-family:monospace;font-size:12px;border:none;")
        slider_row.addWidget(self._time_start_label)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, int(self.duration_sec))
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal{{height:8px;background:{COL_DARK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}
            QSlider::handle:horizontal{{background:{COL_GREEN};border:1px solid {COL_WHITE};border-radius:9px;
                width:18px;height:18px;margin:-6px 0;}}
            QSlider::sub-page:horizontal{{background:{COL_GREEN_DRK};border-radius:4px;}}
        """)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._slider, 1)
        self._pos_label = QLabel("00:00:00")
        self._pos_label.setStyleSheet(f"color:{COL_TIMESTAMP};font-family:monospace;font-size:14px;font-weight:bold;"
                                      f"background:{COL_BLACK};padding:4px;border:1px solid {COL_GREEN_DRK};border-radius:4px;border:none;")
        slider_row.addWidget(self._pos_label)
        layout.addLayout(slider_row)

        # Transport + controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)

        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setStyleSheet(_style_btn())
        self._play_btn.setFixedHeight(30)
        self._play_btn.setMinimumWidth(80)
        self._play_btn.clicked.connect(self._toggle_playback)
        ctrl_row.addWidget(self._play_btn)

        for speed_lbl in ["0.5x", "1x", "2x", "4x"]:
            btn = QPushButton(speed_lbl)
            btn.setStyleSheet(_style_btn(COL_DARK, COL_GREEN, COL_GREEN_DRK))
            btn.setFixedHeight(28)
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda _, s=speed_lbl: self._set_speed(s))
            ctrl_row.addWidget(btn)

        sep = QLabel("|")
        sep.setStyleSheet(f"color:{COL_GREEN_DRK};")
        ctrl_row.addWidget(sep)

        lbl_lead = QLabel("Lead:")
        lbl_lead.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;border:none;")
        ctrl_row.addWidget(lbl_lead)
        self._lead_combo = QComboBox()
        self._lead_combo.addItems(["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"])
        self._lead_combo.setCurrentIndex(1)
        self._lead_combo.setStyleSheet(f"background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                                       f"padding:4px;border-radius:4px;font-weight:bold;")
        self._lead_combo.currentIndexChanged.connect(self.lead_changed)
        ctrl_row.addWidget(self._lead_combo)

        ctrl_row.addSpacing(12)

        # Event jump buttons
        for lbl_txt, ev, d in [("◀ AF","AF","prev"),("AF ▶","AF","next"),
                               ("◀ Brady","Brady","prev"),("Brady ▶","Brady","next"),
                               ("◀ Tachy","Tachy","prev"),("Tachy ▶","Tachy","next")]:
            btn = QPushButton(lbl_txt)
            btn.setStyleSheet(_style_btn(COL_BLACK, COL_GREEN, COL_GREEN_DRK))
            btn.setFixedHeight(28)
            ev_c, d_c = ev, d
            btn.clicked.connect(lambda _, e=ev_c, dd=d_c: self._jump_event(e, dd))
            ctrl_row.addWidget(btn)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Bottom toolbar (like reference image)
        toolbar = QFrame()
        toolbar.setStyleSheet(f"QFrame{{background:{COL_BLACK};border-top:1px solid {COL_GREEN_DRK};}}")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(4, 4, 4, 4)
        toolbar_layout.setSpacing(4)
        self._tool_btns = {}
        for tool in ["Patient information", "Full Disc.", "Goto Template", "Measuring Ruler",
                     "Parallel Ruler", "Magnifying Glass", "Gain Settings",
                     "Paper speed:25mm/s", "Add Event(space)", "Adjust strip position", "Strip Length:10s"]:
            tbtn = QPushButton(tool)
            tbtn.setStyleSheet(f"QPushButton{{background:{COL_DARK};color:{COL_TEXT};border:1px solid {COL_GREEN_DRK};"
                               f"border-radius:3px;padding:3px 6px;font-size:10px;}}"
                               f"QPushButton:hover{{background:#202020;color:{COL_WHITE};}}")
            tbtn.clicked.connect(lambda _, t=tool, b=tbtn: self._set_tool_mode(t, b))
            toolbar_layout.addWidget(tbtn)
            self._tool_btns[tool] = tbtn
        self._tool_btns["Measuring Ruler"].setToolTip(
            "Measure interval in ms and approximate BPM between two points."
        )
        self._tool_btns["Parallel Ruler"].setToolTip(
            "Compare beat-to-beat interval regularity using two vertical markers."
        )
        self._tool_btns["Magnifying Glass"].setToolTip(
            "Drag to zoom-highlight a detailed ECG segment for review."
        )
        self._tool_btns["Gain Settings"].setToolTip(
            "Cycle gain (5/10/20/40 mm/mV equivalent) to improve waveform visibility."
        )
        toolbar_layout.addStretch()
        layout.addWidget(toolbar)

    def _set_tool_mode(self, tool_name: str, btn: QPushButton = None):
        if "Goto Template" in tool_name:
            QMessageBox.information(
                self,
                "Holter ECG Software Tools — Explained",
                "Measuring Ruler: measure interval/amplitude and BPM.\n"
                "Parallel Ruler: compare regularity/coupling across beats.\n"
                "Magnifying Glass: zoom-highlight subtle waveform details.\n"
                "Gain Settings: cycle 5/10/20/40 mm/mV-equivalent scaling.\n\n"
                "End-to-end flow:\n"
                "Raw recording → Gain optimization → Magnify flagged events → "
                "Measure intervals (QT/PR/pause) → Parallel comparison → Final report."
            )
            return

        # Handle state cycles for Gain, Speed, Length
        if "Gain Settings" in tool_name:
            # Cycle: 5 -> 10 -> 20 -> 40
            gains = [0.5, 1.0, 2.0, 4.0]
            curr_g = getattr(self, '_curr_gain_idx', 1)
            next_g = (curr_g + 1) % len(gains)
            self._curr_gain_idx = next_g
            val = gains[next_g]
            for s in getattr(self, "_ch_strips", []):
                s.set_gain(val)
            if hasattr(self, "_mini_strip"):
                self._mini_strip.set_gain(val)
            if btn: btn.setText(f"Gain: {int(val*10)}mm/mV")
            return
        elif "Paper speed" in tool_name:
            speeds = [12.5, 25, 50, 100]
            curr_s = getattr(self, '_curr_speed_idx', 1)
            next_s = (curr_s + 1) % len(speeds)
            self._curr_speed_idx = next_s
            val = speeds[next_s]
            for s in getattr(self, "_ch_strips", []):
                s.set_paper_speed(int(val))
            if hasattr(self, "_mini_strip"):
                self._mini_strip.set_paper_speed(int(val))
            if btn: btn.setText(f"Paper speed:{val}mm/s")
            # In a real app, this would change self.duration_shown or similar
            return
        elif "Strip Length" in tool_name:
            lengths = [3, 7, 10, 15, 30]
            curr_l = getattr(self, '_curr_length_idx', 1)
            next_l = (curr_l + 1) % len(lengths)
            self._curr_length_idx = next_l
            val = lengths[next_l]
            self._strip_length_sec = float(val)
            if btn: btn.setText(f"Strip Length:{val}s")
            return

        mode = tool_name if tool_name in ["Measuring Ruler", "Parallel Ruler", "Magnifying Glass"] else "Normal"
        for strip in getattr(self, "_ch_strips", []):
            if hasattr(strip, 'set_mode'):
                strip.set_mode(mode)
        if hasattr(self._mini_strip, 'set_mode'):
            self._mini_strip.set_mode(mode)

    def set_replay_engine(self, engine):
        self._replay_engine = engine
        self._slider.setRange(0, int(engine.duration_sec))
        engine.set_position_callback(self._on_position_update)

    def _on_slider(self, value):
        self._pos_label.setText(_sec_to_hms(value))
        self.seek_requested.emit(float(value))

    def _on_position_update(self, current_sec, duration_sec):
        self._slider.blockSignals(True)
        self._slider.setValue(int(current_sec))
        self._slider.blockSignals(False)
        self._pos_label.setText(_sec_to_hms(current_sec))

    def _toggle_playback(self):
        if not self._replay_engine: return
        if self._replay_engine.is_playing():
            self._replay_engine.pause()
            self._play_btn.setText("▶ Play")
        else:
            self._replay_engine.play()
            self._play_btn.setText("⏸ Pause")

    def _set_speed(self, text: str):
        if self._replay_engine:
            try:
                self._replay_engine.set_speed(float(text.replace("x", "")))
            except Exception:
                pass

    def _jump_event(self, ev_type: str, direction: str):
        if self._replay_engine:
            t = self._replay_engine.seek_to_event(ev_type, direction)
            self.seek_requested.emit(t)

    def update_lorenz(self, metrics_list: list):
        """Update the Lorenz/scatter plot from all individual RR data."""
        rr_all = []
        for m in metrics_list:
            if 'rr_intervals_list' in m:
                rr_all.extend(m['rr_intervals_list'])
            elif m.get('rr_ms', 0) > 200:
                # Fallback to average if individual beats missing
                rr_all.append(m['rr_ms'])
                
        rr_n = [r for r in rr_all if r > 200]
        if len(rr_n) >= 2:
            rr_x = rr_n[:-1]
            rr_y = rr_n[1:]
            self._lorenz_canvas.set_data(rr_x, rr_y)
            
    def set_replay_frame(self, data):
        """Update the 12 channel strips inside the Replay tab."""
        if data is None or data.shape[0] < 12:
            return

        strip_len = int(max(1, getattr(self, "_strip_length_sec", 10.0)) * 500)
        if data.shape[1] > strip_len:
            data = data[:, -strip_len:]

        N = data.shape[1]
        x = np.linspace(0, N/500.0, N) if N > 0 else []
        for i, strip in enumerate(self._ch_strips):
            if i < data.shape[0] and N > 0:
                y = data[i].copy()
                strip.set_data(x, y)


# ── Helper canvas widgets ─────────────────────────────────────────────────────

class LorenzCanvas(QWidget):
    """Simple RR scatter / Poincaré plot."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._x = []
        self._y = []
        self.setMinimumSize(200, 180)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")

    def set_data(self, x, y):
        self._x = list(x)
        self._y = list(y)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        # Axes
        pen = QPen(QColor(COL_GREEN_DRK))
        pen.setWidth(1)
        painter.setPen(pen)
        w, h = self.width(), self.height()
        painter.drawLine(10, h - 10, w - 10, h - 10)
        painter.drawLine(10, 10, 10, h - 10)
        # Identity line
        pen.setColor(QColor("#003300"))
        painter.setPen(pen)
        painter.drawLine(10, h - 10, w - 10, 10)

        if not self._x or not self._y:
            painter.setPen(QPen(QColor(COL_GREEN_DRK)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No RR data")
            return

        all_vals = self._x + self._y
        mn, mx = min(all_vals), max(all_vals)
        rng = max(mx - mn, 1)

        def to_px(val_x, val_y):
            px = int(10 + (val_x - mn) / rng * (w - 20))
            py = int(h - 10 - (val_y - mn) / rng * (h - 20))
            return px, py

        if len(self._x) >= 2:
            tpen = QPen(QColor(COL_WAVE_RED))
            tpen.setWidth(1)
            painter.setPen(tpen)
            prev = None
            for x, y in zip(self._x, self._y):
                px, py = to_px(x, y)
                if prev is not None:
                    painter.drawLine(prev[0], prev[1], px, py)
                prev = (px, py)

        pen = QPen(QColor(COL_GREEN_MID))
        pen.setWidth(1)
        painter.setPen(pen)
        brush = QBrush(QColor(COL_GREEN_MID))
        painter.setBrush(brush)
        for x, y in zip(self._x, self._y):
            px, py = to_px(x, y)
            painter.drawEllipse(px - 2, py - 2, 4, 4)

        # Axis labels
        pen.setColor(QColor(COL_GREEN_DRK))
        painter.setPen(pen)
        painter.drawText(w // 2 - 20, h - 2, f"{int(mn)}-{int(mx)}ms")


class ECGStripCanvas(QWidget):
    """Simple ECG strip renderer with interactive measurement tools."""
    def __init__(self, parent=None, height: int = 80, color: str = "#00FF00", pen_width: float = 0.7):
        super().__init__(parent)
        self._data = np.zeros(200)
        self._color = color
        self._pen_width = pen_width
        self._gain = 1.0
        self._speed = 25
        self.setFixedHeight(height)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")
        self.setMouseTracking(True)
        self._mode = "Normal"
        self._start_pos = None
        self._curr_pos = None
        self._hover_pos = None
        self._fs = 500.0

    def set_gain(self, gain: float):
        self._gain = gain
        self.update()

    def set_paper_speed(self, speed: int):
        self._speed = speed
        self.update()

    def set_mode(self, mode: str):
        self._mode = mode
        self._start_pos = None
        self._curr_pos = None
        self._hover_pos = None
        self.update()

    def set_data(self, *args):
        if len(args) == 2:
            self._data = np.asarray(args[1], dtype=float)
        elif len(args) == 1:
            self._data = np.asarray(args[0], dtype=float)
        self.update()

    def mousePressEvent(self, event):
        if self._mode != "Normal":
            self._start_pos = event.pos()
            self._curr_pos = event.pos()
            self._hover_pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        self._hover_pos = event.pos()
        if self._mode == "Magnifying Glass":
            self.update()
            return
        if self._mode != "Normal" and self._start_pos is not None:
            self._curr_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._mode == "Magnifying Glass":
            self._hover_pos = event.pos()
            self.update()
            return
        if self._mode != "Normal":
            self._curr_pos = event.pos()
            self.update()

    def leaveEvent(self, event):
        self._hover_pos = None
        self.update()
        super().leaveEvent(event)

    def _get_display_signal(self):
        if self._data.size < 2:
            return np.array([]), 0.0, 1.0
        mn, mx = 0.0, 4096.0
        rng = 4096.0
        d = np.clip(((self._data - 2048.0) * self._gain) + 2048.0, mn, mx)
        return d, mn, rng

    def _x_to_index(self, x: int, width: int, n: int) -> int:
        if n <= 1 or width <= 1:
            return 0
        x = max(0, min(x, width - 1))
        return int(round((x / float(width - 1)) * (n - 1)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()
        minor_pen = QPen(QColor(COL_GRID_MINOR))
        minor_pen.setWidth(1)
        major_pen = QPen(QColor(COL_GRID_MAJOR))
        major_pen.setWidth(1)
        for gx in range(0, w, 20):
            painter.setPen(major_pen if gx % 100 == 0 else minor_pen)
            painter.drawLine(gx, 0, gx, h)
        for gy in range(0, h, 20):
            painter.setPen(major_pen if gy % 100 == 0 else minor_pen)
            painter.drawLine(0, gy, w, gy)

        if self._data.size < 2:
            return
        d, mn, rng = self._get_display_signal()
        if d.size < 2:
            return
        
        pen = QPen(QColor(self._color))
        pen.setWidthF(self._pen_width)
        painter.setPen(pen)
        x_scale = w / (len(d) - 1)
        for i in range(1, len(d)):
            x1 = int((i - 1) * x_scale)
            y1 = int(h - (d[i-1] - mn) / rng * h)
            x2 = int(i * x_scale)
            y2 = int(h - (d[i] - mn) / rng * h)
            painter.drawLine(x1, y1, x2, y2)

        if self._mode == "Measuring Ruler" and self._start_pos and self._curr_pos:
            rpen = QPen(QColor("#00FFFF"), 2, Qt.DashLine)
            painter.setPen(rpen)
            painter.drawLine(self._start_pos, self._curr_pos)
            dx = abs(self._curr_pos.x() - self._start_pos.x())
            ms = (dx / max(1, w)) * (len(d) / self._fs) * 1000 if len(d) > 0 else 0
            bpm = 60000 / ms if ms > 0 else 0
            dy_adc = abs(self._curr_pos.y() - self._start_pos.y()) / max(1, h) * rng
            dy_mv = dy_adc / 200.0
            painter.setPen(QPen(QColor("#00FFFF")))
            painter.drawText(self._curr_pos.x(), max(12, self._curr_pos.y() - 6), f"{ms:.0f} ms  {dy_mv:.2f} mV  {bpm:.0f} BPM")
        elif self._mode == "Parallel Ruler" and self._start_pos and self._curr_pos:
            ppen = QPen(QColor("#FFFF00"), 1)
            painter.setPen(ppen)
            painter.drawLine(self._start_pos.x(), 0, self._start_pos.x(), h)
            painter.drawLine(self._curr_pos.x(), 0, self._curr_pos.x(), h)
            dx = abs(self._curr_pos.x() - self._start_pos.x())
            ms = (dx / max(1, w)) * (len(d) / self._fs) * 1000 if len(d) > 0 else 0
            bpm = 60000 / ms if ms > 0 else 0
            painter.drawText(min(self._start_pos.x(), self._curr_pos.x()) + dx//2, 12, f"{ms:.0f} ms  {bpm:.0f} BPM")
        elif self._mode == "Magnifying Glass" and self._hover_pos is not None:
            hover_x = max(0, min(self._hover_pos.x(), w - 1))
            hover_y = max(0, min(self._hover_pos.y(), h - 1))
            src_center = self._x_to_index(hover_x, w, len(d))
            span = max(12, int(len(d) / max(2.0, self._speed / 12.5) / max(2, getattr(self.parent(), "_curr_length_idx", 1) + 2)))
            half = max(8, int(span / max(2, self._gain * 1.5)))
            i0 = max(0, src_center - half)
            i1 = min(len(d), src_center + half)
            sub = d[i0:i1]

            panel_w = min(320, max(220, int(w * 0.34)))
            panel_h = min(180, max(120, int(h * 0.72)))
            panel_x = min(w - panel_w - 10, hover_x + 24)
            panel_y = max(8, hover_y - panel_h - 18)
            if panel_x < 8:
                panel_x = 8
            if panel_y < 8:
                panel_y = min(h - panel_h - 8, hover_y + 18)

            panel_rect = QRect(panel_x, panel_y, panel_w, panel_h)
            inner = panel_rect.adjusted(10, 10, -10, -10)

            painter.setBrush(QColor(8, 8, 8, 235))
            painter.setPen(QPen(QColor(COL_YELLOW), 3))
            painter.drawRoundedRect(panel_rect, 12, 12)

            painter.setPen(QPen(QColor(COL_GRID_MINOR), 1))
            for frac in (0.25, 0.5, 0.75):
                gx = int(inner.left() + inner.width() * frac)
                gy = int(inner.top() + inner.height() * frac)
                painter.drawLine(gx, inner.top(), gx, inner.bottom())
                painter.drawLine(inner.left(), gy, inner.right(), gy)

            if len(sub) > 1:
                sub_min = max(mn, float(np.min(sub)))
                sub_max = min(mn + rng, float(np.max(sub)))
                pad = max(80.0, (sub_max - sub_min) * 0.35)
                view_min = max(mn, sub_min - pad)
                view_max = min(mn + rng, sub_max + pad)
                view_rng = max(1.0, view_max - view_min)

                path_pen = QPen(QColor(self._color))
                path_pen.setWidthF(2.0)
                painter.setPen(path_pen)
                x_scale_sub = inner.width() / max(1, len(sub) - 1)
                prev = None
                for i in range(len(sub)):
                    xx = int(inner.left() + i * x_scale_sub)
                    yy = int(inner.bottom() - ((sub[i] - view_min) / view_rng) * inner.height())
                    if prev is not None:
                        painter.drawLine(prev[0], prev[1], xx, yy)
                    prev = (xx, yy)

                focus_x = int(inner.left() + ((src_center - i0) / max(1, len(sub) - 1)) * inner.width())
                focus_y = int(inner.bottom() - ((d[src_center] - view_min) / view_rng) * inner.height())
                painter.setPen(QPen(QColor("#ffffff"), 1))
                painter.drawLine(focus_x, inner.top(), focus_x, inner.bottom())
                painter.drawLine(inner.left(), focus_y, inner.right(), focus_y)

            painter.setPen(QPen(QColor(COL_WHITE)))
            painter.drawText(panel_rect.left() + 10, panel_rect.bottom() - 10, f"{getattr(self.parent(), '_curr_gain_idx', 1) + 2}x")


# ══════════════════════════════════════════════════════════════════════════════
# 7. HOLTER EVENTS PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterEventsPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._events = []
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: event list + stats
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        ev_title = QLabel("Events")
        ev_title.setStyleSheet(f"color:{COL_BLACK};font-size:13px;font-weight:bold;background:{COL_GREEN};padding:5px;border-radius:4px;")
        left_layout.addWidget(ev_title)

        cols = ["Event name", "Start Time", "Chan.", "Print Len."]
        self._ev_table = QTableWidget(0, len(cols))
        self._ev_table.setHorizontalHeaderLabels(cols)
        self._ev_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._ev_table.setStyleSheet(_table_style())
        self._ev_table.verticalHeader().setVisible(False)
        self._ev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._ev_table.cellClicked.connect(self._on_event_clicked)
        left_layout.addWidget(self._ev_table, 1)

        # Stats below table
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"QFrame{{background:{COL_DARK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(8, 6, 8, 6)
        sf_layout.setSpacing(6)
        self._stat_labels = {}
        for i, (key, label) in enumerate([
            ("hr_max","HR Max"),("hr_min","HR Min"),("hr_smax","Sinus Max HR"),
            ("hr_smin","Sinus Min HR"),("brady","Bradycardia"),("user_ev","User Event"),
        ]):
            row, col = divmod(i, 2)
            l = QLabel(f"{label}:")
            l.setStyleSheet(f"color:{COL_GREEN_DRK};font-size:10px;font-weight:bold;border:none;")
            v = QLabel("—")
            v.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
            sf_layout.addWidget(l, row * 2, col)
            sf_layout.addWidget(v, row * 2 + 1, col)
            self._stat_labels[key] = v
        left_layout.addWidget(stats_frame)
        layout.addWidget(left, 1)

        # Right: navigation
        nav = QWidget()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)
        for label in ["⟵ Prev Event", "Next Event ⟶", "Remove All", "Remove"]:
            btn = QPushButton(label)
            btn.setStyleSheet(_style_btn())
            btn.setFixedHeight(38)
            nav_layout.addWidget(btn)
        nav_layout.addStretch()
        layout.addWidget(nav)

    def load_events(self, events: list, summary: dict):
        self._events = events
        self._ev_table.setRowCount(len(events))
        for i, ev in enumerate(events):
            t_str = _sec_to_hms(ev['timestamp'])
            for j, val in enumerate([ev['label'], t_str, "3", "7s"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(COL_WHITE))
                self._ev_table.setItem(i, j, item)
        s = summary
        for key, fmt in [("hr_max",f"{s.get('max_hr',0):.0f} bpm"),
                          ("hr_min",f"{s.get('min_hr',0):.0f} bpm"),
                          ("hr_smax",f"{s.get('max_hr',0):.0f} bpm"),
                          ("hr_smin",f"{s.get('min_hr',0):.0f} bpm"),
                          ("brady",str(s.get('brady_beats',0))),
                          ("user_ev","1")]:
            if key in self._stat_labels:
                self._stat_labels[key].setText(fmt)

    def _on_event_clicked(self, row, col):
        if row < len(self._events):
            self.seek_requested.emit(self._events[row]['timestamp'])


# ══════════════════════════════════════════════════════════════════════════════
# 8. HOLTER WAVE GRID PANEL  (12-lead)
# ══════════════════════════════════════════════════════════════════════════════

class HolterWaveGridPanel(QFrame):
    LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    def __init__(self, parent=None, live_source=None, replay_engine=None):
        super().__init__(parent)
        self.live_source = live_source
        self.replay_engine = replay_engine
        self.window_sec = 10.0
        self._lead_widgets = []
        self._replay_buffer = None
        self.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:12px;}}")
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_waveforms)
        self._timer.start(150)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("12‑Lead Live Workspace")
        title.setStyleSheet(f"color:{COL_GREEN};font-size:16px;font-weight:bold;border:none;")
        subtitle = QLabel("Professional Comphrensive ECG Analysis view with synchronized moving strips.")
        subtitle.setStyleSheet(f"color:{COL_GREEN_DRK};font-size:11px;font-weight:bold;border:none;")
        hcol = QVBoxLayout()
        hcol.addWidget(title)
        hcol.addWidget(subtitle)
        header.addLayout(hcol)
        header.addStretch()
        speed = QLabel("Paper Speed 25mm/s  |  Gain 10mm/mV")
        speed.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;background:{COL_DARK};"
                            f"padding:5px;border:1px solid {COL_GREEN_DRK};border-radius:4px;")
        header.addWidget(speed)
        layout.addLayout(header)

        if not HAS_PG:
            fb = QLabel("pyqtgraph not available — install it for live waveforms: pip install pyqtgraph")
            fb.setWordWrap(True)
            fb.setStyleSheet(f"color:{COL_GREEN};font-size:12px;padding:16px;border:none;")
            layout.addWidget(fb)
            return

        # Use a QVBoxLayout for a single vertical column (12:1 format)
        self.grid_layout = QVBoxLayout()
        self.grid_layout.setSpacing(2) # Minimal spacing between leads

        # Create a container widget and a scroll area
        container = QWidget()
        container.setLayout(self.grid_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setStyleSheet("QScrollArea { border: none; background: black; }")

        if HAS_PG:
            pg.setConfigOptions(antialias=True, useOpenGL=False, background=COL_BLACK, foreground=COL_GREEN)

        for idx, lead in enumerate(self.LEADS):
            card = QFrame()
            # Use a darker green for borders to make the wave pop
            card.setStyleSheet(f"QFrame{{background:{COL_GRAY};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(4, 2, 4, 2)
            cl.setSpacing(0)
            lbl = QLabel(lead)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;padding-left:4px;")
            cl.addWidget(lbl)
            plot = pg.PlotWidget()
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            plot.setBackground(COL_BLACK)
            # Use a more subtle grid
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.getAxis("left").setStyle(showValues=False)
            plot.getAxis("bottom").setStyle(showValues=False)
            plot.getAxis("left").setPen(pg.mkPen(color='#004400'))
            plot.getAxis("bottom").setPen(pg.mkPen(color='#004400'))
            if lead == "aVR":
                plot.setYRange(0, -4096, padding=0)
            else:
                plot.setYRange(0, 4096, padding=0)
            plot.setMinimumHeight(100) # Taller strips for better visibility of III/aVR
            # Set wave thickness to 0.7mm (approx 0.7 pixels for standard displays)
            curve = plot.plot(pen=pg.mkPen(COL_WAVE_ORANGE, width=0.8))
            cl.addWidget(plot, 1)
            self._lead_widgets.append((curve, plot))
            self.grid_layout.addWidget(card)

        layout.addWidget(scroll, 1)

    def set_replay_engine(self, e): self.replay_engine = e
    def set_live_source(self, s): self.live_source = s

    def set_replay_frame(self, data):
        self._replay_buffer = data
        self.refresh_waveforms()

    def _normalize(self, sig):
        arr = np.asarray(sig, dtype=float).flatten()
        if arr.size == 0:
            return np.full(400, 2048.0, dtype=float)
        # Keep raw values, just ensure length and handle NaNs
        arr = arr[-max(300, int(500 * self.window_sec)):]
        arr = np.nan_to_num(arr, nan=2048.0)
        return arr

    def _to_display_space(self, sig: np.ndarray, lead_name: str) -> np.ndarray:
        """
        Recenter leads for readability.
        - Standard leads centered near midline (2048) in 0..4096 window.
        - aVR centered near -2048 in 0..-4096 window.
        """
        baseline = float(np.median(sig)) if sig.size else 2048.0
        centered = sig - baseline
        if lead_name == "aVR":
            return np.clip((-centered) - 2048.0, -4096.0, 0.0)
        return np.clip(centered + 2048.0, 0.0, 4096.0)

    def refresh_waveforms(self):
        if not self._lead_widgets:
            return
        if self._replay_buffer is not None:
            lead_data = [self._normalize(s) for s in self._replay_buffer]
        elif self.replay_engine is not None:
            try:
                data = self.replay_engine.get_all_leads_data(window_sec=self.window_sec)
                lead_data = [self._normalize(s) for s in data]
            except Exception:
                return
        else:
            src = getattr(self.live_source, "data", None)
            if not src: return
            lead_data = [self._normalize(src[i]) for i in range(min(len(self.LEADS), len(src)))]
            while len(lead_data) < len(self.LEADS):
                lead_data.append(np.full(400, 2048.0, dtype=float))

        for idx, (curve, plot) in enumerate(self._lead_widgets):
            sig = lead_data[idx] if idx < len(lead_data) else np.full(400, 2048.0)
            lead_name = self.LEADS[idx] if idx < len(self.LEADS) else ""
            sig = self._to_display_space(np.asarray(sig, dtype=float), lead_name)
            # Create time axis based on 500 SPS
            time_axis = np.arange(sig.size, dtype=float) / 500.0
            curve.setData(time_axis, sig)
            # Auto-scroll X axis to show the latest window_sec
            if sig.size > 0:
                max_t = time_axis[-1]
                plot.setXRange(max(0, max_t - self.window_sec), max_t, padding=0)


# ══════════════════════════════════════════════════════════════════════════════
# 9. HOLTER INSIGHT PANEL  (report preview)
# ══════════════════════════════════════════════════════════════════════════════

class HolterInsightPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:10px;}}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        title = QLabel("Comprehensive Report Preview")
        title.setStyleSheet(f"color:{COL_GREEN};font-size:14px;font-weight:bold;background:{COL_DARK};"
                            f"padding:7px;border-radius:4px;border:none;")
        layout.addWidget(title)
        self._report = QTextEdit()
        self._report.setReadOnly(True)
        self._report.setStyleSheet(f"""
            QTextEdit{{background:{COL_BLACK};color:{COL_GREEN};
              border:1px solid {COL_GREEN_DRK};border-radius:8px;padding:10px;font-size:13px;}}
        """)
        layout.addWidget(self._report)

    def update_text(self, patient_info: dict, summary: dict):
        name = patient_info.get("patient_name") or patient_info.get("name") or "Unknown patient"
        age = patient_info.get("age", "—")
        sex = patient_info.get("gender") or patient_info.get("sex") or "—"
        email = patient_info.get("email", "—")
        dur = summary.get("duration_sec", 0) / 3600
        avg_hr = summary.get("avg_hr", 0)
        min_hr = summary.get("min_hr", 0)
        max_hr = summary.get("max_hr", 0)
        quality = summary.get("avg_quality", 0) * 100
        arrhythmias = summary.get("arrhythmia_counts", {})
        top = ", ".join(f"{k} ({v})" for k, v in sorted(arrhythmias.items(), key=lambda x: -x[1])[:4]) \
              or "No clinically significant arrhythmia burden detected."
        rhythm = ("predominantly tachycardic trend" if avg_hr >= 100
                  else "predominantly bradycardic trend" if 0 < avg_hr <= 60
                  else "predominantly sinus-range rhythm")
        text = (
            f"Patient: {name} | Age/Sex: {age}/{sex} | Email: {email}\n\n"
            f"Study summary:\n"
            f"• Recording duration: {dur:.1f} hours\n"
            f"• Average heart rate: {avg_hr:.0f} bpm (range {min_hr:.0f}–{max_hr:.0f} bpm)\n"
            f"• Signal quality: {quality:.1f}%\n"
            f"• Longest RR interval: {summary.get('longest_rr_ms',0):.0f} ms\n"
            f"• HRV profile: SDNN {summary.get('sdnn',0):.1f} ms, "
            f"rMSSD {summary.get('rmssd',0):.1f} ms, pNN50 {summary.get('pnn50',0):.2f}%\n\n"
            f"Interpretation:\n"
            f"The recording demonstrates a {rhythm}. Key events: {top}\n\n"
            f"Suggested final report wording:\n"
            f'"Comphrensive ECG Analysis monitoring for {name} shows {rhythm} with an average heart rate of '
            f'{avg_hr:.0f} bpm. The minimum recorded rate was {min_hr:.0f} bpm and the '
            f'maximum recorded rate was {max_hr:.0f} bpm. Overall signal quality was '
            f'{quality:.1f}%, enabling comprehensive review of the 12-lead trends and event strips."'
        )
        self._report.setPlainText(text)


# ══════════════════════════════════════════════════════════════════════════════
# 10. RECORD MANAGEMENT PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterRecordManagementPanel(QWidget):
    session_selected = pyqtSignal(str)  # session dir path

    def __init__(self, output_dir: str = "recordings"):
        super().__init__()
        self.output_dir = output_dir
        self._build_ui()
        self.refresh_records()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search patient / reporter / status")
        self._search.setStyleSheet(f"QLineEdit{{background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                                   f"border-radius:4px;padding:6px;font-size:12px;}}")
        self._search.textChanged.connect(self.refresh_records)
        self._filter = QComboBox()
        self._filter.addItems(["All", "Today", "Yesterday", "This Week", "This Month", "This Year"])
        self._filter.setStyleSheet(f"QComboBox{{background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                                   f"border-radius:4px;padding:6px;font-size:12px;}}")
        self._filter.currentTextChanged.connect(self.refresh_records)
        actions.addWidget(QLabel("Search:", styleSheet=f"color:{COL_GREEN};font-size:12px;"))
        actions.addWidget(self._search, 2)
        actions.addWidget(QLabel("Filter:", styleSheet=f"color:{COL_GREEN};font-size:12px;"))
        actions.addWidget(self._filter)
        for txt in ["Browse", "Import", "Export", "Backup", "Delete"]:
            btn = QPushButton(txt)
            btn.setStyleSheet(_style_btn())
            actions.addWidget(btn)
        layout.addLayout(actions)

        cols = ["Name","Age","Gender","Record Time","Duration","Channel","Import Time","Status","Reporter","Conclusion"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setStyleSheet(_table_style())
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self._table, 1)

    def refresh_records(self):
        self._table.setRowCount(0)
        if not os.path.isdir(self.output_dir): return
        query = self._search.text().strip().lower()
        rows = []
        for name in sorted(os.listdir(self.output_dir), reverse=True):
            session_dir = os.path.join(self.output_dir, name)
            if not os.path.isdir(session_dir): continue
            if not os.path.exists(os.path.join(session_dir, "recording.ecgh")): continue
            parts = name.split("_", 3)
            rec_time = "_".join(parts[:2]).replace("_", " ") if len(parts) >= 2 else name[:19]
            p_name = parts[-1].replace("_", " ") if len(parts) >= 3 else "Unknown"
            row_values = [p_name, "-", "-", rec_time, "-", "3", rec_time, "Completed", "System", "-"]
            if query and not any(query in str(v).lower() for v in row_values): continue
            rows.append((row_values, session_dir))

        for row_values, session_dir in rows:
            r = self._table.rowCount()
            self._table.insertRow(r)
            for c, v in enumerate(row_values):
                item = QTableWidgetItem(str(v))
                item.setForeground(QColor(COL_GREEN if c == 0 else COL_WHITE))
                item.setData(Qt.UserRole, session_dir)
                self._table.setItem(r, c, item)

    def _on_double_click(self, index):
        row = index.row()
        item = self._table.item(row, 0)
        if item:
            path = item.data(Qt.UserRole)
            if path:
                self.session_selected.emit(path)


# ══════════════════════════════════════════════════════════════════════════════
# 11. HISTOGRAM PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterHistogramPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._metrics = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        title = QLabel("Histogram — RR Interval Distribution")
        title.setStyleSheet(f"color:{COL_GREEN};font-size:14px;font-weight:bold;border:none;")
        title_row.addWidget(title, 1)
        self._type_combo = QComboBox()
        self._type_combo.addItems(["RR Interval", "Heart Rate", "RRI Ratio"])
        self._type_combo.setStyleSheet(f"QComboBox{{background:{COL_DARK};color:{COL_GREEN};"
                                       f"border:1px solid {COL_GREEN_DRK};padding:4px;border-radius:4px;}}")
        self._type_combo.currentTextChanged.connect(lambda _: self._draw())
        title_row.addWidget(self._type_combo)
        layout.addLayout(title_row)

        # Mode buttons
        btn_row = QHBoxLayout()
        for lbl in ["RRI Ranking", "Time Ranking", "Prematurity Ranking", "Similarity Ranking"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            btn_row.addWidget(btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Histogram canvas
        self._hist_canvas = HistogramCanvas()
        layout.addWidget(self._hist_canvas, 1)

        # Bottom stats
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}")
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setContentsMargins(10, 6, 10, 6)
        self._hist_stats = {}
        for i, (key, lbl) in enumerate([("nns","NNs"),("mean_nn","Mean NN"),
                                         ("sdnn","SDNN"),("sdann","SDANN"),
                                         ("rmssd","rMSSD"),("pnn50","pNN50"),
                                         ("triidx","TRIIDX"),("sdnnidx","SDNNIDX")]):
            col = i % 4
            row = i // 4
            l = QLabel(f"{lbl}:")
            l.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
            v = QLabel("—")
            v.setStyleSheet(f"color:{COL_WHITE};font-size:13px;font-weight:bold;border:none;")
            stats_layout.addWidget(l, row*2, col)
            stats_layout.addWidget(v, row*2+1, col)
            self._hist_stats[key] = v
        layout.addWidget(stats_frame)

        # ECG strip
        self._strip = ECGStripCanvas(height=60)
        layout.addWidget(self._strip)

    def update_from_metrics(self, metrics_list: list):
        self._metrics = metrics_list
        self._draw()

    def _draw(self):
        rr_all = []
        for m in self._metrics:
            if 'rr_intervals_list' in m:
                rr_all.extend(m['rr_intervals_list'])
            elif m.get('rr_ms', 0) > 200:
                rr_all.append(m['rr_ms'])
                
        rr_vals = [r for r in rr_all if r > 200]
        self._hist_canvas.set_data(rr_vals)
        if rr_vals:
            arr = np.array(rr_vals)
            self._hist_stats["nns"].setText(str(len(arr)))
            self._hist_stats["mean_nn"].setText(f"{arr.mean():.0f}ms")
            self._hist_stats["sdnn"].setText(f"{arr.std():.0f}ms")
            self._hist_stats["sdann"].setText("—")
            d = np.diff(arr)
            rmssd = np.sqrt(np.mean(d**2)) if len(d) > 0 else 0
            self._hist_stats["rmssd"].setText(f"{rmssd:.0f}ms")
            pnn50 = 100.0 * np.sum(np.abs(d) > 50) / len(d) if len(d) > 0 else 0
            self._hist_stats["pnn50"].setText(f"{pnn50:.2f}%")
            self._hist_stats["triidx"].setText("—")
            self._hist_stats["sdnnidx"].setText("—")


class HistogramCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")

    def set_data(self, rr_values):
        self._data = rr_values
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()

        # Grid
        grid_pen = QPen(QColor("#001100"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        for gy in [h//4, h//2, 3*h//4]:
            painter.drawLine(0, gy, w, gy)

        if not self._data or len(self._data) < 2:
            painter.setPen(QPen(QColor(COL_GREEN_DRK)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No RR data")
            return

        arr = np.array(self._data)
        mn, mx = int(arr.min()), int(arr.max())
        n_bins = min(30, (mx - mn) // 20 + 1) if mx > mn else 10
        counts, edges = np.histogram(arr, bins=n_bins, range=(mn, mx))
        if counts.max() == 0: return

        bar_w = max(1, w // n_bins - 2)
        x_scale = w / (mx - mn + 1)
        y_scale = (h - 20) / counts.max()

        brush = QBrush(QColor("#4466AA"))
        painter.setBrush(brush)
        pen = QPen(QColor("#6688CC"))
        pen.setWidth(1)
        painter.setPen(pen)
        for i, count in enumerate(counts):
            bx = int((edges[i] - mn) * x_scale)
            bh = int(count * y_scale)
            painter.drawRect(bx, h - bh - 10, bar_w, bh)

        # Axis labels
        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        painter.drawText(5, h - 2, f"{mn}ms")
        painter.drawText(w - 60, h - 2, f"{mx}ms")


# ══════════════════════════════════════════════════════════════════════════════
# 12. AF ANALYSIS PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterAFPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: AF event list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        title = QLabel("AF Analysis")
        title.setStyleSheet(f"color:{COL_BLACK};font-size:13px;font-weight:bold;background:{COL_GREEN};padding:5px;border-radius:4px;")
        left_layout.addWidget(title)

        cols = ["Start time", "Duration", "Type"]
        self._af_table = QTableWidget(0, len(cols))
        self._af_table.setHorizontalHeaderLabels(cols)
        self._af_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._af_table.setStyleSheet(_table_style())
        self._af_table.verticalHeader().setVisible(False)
        self._af_table.setEditTriggers(QTableWidget.NoEditTriggers)
        left_layout.addWidget(self._af_table, 1)

        no_items = QLabel("There are no items to show.")
        no_items.setStyleSheet(f"color:{COL_GREEN_DRK};font-style:italic;padding:8px;border:none;")
        self._no_items_lbl = no_items
        left_layout.addWidget(no_items)

        nav_row = QHBoxLayout()
        for lbl in ["Analysis Af.AF", "Parameters", "Prev Event", "Next Event", "Remove All", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            nav_row.addWidget(btn)
        left_layout.addLayout(nav_row)
        layout.addWidget(left, 2)

        # Right: ECG strip + Lorenz
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Beat thumbnail grid (like reference image)
        thumb_frame = QFrame()
        thumb_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        thumb_layout = QGridLayout(thumb_frame)
        thumb_layout.setContentsMargins(4, 4, 4, 4)
        thumb_layout.setSpacing(4)
        self._thumb_strips = []
        for row in range(4):
            for col in range(3):
                strip = ECGStripCanvas(height=60)
                thumb_layout.addWidget(strip, row, col)
                self._thumb_strips.append(strip)
        right_layout.addWidget(thumb_frame, 1)

        self._af_ecg_strip = ECGStripCanvas(height=70)
        right_layout.addWidget(self._af_ecg_strip)

        self._af_lorenz = LorenzCanvas()
        self._af_lorenz.setFixedHeight(160)
        right_layout.addWidget(self._af_lorenz)
        layout.addWidget(right, 3)

    def update_from_metrics(self, metrics_list: list, duration_sec: float = 0):
        af_events = [(m['t'], m.get('arrhythmias', [])) for m in metrics_list
                     if any('AF' in a or 'Fibrill' in a for a in m.get('arrhythmias', []))]
        self._af_table.setRowCount(len(af_events))
        if af_events:
            self._no_items_lbl.hide()
            for i, (t, arrhy) in enumerate(af_events):
                for j, val in enumerate([_sec_to_hms(t), "30s", "AF/Af"]):
                    item = QTableWidgetItem(val)
                    item.setForeground(QColor(COL_WHITE))
                    self._af_table.setItem(i, j, item)
        else:
            self._no_items_lbl.show()

    def set_replay_frame(self, data):
        if data is None or data.shape[0] < 1: return
        N = data.shape[1]
        x = np.linspace(0, N/500.0, N) if N > 0 else []
        if N > 0:
            self._af_ecg_strip.set_data(x, data[0].copy())
            for i, ts in enumerate(self._thumb_strips):
                if i < data.shape[0]:
                    ts.set_data(x[:500], data[i,:500].copy() if 500 < N else data[i].copy())


# ══════════════════════════════════════════════════════════════════════════════
# 13. ST TENDENCY PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterSTPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._metrics = []
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: ECG strip + controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # CH1, CH2, CH3 strips
        self._ch_strips = []
        for ch in ["CH1", "CH2", "CH3"]:
            lbl = QLabel(ch)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
            left_layout.addWidget(lbl)
            strip = ECGStripCanvas(height=80)
            left_layout.addWidget(strip)
            self._ch_strips.append(strip)

        # Mini overview
        self._mini_strip = ECGStripCanvas(height=40, color="#00AA00")
        left_layout.addWidget(self._mini_strip)

        nav_row = QHBoxLayout()
        for lbl in ["ReScan", "Next Event", "Remove All", "Remove", "Reset"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            btn.setFixedHeight(30)
            nav_row.addWidget(btn)
        left_layout.addLayout(nav_row)
        layout.addWidget(left, 3)

        # Right: ST tendency charts + conclusion
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        mode_row = QHBoxLayout()
        st_btn = QPushButton("ST")
        st_btn.setStyleSheet(_style_active_btn())
        st_btn.setFixedWidth(50)
        t_btn = QPushButton("T")
        t_btn.setStyleSheet(_style_btn())
        t_btn.setFixedWidth(40)
        mode_row.addWidget(st_btn)
        mode_row.addWidget(t_btn)
        mode_row.addStretch()
        right_layout.addLayout(mode_row)

        st_title = QLabel("ST tendency(mV)")
        st_title.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
        right_layout.addWidget(st_title)

        self._st_canvases = []
        for ch in ["CH1", "CH2", "CH3"]:
            ch_lbl = QLabel(f"{ch}  0 ─────────────────")
            ch_lbl.setStyleSheet(f"color:{COL_GREEN};font-size:10px;border:none;")
            right_layout.addWidget(ch_lbl)
            canvas = STCanvas(height=70)
            right_layout.addWidget(canvas)
            self._st_canvases.append(canvas)

        conclusion_frame = QFrame()
        conclusion_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}")
        cf_layout = QVBoxLayout(conclusion_frame)
        cf_layout.setContentsMargins(8, 6, 8, 6)
        cf_lbl = QLabel("Conclusion")
        cf_lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
        cf_layout.addWidget(cf_lbl)
        self._conclusion_edit = QTextEdit()
        self._conclusion_edit.setFixedHeight(60)
        self._conclusion_edit.setStyleSheet(f"QTextEdit{{background:{COL_DARK};color:{COL_GREEN};"
                                             f"border:none;font-size:11px;padding:4px;}}")
        cf_layout.addWidget(self._conclusion_edit)
        save_row = QHBoxLayout()
        for lbl in ["Save as template", "Quote templates"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            save_row.addWidget(btn)
        cf_layout.addLayout(save_row)
        right_layout.addWidget(conclusion_frame)
        layout.addWidget(right, 2)

    def update_from_metrics(self, metrics_list: list):
        self._metrics = metrics_list
        st_vals = [m.get('st_mv', 0.0) for m in metrics_list]
        for canvas in self._st_canvases:
            canvas.set_data(st_vals)

    def set_replay_frame(self, data):
        if data is None or data.shape[0] < 3: return
        N = data.shape[1]
        x = np.linspace(0, N/500.0, N) if N > 0 else []
        for i, strip in enumerate(self._ch_strips):
            if i < data.shape[0] and N > 0:
                strip.set_data(x, data[i].copy())


class STCanvas(QWidget):
    def __init__(self, parent=None, height: int = 70):
        super().__init__(parent)
        self._data = []
        self.setFixedHeight(height)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")

    def set_data(self, vals):
        self._data = vals
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()
        # Zero line
        pen = QPen(QColor(COL_GREEN_DRK))
        pen.setWidth(1)
        painter.setPen(pen)
        mid = h // 2
        painter.drawLine(0, mid, w, mid)
        if not self._data: return
        d = np.array(self._data)
        mn, mx = min(d.min(), -0.1), max(d.max(), 0.1)
        rng = max(mx - mn, 0.2)
        pen = QPen(QColor(COL_GREEN))
        pen.setWidth(2)
        painter.setPen(pen)
        n = len(d)
        x_scale = w / max(n - 1, 1)
        for i in range(1, n):
            x1 = int((i-1) * x_scale)
            y1 = int(h - 5 - (d[i-1] - mn) / rng * (h - 10))
            x2 = int(i * x_scale)
            y2 = int(h - 5 - (d[i] - mn) / rng * (h - 10))
            painter.drawLine(x1, y1, x2, y2)
        # mV label
        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        if len(d) > 0:
            painter.drawText(w - 70, 14, f"{d[min(len(d)//2,len(d)-1)]:.3f}mV")


# ══════════════════════════════════════════════════════════════════════════════
# 14. EDIT EVENT PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterEditEventPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._events = []
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: event list + stats
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        cols = ["Event name", "Start Time", "Chan.", "Print Len."]
        self._ev_table = QTableWidget(0, len(cols))
        self._ev_table.setHorizontalHeaderLabels(cols)
        self._ev_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._ev_table.setStyleSheet(_table_style())
        self._ev_table.verticalHeader().setVisible(False)
        self._ev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._ev_table.cellClicked.connect(self._on_click)
        left_layout.addWidget(self._ev_table, 1)

        # Stats
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"QFrame{{background:{COL_DARK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(8, 6, 8, 6)
        sf_layout.setSpacing(4)
        self._stat_labels = {}
        for i, (key, lbl) in enumerate([
            ("atrial_ecto","Atrial Ectopic"),("rr_int","Longest RR Interval"),
            ("hr","HR"),("max_hr","HR Max"),("min_hr","HR Min"),
            ("smax_hr","Sinus Max HR"),("smin_hr","Sinus Min HR"),
            ("brady","Bradycardia"),("user_ev","User Event"),("event","Event"),
        ]):
            r, c = divmod(i, 2)
            l = QLabel(f"{lbl}:")
            l.setStyleSheet(f"color:{COL_GREEN_DRK};font-size:10px;border:none;")
            v = QLabel("—")
            v.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
            sf_layout.addWidget(l, r*2, c)
            sf_layout.addWidget(v, r*2+1, c)
            self._stat_labels[key] = v
        left_layout.addWidget(stats_frame)
        layout.addWidget(left, 1)

        # Right: ECG strip + thumbnail
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Tool buttons row
        tool_row = QHBoxLayout()
        for icon in ["↻", "☰", "⏕", "⏖", "⏜", "⏝", "⏟", "⏪"]:
            btn = QPushButton(icon)
            btn.setStyleSheet(_style_btn())
            btn.setFixedSize(30, 30)
            tool_row.addWidget(btn)
        tool_row.addStretch()
        right_layout.addLayout(tool_row)

        # Thumbnail strip
        self._thumb = ECGStripCanvas(height=80)
        right_layout.addWidget(self._thumb)

        # Main ECG strip
        self._ch_strips = []
        for ch in ["CH1", "CH2", "CH3"]:
            lbl = QLabel(ch)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
            right_layout.addWidget(lbl)
            strip = ECGStripCanvas(height=80)
            right_layout.addWidget(strip)
            self._ch_strips.append(strip)

        self._mini = ECGStripCanvas(height=40, color="#00AA00")
        right_layout.addWidget(self._mini)

        nav_row = QHBoxLayout()
        for lbl in ["⟵ Prev Event", "Next Event ⟶", "Remove All", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            nav_row.addWidget(btn)
        right_layout.addLayout(nav_row)
        layout.addWidget(right, 2)

    def load_events(self, events: list, summary: dict):
        self._events = events
        self._ev_table.setRowCount(len(events))
        for i, ev in enumerate(events):
            for j, val in enumerate([ev['label'], _sec_to_hms(ev['timestamp']), "3", "7s"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(COL_WHITE))
                self._ev_table.setItem(i, j, item)
        s = summary
        for key, fmt in [("hr",f"{s.get('avg_hr',0):.0f} bpm"),
                          ("max_hr",f"{s.get('max_hr',0):.0f} bpm"),
                          ("min_hr",f"{s.get('min_hr',0):.0f} bpm"),
                          ("smax_hr",f"{s.get('max_hr',0):.0f} bpm"),
                          ("smin_hr",f"{s.get('min_hr',0):.0f} bpm"),
                          ("brady",str(s.get('brady_beats',0))),
                          ("user_ev","1"),("event","1"),
                          ("rr_int",f"{s.get('longest_rr_ms',0):.0f} ms"),
                          ("atrial_ecto","1")]:
            if key in self._stat_labels:
                self._stat_labels[key].setText(fmt)

    def _on_click(self, row, col):
        if row < len(self._events):
            self.seek_requested.emit(self._events[row]['timestamp'])

    def set_replay_frame(self, data):
        if data is None or data.shape[0] < 3: return
        N = data.shape[1]
        x = np.linspace(0, N/500.0, N) if N > 0 else []
        for i, strip in enumerate(self._ch_strips):
            if i < data.shape[0] and N > 0:
                strip.set_data(x, data[i].copy())
        if N > 0:
            self._thumb.set_data(x[:500], data[0,:500].copy() if 500 < N else data[0].copy())
            self._mini.set_data(x, data[0].copy())


# ══════════════════════════════════════════════════════════════════════════════
# 15. EDIT STRIPS PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterEditStripsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: event list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        cols = ["Event name", "Start Time", "Chan.", "Print Len."]
        self._ev_table = QTableWidget(0, len(cols))
        self._ev_table.setHorizontalHeaderLabels(cols)
        self._ev_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._ev_table.setStyleSheet(_table_style())
        self._ev_table.verticalHeader().setVisible(False)
        left_layout.addWidget(self._ev_table, 1)

        nav_row = QHBoxLayout()
        for lbl in ["⟵", "⟶", "Remove All", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            nav_row.addWidget(btn)
        left_layout.addLayout(nav_row)
        layout.addWidget(left, 1)

        # Right: 4 thumbnail grids (Max HR, Min HR, Sinus Max, Sinus Min)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Tool buttons
        tool_row = QHBoxLayout()
        for icon in ["↻", "☰", "⏕", "⏖", "⏜", "⏝", "⏟", "⏪"]:
            btn = QPushButton(icon)
            btn.setStyleSheet(_style_btn())
            btn.setFixedSize(30, 30)
            tool_row.addWidget(btn)
        tool_row.addStretch()
        right_layout.addLayout(tool_row)

        # 2×2 grid of thumbnail strips
        thumb_grid = QGridLayout()
        thumb_grid.setSpacing(8)
        self._thumb_frames = []
        for row, col, title in [(0,0,"Maximum Heart Rate"),(0,1,"Minimum Heart Rate"),
                                 (1,0,"Sinus Max HR"),(1,1,"Sinus Min HR")]:
            frame = QFrame()
            frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(6, 6, 6, 6)
            fl.setSpacing(2)
            title_lbl = QLabel(title)
            title_lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
            fl.addWidget(title_lbl)
            strip = ECGStripCanvas(height=80)
            fl.addWidget(strip)
            thumb_grid.addWidget(frame, row, col)
            self._thumb_frames.append(strip)
        right_layout.addLayout(thumb_grid, 1)

        self._mini = ECGStripCanvas(height=40, color="#00AA00")
        right_layout.addWidget(self._mini)
        layout.addWidget(right, 2)

    def load_events(self, events: list, summary: dict):
        self._ev_table.setRowCount(len(events))
        for i, ev in enumerate(events):
            for j, val in enumerate([ev['label'], _sec_to_hms(ev['timestamp']), "3", "7s"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(COL_WHITE))
                self._ev_table.setItem(i, j, item)

    def set_replay_frame(self, data):
        if data is None or data.shape[0] < 1: return
        N = data.shape[1]
        x = np.linspace(0, N/500.0, N) if N > 0 else []
        if N > 0:
            for strip in self._thumb_frames:
                strip.set_data(x, data[0].copy())
            self._mini.set_data(x, data[0].copy())


# ══════════════════════════════════════════════════════════════════════════════
# 16. REPORT TABLE PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterReportTablePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Hour-by-Hour Report Table")
        title.setStyleSheet(f"color:{COL_GREEN};font-size:14px;font-weight:bold;border:none;")
        layout.addWidget(title)

        cols = ["Hour", "Beats", "HR Min", "HR Avg", "HR Max",
                "VE Iso.", "VE Coup.", "VE Runs", "VE Total", "VE %",
                "SVE Iso.", "SVE Coup.", "SVE Total", "Pauses"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setStyleSheet(_table_style())
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, 1)

    def update_from_metrics(self, metrics_list: list):
        hourly: Dict[int, list] = {}
        for m in metrics_list:
            h = int(m.get('t', 0) // 3600)
            hourly.setdefault(h, []).append(m)

        rows = []
        total_beats = total_pauses = 0
        for h in sorted(hourly.keys()):
            chunks = hourly[h]
            beats = sum(c.get('beat_count', 0) for c in chunks)
            hr_vals = [c.get('hr_mean', 0) for c in chunks if c.get('hr_mean', 0) > 0]
            hr_min_vals = [c.get('hr_min', 0) for c in chunks if c.get('hr_min', 0) > 0]
            hr_max_vals = [c.get('hr_max', 0) for c in chunks if c.get('hr_max', 0) > 0]
            pauses = sum(c.get('pauses', 0) for c in chunks)
            avg_hr = int(np.mean(hr_vals)) if hr_vals else 0
            min_hr = int(np.min(hr_min_vals)) if hr_min_vals else 0
            max_hr = int(np.max(hr_max_vals)) if hr_max_vals else 0
            total_beats += beats
            total_pauses += pauses
            rows.append([f"{h:02d}:00", str(beats), str(min_hr), str(avg_hr), str(max_hr),
                          "0","0","0","0","0%","0","0","0", str(pauses)])

        # Total row
        rows.append(["Total", str(total_beats), "—", "—", "—",
                      "0","0","0","0","0%","0","0","0", str(total_pauses)])

        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            is_total = (i == len(rows) - 1)
            for j, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(COL_GREEN if j == 0 or is_total else COL_WHITE))
                if is_total:
                    item.setBackground(QColor(COL_GREEN_DRK))
                self._table.setItem(i, j, item)


# ══════════════════════════════════════════════════════════════════════════════
# 17. HOLTER MAIN WINDOW  — Orchestrates everything
# ══════════════════════════════════════════════════════════════════════════════

class HolterMainWindow(QDialog):
    def __init__(self, parent=None, session_dir: str = "",
                 patient_info: dict = None,
                 writer=None,
                 live_source=None,
                 duration_hours: int = 24):
        super().__init__(parent)
        self.setWindowTitle("Comphrensive ECG Analysis Monitor & Analysis")
        self.setMinimumSize(1100, 750)

        screen = QApplication.primaryScreen()
        if screen:
            g = screen.availableGeometry()
            self.resize(max(1100, int(g.width() * 0.92)), max(750, int(g.height() * 0.92)))
        else:
            self.resize(1400, 900)

        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint |
                            Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setStyleSheet(f"QDialog{{background:{COL_BLACK};}}")

        self.session_dir = session_dir
        self.patient_info = patient_info or (writer.patient_info if writer else {})
        self._writer = writer
        self._live_source = live_source
        self._duration_hours = duration_hours
        self._replay_engine = None
        self._metrics_list = []
        self._summary = {}

        if not self.session_dir and writer:
            self.session_dir = getattr(writer, 'session_dir', '')

        self._load_session()
        self._build_ui()

        if self._writer:
            self._live_timer = QTimer(self)
            self._live_timer.timeout.connect(self._update_live_ui)
            self._live_timer.start(1000)

    # ── Session loading ────────────────────────────────────────────────────────

    def _load_session(self):
        self._metrics_list = []
        jsonl_path = os.path.join(self.session_dir, 'metrics.jsonl') if self.session_dir else ''
        if os.path.exists(jsonl_path):
            try:
                with open(jsonl_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._metrics_list.append(json.loads(line))
            except Exception as e:
                print(f"[HolterUI] Could not load metrics: {e}")

        ecgh_path = os.path.join(self.session_dir, 'recording.ecgh') if self.session_dir else ''
        if os.path.exists(ecgh_path):
            try:
                from .file_format import ECGHFileReader
                from .replay_engine import HolterReplayEngine
                self._replay_engine = HolterReplayEngine(ecgh_path)
                self._summary = self._replay_engine.get_summary()
            except Exception as e:
                print(f"[HolterUI] Replay engine error: {e}")
                self._summary = self._build_summary_from_metrics()
        else:
            self._summary = self._build_summary_from_metrics()

    def _build_summary_from_metrics(self) -> dict:
        if not self._metrics_list:
            return {}
        ml = self._metrics_list
        hr_vals = [m['hr_mean'] for m in ml if m.get('hr_mean', 0) > 0]
        beat_counts = [m.get('beat_count', 0) for m in ml]
        rr_stds = [m['rr_std'] for m in ml if m.get('rr_std', 0) > 0]
        rmssds = [m['rmssd'] for m in ml if m.get('rmssd', 0) > 0]
        pnn50s = [m['pnn50'] for m in ml if m.get('pnn50', 0) >= 0]
        qualities = [m['quality'] for m in ml if m.get('quality', 0) > 0]
        arrhy_counts: Dict[str, int] = {}
        for m in ml:
            for a in m.get('arrhythmias', []):
                arrhy_counts[a] = arrhy_counts.get(a, 0) + 1
        all_rr = [m.get('longest_rr', 0) for m in ml]
        return {
            'duration_sec': len(ml) * 30,
            'total_beats': sum(beat_counts),
            'avg_hr': float(np.mean(hr_vals)) if hr_vals else 0.0,
            'max_hr': float(np.max(hr_vals)) if hr_vals else 0.0,
            'min_hr': float(np.min(hr_vals)) if hr_vals else 0.0,
            'sdnn': float(np.mean(rr_stds)) if rr_stds else 0.0,
            'rmssd': float(np.mean(rmssds)) if rmssds else 0.0,
            'pnn50': float(np.mean(pnn50s)) if pnn50s else 0.0,
            'avg_quality': float(np.mean(qualities)) if qualities else 1.0,
            'arrhythmia_counts': arrhy_counts,
            'longest_rr_ms': max(all_rr) if all_rr else 0,
            'tachy_beats': sum(m.get('tachy_beats', 0) for m in ml),
            'brady_beats': sum(m.get('brady_beats', 0) for m in ml),
            'pauses': sum(m.get('pauses', 0) for m in ml),
            'avg_st_mv': float(np.mean([m.get('st_mv', 0) for m in ml])),
            'patient_info': self.patient_info,
            'chunks_analyzed': len(ml),
        }

    # ── Build UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Top title bar ──
        title_bar = QFrame()
        title_bar.setStyleSheet(f"QFrame{{background:{COL_BLACK};border-bottom:2px solid {COL_GREEN};}}")
        title_bar.setFixedHeight(44)
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(14, 0, 14, 0)
        tb_layout.setSpacing(14)
        app_title = QLabel("COMPHRENSIVE ECG ANALYSIS SUITE")
        app_title.setStyleSheet(f"color:{COL_GREEN};font-size:18px;font-weight:bold;border:none;")
        tb_layout.addWidget(app_title)
        sep_v = QLabel("|")
        sep_v.setStyleSheet(f"color:{COL_GREEN_DRK};border:none;")
        tb_layout.addWidget(sep_v)
        dur_text = self._summary.get('duration_sec', 0)
        dur_h = int(dur_text // 3600)
        dur_m = int((dur_text % 3600) // 60)
        self._dur_label = QLabel(f"Duration: {dur_h:02d}h {dur_m:02d}m")
        self._dur_label.setStyleSheet(f"color:{COL_GREEN};font-size:14px;border:none;border-bottom:2px solid {COL_GREEN};padding-bottom:2px;")
        tb_layout.addWidget(self._dur_label)
        tb_layout.addStretch()
        gen_report_btn = QPushButton("📄  Generate Report")
        gen_report_btn.setStyleSheet(_style_btn(COL_GREEN, COL_BLACK, COL_GREEN_MID))
        gen_report_btn.setFixedHeight(32)
        gen_report_btn.clicked.connect(self._generate_report)
        tb_layout.addWidget(gen_report_btn)
        close_btn = QPushButton("✕  Close")
        close_btn.setStyleSheet(_style_btn(COL_GREEN_DRK, COL_WHITE, "#880000"))
        close_btn.setFixedHeight(32)
        close_btn.clicked.connect(self.close)
        tb_layout.addWidget(close_btn)
        main_layout.addWidget(title_bar)

        # ── Status bar (if recording) ──
        if self._writer:
            self._status_bar = HolterStatusBar(self, target_hours=self._duration_hours)
            self._status_bar.stop_requested.connect(self._stop_recording)
            main_layout.addWidget(self._status_bar)

        # ── Summary KPI cards ──
        self._summary_cards = HolterSummaryCards()
        self._summary_cards.setFixedHeight(190)
        self._summary_cards.update_summary(self._summary)
        main_layout.addWidget(self._summary_cards)

        # ── Body: 12-lead grid (left) + tabs (right) ──
        body = QSplitter(Qt.Horizontal)
        body.setStyleSheet(f"QSplitter{{background:{COL_BLACK};}}")

        # Left: 12-lead wave grid
        self._wave_panel = HolterWaveGridPanel(
            parent=self,
            live_source=self._live_source,
            replay_engine=self._replay_engine
        )
        body.addWidget(self._wave_panel)

        # Right: tabs
        right_frame = QFrame()
        right_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};}}")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background: {COL_BLACK};
                border: none;
            }}
            QTabBar::tab {{
                background: {COL_DARK};
                color: {COL_GREEN_DRK};
                border: 1px solid {COL_GREEN_DRK};
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
                border-bottom: none;
                margin-right: 1px;
                min-width: 85px;
                text-align: center;
            }}
            QTabBar::tab:selected {{
                color: {COL_BLACK};
                background: {COL_GREEN};
                border-color: {COL_GREEN};
            }}
            QTabBar::tab:hover:!selected {{
                background: {COL_GREEN_DRK};
                color: {COL_WHITE};
            }}
        """)

        # Overview
        self._overview_panel = HolterOverviewPanel()
        self._overview_panel.update_summary(self._summary)
        self._tabs.addTab(self._overview_panel, "📊 Overview")

        # Record Management
        record_dir = self.session_dir if self.session_dir and os.path.isdir(self.session_dir) \
                     else os.path.join(os.getcwd(), "recordings")
        if os.path.isfile(record_dir):
            record_dir = os.path.dirname(record_dir)
        self._record_mgmt_panel = HolterRecordManagementPanel(output_dir=record_dir)
        self._record_mgmt_panel.session_selected.connect(self.load_completed_session)
        self._tabs.addTab(self._record_mgmt_panel, "📋 Record Mgmt")

        # HRV Analysis
        self._hrv_panel = HolterHRVPanel()
        self._hrv_panel.update_hrv(self._metrics_list, self._summary)
        self._tabs.addTab(self._hrv_panel, "📈 HRV Analysis")

        # Replay (Lorenz + scrub)
        duration = self._summary.get('duration_sec', self._duration_hours * 3600)
        self._replay_panel = HolterReplayPanel(duration_sec=duration)
        if self._replay_engine:
            self._replay_panel.set_replay_engine(self._replay_engine)
            self._replay_panel.seek_requested.connect(self._on_seek_requested)
        self._replay_panel.update_lorenz(self._metrics_list)
        self._tabs.addTab(self._replay_panel, "▶ Replay")

        # Events
        self._events_panel = HolterEventsPanel()
        events = self._replay_engine.get_events_list() if self._replay_engine else []
        self._events_panel.load_events(events, self._summary)
        self._events_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._events_panel, "⚡ Events")

        # Histogram
        self._hist_panel = HolterHistogramPanel()
        self._hist_panel.update_from_metrics(self._metrics_list)
        self._tabs.addTab(self._hist_panel, "📊 Histogram")

        # AF Analysis
        self._af_panel = HolterAFPanel()
        self._af_panel.update_from_metrics(self._metrics_list, duration)
        self._tabs.addTab(self._af_panel, "🔬 AF Analysis")

        # ST Tendency
        self._st_panel = HolterSTPanel()
        self._st_panel.update_from_metrics(self._metrics_list)
        self._tabs.addTab(self._st_panel, "📉 ST Tendency")

        # Edit Event
        self._edit_event_panel = HolterEditEventPanel()
        self._edit_event_panel.load_events(events, self._summary)
        self._edit_event_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._edit_event_panel, "✏️ Edit Event")

        # Edit Strips
        self._edit_strips_panel = HolterEditStripsPanel()
        self._edit_strips_panel.load_events(events, self._summary)
        self._tabs.addTab(self._edit_strips_panel, "🎞️ Edit Strips")

        # Report Table
        self._report_table_panel = HolterReportTablePanel()
        self._report_table_panel.update_from_metrics(self._metrics_list)
        self._tabs.addTab(self._report_table_panel, "📑 Report Table")

        # Report Preview (insight)
        scroll_insight = QScrollArea()
        scroll_insight.setWidgetResizable(True)
        scroll_insight.setFrameShape(QFrame.NoFrame)
        scroll_insight.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:none;}}")
        self._insight_panel = HolterInsightPanel()
        self._insight_panel.update_text(self.patient_info, self._summary)
        scroll_insight.setWidget(self._insight_panel)
        self._tabs.addTab(scroll_insight, "📄 Report Preview")

        right_layout.addWidget(self._tabs)
        body.addWidget(right_frame)
        body.setSizes([720, 480])

        body_scroll = QScrollArea()
        body_scroll.setWidgetResizable(True)
        body_scroll.setFrameShape(QFrame.NoFrame)
        body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        body_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        body_scroll.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:none;}}")
        body_scroll.setWidget(body)
        main_layout.addWidget(body_scroll, 1)

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _on_seek_requested(self, target_sec: float):
        if self._replay_engine:
            self._replay_engine.seek(target_sec)
            try:
                data = self._replay_engine.get_all_leads_data(window_sec=10.0)
                if hasattr(self, '_wave_panel'):
                    self._wave_panel.set_replay_frame(data)
                
                # Broadcast data to all panels that support it
                for panel in [getattr(self, p, None) for p in [
                    '_replay_panel', '_hist_panel', '_af_panel', 
                    '_st_panel', '_edit_event_panel', '_edit_strips_panel', '_events_panel'
                ]]:
                    if panel and hasattr(panel, 'set_replay_frame'):
                        panel.set_replay_frame(data)
                        
            except Exception:
                pass

    def _update_live_ui(self):
        if not self._writer or not self._writer.is_running:
            if hasattr(self, '_live_timer'):
                self._live_timer.stop()
            self._load_session()
            self._refresh_ui()
            return
        stats = self._writer.get_live_stats()
        if hasattr(self, '_status_bar'):
            self._status_bar.update_stats(stats['bpm'], stats['arrhythmias'])
        if hasattr(self, '_wave_panel'):
            self._wave_panel.refresh_waveforms()
        if stats['elapsed'] % 30 < 2:
            self._load_session()
            self._refresh_ui()

    def _refresh_ui(self):
        if hasattr(self, '_summary_cards'):
            self._summary_cards.update_summary(self._summary)
        if hasattr(self, '_insight_panel'):
            self._insight_panel.update_text(self.patient_info, self._summary)
        if hasattr(self, '_overview_panel'):
            self._overview_panel.update_summary(self._summary)
        if hasattr(self, '_hrv_panel'):
            self._hrv_panel.update_hrv(self._metrics_list, self._summary)
        if hasattr(self, '_replay_panel'):
            self._replay_panel.update_lorenz(self._metrics_list)
        if hasattr(self, '_hist_panel'):
            self._hist_panel.update_from_metrics(self._metrics_list)
        if hasattr(self, '_af_panel'):
            self._af_panel.update_from_metrics(self._metrics_list, self._summary.get('duration_sec', 0))
        if hasattr(self, '_st_panel'):
            self._st_panel.update_from_metrics(self._metrics_list)
        if hasattr(self, '_report_table_panel'):
            self._report_table_panel.update_from_metrics(self._metrics_list)
        events = self._replay_engine.get_events_list() if self._replay_engine else []
        if hasattr(self, '_events_panel'):
            self._events_panel.load_events(events, self._summary)
        if hasattr(self, '_edit_event_panel'):
            self._edit_event_panel.load_events(events, self._summary)
        if hasattr(self, '_edit_strips_panel'):
            self._edit_strips_panel.load_events(events, self._summary)
        if hasattr(self, '_wave_panel'):
            self._wave_panel.set_live_source(self._live_source)
            self._wave_panel.set_replay_engine(self._replay_engine)
            self._wave_panel.refresh_waveforms()

        # Update duration label
        dur = self._summary.get('duration_sec', 0)
        dur_h = int(dur // 3600)
        dur_m = int((dur % 3600) // 60)
        if hasattr(self, '_dur_label'):
            self._dur_label.setText(f"Duration: {dur_h:02d}h {dur_m:02d}m")

    def _stop_recording(self):
        if self._writer:
            summary = self._writer.stop()
            self._writer = None
            if hasattr(self, '_status_bar') and self._status_bar is not None:
                self._status_bar.setVisible(False)
                if hasattr(self._status_bar, 'cleanup'):
                    self._status_bar.cleanup()
            
            # Show dialog to collect patient info AFTER recording
            dialog = HolterStartDialog(self, patient_info=self.patient_info or {}, output_dir=self.session_dir)
            dialog.setWindowTitle("Save Comphrensive ECG Analysis Recording Details")
            if dialog.exec_() == QDialog.Accepted:
                patient_info, dur, out_dir = dialog.get_result()
                summary['patient_info'] = patient_info
                self.patient_info = patient_info
                import json
                try:
                    with open(os.path.join(summary.get('session_dir', ''), "patient.json"), 'w') as f:
                        json.dump(patient_info, f, indent=4)
                except Exception as e:
                    print(f"Failed to save patient.json: {e}")

            QMessageBox.information(self, "Recording Complete",
                                    f"Comphrensive ECG Analysis recording saved to:\n{summary.get('session_dir', '')}")
            self.load_completed_session(summary.get('session_dir', ''), summary.get('patient_info', {}))
            
            # Auto-generate report when recording is stopped
            self._generate_report()

    def _generate_report(self):
        from PyQt5.QtWidgets import QProgressDialog
        progress = QProgressDialog("Generating Comphrensive ECG Analysis Report...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setStyleSheet(f"QProgressDialog{{background:{COL_DARK};color:{COL_GREEN};}}")
        progress.show()
        QApplication.processEvents()
        try:
            from .report_generator import generate_holter_report
            path = generate_holter_report(
                session_dir=self.session_dir,
                patient_info=self.patient_info,
                summary=self._summary,
            )
            
            # Save to history
            try:
                from dashboard.history_window import append_history_entry
                h_pat = self.patient_info.copy() if self.patient_info else {}
                if 'patient_name' not in h_pat and 'name' in h_pat:
                    h_pat['patient_name'] = h_pat['name']
                append_history_entry(h_pat, path, report_type="Comphrensive ECG Analysis")
            except Exception as h_err:
                print(f"Failed to append Holter history: {h_err}")
                
            progress.close()
            QMessageBox.information(self, "Report Generated", f"Comphrensive ECG Analysis report saved:\n{path}")
        except Exception as e:
            progress.close()
            QMessageBox.warning(self, "Report Error", f"Could not generate report:\n{e}")

    def attach_writer(self, writer, session_dir: str = "", patient_info: dict = None):
        self._writer = writer
        if session_dir:
            self.session_dir = session_dir
        if patient_info:
            self.patient_info = patient_info
        if writer and not hasattr(self, '_status_bar'):
            self._status_bar = HolterStatusBar(self, target_hours=self._duration_hours)
            self._status_bar.stop_requested.connect(self._stop_recording)
            self.layout().insertWidget(1, self._status_bar)
        if writer and not hasattr(self, '_live_timer'):
            self._live_timer = QTimer(self)
            self._live_timer.timeout.connect(self._update_live_ui)
        if writer and hasattr(self, '_live_timer') and not self._live_timer.isActive():
            self._live_timer.start(1000)
        self._refresh_ui()

    def load_completed_session(self, session_dir: str, patient_info: dict = None):
        self.session_dir = session_dir
        if patient_info:
            self.patient_info = patient_info
        self._writer = None
        self._load_session()
        if hasattr(self, '_replay_panel') and getattr(self, '_replay_engine', None):
            self._replay_panel.set_replay_engine(self._replay_engine)
            try:
                self._replay_panel.seek_requested.disconnect(self._on_seek_requested)
            except Exception:
                pass
            self._replay_panel.seek_requested.connect(self._on_seek_requested)
        self._refresh_ui()
        if hasattr(self, '_tabs'):
            self._tabs.setCurrentIndex(0)

    def closeEvent(self, event):
        if self._replay_engine:
            try:
                self._replay_engine.close()
            except Exception:
                pass
        super().closeEvent(event)

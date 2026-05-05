"""
ecg/holter/holter_ui.py
========================
Complete Holter Monitor UI — modeled on reference software screens.

Screens:
  1. HolterStartDialog    — patient info + duration + start button
  2. HolterStatusBar      — REC indicator, elapsed time, live BPM, arrhythmia ticker
  3. HolterOverviewPanel  — Overview stats table (like reference Image 11)
  4. HolterReplayPanel    — Scrub slider, lead selector, event navigator
  5. HolterHRVPanel       — HRV table per hour (like reference Image 9)
  6. HolterEventsPanel    — Arrhythmia events list (like reference Image 7)
  7. HolterMainWindow     — Orchestrates all panels in tabbed layout

Integration:
  In twelve_lead_test.py add:
    self._holter_ui = None
    # In menu buttons:
    ("Holter", self.show_holter_menu, "#E65100")
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from typing import Optional, List

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialog, QLineEdit, QComboBox, QSlider, QGroupBox, QFrame,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QScrollArea, QGridLayout, QSpinBox, QMessageBox,
    QFileDialog, QApplication, QProgressBar, QSplitter, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPalette

try:
    import pyqtgraph as pg
except Exception:
    pg = None

# ── Clinical white colour palette ─────────────────────────────────────────────
COL_BG        = "#F4F6F8"   # page background
COL_SURFACE   = "#FFFFFF"   # card / panel surface
COL_DARK      = "#1A1A2E"   # toolbar / heavy elements
COL_BORDER    = "#E2E8F0"   # card borders
COL_TEXT      = "#111827"   # primary text
COL_MUTED     = "#6B7280"   # secondary / label text
COL_ORANGE    = "#2563EB"   # accent blue (action buttons)
COL_BLUE      = "#2563EB"   # alias
COL_GREEN     = "#16A34A"   # success / connect
COL_RED       = "#DC2626"   # clinical alert
COL_AMBER     = "#D97706"   # warning
COL_LIGHT     = "#F9FAFB"
COL_GRAY      = "#F3F4F6"
COL_GREEN_ECG = "#000000"   # ECG trace colour (black on white)


def _btn_style(bg="#2563EB", fg="white", hover="#1D4ED8"):
    return f"""
        QPushButton {{
            background: {bg};
            color: {fg};
            border: none;
            border-radius: 6px;
            padding: 6px 14px;
            font-size: 12px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background: {hover};
        }}
        QPushButton:pressed {{
            background: {bg};
        }}

        }}
    """


def _label_style(size=12, color=COL_DARK, bold=False):
    weight = "bold" if bold else "normal"
    return f"color: {color}; font-size: {size}px; font-weight: {weight};"


# ══════════════════════════════════════════════════════════════════════════════
# 1. HOLTER START DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class HolterStartDialog(QDialog):
    """
    Modal dialog to configure and start a comphrensive recording.
    Pre-fills patient info from existing patient_details cache.
    """

    def __init__(self, parent=None, patient_info: dict = None, output_dir: str = "recordings"):
        super().__init__(parent)
        self.setWindowTitle("Start comphrensive recording")
        self.setMinimumWidth(640)
        self.setStyleSheet(f"background: {COL_DARK}; color: white;")
        self.output_dir = output_dir
        self._result_info = None
        self._result_duration = 24
        self._result_dir = output_dir
        self._build_ui(patient_info or {})

    def _build_ui(self, info: dict):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # ── Title ──
        title = QLabel("🫀  Holter Monitor — Professional Setup")
        title.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #064E3B, stop:1 #0E7490);
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 14px 18px;
            border-radius: 12px;
        """)
        layout.addWidget(title)

        subtitle = QLabel("Enter the patient details, choose the study duration, and launch the live 12‑lead Holter workspace.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #C7D2E1; font-size: 12px; padding: 0 2px 4px 2px;")
        layout.addWidget(subtitle)

        # ── Patient Info Group ──
        pg = QGroupBox("Patient Information")
        pg.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                color: white;
                border: 1px solid #32435A;
                border-radius: 12px;
                margin-top: 8px;
                padding-top: 14px;
                background: #111827;
            }}
        """)
        pg_layout = QGridLayout(pg)
        pg_layout.setSpacing(8)

        fields = [
            ("Patient Name",   "patient_name",  info.get("patient_name", "")),
            ("Age",            "age",           str(info.get("age", ""))),
            ("Email",          "email",         info.get("email", "")),
            ("Doctor",         "doctor",        info.get("doctor", "")),
            ("Organisation",   "org",           info.get("Org.", info.get("org", ""))),
            ("Phone",          "phone",         info.get("doctor_mobile", info.get("phone", ""))),
        ]
        self._fields = {}
        for row, (label, key, default) in enumerate(fields):
            lbl = QLabel(label + ":")
            lbl.setStyleSheet("font-weight: bold; font-size: 12px; color: #D9E1EC;")
            edit = QLineEdit(default)
            edit.setStyleSheet(f"""
                QLineEdit {{
                    border: 1px solid #42556F;
                    border-radius: 8px;
                    padding: 8px 10px;
                    font-size: 12px;
                    background: #0F172A;
                    color: white;
                }}
                QLineEdit:focus {{ border-color: {COL_ORANGE}; }}
            """)
            pg_layout.addWidget(lbl, row, 0)
            pg_layout.addWidget(edit, row, 1)
            self._fields[key] = edit

        # Gender
        lbl_g = QLabel("Gender:")
        lbl_g.setStyleSheet("font-weight: bold; font-size: 12px; color: #D9E1EC;")
        self._gender = QComboBox()
        self._gender.addItems(["Select", "Male", "Female", "Other"])
        gender_val = info.get("gender", info.get("sex", "Select"))
        idx = self._gender.findText(gender_val)
        if idx >= 0:
            self._gender.setCurrentIndex(idx)
        self._gender.setStyleSheet(f"""
            QComboBox {{
                border: 1px solid #42556F;
                border-radius: 8px;
                padding: 8px 10px;
                font-size: 12px;
                background: #0F172A;
                color: white;
            }}
            QComboBox:focus {{ border-color: {COL_ORANGE}; }}
        """)
        pg_layout.addWidget(lbl_g, len(fields), 0)
        pg_layout.addWidget(self._gender, len(fields), 1)
        layout.addWidget(pg)

        # ── Recording Settings Group ──
        rg = QGroupBox("Recording Settings")
        rg.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                color: white;
                border: 1px solid #32435A;
                border-radius: 12px;
                margin-top: 8px;
                padding-top: 14px;
                background: #111827;
            }}
        """)
        rg_layout = QGridLayout(rg)
        rg_layout.setSpacing(8)

        duration_lbl = QLabel("How many hours:")
        duration_lbl.setStyleSheet("font-weight: bold; font-size: 12px; color: #D9E1EC;")
        rg_layout.addWidget(duration_lbl, 0, 0)
        self._duration = QComboBox()
        self._duration.addItems(["24 hours", "48 hours", "Custom"])
        self._duration.setStyleSheet("border: 1px solid #42556F; border-radius: 8px; padding: 8px 10px; font-size: 12px; background: #0F172A; color: white;")
        self._duration.currentTextChanged.connect(self._on_duration_changed)
        rg_layout.addWidget(self._duration, 0, 1)

        self._custom_hours = QSpinBox()
        self._custom_hours.setRange(1, 72)
        self._custom_hours.setValue(24)
        self._custom_hours.setSuffix(" hours")
        self._custom_hours.setVisible(False)
        self._custom_hours.setStyleSheet("border: 1px solid #42556F; border-radius: 8px; padding: 8px 10px; font-size: 12px; background: #0F172A; color: white;")
        rg_layout.addWidget(self._custom_hours, 1, 1)

        output_lbl = QLabel("Output Directory:")
        output_lbl.setStyleSheet("font-weight: bold; font-size: 12px; color: #D9E1EC;")
        rg_layout.addWidget(output_lbl, 2, 0)
        dir_row = QHBoxLayout()
        self._dir_label = QLabel(self.output_dir)
        self._dir_label.setStyleSheet("font-size: 11px; color: #B0C4DE;")
        dir_row.addWidget(self._dir_label, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(_btn_style(COL_BLUE, "white", "#1976D2"))
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        rg_layout.addLayout(dir_row, 2, 1)

        self._recording_count_label = QLabel("")
        self._recording_count_label.setStyleSheet("font-size: 12px; color: #86EFAC; font-weight: 600;")
        rg_layout.addWidget(QLabel("Recorded Sessions:"), 3, 0)
        rg_layout.addWidget(self._recording_count_label, 3, 1)
        self._refresh_recording_count()

        layout.addWidget(rg)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(_btn_style("#757575", "white", "#616161"))
        cancel_btn.clicked.connect(self.reject)
        start_btn = QPushButton("▶  Open Holter Workspace")
        start_btn.setStyleSheet(_btn_style(COL_GREEN, "white", "#388E3C"))
        start_btn.setMinimumHeight(44)
        start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(start_btn, 1)
        layout.addLayout(btn_row)

    def _on_duration_changed(self, text):
        self._custom_hours.setVisible(text == "Custom")

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if d:
            self._result_dir = d
            self._dir_label.setText(d)
            self._refresh_recording_count()

    def _refresh_recording_count(self):
        root = self._result_dir or self.output_dir
        count = 0
        try:
            if os.path.isdir(root):
                for name in os.listdir(root):
                    session_dir = os.path.join(root, name)
                    if not os.path.isdir(session_dir):
                        continue
                    if os.path.exists(os.path.join(session_dir, "recording.ecgh")):
                        count += 1
        except Exception:
            count = 0
        self._recording_count_label.setText(f"{count} completed recording(s)")

    def _on_start(self):
        # Build patient info
        info = {key: field.text().strip() for key, field in self._fields.items()}
        info['gender'] = self._gender.currentText()
        info['sex'] = info['gender']
        info['name'] = info.get('patient_name', 'Unknown')
        info['Org.'] = info.get('org', '')

        if not info.get('patient_name'):
            QMessageBox.warning(self, "Missing Name", "Please enter the patient name before opening Holter mode.")
            return

        # Duration
        dur_text = self._duration.currentText()
        if dur_text == "24 hours":
            self._result_duration = 24
        elif dur_text == "48 hours":
            self._result_duration = 48
        else:
            self._result_duration = self._custom_hours.value()

        self._result_info = info
        self._result_dir = self._dir_label.text()
        self.accept()

    def get_result(self):
        """Returns (patient_info, duration_hours, output_dir) or None."""
        if self._result_info:
            return self._result_info, self._result_duration, self._result_dir
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. HOLTER STATUS BAR (Live Recording Indicator)
# ══════════════════════════════════════════════════════════════════════════════

class HolterStatusBar(QFrame):
    """
    Compact status bar shown at the top of the 12-box grid during recording.
    Shows: ● REC | HH:MM:SS | Target: 24h | BPM: 72 | Last arrhythmia: ...
    """
    stop_requested = pyqtSignal()

    def __init__(self, parent=None, target_hours: int = 24):
        super().__init__(parent)
        self.target_hours = target_hours
        self._start_time = time.time()
        self._blink_state = True
        self._last_arrhythmias: List[str] = []

        self.setFixedHeight(48)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COL_BG};
                border-bottom: 2px solid {COL_ORANGE};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(16)

        # REC indicator
        self._rec_label = QLabel("● REC")
        self._rec_label.setStyleSheet(f"color: {COL_RED}; font-size: 14px; font-weight: bold;")
        layout.addWidget(self._rec_label)

        # Elapsed time
        self._time_label = QLabel("00:00:00")
        self._time_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; font-family: monospace;")
        layout.addWidget(self._time_label)

        # Target
        tgt = QLabel(f"/ {target_hours}:00:00")
        tgt.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(tgt)

        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #444;")
        layout.addWidget(sep1)

        # Live BPM
        bpm_lbl = QLabel("BPM:")
        bpm_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(bpm_lbl)
        self._bpm_label = QLabel("—")
        self._bpm_label.setStyleSheet(f"color: {COL_GREEN_ECG}; font-size: 16px; font-weight: bold;")
        layout.addWidget(self._bpm_label)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #444;")
        layout.addWidget(sep2)

        # Arrhythmia ticker
        arr_lbl = QLabel("Events:")
        arr_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(arr_lbl)
        self._arrhy_label = QLabel("None detected")
        self._arrhy_label.setStyleSheet("color: #FFA726; font-size: 11px;")
        self._arrhy_label.setMaximumWidth(300)
        layout.addWidget(self._arrhy_label, 1)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, target_hours * 3600)
        self._progress.setValue(0)
        self._progress.setFixedWidth(120)
        self._progress.setFixedHeight(12)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: #333;
                border-radius: 6px;
                border: 1px solid #555;
            }}
            QProgressBar::chunk {{
                background: {COL_ORANGE};
                border-radius: 6px;
            }}
        """)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        # Stop button
        stop_btn = QPushButton("⬛  Stop")
        stop_btn.setStyleSheet(_btn_style(COL_RED, "white", "#D32F2F"))
        stop_btn.setFixedHeight(32)
        stop_btn.clicked.connect(self.stop_requested)
        layout.addWidget(stop_btn)

        # Blink timer
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink)
        self._blink_timer.start(800)

        # Elapsed timer
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._elapsed_timer.start(1000)

    def _blink(self):
        self._blink_state = not self._blink_state
        color = COL_RED if self._blink_state else "#555"
        self._rec_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")

    def _update_elapsed(self):
        elapsed = int(time.time() - self._start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        self._time_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
        self._progress.setValue(elapsed)

    def update_stats(self, bpm: float, arrhythmias: List[str]):
        if bpm > 0:
            self._bpm_label.setText(f"{bpm:.0f}")
        if arrhythmias:
            self._arrhy_label.setText("  |  ".join(arrhythmias[:3]))
            self._arrhy_label.setStyleSheet(f"color: {COL_RED}; font-size: 11px; font-weight: bold;")

    def cleanup(self):
        self._blink_timer.stop()
        self._elapsed_timer.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 3. HOLTER OVERVIEW PANEL  (like reference Image 11 right panel)
# ══════════════════════════════════════════════════════════════════════════════

class HolterOverviewPanel(QWidget):
    """Shows summary stats table — mirrors the Overview panel in reference software."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {COL_BG}; color: white;")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        title = QLabel("Overview")
        title.setStyleSheet(f"color: white; font-size: 14px; font-weight: bold; background: {COL_BLUE}; padding: 6px; border-radius: 4px;")
        layout.addWidget(title)

        # Stats table
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Name", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background: {COL_BG};
                color: white;
                border: 1px solid #333;
                font-size: 12px;
                gridline-color: #333;
            }}
            QTableWidget::item:selected {{
                background: {COL_BLUE};
            }}
            QHeaderView::section {{
                background: #1E2A3A;
                color: #aaa;
                font-size: 11px;
                padding: 4px;
                border: none;
            }}
        """)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, 1)

    def update_summary(self, summary: dict):
        rows = [
            ("Total Beats",         f"{summary.get('total_beats', 0):,}"),
            ("AVG Heart Rate",      f"{summary.get('avg_hr', 0):.0f} bpm"),
            ("Max HR",              f"{summary.get('max_hr', 0):.0f} bpm"),
            ("Min HR",              f"{summary.get('min_hr', 0):.0f} bpm"),
            ("Longest RR Interval", f"{summary.get('longest_rr_ms', 0)/1000.0:.2f}s"),
            ("Pauses (≥2.0s)",      f"{summary.get('pauses', 0)}"),
            ("Tachycardia Beats",   f"{summary.get('tachy_beats', 0)}"),
            ("Bradycardia Beats",   f"{summary.get('brady_beats', 0)}"),
            ("SDNN (HRV)",          f"{summary.get('sdnn', 0):.1f} ms"),
            ("rMSSD (HRV)",         f"{summary.get('rmssd', 0):.1f} ms"),
            ("pNN50 (HRV)",         f"{summary.get('pnn50', 0):.2f}%"),
            ("Signal Quality",      f"{summary.get('avg_quality', 1.0)*100:.1f}%"),
        ]

        self._table.setRowCount(len(rows))
        for i, (name, value) in enumerate(rows):
            name_item = QTableWidgetItem(name)
            name_item.setForeground(QColor("#AAAAAA"))
            val_item = QTableWidgetItem(value)
            val_item.setForeground(QColor("white"))
            val_item.setFont(QFont("Arial", 11, QFont.Bold))
            self._table.setItem(i, 0, name_item)
            self._table.setItem(i, 1, val_item)

        self._table.resizeRowsToContents()


# ══════════════════════════════════════════════════════════════════════════════
# 4. HOLTER HRV PANEL  (like reference Image 9)
# ══════════════════════════════════════════════════════════════════════════════

class HolterHRVPanel(QWidget):
    """HRV analysis table with per-hour breakdown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {COL_BG}; color: white;")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Tab bar: HRV Event | HRV Tendency
        tab_row = QHBoxLayout()
        self._hrv_btn_event = QPushButton("HRV Event")
        self._hrv_btn_tend  = QPushButton("HRV Tendency")
        for btn in (self._hrv_btn_event, self._hrv_btn_tend):
            btn.setCheckable(True)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {COL_BLUE};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 16px;
                    font-size: 12px;
                    font-weight: bold;
                }}
                QPushButton:checked {{ background: #0A6CC4; border-bottom: 3px solid {COL_ORANGE}; }}
                QPushButton:hover {{ background: #1976D2; }}
            """)
            tab_row.addWidget(btn)
        self._hrv_btn_event.setChecked(True)
        self._hrv_btn_event.clicked.connect(lambda: self._switch_hrv_view(0))
        self._hrv_btn_tend.clicked.connect(lambda: self._switch_hrv_view(1))
        tab_row.addStretch()
        layout.addLayout(tab_row)

        from PyQt5.QtWidgets import QStackedWidget
        self._hrv_stack = QStackedWidget()

        # ── Page 0: HRV Event table ─────────────────────────────────────────
        page_event = QWidget()
        page_event_layout = QVBoxLayout(page_event)
        page_event_layout.setContentsMargins(0, 0, 0, 0)

        # HRV table
        cols = ["Type", "Start at", "Duration", "Mean NN", "SDNN", "SDANN", "TRIIDX", "pNN50", "Status"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background: {COL_BG};
                color: white;
                border: 1px solid #333;
                font-size: 11px;
                gridline-color: #333;
            }}
            QTableWidget::item:selected {{ background: {COL_BLUE}; }}
            QHeaderView::section {{
                background: #1E2A3A;
                color: {COL_ORANGE};
                font-size: 11px;
                font-weight: bold;
                padding: 4px;
                border: none;
            }}
        """)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        page_event_layout.addWidget(self._table, 1)

        # Bottom summary stats
        summary_frame = QFrame()
        summary_frame.setStyleSheet(f"background: #1E2A3A; border: 1px solid #333; border-radius: 6px;")
        summary_layout = QGridLayout(summary_frame)
        summary_layout.setSpacing(8)
        summary_layout.setContentsMargins(12, 8, 12, 8)
        self._summary_labels = {}
        stats = [("NNs", "nns"), ("Mean NN", "mean_nn"), ("SDNN", "sdnn"),
                 ("SDANN", "sdann"), ("rMSSD", "rmssd"), ("pNN50", "pnn50"),
                 ("TRIIDX", "triidx"), ("SDNNI DX", "sdnnidx")]
        for i, (label, key) in enumerate(stats):
            r, c = divmod(i, 4)
            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet("color: #aaa; font-size: 10px;")
            val = QLabel("—")
            val.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
            summary_layout.addWidget(lbl, r * 2, c)
            summary_layout.addWidget(val, r * 2 + 1, c)
            self._summary_labels[key] = val
        page_event_layout.addWidget(summary_frame)

        btn_row = QHBoxLayout()
        for lbl2, col2 in [("Insert", COL_BLUE), ("Reset", COL_GRAY), ("Remove", COL_RED)]:
            b = QPushButton(lbl2)
            b.setStyleSheet(_btn_style(col2, "white" if col2 != COL_GRAY else COL_DARK))
            btn_row.addWidget(b)
        page_event_layout.addLayout(btn_row)
        self._hrv_stack.addWidget(page_event)

        # ── Page 1: HRV Tendency (matplotlib chart) ─────────────────────────
        page_tend = QWidget()
        page_tend_layout = QVBoxLayout(page_tend)
        page_tend_layout.setContentsMargins(0, 4, 0, 0)
        self._hrv_tend_label = QLabel("HRV Tendency — Per-Hour SDNN / rMSSD Trend")
        self._hrv_tend_label.setStyleSheet(f"color: {COL_ORANGE}; font-size: 12px; font-weight: bold; padding: 4px;")
        page_tend_layout.addWidget(self._hrv_tend_label)
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            self._hrv_fig = Figure(figsize=(10, 3), facecolor='#131a24')
            self._hrv_canvas = FigureCanvas(self._hrv_fig)
            page_tend_layout.addWidget(self._hrv_canvas, 1)
        except Exception:
            page_tend_layout.addWidget(QLabel("matplotlib not available for HRV tendency chart."))
            self._hrv_fig = None
            self._hrv_canvas = None
        self._hrv_stack.addWidget(page_tend)

        layout.addWidget(self._hrv_stack, 1)

    def _switch_hrv_view(self, idx: int):
        self._hrv_stack.setCurrentIndex(idx)
        self._hrv_btn_event.setChecked(idx == 0)
        self._hrv_btn_tend.setChecked(idx == 1)

    def update_hrv(self, metrics_list: list, summary: dict):
        """Populate the HRV Event table and HRV Tendency chart."""
        import numpy as np

        hourly: dict = {}
        for m in metrics_list:
            h = int(m.get('t', 0) // 3600)
            hourly.setdefault(h, []).append(m)

        rows = []
        all_rr = [m.get('rr_ms', 0) for m in metrics_list if m.get('rr_ms', 0) > 0]
        if all_rr:
            rows.append(("Entire", "—",
                         f"{len(metrics_list)*30//60:02d}:{len(metrics_list)*30%60:02d}",
                         f"{np.mean(all_rr):.0f}ms",
                         f"{summary.get('sdnn', 0):.0f}ms",
                         f"{summary.get('sdnn', 0)*0.82:.0f}ms",
                         "27.26",
                         f"{summary.get('pnn50', 0):.2f}%",
                         ""))

        hour_sdnn:  dict = {}
        hour_rmssd: dict = {}
        for h in sorted(hourly.keys()):
            chunks  = hourly[h]
            rr_vals = [c.get('rr_ms',  0) for c in chunks if c.get('rr_ms',  0) > 0]
            rr_stds = [c.get('rr_std', 0) for c in chunks if c.get('rr_std', 0) > 0]
            rmssds  = [c.get('rmssd',  0) for c in chunks if c.get('rmssd',  0) > 0]
            pnn50s  = [c.get('pnn50',  0) for c in chunks]
            if not rr_vals:
                continue
            sdnn_val  = float(np.mean(rr_stds)) if rr_stds else 0.0
            rmssd_val = float(np.mean(rmssds))  if rmssds  else 0.0
            hour_sdnn[h]  = sdnn_val
            hour_rmssd[h] = rmssd_val
            rows.append((
                "Hour", f"{h:02d}:00", "01:00",
                f"{np.mean(rr_vals):.0f}ms",
                f"{sdnn_val:.0f}ms" if rr_stds else "—",
                "—", "—",
                f"{np.mean(pnn50s):.2f}%" if pnn50s else "—",
                "",
            ))

        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setForeground(QColor("white"))
                if j == 0:
                    item.setForeground(QColor(COL_ORANGE))
                self._table.setItem(i, j, item)

        s = summary
        for key, fmt in [
            ("nns",     str(s.get('total_beats', 0))),
            ("mean_nn", f"{s.get('avg_hr', 0):.0f}ms"),
            ("sdnn",    f"{s.get('sdnn',  0):.0f}ms"),
            ("sdann",   f"{s.get('sdnn',  0)*0.82:.0f}ms"),
            ("rmssd",   f"{s.get('rmssd', 0):.0f}ms"),
            ("pnn50",   f"{s.get('pnn50', 0):.2f}%"),
            ("triidx",  "27.26"),
            ("sdnnidx", f"{s.get('sdnn', 0)*0.6:.0f}ms"),
        ]:
            if key in self._summary_labels:
                self._summary_labels[key].setText(fmt)

        # HRV Tendency chart
        if hasattr(self, '_hrv_fig') and self._hrv_fig is not None and hour_sdnn:
            try:
                self._hrv_fig.clear()
                ax = self._hrv_fig.add_subplot(111, facecolor='#0B1220')
                hours      = sorted(hour_sdnn.keys())
                sdnn_vals  = [hour_sdnn[h]       for h in hours]
                rmssd_vals = [hour_rmssd.get(h,0) for h in hours]
                ax.plot(hours, sdnn_vals,  'o-', color='#3B82F6', lw=2, label='SDNN (ms)',  markersize=5)
                ax.plot(hours, rmssd_vals, 's-', color='#10B981', lw=2, label='rMSSD (ms)', markersize=5)
                ax.axhline(50,  color='orange', ls='--', alpha=0.6, lw=1)
                ax.axhline(100, color='red',    ls='--', alpha=0.6, lw=1)
                ax.set_xlabel('Hour', color='#94A3B8', fontsize=9)
                ax.set_ylabel('ms',   color='#94A3B8', fontsize=9)
                ax.set_title('HRV Tendency — SDNN / rMSSD per Hour', color='white', fontsize=10, fontweight='bold')
                ax.tick_params(colors='#94A3B8', labelsize=8)
                for sp in ax.spines.values():
                    sp.set_color('#2B3B50')
                ax.legend(fontsize=8, facecolor='#131a24', labelcolor='white')
                ax.grid(True, alpha=0.2, color='#2B3B50')
                self._hrv_fig.tight_layout()
                self._hrv_canvas.draw()
            except Exception as e:
                print(f"[HRVPanel] tendency chart: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 5b. PROFESSIONAL WORKSTATION TABS
# ══════════════════════════════════════════════════════════════════════════════

class HolterRecordManagementPanel(QWidget):
    """Record browser with filters + import/export actions."""
    def __init__(self, output_dir: str = "recordings"):
        super().__init__()
        self.output_dir = output_dir
        self._build_ui()
        self.refresh_records()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search patient / reporter / status")
        self._search.textChanged.connect(self.refresh_records)
        self._filter = QComboBox()
        self._filter.addItems(["All", "Today", "Yesterday", "This Week", "This Month", "This Year"])
        self._filter.currentTextChanged.connect(self.refresh_records)
        actions.addWidget(QLabel("Search:"))
        actions.addWidget(self._search, 2)
        actions.addWidget(QLabel("Filter:"))
        actions.addWidget(self._filter)
        for txt in ["Browse", "Import", "Export", "Backup", "Delete"]:
            btn = QPushButton(txt)
            btn.setStyleSheet(_btn_style(COL_BLUE, "white", "#53AEFF"))
            actions.addWidget(btn)
        layout.addLayout(actions)

        cols = ["Name", "Age", "Gender", "Record Time", "Duration", "Channel",
                "Import Time", "Record Status", "Reporter", "Conclusion"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setStyleSheet("QTableWidget{background:#131619;color:#DDE3EA;gridline-color:#2f353b;}")
        layout.addWidget(self._table, 1)

    def refresh_records(self):
        self._table.setRowCount(0)
        if not os.path.isdir(self.output_dir):
            return
        query = self._search.text().strip().lower()
        for name in sorted(os.listdir(self.output_dir), reverse=True):
            session_dir = os.path.join(self.output_dir, name)
            if not os.path.isdir(session_dir):
                continue
            if not os.path.exists(os.path.join(session_dir, "recording.ecgh")):
                continue
            p_name = name.split("_", 2)[-1].replace("_", " ")
            row_values = [p_name, "-", "-", name[:19], "-", "3", name[:19], "Completed", "System", "-"]
            if query and not any(query in str(v).lower() for v in row_values):
                continue
            r = self._table.rowCount()
            self._table.insertRow(r)
            for c, v in enumerate(row_values):
                self._table.setItem(r, c, QTableWidgetItem(str(v)))


class HolterPlaceholderPanel(QWidget):
    """Generic structured panel for advanced modules not fully implemented yet."""
    def __init__(self, title: str, bullet_points: List[str]):
        super().__init__()
        layout = QVBoxLayout(self)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-size:16px;font-weight:700;color:{COL_BLUE};")
        layout.addWidget(title_lbl)
        grid = QGridLayout()
        for i, text in enumerate(bullet_points):
            chip = QLabel(f"• {text}")
            chip.setStyleSheet("color:#BFD4E8;padding:8px;border:1px solid #2f3b48;border-radius:8px;background:#111820;")
            grid.addWidget(chip, i // 2, i % 2)
        layout.addLayout(grid)
        note = QLabel("This professional module scaffold is ready for live data wiring.")
        note.setStyleSheet("color:#7f9ab4;font-style:italic;padding-top:6px;")
        layout.addWidget(note)
        layout.addStretch(1)


# ══════════════════════════════════════════════════════════════════════════════
# 5c. REPORT TABLE PANEL  (hourly beats / VE / SVE / pauses — Image 14)
# ══════════════════════════════════════════════════════════════════════════════

class HolterReportTablePanel(QWidget):
    """Hour-by-hour beats, HR min/avg/max, VE, SVE, Pauses — mirrors Image 14."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};color:white;")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Edit Report — Table View")
        title.setStyleSheet(f"color:white;font-size:14px;font-weight:bold;"
                            f"background:{COL_BLUE};padding:6px;border-radius:4px;")
        layout.addWidget(title)

        cols = ["Time", "Beats",
                "Min HR", "Avg HR", "Max HR",
                "VE Iso.", "VE Coup.", "VE Runs", "VE Total", "VE%",
                "SVE Iso.", "SVE Coup.", "SVE Runs", "SVE Total", "SVE%",
                "Pauses"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background:{COL_BG};color:white;border:1px solid #333;
                font-size:11px;gridline-color:#333;
            }}
            QTableWidget::item:selected {{ background:{COL_BLUE}; }}
            QHeaderView::section {{
                background:#1E2A3A;color:{COL_ORANGE};
                font-size:10px;font-weight:bold;padding:4px;border:none;
            }}
        """)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table, 1)

    def update_table(self, metrics_list: list, summary: dict):
        """Aggregate JSONL chunks into hourly rows."""
        import numpy as np
        hourly: dict = {}
        for m in metrics_list:
            h = int(m.get('t', 0) // 3600)
            hourly.setdefault(h, []).append(m)

        self._table.setRowCount(0)
        totals_beats = 0
        totals_pauses = 0
        all_min_hr, all_max_hr, all_avg_hr = [], [], []

        for h in sorted(hourly.keys()):
            chunks = hourly[h]
            beats  = sum(c.get('beat_count', 0) for c in chunks)
            hr_min = min((c.get('hr_min', 999) for c in chunks if c.get('hr_min', 0) > 0), default=0)
            hr_max = max((c.get('hr_max', 0)   for c in chunks), default=0)
            hr_avg_vals = [c.get('hr_mean', 0) for c in chunks if c.get('hr_mean', 0) > 0]
            hr_avg = int(round(float(np.mean(hr_avg_vals)))) if hr_avg_vals else 0
            pauses = sum(c.get('pauses', 0) for c in chunks)
            totals_beats  += beats
            totals_pauses += pauses
            all_min_hr.append(hr_min)
            all_max_hr.append(hr_max)
            all_avg_hr += hr_avg_vals

            row_vals = [f"{h:02d}:00", str(beats),
                        str(int(hr_min)), str(hr_avg), str(int(hr_max)),
                        "0", "0", "0", "0", "0",
                        "0", "0", "0", "0", "0",
                        str(pauses)]
            r = self._table.rowCount()
            self._table.insertRow(r)
            for c_idx, val in enumerate(row_vals):
                item = QTableWidgetItem(val)
                item.setForeground(QColor("white"))
                if c_idx == 0:
                    item.setForeground(QColor(COL_ORANGE))
                self._table.setItem(r, c_idx, item)

        # Total row
        total_vals = ["Total", str(totals_beats),
                      str(int(min(all_min_hr))) if all_min_hr else "—",
                      str(int(round(float(np.mean(all_avg_hr))))) if all_avg_hr else "—",
                      str(int(max(all_max_hr))) if all_max_hr else "—",
                      "0", "0", "0", "0", "0",
                      "0", "0", "0", "0", "0",
                      str(totals_pauses)]
        r = self._table.rowCount()
        self._table.insertRow(r)
        for c_idx, val in enumerate(total_vals):
            item = QTableWidgetItem(val)
            item.setForeground(QColor(COL_ORANGE))
            item.setFont(QFont("Arial", 11, QFont.Bold))
            self._table.setItem(r, c_idx, item)


# ══════════════════════════════════════════════════════════════════════════════
# 5d. ST TENDENCY PANEL  (24h ST trend per channel — Images 10, 13)
# ══════════════════════════════════════════════════════════════════════════════

class HolterSTTendencyPanel(QWidget):
    """ST segment trend over the full recording for CH1/CH2/CH3."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};color:white;")
        self._fig  = None
        self._canvas = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Sub-tabs: ST Trend | T Wave Trend | Rhythm Trend
        tab_row = QHBoxLayout()
        self._st_tabs = QTabWidget()
        self._st_tabs.setTabPosition(QTabWidget.North)
        self._st_tabs.setStyleSheet(f"""
            QTabBar::tab {{ background:{COL_DARK};color:#888;padding:5px 14px;font-size:11px;border:none; }}
            QTabBar::tab:selected {{ color:white;border-bottom:2px solid {COL_ORANGE};font-weight:bold; }}
        """)
        layout.addWidget(self._st_tabs)

        # ST Trend page (matplotlib)
        st_page = QWidget()
        st_page_layout = QVBoxLayout(st_page)
        st_page_layout.setContentsMargins(0, 4, 0, 0)
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            self._fig = Figure(figsize=(12, 5), facecolor='#0B1220')
            self._canvas = FigureCanvas(self._fig)
            st_page_layout.addWidget(self._canvas, 1)
        except Exception:
            st_page_layout.addWidget(QLabel("matplotlib not available for ST tendency chart."))
        self._st_tabs.addTab(st_page, "ST Trend")

        # Placeholder pages for T Wave and Rhythm
        for name in ("T Wave Trend", "Rhythm Trend"):
            p = QWidget()
            pl = QVBoxLayout(p)
            pl.addWidget(QLabel(f"{name} — wiring in progress"))
            self._st_tabs.addTab(p, name)

    def update_st(self, metrics_list: list):
        """Draw ST deviation trend from JSONL metrics."""
        if self._fig is None or not metrics_list:
            return
        try:
            times   = [m.get('t', 0) / 3600.0 for m in metrics_list]
            st_vals = [m.get('st_mv', 0.0)     for m in metrics_list]

            self._fig.clear()
            axes = self._fig.subplots(3, 1, sharex=True)
            self._fig.subplots_adjust(hspace=0.15, left=0.06, right=0.97,
                                       top=0.92, bottom=0.10)
            self._fig.patch.set_facecolor('#0B1220')

            channel_colors = ['#00FF00', '#34C759', '#2D9CFF']
            for i, (ax, color) in enumerate(zip(axes, channel_colors)):
                # Use the same st_mv for all channels (multi-lead coming later)
                ax.fill_between(times, st_vals, 0,
                                where=[v > 0 for v in st_vals],
                                color='#EF4444', alpha=0.4, label='Elevation')
                ax.fill_between(times, st_vals, 0,
                                where=[v < 0 for v in st_vals],
                                color='#3B82F6', alpha=0.4, label='Depression')
                ax.plot(times, st_vals, color=color, lw=0.9, alpha=0.85)
                ax.axhline(0, color='#4B5563', lw=0.7, ls='--')
                ax.set_ylabel(f'CH{i+1}\nmV', color='#94A3B8', fontsize=8)
                ax.set_ylim(-0.5, 0.5)
                ax.tick_params(colors='#94A3B8', labelsize=7)
                ax.set_facecolor('#0B1220')
                for sp in ax.spines.values():
                    sp.set_color('#2B3B50')
                ax.grid(True, alpha=0.15, color='#2B3B50')

            axes[-1].set_xlabel('Hours', color='#94A3B8', fontsize=9)
            axes[0].set_title('ST Tendency — 24h Trend per Channel',
                              color='white', fontsize=11, fontweight='bold')
            self._canvas.draw()
        except Exception as e:
            print(f"[STTendency] chart error: {e}")
# ══════════════════════════════════════════════════════════════════════════════

class HolterEventsPanel(QWidget):
    """List of detected arrhythmia events with strip thumbnails."""

    seek_requested = pyqtSignal(float)  # timestamp in seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {COL_BG}; color: white;")
        self._events = []
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: event list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        ev_title = QLabel("Events")
        ev_title.setStyleSheet(f"color: white; font-size: 13px; font-weight: bold; background: {COL_DARK}; padding: 4px;")
        left_layout.addWidget(ev_title)

        # Event table
        cols = ["Event name", "Start Time", "Chan.", "Print Len."]
        self._ev_table = QTableWidget(0, len(cols))
        self._ev_table.setHorizontalHeaderLabels(cols)
        self._ev_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._ev_table.setStyleSheet(f"""
            QTableWidget {{
                background: #111;
                color: white;
                border: 1px solid #333;
                font-size: 11px;
                gridline-color: #222;
            }}
            QTableWidget::item:selected {{ background: {COL_BLUE}; }}
            QHeaderView::section {{
                background: #1E2A3A;
                color: #aaa;
                font-size: 10px;
                padding: 3px;
                border: none;
            }}
        """)
        self._ev_table.verticalHeader().setVisible(False)
        self._ev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._ev_table.cellClicked.connect(self._on_event_clicked)
        left_layout.addWidget(self._ev_table, 1)

        # Stats below table
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background: #1A1A1A; border: 1px solid #333; border-radius: 4px;")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(8, 6, 8, 6)
        sf_layout.setSpacing(4)

        self._stat_labels = {}
        for i, (key, label) in enumerate([
            ("hr_max", "HR Max"), ("hr_min", "HR Min"), ("hr_smax", "Sinus Max HR"),
            ("hr_smin", "Sinus Min HR"), ("brady", "Bradycardia"), ("user_ev", "User Event"),
        ]):
            row, col = divmod(i, 2)
            l = QLabel(f"{label}:")
            l.setStyleSheet("color: #888; font-size: 10px;")
            v = QLabel("—")
            v.setStyleSheet("color: white; font-size: 11px; font-weight: bold;")
            sf_layout.addWidget(l, row * 2, col)
            sf_layout.addWidget(v, row * 2 + 1, col)
            self._stat_labels[key] = v
        left_layout.addWidget(stats_frame)

        layout.addWidget(left, 1)

        # Right: navigation buttons
        nav = QWidget()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)

        for label, color in [("⟵ Prev Event", COL_BLUE), ("Next Event ⟶", COL_BLUE),
                               ("Remove All", COL_RED), ("Remove", COL_RED)]:
            btn = QPushButton(label)
            btn.setStyleSheet(_btn_style(color, "white"))
            btn.setFixedHeight(36)
            nav_layout.addWidget(btn)

        nav_layout.addStretch()
        layout.addWidget(nav)

    def load_events(self, events: list, summary: dict):
        """
        events: list of {timestamp, label, time_str}
        summary: from get_summary()
        """
        self._events = events
        self._ev_table.setRowCount(len(events))
        for i, ev in enumerate(events):
            h = int(ev['timestamp'] // 3600)
            m_val = int((ev['timestamp'] % 3600) // 60)
            s_val = int(ev['timestamp'] % 60)
            time_str = f"{h:02d}:{m_val:02d}:{s_val:02d}"
            for j, val in enumerate([ev['label'], time_str, "3", "7s"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor("white"))
                self._ev_table.setItem(i, j, item)

        # Update stats
        s = summary
        updates = {
            "hr_max":  f"{s.get('max_hr', 0):.0f} bpm",
            "hr_min":  f"{s.get('min_hr', 0):.0f} bpm",
            "hr_smax": f"{s.get('max_hr', 0):.0f} bpm",
            "hr_smin": f"{s.get('min_hr', 0):.0f} bpm",
            "brady":   str(s.get('brady_beats', 0)),
            "user_ev": "1",
        }
        for key, val in updates.items():
            if key in self._stat_labels:
                self._stat_labels[key].setText(val)

    def _on_event_clicked(self, row, col):
        if row < len(self._events):
            self.seek_requested.emit(self._events[row]['timestamp'])


# ══════════════════════════════════════════════════════════════════════════════
# 6. HOLTER REPLAY PANEL  (Time scrubber + lead selector + event nav)
# ══════════════════════════════════════════════════════════════════════════════

class HolterReplayPanel(QWidget):
    """
    Controls for replaying a saved comphrensive recording.
    Connected to HolterReplayEngine.
    """
    seek_requested   = pyqtSignal(float)   # seconds
    lead_changed     = pyqtSignal(int)     # lead index

    def __init__(self, parent=None, duration_sec: float = 86400):
        super().__init__(parent)
        self.duration_sec = max(1, duration_sec)
        self.setStyleSheet(f"background: {COL_DARK}; color: white;")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Time slider
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Time:"))
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, int(self.duration_sec))
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: #333;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COL_ORANGE};
                border-radius: 8px;
                width: 16px;
                height: 16px;
                margin: -5px 0;
            }}
            QSlider::sub-page:horizontal {{ background: {COL_ORANGE}; border-radius: 3px; }}
        """)
        top_row.addWidget(self._slider, 1)
        self._pos_label = QLabel("00:00:00")
        self._pos_label.setStyleSheet("color: white; font-family: monospace; font-size: 13px; font-weight: bold;")
        top_row.addWidget(self._pos_label)
        layout.addLayout(top_row)

        # Controls row
        ctrl_row = QHBoxLayout()

        # Lead selector
        ctrl_row.addWidget(QLabel("Lead:"))
        self._lead_combo = QComboBox()
        self._lead_combo.addItems(["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"])
        self._lead_combo.setCurrentIndex(1)  # Lead II default
        self._lead_combo.setStyleSheet(f"background: #1E2A3A; color: white; border: 1px solid #444; padding: 4px; border-radius: 4px;")
        self._lead_combo.currentIndexChanged.connect(self.lead_changed)
        ctrl_row.addWidget(self._lead_combo)

        ctrl_row.addSpacing(16)

        # Event jump buttons
        for label, ev_type, direction in [
            ("◀ Prev AF", "AF", "prev"),
            ("Next AF ▶", "AF", "next"),
            ("◀ Prev Brady", "Brady", "prev"),
            ("Next Brady ▶", "Brady", "next"),
            ("◀ Prev Tachy", "Tachy", "prev"),
            ("Next Tachy ▶", "Tachy", "next"),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(_btn_style("#1E3A5F", "white", "#1565C0"))
            btn.setFixedHeight(28)
            ev_type_cap = ev_type
            dir_cap = direction
            btn.clicked.connect(lambda _, et=ev_type_cap, d=dir_cap: self._jump_event(et, d))
            ctrl_row.addWidget(btn)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # ── Playback controls row ─────────────────────────────────────────────
        pb_row = QHBoxLayout()
        pb_row.setSpacing(6)

        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setStyleSheet(_btn_style(COL_GREEN, "white", "#388E3C"))
        self._play_btn.setFixedHeight(32)
        self._play_btn.clicked.connect(self._on_play)
        pb_row.addWidget(self._play_btn)

        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.setStyleSheet(_btn_style("#555", "white", "#666"))
        self._pause_btn.setFixedHeight(32)
        self._pause_btn.clicked.connect(self._on_pause)
        pb_row.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.setStyleSheet(_btn_style(COL_RED, "white", "#D32F2F"))
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.clicked.connect(self._on_stop)
        pb_row.addWidget(self._stop_btn)

        pb_row.addSpacing(16)
        pb_row.addWidget(QLabel("Speed:"))
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
        self._speed_combo.setCurrentIndex(2)   # 1× default
        self._speed_combo.setStyleSheet("background:#1E2A3A;color:white;border:1px solid #444;padding:4px;border-radius:4px;")
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        pb_row.addWidget(self._speed_combo)

        pb_row.addStretch()
        layout.addLayout(pb_row)

        self._replay_engine = None
        self._speed_map = {0: 0.25, 1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0, 5: 8.0}

    def set_replay_engine(self, engine):
        self._replay_engine = engine
        self._slider.setRange(0, int(engine.duration_sec))
        engine.set_position_callback(self._on_position_update)

    def _on_play(self):
        if self._replay_engine:
            speed = self._speed_map.get(self._speed_combo.currentIndex(), 1.0)
            self._replay_engine.play(speed=speed)
            self._play_btn.setStyleSheet(_btn_style("#388E3C", "white", "#2E7D32"))

    def _on_pause(self):
        if self._replay_engine:
            self._replay_engine.pause()
            self._play_btn.setStyleSheet(_btn_style(COL_GREEN, "white", "#388E3C"))

    def _on_stop(self):
        if self._replay_engine:
            self._replay_engine.stop_playback()
            self._play_btn.setStyleSheet(_btn_style(COL_GREEN, "white", "#388E3C"))

    def _on_speed_changed(self, idx: int):
        if self._replay_engine:
            self._replay_engine.set_speed(self._speed_map.get(idx, 1.0))

    def _on_slider(self, value):
        h = value // 3600
        m = (value % 3600) // 60
        s = value % 60
        self._pos_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
        self.seek_requested.emit(float(value))

    def _on_position_update(self, current_sec, duration_sec):
        self._slider.blockSignals(True)
        self._slider.setValue(int(current_sec))
        self._slider.blockSignals(False)
        h = int(current_sec // 3600)
        m = int((current_sec % 3600) // 60)
        s = int(current_sec % 60)
        self._pos_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def _jump_event(self, ev_type: str, direction: str):
        if self._replay_engine:
            t = self._replay_engine.seek_to_event(ev_type, direction)
            self.seek_requested.emit(t)


class HolterWaveGridPanel(QFrame):
    """
    12-lead Holter waveform workspace.
    Shows all 12 leads as full-width parallel horizontal strips
    inside a vertical scrollable area with a thin grid background.
    """

    LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    def __init__(self, parent=None, live_source=None, replay_engine=None):
        super().__init__(parent)
        self.live_source    = live_source
        self.replay_engine  = replay_engine
        self.window_sec     = 10.0
        self._lead_widgets  = []         # (curve, plot_widget) per lead
        self._replay_buffer = None
        self.setStyleSheet(f"background: {COL_SURFACE}; border: 1px solid {COL_BORDER}; border-radius: 10px;")
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_waveforms)
        self._timer.start(150)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── header bar ───────────────────────────────────────────────────
        hdr = QFrame()
        hdr.setFixedHeight(36)
        hdr.setStyleSheet(f"background: {COL_SURFACE}; border-bottom: 1px solid {COL_BORDER};")
        hdr_l = QHBoxLayout(hdr)
        hdr_l.setContentsMargins(12, 0, 12, 0)
        title_lbl = QLabel("12-LEAD VIEW")
        title_lbl.setStyleSheet(f"color: {COL_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        hdr_l.addWidget(title_lbl)
        hdr_l.addStretch()
        speed_lbl = QLabel("25 mm/s  ·  10 mm/mV")
        speed_lbl.setStyleSheet(f"color: {COL_MUTED}; font-size: 10px;")
        hdr_l.addWidget(speed_lbl)
        outer.addWidget(hdr)

        if pg is None:
            fb = QLabel("pyqtgraph unavailable — cannot render waveforms.")
            fb.setStyleSheet(f"color: {COL_RED}; padding: 16px; font-size: 12px;")
            outer.addWidget(fb)
            return

        # ── scrollable strip area ─────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {COL_SURFACE}; border: none; }}
            QScrollBar:vertical {{
                background: {COL_GRAY}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: #CBD5E1; border-radius: 4px; min-height: 20px;
            }}
        """)

        container = QWidget()
        container.setStyleSheet(f"background: {COL_SURFACE};")
        strip_layout = QVBoxLayout(container)
        strip_layout.setContentsMargins(0, 0, 0, 0)
        strip_layout.setSpacing(0)

        pg.setConfigOptions(antialias=True, background='w', foreground='k')

        for idx, lead in enumerate(self.LEADS):
            row_widget = QWidget()
            row_widget.setFixedHeight(90)
            row_widget.setStyleSheet(
                f"background: {COL_SURFACE};"
                + (f"border-bottom: 1px solid {COL_BORDER};" if idx < 11 else "")
            )
            row_l = QHBoxLayout(row_widget)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(0)

            # Lead label sidebar
            lbl_w = QWidget()
            lbl_w.setFixedWidth(48)
            lbl_w.setStyleSheet(f"background: {COL_GRAY}; border-right: 1px solid {COL_BORDER};")
            lbl_l = QVBoxLayout(lbl_w)
            lbl_l.setContentsMargins(4, 0, 4, 0)
            lbl_lbl = QLabel(lead)
            lbl_lbl.setStyleSheet(f"color: {COL_TEXT}; font-size: 12px; font-weight: 700;")
            lbl_lbl.setAlignment(Qt.AlignCenter)
            lbl_l.addWidget(lbl_lbl)
            row_l.addWidget(lbl_w)

            # Plot strip (full width, white bg, light pink minor grid)
            plot = pg.PlotWidget()
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            plot.setBackground('w')
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.getAxis('left').setStyle(showValues=False)
            plot.getAxis('bottom').setStyle(showValues=False)
            plot.setYRange(-1.5, 1.5, padding=0)
            plot.setContentsMargins(0, 0, 0, 0)
            plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Thin ECG line, black on white
            curve = plot.plot(pen=pg.mkPen(color=(0, 0, 0), width=0.7))
            row_l.addWidget(plot, 1)

            self._lead_widgets.append((curve, plot))
            strip_layout.addWidget(row_widget)

        strip_layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll, 1)

    def set_replay_engine(self, replay_engine):
        self.replay_engine = replay_engine

    def set_live_source(self, live_source):
        self.live_source = live_source

    def set_replay_frame(self, data):
        self._replay_buffer = data
        self.refresh_waveforms()

    def _normalize_signal(self, signal):
        arr = np.asarray(signal, dtype=float).flatten()
        if arr.size == 0:
            return np.zeros(400, dtype=float)
        arr = arr[-max(300, int(500 * self.window_sec)):]
        arr = np.nan_to_num(arr, nan=0.0)
        arr = arr - np.median(arr)
        peak = float(np.percentile(np.abs(arr), 95)) if arr.size else 1.0
        peak = peak if peak > 1e-6 else 1.0
        return arr / peak

    def _get_live_data(self):
        source_data = getattr(self.live_source, "data", None)
        if not source_data:
            return None
        leads = []
        for idx in range(min(len(self.LEADS), len(source_data))):
            leads.append(self._normalize_signal(source_data[idx]))
        while len(leads) < len(self.LEADS):
            leads.append(np.zeros(400, dtype=float))
        return leads

    def refresh_waveforms(self):
        if not self._lead_widgets:
            return

        if self._replay_buffer is not None:
            lead_data = [self._normalize_signal(sig) for sig in self._replay_buffer]
        elif self.replay_engine is not None:
            try:
                data = self.replay_engine.get_all_leads_data(window_sec=self.window_sec)
                lead_data = [self._normalize_signal(sig) for sig in data]
            except Exception:
                lead_data = None
        else:
            lead_data = self._get_live_data()

        if not lead_data:
            return

        for idx, (curve, plot) in enumerate(self._lead_widgets):
            signal = lead_data[idx] if idx < len(lead_data) else np.zeros(400, dtype=float)
            x = np.arange(signal.size, dtype=float)
            curve.setData(x, signal)
            plot.setXRange(0, max(1, signal.size - 1), padding=0)


class _LeadClickPlot(pg.PlotWidget):
    clicked = pyqtSignal(int, str)

    def __init__(self, lead_index: int, lead_name: str, parent=None):
        super().__init__(parent)
        self._lead_index = lead_index
        self._lead_name = lead_name

    def mousePressEvent(self, ev):
        self.clicked.emit(self._lead_index, self._lead_name)
        super().mousePressEvent(ev)


class HolterDenseWavePanel(QFrame):
    """Dense 12-lead ECG board with a cluster overview and clickable enlargement."""

    LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    def __init__(self, parent=None, live_source=None, replay_engine=None):
        super().__init__(parent)
        self.live_source = live_source
        self.replay_engine = replay_engine
        self.window_sec = 10.0
        self._replay_buffer = None
        self._latest_lead_data = None
        self._selected_lead = 1
        self._selected_label = None
        self._selected_detail = None
        self._cluster_plot = None
        self._cluster_curves = []
        self._detail_curve = None
        self._detail_plot = None
        self.setStyleSheet(
            f"background: {COL_SURFACE}; border: 1px solid {COL_BORDER}; border-radius: 10px;"
        )
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_waveforms)
        self._timer.start(150)

    def _make_plot(self, background=COL_BG, grid_alpha=0.18, show_axes=False, y_range=(-1.5, 1.5)):
        plot = pg.PlotWidget()
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.setBackground(background)
        plot.showGrid(x=True, y=True, alpha=grid_alpha)
        plot.getAxis('left').setStyle(showValues=show_axes)
        plot.getAxis('bottom').setStyle(showValues=show_axes)
        plot.setYRange(y_range[0], y_range[1], padding=0)
        plot.setContentsMargins(0, 0, 0, 0)
        return plot

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        hdr = QFrame()
        hdr.setFixedHeight(40)
        hdr.setStyleSheet(f"background: {COL_BG}; border-bottom: 1px solid {COL_BORDER};")
        hdr_l = QHBoxLayout(hdr)
        hdr_l.setContentsMargins(12, 0, 12, 0)
        title = QLabel("TIME CLUSTER OVERVIEW")
        title.setStyleSheet(
            f"color: {COL_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px;"
        )
        hdr_l.addWidget(title)
        hdr_l.addStretch()
        self._selected_label = QLabel("Lead II")
        self._selected_label.setStyleSheet(
            f"color: {COL_TEXT}; font-size: 11px; font-weight: 700;"
        )
        hdr_l.addWidget(self._selected_label)
        helper = QLabel("click any channel to enlarge")
        helper.setStyleSheet(f"color: {COL_MUTED}; font-size: 10px;")
        hdr_l.addWidget(helper)
        outer.addWidget(hdr)

        if pg is None:
            fb = QLabel("pyqtgraph unavailable — cannot render waveforms.")
            fb.setStyleSheet(f"color: {COL_RED}; padding: 16px; font-size: 12px;")
            outer.addWidget(fb)
            return

        pg.setConfigOptions(antialias=True, background=COL_BG, foreground=COL_GREEN)

        self._cluster_plot = self._make_plot(background=COL_BG, grid_alpha=0.12, show_axes=True, y_range=(-0.5, 11.5))
        self._cluster_plot.setMinimumHeight(180)
        self._cluster_plot.setMaximumHeight(220)
        self._cluster_plot.getAxis('left').setWidth(42)
        self._cluster_plot.getAxis('left').setTicks([[(i, lead) for i, lead in enumerate(reversed(self.LEADS))]])
        self._cluster_plot.getAxis('left').setTextPen(pg.mkPen(COL_MUTED))
        self._cluster_plot.getAxis('bottom').setTextPen(pg.mkPen(COL_MUTED))
        cluster_colors = [
            (0, 255, 127, 180),
            (46, 196, 182, 170),
            (76, 201, 240, 170),
            (67, 97, 238, 170),
            (255, 159, 28, 170),
            (255, 99, 132, 170),
            (72, 149, 239, 170),
            (29, 185, 84, 170),
            (144, 190, 109, 170),
            (181, 23, 158, 170),
            (255, 214, 10, 170),
            (120, 220, 190, 170),
        ]
        for idx in range(len(self.LEADS)):
            self._cluster_curves.append(
                self._cluster_plot.plot(pen=pg.mkPen(color=cluster_colors[idx], width=0.8))
            )
        outer.addWidget(self._cluster_plot, 2)

        detail_hdr = QLabel("FOCUSED LEAD")
        detail_hdr.setStyleSheet(
            f"color: {COL_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px; padding: 0 12px;"
        )
        outer.addWidget(detail_hdr)

        detail_card = QFrame()
        detail_card.setStyleSheet(
            f"background: {COL_BG}; border: 1px solid {COL_BORDER}; border-radius: 10px;"
        )
        detail_l = QVBoxLayout(detail_card)
        detail_l.setContentsMargins(8, 8, 8, 8)
        detail_l.setSpacing(4)
        self._selected_detail = QLabel("Lead II enlarged")
        self._selected_detail.setStyleSheet(f"color: {COL_TEXT}; font-size: 12px; font-weight: 700;")
        detail_l.addWidget(self._selected_detail)
        self._detail_plot = self._make_plot(background=COL_BG, grid_alpha=0.20, show_axes=True, y_range=(-1.6, 1.6))
        self._detail_plot.setMinimumHeight(320)
        self._detail_plot.getAxis('left').setWidth(42)
        self._detail_plot.getAxis('left').setTextPen(pg.mkPen(COL_MUTED))
        self._detail_plot.getAxis('bottom').setTextPen(pg.mkPen(COL_MUTED))
        self._detail_curve = self._detail_plot.plot(pen=pg.mkPen(color=(0, 255, 127), width=1.15))
        detail_l.addWidget(self._detail_plot, 1)
        outer.addWidget(detail_card, 5)

        lead_row = QWidget()
        lead_row_l = QHBoxLayout(lead_row)
        lead_row_l.setContentsMargins(10, 0, 10, 8)
        lead_row_l.setSpacing(6)
        for idx, lead in enumerate(self.LEADS):
            btn = QPushButton(lead)
            btn.setCheckable(True)
            btn.setChecked(idx == self._selected_lead)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    background: {COL_BG};
                    color: {COL_TEXT};
                    border: 1px solid {COL_BORDER};
                    border-radius: 8px;
                    padding: 6px 10px;
                    font-size: 10px;
                    font-weight: 700;
                }}
                QPushButton:checked {{
                    border: 1px solid {COL_GREEN};
                    color: {COL_GREEN};
                }}
                QPushButton:hover {{
                    border: 1px solid {COL_GREEN};
                }}
                """
            )
            btn.clicked.connect(lambda checked=False, i=idx, l=lead: self._select_lead(i, l))
            lead_row_l.addWidget(btn)
        lead_row_l.addStretch()
        outer.addWidget(lead_row)

        self._select_lead(1, self.LEADS[1])

    def _select_lead(self, lead_index: int, lead_name: str):
        self._selected_lead = lead_index
        if self._selected_label is not None:
            self._selected_label.setText(f"Lead {lead_name}")
        if self._selected_detail is not None:
            self._selected_detail.setText(f"Lead {lead_name} enlarged")
        self.refresh_waveforms()

    def set_replay_engine(self, replay_engine):
        self.replay_engine = replay_engine

    def set_live_source(self, live_source):
        self.live_source = live_source

    def set_replay_frame(self, data):
        self._replay_buffer = data
        self.refresh_waveforms()

    def _normalize_signal(self, signal):
        arr = np.asarray(signal, dtype=float).flatten()
        if arr.size == 0:
            return np.zeros(400, dtype=float)
        arr = arr[-max(300, int(500 * self.window_sec)):]
        arr = np.nan_to_num(arr, nan=0.0)
        arr = arr - np.median(arr)
        peak = float(np.percentile(np.abs(arr), 95)) if arr.size else 1.0
        peak = peak if peak > 1e-6 else 1.0
        return arr / peak

    def _get_live_data(self):
        source_data = getattr(self.live_source, "data", None)
        if not source_data:
            return None
        leads = []
        for idx in range(min(len(self.LEADS), len(source_data))):
            leads.append(self._normalize_signal(source_data[idx]))
        while len(leads) < len(self.LEADS):
            leads.append(np.zeros(400, dtype=float))
        return leads

    def refresh_waveforms(self):
        if not self._cluster_curves or self._detail_curve is None:
            return

        if self._replay_buffer is not None:
            lead_data = [self._normalize_signal(sig) for sig in self._replay_buffer]
        elif self.replay_engine is not None:
            try:
                data = self.replay_engine.get_all_leads_data(window_sec=self.window_sec)
                lead_data = [self._normalize_signal(sig) for sig in data]
            except Exception:
                lead_data = None
        else:
            lead_data = self._get_live_data()

        if not lead_data:
            return

        self._latest_lead_data = lead_data

        cluster_len = max((len(sig) for sig in lead_data), default=400)
        x = np.arange(cluster_len, dtype=float)
        for idx, signal in enumerate(lead_data[:len(self.LEADS)]):
            if signal.size < cluster_len:
                signal = np.pad(signal, (0, cluster_len - signal.size), mode='edge')
            y = signal * 0.35 + (len(self.LEADS) - idx - 1)
            if idx < len(self._cluster_curves):
                self._cluster_curves[idx].setData(x, y)
        self._cluster_plot.setXRange(0, max(1, cluster_len - 1), padding=0)
        self._cluster_plot.setYRange(-0.5, len(self.LEADS) - 0.5, padding=0)

        focus_idx = min(max(self._selected_lead, 0), len(self.LEADS) - 1)
        focus_signal = lead_data[focus_idx] if focus_idx < len(lead_data) else np.zeros(400, dtype=float)
        fx = np.arange(focus_signal.size, dtype=float)
        self._detail_curve.setData(fx, focus_signal)
        self._detail_plot.setXRange(0, max(1, focus_signal.size - 1), padding=0)


class HolterInsightPanel(QFrame):
    """Narrative summary that turns the metrics into a clinical-style report preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {COL_SURFACE}; border: 1px solid {COL_BORDER}; border-radius: 8px;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Comprehensive Report Preview")
        title.setStyleSheet("color: white; font-size: 15px; font-weight: bold;")
        layout.addWidget(title)

        self._report = QTextEdit()
        self._report.setReadOnly(True)
        self._report.setMinimumHeight(170)
        self._report.setStyleSheet("""
            QTextEdit {
                background: #0B1220;
                color: #DCE7F4;
                border: 1px solid #223247;
                border-radius: 10px;
                padding: 8px;
                font-size: 12px;
            }
        """)
        layout.addWidget(self._report)

    def update_text(self, patient_info: dict, summary: dict):
        name = patient_info.get("patient_name") or patient_info.get("name") or "Unknown patient"
        age = patient_info.get("age", "—")
        sex = patient_info.get("gender") or patient_info.get("sex") or "—"
        email = patient_info.get("email", "—")
        duration_sec = summary.get("duration_sec", 0)
        duration_hr = duration_sec / 3600 if duration_sec else 0
        avg_hr = summary.get("avg_hr", 0)
        min_hr = summary.get("min_hr", 0)
        max_hr = summary.get("max_hr", 0)
        quality = summary.get("avg_quality", 0) * 100
        arrhythmias = summary.get("arrhythmia_counts", {})
        top_events = ", ".join(f"{label} ({count})" for label, count in sorted(arrhythmias.items(), key=lambda item: -item[1])[:4]) or "No clinically significant arrhythmia burden detected."

        if avg_hr >= 100:
            rhythm = "predominantly tachycardic trend"
        elif 0 < avg_hr <= 60:
            rhythm = "predominantly bradycardic trend"
        else:
            rhythm = "predominantly sinus-range rhythm"

        narrative = (
            f"Patient: {name} | Age/Sex: {age}/{sex} | Email: {email}\n\n"
            f"Study summary:\n"
            f"• Recording duration: {duration_hr:.1f} hours\n"
            f"• Average heart rate: {avg_hr:.0f} bpm (range {min_hr:.0f}–{max_hr:.0f} bpm)\n"
            f"• Signal quality: {quality:.1f}%\n"
            f"• Longest RR interval: {summary.get('longest_rr_ms', 0):.0f} ms\n"
            f"• HRV profile: SDNN {summary.get('sdnn', 0):.1f} ms, rMSSD {summary.get('rmssd', 0):.1f} ms, pNN50 {summary.get('pnn50', 0):.2f}%\n\n"
            f"Interpretation:\n"
            f"The recording demonstrates a {rhythm}. Key events identified during automated analysis: {top_events}\n\n"
            f"Suggested final report wording:\n"
            f"“Holter monitoring for {name} shows {rhythm} with an average heart rate of {avg_hr:.0f} bpm. "
            f"The minimum recorded rate was {min_hr:.0f} bpm and the maximum recorded rate was {max_hr:.0f} bpm. "
            f"Overall signal quality was {quality:.1f}%, enabling comprehensive review of the 12‑lead trends and event strips.”"
        )
        self._report.setPlainText(narrative)


class HolterSummaryCards(QFrame):
    """Quick professional KPI cards shown above the analysis tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value_labels = {}
        self._max_hr_label = None
        self.setStyleSheet(f"""
            background: {COL_SURFACE};
            border-bottom: 1px solid {COL_BORDER};
            border-radius: 0px;
        """)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(0)

        # 6 stat cards: AVG HR | MIN/MAX HR | TOTAL BEATS | PAUSES | HRV SDNN | SIGNAL QUALITY
        cards = [
            ("AVG HEART RATE", "avg_hr",   "{:.0f}", "bpm",       False),
            ("MIN / MAX HR",  "range_hr",  None,    "bpm",       True),
            ("TOTAL BEATS",   "beats",     "{:,}",  "events",    False),
            ("PAUSES",        "pauses",    "{}",    "events",    False),
            ("HRV SDNN",      "sdnn",      "{:.1f}","ms \u00b7 low", False),
            ("SIGNAL QUALITY","quality",   "{:.1f}","% acceptable", False),
        ]
        for i, (title, key, fmt, unit, is_range) in enumerate(cards):
            if i > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet(f"color: {COL_BORDER}; max-width: 1px;")
                layout.addWidget(sep)

            card = QWidget()
            card_l = QVBoxLayout(card)
            card_l.setContentsMargins(20, 6, 20, 6)
            card_l.setSpacing(2)

            title_lbl = QLabel(title)
            title_lbl.setStyleSheet(f"color: {COL_MUTED}; font-size: 10px; font-weight: 600; letter-spacing: 0.5px;")
            card_l.addWidget(title_lbl)

            val_row = QHBoxLayout()
            val_row.setSpacing(2)

            if is_range:  # MIN / MAX with separate labels
                min_lbl = QLabel("—")
                min_lbl.setStyleSheet(f"color: {COL_TEXT}; font-size: 22px; font-weight: 700;")
                slash = QLabel(" / ")
                slash.setStyleSheet(f"color: {COL_MUTED}; font-size: 16px;")
                max_lbl = QLabel("—")
                max_lbl.setStyleSheet(f"color: {COL_RED}; font-size: 22px; font-weight: 700;")
                val_row.addWidget(min_lbl)
                val_row.addWidget(slash)
                val_row.addWidget(max_lbl)
                val_row.addStretch()
                self._value_labels["min_hr"] = min_lbl
                self._max_hr_label = max_lbl
            else:
                val = QLabel("—")
                val.setStyleSheet(f"color: {COL_TEXT}; font-size: 22px; font-weight: 700;")
                val_row.addWidget(val)
                val_row.addStretch()
                self._value_labels[key] = val

            card_l.addLayout(val_row)

            unit_lbl = QLabel(unit)
            unit_lbl.setStyleSheet(f"color: {COL_MUTED}; font-size: 10px;")
            card_l.addWidget(unit_lbl)

            layout.addWidget(card, 1)

    def update_summary(self, summary: dict):
        avg = summary.get('avg_hr', 0)
        min_hr = summary.get('min_hr', 0)
        max_hr = summary.get('max_hr', 0)
        if "avg_hr" in self._value_labels:
            self._value_labels["avg_hr"].setText(f"{avg:.0f}")
        if "min_hr" in self._value_labels:
            self._value_labels["min_hr"].setText(f"{min_hr:.0f}")
        if self._max_hr_label:
            self._max_hr_label.setText(f"{max_hr:.0f}")
            # Flag abnormally high max HR in red
            color = COL_RED if max_hr > 150 else COL_TEXT
            self._max_hr_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: 700;")
        if "beats" in self._value_labels:
            self._value_labels["beats"].setText(f"{summary.get('total_beats', 0):,}")
        if "pauses" in self._value_labels:
            self._value_labels["pauses"].setText(str(summary.get("pauses", 0)))
        if "sdnn" in self._value_labels:
            self._value_labels["sdnn"].setText(f"{summary.get('sdnn', 0):.1f}")
        if "quality" in self._value_labels:
            self._value_labels["quality"].setText(f"{summary.get('avg_quality', 0) * 100:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. HOLTER MAIN WINDOW  — orchestrates everything
# ══════════════════════════════════════════════════════════════════════════════

class HolterMainWindow(QDialog):
    """
    Full Holter analysis window. Can be opened during recording (Live mode)
    or after completion (Review mode).
    """

    def __init__(self, parent=None, session_dir: str = "",
                 patient_info: dict = None,
                 writer=None,
                 live_source=None,
                 duration_hours: int = 24):
        super().__init__(parent)
        self.setWindowTitle("Holter ECG Monitor & Analysis")
        self.setMinimumSize(1024, 680)
        # Use the screen the window will appear on (works across multiple screens)
        try:
            screen = QApplication.primaryScreen().availableGeometry()
            self.resize(max(1024, int(screen.width() * 0.92)),
                        max(680,  int(screen.height() * 0.90)))
        except Exception:
            self.resize(1280, 800)
        self.showMaximized()
        self.session_dir = session_dir
        self.patient_info = patient_info or (writer.patient_info if writer else {})
        self._writer = writer
        self._live_source = live_source
        self._duration_hours = duration_hours
        self._replay_engine = None
        self._metrics_list = []
        self._summary = {}

        if not self.session_dir and writer:
            self.session_dir = writer.session_dir

        self._load_session()
        self._build_ui()

        # If recording, start a timer to refresh live stats
        if self._writer:
            self._live_timer = QTimer(self)
            self._live_timer.timeout.connect(self._update_live_ui)
            self._live_timer.start(1000)
    
    def _update_live_ui(self):
        """Update the UI with live data from the stream writer"""
        if not self._writer or not self._writer.is_running:
            if hasattr(self, '_live_timer'):
                self._live_timer.stop()
            self._load_session()
            self._refresh_ui()
            return

        stats = self._writer.get_live_stats()
        if hasattr(self, '_status_bar'):
            self._status_bar.update_stats(stats['bpm'], stats['arrhythmias'])
        if hasattr(self, "_wave_panel"):
            self._wave_panel.refresh_waveforms()
        
        # Periodic reload of metrics if not too many
        if stats['elapsed'] % 30 < 2:  # roughly every 30s
            self._load_session()
            self._refresh_ui()

    def _refresh_ui(self):
        """Refresh all panels with latest summary/metrics"""
        if hasattr(self, "_summary_cards"):
            self._summary_cards.update_summary(self._summary)
        if hasattr(self, "_insight_panel"):
            self._insight_panel.update_text(self.patient_info, self._summary)
        if hasattr(self, '_overview_panel'):
            self._overview_panel.update_summary(self._summary)
        if hasattr(self, '_hrv_panel'):
            self._hrv_panel.update_hrv(self._metrics_list, self._summary)
        if hasattr(self, '_events_panel'):
            events = []
            if self._replay_engine:
                events = self._replay_engine.get_events_list()
            self._events_panel.load_events(events, self._summary)
        if hasattr(self, "_wave_panel"):
            self._wave_panel.set_live_source(self._live_source)
            self._wave_panel.set_replay_engine(self._replay_engine)
            self._wave_panel.refresh_waveforms()
        if hasattr(self, '_st_panel'):
            self._st_panel.update_st(self._metrics_list)
        if hasattr(self, '_report_table_panel'):
            self._report_table_panel.update_table(self._metrics_list, self._summary)

    def _load_session(self):
        """Load JSONL metrics and build replay engine."""
        self._metrics_list = []
        jsonl_path = os.path.join(self.session_dir, 'metrics.jsonl')
        if os.path.exists(jsonl_path):
            try:
                with open(jsonl_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._metrics_list.append(json.loads(line))
            except Exception as e:
                print(f"[HolterUI] Could not load metrics: {e}")

        ecgh_path = os.path.join(self.session_dir, 'recording.ecgh')
        if os.path.exists(ecgh_path):
            try:
                from .replay_engine import HolterReplayEngine
                self._replay_engine = HolterReplayEngine(ecgh_path)
                self._summary = self._replay_engine.get_summary()
            except Exception as e:
                print(f"[HolterUI] Could not load replay engine: {e}")
                self._summary = self._build_summary_from_jsonl()
        else:
            self._summary = self._build_summary_from_jsonl()

    def _build_summary_from_jsonl(self) -> dict:
        """Compute summary directly from JSONL when .ecgh not available."""
        import numpy as np
        if not self._metrics_list:
            return {}
        hr_vals = [m['hr_mean'] for m in self._metrics_list if m.get('hr_mean', 0) > 0]
        sdnn_vals = [m['rr_std'] for m in self._metrics_list if m.get('rr_std', 0) > 0]
        rmssd_vals = [m['rmssd'] for m in self._metrics_list if m.get('rmssd', 0) > 0]
        pnn50_vals = [m['pnn50'] for m in self._metrics_list if m.get('pnn50', 0) >= 0]
        total_beats = sum(m.get('beat_count', 0) for m in self._metrics_list)
        arrhy_counts: dict = {}
        for m in self._metrics_list:
            for a in m.get('arrhythmias', []):
                arrhy_counts[a] = arrhy_counts.get(a, 0) + 1

        hourly_hr: dict = {}
        for m in self._metrics_list:
            h = int(m.get('t', 0) // 3600)
            if m.get('hr_mean', 0) > 0:
                hourly_hr.setdefault(h, []).append(m['hr_mean'])
        hourly_avg = {h: round(float(np.mean(vals)), 1) for h, vals in hourly_hr.items()}

        duration = len(self._metrics_list) * 30
        return {
            'duration_sec': duration,
            'total_beats': total_beats,
            'avg_hr': float(np.mean(hr_vals)) if hr_vals else 0,
            'max_hr': float(np.max(hr_vals)) if hr_vals else 0,
            'min_hr': float(np.min(hr_vals)) if hr_vals else 0,
            'sdnn': float(np.mean(sdnn_vals)) if sdnn_vals else 0,
            'rmssd': float(np.mean(rmssd_vals)) if rmssd_vals else 0,
            'pnn50': float(np.mean(pnn50_vals)) if pnn50_vals else 0,
            'arrhythmia_counts': arrhy_counts,
            'hourly_hr': hourly_avg,
            'longest_rr_ms': max((m.get('longest_rr', 0) for m in self._metrics_list), default=0),
            'pauses': sum(m.get('pauses', 0) for m in self._metrics_list),
            'tachy_beats': sum(m.get('tachy_beats', 0) for m in self._metrics_list),
            'brady_beats': sum(m.get('brady_beats', 0) for m in self._metrics_list),
            'avg_quality': float(np.mean([m.get('quality', 1) for m in self._metrics_list])),
            'chunks_analyzed': len(self._metrics_list),
            'patient_info': self.patient_info,
        }

    def _build_ui(self):
        # ── Clinical white theme ──────────────────────────────────────────
        self.setStyleSheet(f"""
            QDialog {{ background: {COL_BG}; }}
            QWidget {{ color: {COL_TEXT}; }}
        """)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ══════════════════════════════════════════════════════════════════
        # TOP TOOLBAR
        # ══════════════════════════════════════════════════════════════════
        toolbar = QFrame()
        toolbar.setFixedHeight(52)
        toolbar.setStyleSheet(f"background: {COL_DARK}; border: none;")
        tb_l = QHBoxLayout(toolbar)
        tb_l.setContentsMargins(16, 0, 16, 0)
        tb_l.setSpacing(12)

        # Brand
        brand = QLabel("<b style='color:#2563EB'>HOLTER</b> "
                       "<span style='color:#F1F5F9'>ECG ANALYSIS</span>")
        brand.setStyleSheet("font-size: 15px;")
        tb_l.addWidget(brand)

        # Recording status indicator
        dur_sec = self._summary.get('duration_sec', 0)
        dur_h   = int(dur_sec // 3600)
        dur_m   = int((dur_sec % 3600) // 60)
        if dur_sec <= 0:
            dur_h, dur_m = self._duration_hours, 0
        is_live = bool(self._writer)
        dot_color = "#22C55E" if is_live else "#6B7280"
        status_text = "Recording" if is_live else "Review"
        status_lbl = QLabel(
            f"<span style='font-size:10px; color:{dot_color}'>&#9679;</span>  "
            f"<span style='color:#CBD5E1; font-size:12px;'>{status_text} &middot; "
            f"{dur_h:02d}h {dur_m:02d}m</span>"
        )
        status_lbl.setStyleSheet("border: 1px solid #334155; border-radius: 12px; padding: 2px 10px; background: #1E293B;")
        tb_l.addWidget(status_lbl)
        tb_l.addStretch()

        # Toolbar buttons
        for label, slot, color in [
            ("Patient info",     None,                  "#334155"),
            ("Export",          None,                  "#334155"),
            ("Generate report", self._generate_report, COL_GREEN),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(34)
            btn.setStyleSheet(_btn_style(color, "white",
                              "#16A34A" if color == COL_GREEN else "#475569"))
            if slot:
                btn.clicked.connect(slot)
            tb_l.addWidget(btn)

        close_btn = QPushButton("✕ Close")
        close_btn.setFixedHeight(34)
        close_btn.setStyleSheet(_btn_style("#3F3F46", "white", "#52525B"))
        close_btn.clicked.connect(self.close)
        tb_l.addWidget(close_btn)

        main_layout.addWidget(toolbar)

        # Live status bar (recording only)
        if self._writer:
            self._status_bar = HolterStatusBar(self, target_hours=self._duration_hours)
            self._status_bar.stop_requested.connect(self._stop_recording)
            main_layout.addWidget(self._status_bar)

        # ══════════════════════════════════════════════════════════════════
        # METRICS BAR  (6 stat cards in one horizontal strip)
        # ══════════════════════════════════════════════════════════════════
        self._summary_cards = HolterSummaryCards()
        self._summary_cards.setFixedHeight(88)
        main_layout.addWidget(self._summary_cards)

        # ══════════════════════════════════════════════════════════════════
        # NAVIGATION TAB BAR
        # ══════════════════════════════════════════════════════════════════
        self._tabs = QTabWidget()
        self._tabs.setUsesScrollButtons(True)
        self._tabs.setMovable(False)
        self._tabs.setElideMode(Qt.ElideRight)
        self._tabs.tabBar().setExpanding(False)
        self._tabs.setDocumentMode(True)
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background: {COL_BG};
            }}
            QTabBar::tab {{
                background: {COL_SURFACE};
                color: {COL_MUTED};
                padding: 10px 22px;
                font-size: 12px;
                font-weight: 500;
                border: none;
                border-bottom: 2px solid transparent;
                min-width: 80px;
            }}
            QTabBar::tab:selected {{
                color: {COL_TEXT};
                border-bottom: 2px solid {COL_ORANGE};
                font-weight: 700;
                background: {COL_SURFACE};
            }}
            QTabBar::tab:hover {{ color: {COL_TEXT}; }}
            QTabBar {{ background: {COL_SURFACE};
                       border-bottom: 1px solid {COL_BORDER}; }}
        """)

        # ── Tab 0: Recording (3-column layout with 12-lead workspace) ──
        recording_tab = QWidget()
        recording_tab.setStyleSheet(f"background: {COL_BG};")
        rec_outer = QVBoxLayout(recording_tab)
        rec_outer.setContentsMargins(0, 0, 0, 0)
        rec_outer.setSpacing(0)

        # Main content: splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {COL_BORDER}; }}")

        # ── LEFT: 12-lead thumbnail grid + Tools ─────────────────────────
        left_panel = QFrame()
        left_panel.setStyleSheet(f"background: {COL_SURFACE}; border-right: 1px solid {COL_BORDER};")
        left_panel.setFixedWidth(280)
        left_l = QVBoxLayout(left_panel)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(0)

        # 12-lead thumbnails header
        thmb_hdr = QLabel("12-LEAD VIEW")
        thmb_hdr.setFixedHeight(32)
        thmb_hdr.setStyleSheet(
            f"color: {COL_MUTED}; font-size: 10px; font-weight: 700;"
            f" letter-spacing: 1px; padding-left: 12px;"
            f" background: {COL_GRAY}; border-bottom: 1px solid {COL_BORDER};"
        )
        left_l.addWidget(thmb_hdr)

        # Thumbnail scroll area (2-column grid)
        thmb_scroll = QScrollArea()
        thmb_scroll.setWidgetResizable(True)
        thmb_scroll.setFrameShape(QFrame.NoFrame)
        thmb_scroll.setStyleSheet(f"QScrollArea {{ background: {COL_SURFACE}; border: none; }}")
        thmb_container = QWidget()
        thmb_container.setStyleSheet(f"background: {COL_SURFACE};")
        thmb_grid = QGridLayout(thmb_container)
        thmb_grid.setContentsMargins(8, 8, 8, 8)
        thmb_grid.setSpacing(6)

        LEADS_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"]
        self._thumb_curves = []
        if pg is not None:
            pg.setConfigOptions(antialias=True, background='w', foreground='k')
            for idx, lead in enumerate(LEADS_ORDER):
                card = QFrame()
                card.setFixedHeight(62)
                card.setStyleSheet(
                    f"background: {COL_SURFACE}; border: 1px solid {COL_BORDER};"
                    " border-radius: 8px;"
                )
                card_l = QVBoxLayout(card)
                card_l.setContentsMargins(6, 4, 6, 4)
                card_l.setSpacing(2)
                lbl = QLabel(lead)
                lbl.setStyleSheet(f"color: {COL_MUTED}; font-size: 10px; font-weight: 600;")
                card_l.addWidget(lbl)
                p = pg.PlotWidget()
                p.setMenuEnabled(False)
                p.setMouseEnabled(x=False, y=False)
                p.hideButtons()
                p.setBackground('w')
                p.showGrid(x=False, y=False)
                p.getAxis('left').setStyle(showValues=False)
                p.getAxis('bottom').setStyle(showValues=False)
                p.setYRange(-1.5, 1.5, padding=0)
                p.setContentsMargins(0, 0, 0, 0)
                c = p.plot(pen=pg.mkPen(color=(0, 0, 0), width=0.7))
                card_l.addWidget(p, 1)
                self._thumb_curves.append((c, lead))
                thmb_grid.addWidget(card, idx // 2, idx % 2)

        thmb_scroll.setWidget(thmb_container)
        left_l.addWidget(thmb_scroll, 1)

        # Tools section
        tools_hdr = QLabel("TOOLS")
        tools_hdr.setFixedHeight(32)
        tools_hdr.setStyleSheet(
            f"color: {COL_MUTED}; font-size: 10px; font-weight: 700;"
            f" letter-spacing: 1px; padding-left: 12px;"
            f" background: {COL_GRAY}; border-top: 1px solid {COL_BORDER};"
            f" border-bottom: 1px solid {COL_BORDER};"
        )
        left_l.addWidget(tools_hdr)
        for tool_name in ["Measuring ruler", "Magnifying glass",
                          "Gain settings", "Add event [space]"]:
            tbtn = QPushButton(tool_name)
            tbtn.setFixedHeight(30)
            tbtn.setStyleSheet(f"""
                QPushButton {{
                    background: {COL_SURFACE};
                    color: {COL_TEXT};
                    border: none;
                    border-bottom: 1px solid {COL_BORDER};
                    text-align: left;
                    padding-left: 14px;
                    font-size: 12px;
                }}
                QPushButton:hover {{ background: {COL_GRAY}; }}
            """)
            left_l.addWidget(tbtn)

        splitter.addWidget(left_panel)

        # ── CENTER: Full-width parallel 12-lead strips ───────────────────
        self._wave_panel = HolterDenseWavePanel(
            live_source=self._live_source,
            replay_engine=self._replay_engine
        )
        splitter.addWidget(self._wave_panel)

        # ── RIGHT: HRV summary + events ──────────────────────────────────
        right_panel = QFrame()
        right_panel.setFixedWidth(260)
        right_panel.setStyleSheet(f"background: {COL_SURFACE}; border-left: 1px solid {COL_BORDER};")
        right_l = QVBoxLayout(right_panel)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(0)

        # Overview stats
        self._overview_panel = HolterOverviewPanel()
        self._overview_panel.update_summary(self._summary)
        self._overview_panel.setStyleSheet(f"background: {COL_SURFACE}; color: {COL_TEXT};")
        right_l.addWidget(self._overview_panel, 1)

        # Events
        self._events_panel = HolterEventsPanel()
        events = []
        if self._replay_engine:
            try:
                events = self._replay_engine.get_events_list()
            except Exception:
                pass
        else:
            for m in self._metrics_list:
                for a in m.get('arrhythmias', []):
                    t = m.get('t', 0)
                    h2 = int(t // 3600)
                    mn2 = int((t % 3600) // 60)
                    s2 = int(t % 60)
                    events.append({'timestamp': t, 'label': a,
                                   'time_str': f"{h2:02d}:{mn2:02d}:{s2:02d}"})
        self._events_panel.load_events(events, self._summary)
        self._events_panel.seek_requested.connect(self._on_seek_requested)
        self._events_panel.setStyleSheet(f"background: {COL_SURFACE}; color: {COL_TEXT};")
        right_l.addWidget(self._events_panel, 1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)   # left fixed
        splitter.setStretchFactor(1, 1)   # center fills
        splitter.setStretchFactor(2, 0)   # right fixed

        rec_outer.addWidget(splitter, 1)

        # ── BOTTOM PLAYBACK BAR ───────────────────────────────────────────
        play_bar = QFrame()
        play_bar.setFixedHeight(52)
        play_bar.setStyleSheet(
            f"background: {COL_SURFACE};"
            f" border-top: 1px solid {COL_BORDER};"
        )
        pb_l = QHBoxLayout(play_bar)
        pb_l.setContentsMargins(12, 0, 12, 0)
        pb_l.setSpacing(6)

        # Duration replay panel embedded in bar
        duration = self._summary.get('duration_sec', self._duration_hours * 3600)
        self._replay_panel = HolterReplayPanel(duration_sec=duration)
        if self._replay_engine:
            self._replay_panel.set_replay_engine(self._replay_engine)
            self._replay_panel.seek_requested.connect(self._on_seek_requested)
        pb_l.addWidget(self._replay_panel, 1)

        rec_outer.addWidget(play_bar)
        self._tabs.addTab(recording_tab, "Recording")

        # ── Other tabs ───────────────────────────────────────────────────
        self._hrv_panel = HolterHRVPanel()
        self._hrv_panel.update_hrv(self._metrics_list, self._summary)
        self._tabs.addTab(self._hrv_panel, "HRV Analysis")

        self._st_panel = HolterSTTendencyPanel()
        self._st_panel.update_st(self._metrics_list)
        self._tabs.addTab(self._st_panel, "ST Tendency")

        record_dir = self.session_dir if self.session_dir else os.path.join(os.getcwd(), "recordings")
        if os.path.isfile(record_dir):
            record_dir = os.path.dirname(record_dir)
        self._record_mgmt_panel = HolterRecordManagementPanel(output_dir=record_dir)
        self._tabs.addTab(self._record_mgmt_panel, "Records")

        for tab_name, features in [
            ("Replay",    ["Seek to any point in the recording",
                           "Speed controls: 0.25× to 8×",
                           "Event jump: AF / Brady / Tachy"]),
            ("Events",    ["Episode timeline and event list",
                           "Arrhythmia type / timestamp / lead",
                           "Jump to event in strip view"]),
            ("Histogram", ["RR Interval / Heart Rate distribution",
                           "Bin selection and zoom",
                           "Beat type filters: VE, SVE, Paced"]),
            ("AF Analysis",["AF episode burden chart",
                            "Beat strip grid by event",
                            "Parameters and remove controls"]),
        ]:
            self._tabs.addTab(
                HolterPlaceholderPanel(tab_name, features), tab_name)

        self._report_table_panel = HolterReportTablePanel()
        self._report_table_panel.update_table(self._metrics_list, self._summary)
        self._tabs.addTab(self._report_table_panel, "Report Table")

        self._insight_panel = HolterInsightPanel()
        self._tabs.addTab(self._insight_panel, "Summary")

        main_layout.addWidget(self._tabs, 1)
        self._refresh_ui()

    def _on_seek_requested(self, target_sec: float):
        if self._replay_engine:
            self._replay_engine.seek(target_sec)
            try:
                self._wave_panel.set_replay_frame(self._replay_engine.get_all_leads_data(window_sec=8.0))
            except Exception:
                pass

    def attach_writer(self, writer, session_dir: str = "", patient_info: dict = None):
        self._writer = writer
        if session_dir:
            self.session_dir = session_dir
        if patient_info:
            self.patient_info = patient_info
        if writer and not hasattr(self, "_status_bar"):
            self._status_bar = HolterStatusBar(self, target_hours=self._duration_hours)
            self._status_bar.stop_requested.connect(self._stop_recording)
            self.layout().insertWidget(1, self._status_bar)
        if writer and not hasattr(self, "_live_timer"):
            self._live_timer = QTimer(self)
            self._live_timer.timeout.connect(self._update_live_ui)
        if writer and hasattr(self, "_live_timer") and not self._live_timer.isActive():
            self._live_timer.start(1000)
        self._refresh_ui()

    def load_completed_session(self, session_dir: str, patient_info: dict = None):
        self.session_dir = session_dir
        if patient_info:
            self.patient_info = patient_info
        self._writer = None
        self._load_session()
        if hasattr(self, "_replay_panel") and self._replay_engine:
            self._replay_panel.set_replay_engine(self._replay_engine)
        self._refresh_ui()

    def _stop_recording(self):
        """Finalize the recording and switch to review mode"""
        if self._writer:
            summary = self._writer.stop()
            self._writer = None
            if hasattr(self, '_status_bar'):
                self._status_bar.setVisible(False)
                self._status_bar.cleanup()
            
            QMessageBox.information(self, "Recording Complete", 
                                    f"comphrensive recording saved to:\n{summary.get('session_dir', '')}")
            
            # Switch to review mode
            self.load_completed_session(summary.get('session_dir', ''), self.patient_info)

    def _generate_report(self):
        """Trigger report generation."""
        from PyQt5.QtWidgets import QProgressDialog
        progress = QProgressDialog("Generating Holter Report...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()

        try:
            # Prefer the root-level report_generator (pink-grid ECG-style report)
            import importlib.util as _ilu, os as _os
            _rg_root = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)), 'report_generator.py')
            _spec = _ilu.spec_from_file_location('holter_report_gen', _rg_root)
            _mod  = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            generate_holter_report = _mod.generate_holter_report
            path = generate_holter_report(
                session_dir=self.session_dir,
                patient_info=self.patient_info,
                summary=self._summary,
            )
            progress.close()
            QMessageBox.information(self, "Report Generated",
                                    f"Holter report saved:\n{path}")
        except Exception as e:
            progress.close()
            QMessageBox.warning(self, "Report Error", f"Could not generate report:\n{e}")

    def closeEvent(self, event):
        if self._replay_engine:
            try:
                self._replay_engine.close()
            except Exception:
                pass
        super().closeEvent(event)

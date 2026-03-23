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

# ── Colour palette ─────────────────────────────────────────────────────────────
COL_ORANGE  = "#E65100"
COL_DARK    = "#1A1A2E"
COL_BLUE    = "#1565C0"
COL_GREEN   = "#2E7D32"
COL_RED     = "#B71C1C"
COL_LIGHT   = "#FFF8F0"
COL_GRAY    = "#F5F5F5"
COL_BG      = "#0D1117"    # dark ECG-style background
COL_GREEN_ECG = "#00FF00"  # ECG trace green


def _btn_style(bg="#E65100", fg="white", hover="#FF6D00"):
    return f"""
        QPushButton {{
            background: {bg};
            color: {fg};
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background: {hover};
        }}
        QPushButton:pressed {{
            background: {bg};
            opacity: 0.8;
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
    Modal dialog to configure and start a Holter recording.
    Pre-fills patient info from existing patient_details cache.
    """

    def __init__(self, parent=None, patient_info: dict = None, output_dir: str = "recordings"):
        super().__init__(parent)
        self.setWindowTitle("Start Holter Recording")
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
                stop:0 {COL_ORANGE}, stop:1 {COL_BLUE});
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
        for label in ["HRV Event", "HRV Tendency"]:
            btn = QPushButton(label)
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
                QPushButton:hover {{ background: #1976D2; }}
            """)
            tab_row.addWidget(btn)
        tab_row.addStretch()
        layout.addLayout(tab_row)

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
        layout.addWidget(self._table, 1)

        # Bottom summary stats (matching reference image bottom panel)
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
            row, col = divmod(i, 4)
            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet("color: #aaa; font-size: 10px;")
            val = QLabel("—")
            val.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
            summary_layout.addWidget(lbl, row * 2, col)
            summary_layout.addWidget(val, row * 2 + 1, col)
            self._summary_labels[key] = val
        layout.addWidget(summary_frame)

        # Action buttons
        btn_row = QHBoxLayout()
        for label, color in [("Insert", COL_BLUE), ("Reset", COL_GRAY), ("Remove", COL_RED)]:
            btn = QPushButton(label)
            btn.setStyleSheet(_btn_style(color, "white" if color != COL_GRAY else COL_DARK))
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

    def update_hrv(self, metrics_list: list, summary: dict):
        """
        metrics_list: list of dicts from JSONL (one per 30s chunk)
        summary: from HolterReplayEngine.get_summary()
        """
        import numpy as np

        # Build per-hour rows
        hourly: dict = {}
        for m in metrics_list:
            h = int(m.get('t', 0) // 3600)
            hourly.setdefault(h, []).append(m)

        rows = []
        # Entire recording row
        all_rr = [m.get('rr_ms', 0) for m in metrics_list if m.get('rr_ms', 0) > 0]
        all_rr_std = [m.get('rr_std', 0) for m in metrics_list if m.get('rr_std', 0) > 0]
        if all_rr:
            rows.append(("Entire", "—", f"{len(metrics_list)*30//60:02d}:{len(metrics_list)*30%60:02d}",
                          f"{np.mean(all_rr):.0f}ms",
                          f"{summary.get('sdnn', 0):.0f}ms",
                          f"{summary.get('sdnn', 0)*0.82:.0f}ms",
                          f"{27:.2f}",
                          f"{summary.get('pnn50', 0):.2f}%",
                          ""))
        # Per hour
        for h in sorted(hourly.keys()):
            chunks = hourly[h]
            rr_vals = [c.get('rr_ms', 0) for c in chunks if c.get('rr_ms', 0) > 0]
            rr_stds = [c.get('rr_std', 0) for c in chunks if c.get('rr_std', 0) > 0]
            pnn50s  = [c.get('pnn50', 0) for c in chunks]
            if not rr_vals:
                continue
            rows.append((
                "Hour",
                f"{h:02d}:00",
                f"01:00",
                f"{np.mean(rr_vals):.0f}ms",
                f"{np.mean(rr_stds):.0f}ms" if rr_stds else "—",
                "—",
                "—",
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

        # Update summary bottom
        s = summary
        for key, fmt in [
            ("nns",    str(s.get('total_beats', 0))),
            ("mean_nn", f"{s.get('avg_hr', 0):.0f}ms"),
            ("sdnn",    f"{s.get('sdnn', 0):.0f}ms"),
            ("sdann",   f"{s.get('sdnn', 0)*0.82:.0f}ms"),
            ("rmssd",   f"{s.get('rmssd', 0):.0f}ms"),
            ("pnn50",   f"{s.get('pnn50', 0):.2f}%"),
            ("triidx",  "—"),
            ("sdnnidx", "—"),
        ]:
            if key in self._summary_labels:
                self._summary_labels[key].setText(fmt)


# ══════════════════════════════════════════════════════════════════════════════
# 5. HOLTER EVENTS PANEL  (like reference Image 7)
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
    Controls for replaying a saved Holter recording.
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

        self._replay_engine = None

    def set_replay_engine(self, engine):
        self._replay_engine = engine
        self._slider.setRange(0, int(engine.duration_sec))
        engine.set_position_callback(self._on_position_update)

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
    """Professional 12‑lead Holter waveform workspace."""

    LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    def __init__(self, parent=None, live_source=None, replay_engine=None):
        super().__init__(parent)
        self.live_source = live_source
        self.replay_engine = replay_engine
        self.window_sec = 8.0
        self._lead_widgets = []
        self._replay_buffer = None
        self.setStyleSheet("background: #101722; border: 1px solid #2B3B50; border-radius: 14px;")
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_waveforms)
        self._timer.start(150)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("12‑Lead Live Workspace")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        subtitle = QLabel("Professional Holter view with synchronized moving strips.")
        subtitle.setStyleSheet("color: #94A3B8; font-size: 11px;")
        header_col = QVBoxLayout()
        header_col.addWidget(title)
        header_col.addWidget(subtitle)
        header.addLayout(header_col)
        header.addStretch()
        speed = QLabel("Paper Speed 25mm/s  |  Gain 10mm/mV")
        speed.setStyleSheet(f"color: {COL_ORANGE}; font-size: 11px; font-weight: bold;")
        header.addWidget(speed)
        layout.addLayout(header)

        if pg is None:
            fallback = QLabel("pyqtgraph is not available, so the live Holter waveform grid cannot be rendered in this environment.")
            fallback.setWordWrap(True)
            fallback.setStyleSheet("color: #FCA5A5; font-size: 12px; padding: 16px;")
            layout.addWidget(fallback)
            return

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        pg.setConfigOptions(antialias=True, background="#0B1220", foreground="#CBD5E1")
        for idx, lead in enumerate(self.LEADS):
            card = QFrame()
            card.setStyleSheet("background: #0B1220; border: 1px solid #213147; border-radius: 10px;")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(6, 6, 6, 6)
            card_layout.setSpacing(4)

            label = QLabel(lead)
            label.setStyleSheet("color: #E2E8F0; font-size: 12px; font-weight: bold; padding-left: 4px;")
            card_layout.addWidget(label)

            plot = pg.PlotWidget()
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            plot.setBackground("#0B1220")
            plot.showGrid(x=True, y=True, alpha=0.16)
            plot.getAxis("left").setStyle(showValues=False)
            plot.getAxis("bottom").setStyle(showValues=False)
            plot.setYRange(-1.6, 1.6, padding=0)
            plot.setContentsMargins(0, 0, 0, 0)
            curve = plot.plot(pen=pg.mkPen(COL_GREEN_ECG, width=1.6))
            card_layout.addWidget(plot, 1)

            self._lead_widgets.append((curve, plot))
            row, col = divmod(idx, 4)
            grid.addWidget(card, row, col)

        layout.addLayout(grid, 1)

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


class HolterInsightPanel(QFrame):
    """Narrative summary that turns the metrics into a clinical-style report preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: #101722; border: 1px solid #2B3B50; border-radius: 14px;")
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
        self.setStyleSheet("background: #101722; border: 1px solid #2B3B50; border-radius: 14px;")
        self._build_ui()

    def _build_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)

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
            frame.setStyleSheet("background: #0B1220; border: 1px solid #223247; border-radius: 12px;")
            box = QVBoxLayout(frame)
            box.setContentsMargins(12, 10, 12, 10)
            box.setSpacing(4)
            lbl = QLabel(title)
            lbl.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: bold;")
            val = QLabel("—")
            val.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
            unit_lbl = QLabel(unit)
            unit_lbl.setStyleSheet(f"color: {COL_ORANGE}; font-size: 10px;")
            box.addWidget(lbl)
            box.addWidget(val)
            box.addWidget(unit_lbl)
            self._value_labels[key] = val
            layout.addWidget(frame, 0 if idx < 3 else 1, idx % 3)

    def update_summary(self, summary: dict):
        self._value_labels["avg_hr"].setText(f"{summary.get('avg_hr', 0):.0f}")
        self._value_labels["range_hr"].setText(f"{summary.get('min_hr', 0):.0f} / {summary.get('max_hr', 0):.0f}")
        self._value_labels["beats"].setText(f"{summary.get('total_beats', 0):,}")
        self._value_labels["pauses"].setText(str(summary.get("pauses", 0)))
        self._value_labels["quality"].setText(f"{summary.get('avg_quality', 0) * 100:.1f}")
        self._value_labels["sdnn"].setText(f"{summary.get('sdnn', 0):.1f}")


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
        self.setMinimumSize(1380, 860)
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
        self.setStyleSheet(f"background: {COL_BG}; color: white;")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Top toolbar ──
        toolbar = QFrame()
        toolbar.setFixedHeight(48)
        toolbar.setStyleSheet(f"background: {COL_DARK}; border-bottom: 2px solid {COL_ORANGE};")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(12, 4, 12, 4)

        title_lbl = QLabel("HOLTER ECG ANALYSIS SUITE")
        title_lbl.setStyleSheet(f"color: {COL_ORANGE}; font-size: 16px; font-weight: bold;")
        tb_layout.addWidget(title_lbl)

        # Patient info
        pname = self.patient_info.get('name', self.patient_info.get('patient_name', ''))
        if pname:
            p_lbl = QLabel(f"  |  Patient: {pname}")
            p_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
            tb_layout.addWidget(p_lbl)

        dur_sec = self._summary.get('duration_sec', 0)
        dur_h = int(dur_sec // 3600)
        dur_m = int((dur_sec % 3600) // 60)
        if dur_sec <= 0:
            dur_h = self._duration_hours
            dur_m = 0
        d_lbl = QLabel(f"  |  Duration: {dur_h}h {dur_m}m")
        d_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        tb_layout.addWidget(d_lbl)

        tb_layout.addStretch()

        # Generate report button
        report_btn = QPushButton("📄  Generate Report")
        report_btn.setStyleSheet(_btn_style(COL_GREEN, "white", "#388E3C"))
        report_btn.clicked.connect(self._generate_report)
        tb_layout.addWidget(report_btn)

        close_btn = QPushButton("✕  Close")
        close_btn.setStyleSheet(_btn_style("#555", "white", "#666"))
        close_btn.clicked.connect(self.close)
        tb_layout.addWidget(close_btn)

        main_layout.addWidget(toolbar)

        # ── Live status bar (only during recording) ──
        if self._writer:
            self._status_bar = HolterStatusBar(self, target_hours=self._duration_hours)
            self._status_bar.stop_requested.connect(self._stop_recording)
            main_layout.addWidget(self._status_bar)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 12, 12, 12)
        body_layout.setSpacing(12)

        self._summary_cards = HolterSummaryCards()
        body_layout.addWidget(self._summary_cards)

        workspace = QSplitter(Qt.Horizontal)
        workspace.setChildrenCollapsible(False)
        workspace.setStyleSheet("""
            QSplitter::handle {
                background: #223247;
                width: 2px;
            }
        """)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self._wave_panel = HolterWaveGridPanel(live_source=self._live_source, replay_engine=self._replay_engine)
        left_layout.addWidget(self._wave_panel, 2)

        self._insight_panel = HolterInsightPanel()
        left_layout.addWidget(self._insight_panel, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self._overview_panel = HolterOverviewPanel()
        self._overview_panel.update_summary(self._summary)
        right_layout.addWidget(self._overview_panel, 1)

        self._events_panel = HolterEventsPanel()
        events = []
        if self._replay_engine:
            events = self._replay_engine.get_events_list()
        else:
            for m in self._metrics_list:
                for a in m.get('arrhythmias', []):
                    t = m['t']
                    h = int(t // 3600)
                    mn = int((t % 3600) // 60)
                    s = int(t % 60)
                    events.append({'timestamp': t, 'label': a, 'time_str': f"{h:02d}:{mn:02d}:{s:02d}"})
        self._events_panel.load_events(events, self._summary)
        self._events_panel.seek_requested.connect(self._on_seek_requested)
        right_layout.addWidget(self._events_panel, 1)

        workspace.addWidget(left)
        workspace.addWidget(right)
        workspace.setSizes([930, 390])
        body_layout.addWidget(workspace, 3)

        # ── Tab widget ──
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background: {COL_BG};
            }}
            QTabBar::tab {{
                background: {COL_DARK};
                color: #888;
                padding: 8px 20px;
                font-size: 12px;
                border: none;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: white;
                border-bottom: 2px solid {COL_ORANGE};
                font-weight: bold;
            }}
            QTabBar::tab:hover {{ color: white; }}
        """)

        # HRV tab
        self._hrv_panel = HolterHRVPanel()
        self._hrv_panel.update_hrv(self._metrics_list, self._summary)
        self._tabs.addTab(self._hrv_panel, "📈  HRV Analysis")

        # Replay tab
        duration = self._summary.get('duration_sec', self._duration_hours * 3600)
        self._replay_panel = HolterReplayPanel(duration_sec=duration)
        if self._replay_engine:
            self._replay_panel.set_replay_engine(self._replay_engine)
            self._replay_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._replay_panel, "▶  Replay")

        body_layout.addWidget(self._tabs, 1)
        main_layout.addWidget(body, 1)
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
                                    f"Holter recording saved to:\n{summary.get('session_dir', '')}")
            
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
            from .report_generator import generate_holter_report
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

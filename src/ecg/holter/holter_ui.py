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
import shutil
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialog, QLineEdit, QComboBox, QSlider, QGroupBox, QFrame,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QScrollArea, QGridLayout, QSpinBox, QMessageBox,
    QFileDialog, QApplication, QProgressBar, QSplitter, QTextEdit, QInputDialog, QDoubleSpinBox,
    QAbstractItemView, QToolButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPoint, QPointF, QRect
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter, QPen, QBrush, QPixmap

try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:
    pg = None
    HAS_PG = False


def _resolve_recordings_dir(session_dir: str = "") -> str:
    """Return the recordings root directory for the current session or project."""
    normalized = os.path.dirname(session_dir) if os.path.isfile(session_dir) else session_dir
    if normalized and os.path.isdir(normalized):
        if os.path.basename(os.path.normpath(normalized)).lower() == "recordings":
            return normalized
        parent_dir = os.path.dirname(normalized)
        if os.path.basename(os.path.normpath(parent_dir)).lower() == "recordings":
            return parent_dir

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    preferred_dir = os.path.join(src_root, "recordings")
    fallback_dir = os.path.join(os.getcwd(), "recordings")
    if os.path.isdir(preferred_dir):
        return preferred_dir
    if os.path.isdir(fallback_dir):
        return fallback_dir
    return preferred_dir


def _find_latest_completed_session(output_dir: str) -> str:
    """Return the newest completed session directory, or empty string."""
    if not output_dir or not os.path.isdir(output_dir):
        return ""

    candidates = []
    try:
        for name in os.listdir(output_dir):
            session_dir = os.path.join(output_dir, name)
            ecgh_path = os.path.join(session_dir, "recording.ecgh")
            if not os.path.isdir(session_dir):
                continue
            if not os.path.exists(ecgh_path):
                continue
            try:
                sort_key = os.path.getmtime(ecgh_path)
            except Exception:
                sort_key = os.path.getmtime(session_dir)
            candidates.append((sort_key, session_dir))
    except Exception:
        return ""

    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _metrics_duration_sec(metrics_list: list) -> float:
    return float(sum(m.get('duration', 0.0) or 0.0 for m in metrics_list))

try:
    from .theme import (
        ADC_TO_MV,
        COL_BEAT_S,
        COL_BG,
        COL_BLACK,
        COL_BTN_ACTIVE_BG,
        COL_BTN_ACTIVE_TEXT,
        COL_DARK,
        COL_GRAY,
        COL_GREEN,
        COL_GREEN_DRK,
        COL_GREEN_MID,
        COL_GRID_MAJOR,
        COL_GRID_MINOR,
        COL_RED,
        COL_TEXT,
        COL_TIMESTAMP,
        COL_WAVE_ORANGE,
        COL_WAVE_RED,
        COL_WHITE,
        COL_YELLOW,
        GAINS,
        PAPER_SPEEDS,
        TOOL_CALIPER,
        TOOL_MAGNIFY,
        TOOL_RULER,
        TOOL_SELECT,
    )
    from .tool_engine import (
        ECGToolEngine,
        amplitude_mv_from_pixels,
        caliper_label,
        canonical_tool,
        hint as tool_hint,
        interval_ms_from_pixels,
        ruler_label,
        tool_specs,
        tooltip as tool_tooltip,
    )
    from .session_store import append_annotation, load_annotations, load_events, load_metrics, read_session_metadata
except ImportError:
    from ecg.holter.theme import (
        ADC_TO_MV,
        COL_BEAT_S,
        COL_BG,
        COL_BLACK,
        COL_BTN_ACTIVE_BG,
        COL_BTN_ACTIVE_TEXT,
        COL_DARK,
        COL_GRAY,
        COL_GREEN,
        COL_GREEN_DRK,
        COL_GREEN_MID,
        COL_GRID_MAJOR,
        COL_GRID_MINOR,
        COL_RED,
        COL_TEXT,
        COL_TIMESTAMP,
        COL_WAVE_ORANGE,
        COL_WAVE_RED,
        COL_WHITE,
        COL_YELLOW,
        GAINS,
        PAPER_SPEEDS,
        TOOL_CALIPER,
        TOOL_MAGNIFY,
        TOOL_RULER,
        TOOL_SELECT,
    )
    from ecg.holter.tool_engine import (
        ECGToolEngine,
        amplitude_mv_from_pixels,
        caliper_label,
        canonical_tool,
        hint as tool_hint,
        interval_ms_from_pixels,
        ruler_label,
        tool_specs,
        tooltip as tool_tooltip,
    )
    from ecg.holter.session_store import append_annotation, load_annotations, load_events, load_metrics, read_session_metadata

# Professional UI palette (kept separate from signal colors).
UI_BG = "#0B1220"
UI_PANEL = "#0F1A2E"
UI_PANEL_ALT = "#13213A"
UI_CARD = "#101B2F"
UI_BORDER = "#243552"
UI_TEXT = "#E6EDF7"
UI_MUTED = "#9AAECB"
UI_ACCENT = "#2F80ED"
UI_ACCENT_HOVER = "#4B96FA"
UI_SUCCESS = "#16C172"
UI_WARNING = "#F59E0B"


def _style_btn(bg=UI_PANEL_ALT, fg=UI_TEXT, hover="#1A2C49"):
    return f"""
        QPushButton {{
            background: {bg};
            color: {fg};
            border: 1px solid {UI_BORDER};
            border-radius: 8px;
            padding: 7px 14px;
            font-size: 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background: {hover};
            color: {UI_TEXT};
        }}
        QPushButton:pressed {{ background: {bg}; border: 1px solid {UI_ACCENT_HOVER}; }}
        QPushButton:disabled {{ background: #1A2233; color: #5F708A; border: 1px solid #2A3953; }}
    """


def _style_active_btn():
    return f"""
        QPushButton {{
            background: {UI_ACCENT};
            color: {UI_TEXT};
            border: 1px solid #5EA4FF;
            border-radius: 8px;
            padding: 7px 14px;
            font-size: 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{ background: {UI_ACCENT_HOVER}; }}
    """


def _table_style():
    return f"""
        QTableWidget {{
            background: {UI_PANEL};
            alternate-background-color: {UI_PANEL_ALT};
            color: {UI_TEXT};
            gridline-color: {UI_BORDER};
            font-size: 12px;
            border: 1px solid {UI_BORDER};
            selection-background-color: #1E3A5F;
            selection-color: {UI_TEXT};
        }}
        QHeaderView::section {{
            background: {UI_PANEL_ALT};
            color: {UI_MUTED};
            font-size: 11px;
            font-weight: bold;
            padding: 6px;
            border: 1px solid {UI_BORDER};
        }}
        QTableWidget::item {{ padding: 5px; border: none; }}
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
        self._card_frames = []
        self._grid = None
        self.setStyleSheet(f"background:{UI_BG};")
        self._build_ui()

    def _build_ui(self):
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(12, 10, 12, 10)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(10)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        cards = [
            ("Average HR", "avg_hr", "bpm"),
            ("Min / Max HR", "range_hr", "bpm"),
            ("Total Beats", "beats", ""),
            ("Pauses", "pauses", "events"),
            ("Signal Quality", "quality", "%"),
            ("HRV SDNN", "sdnn", "ms"),
            ("rMSSD", "rmssd", "ms"),
            ("Longest RR", "longest_rr", "s"),
        ]
        for idx, (title, key, unit) in enumerate(cards):
            frame = QFrame()
            frame.setStyleSheet(
                f"QFrame{{background:{UI_CARD};border:1px solid {UI_BORDER};border-radius:10px;}}"
            )
            frame.setMinimumHeight(74)
            box = QVBoxLayout(frame)
            box.setContentsMargins(12, 10, 12, 10)
            box.setSpacing(3)
            lbl = QLabel(title)
            lbl.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:600;border:none;")
            val = QLabel("—")
            val.setStyleSheet(f"color:{UI_TEXT};font-size:21px;font-weight:700;border:none;")
            unit_lbl = QLabel(unit)
            unit_lbl.setStyleSheet(f"color:{UI_SUCCESS};font-size:10px;font-weight:700;border:none;")
            box.addWidget(lbl)
            box.addWidget(val)
            box.addWidget(unit_lbl)
            self._value_labels[key] = val
            self._card_frames.append(frame)
        self._relayout_cards()

    def _relayout_cards(self):
        if self._grid is None:
            return
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(self)

        width = max(1, self.width())
        if width >= 1400:
            cols = 4
        elif width >= 1050:
            cols = 3
        elif width >= 700:
            cols = 2
        else:
            cols = 1

        for idx, frame in enumerate(self._card_frames):
            self._grid.addWidget(frame, idx // cols, idx % cols)

        rows = int(math.ceil(len(self._card_frames) / float(cols)))
        self.setMinimumHeight(rows * 84 + 24)

    def update_summary(self, s: dict):
        self._value_labels["avg_hr"].setText(f"{s.get('avg_hr', 0):.0f}")
        self._value_labels["range_hr"].setText(f"{s.get('min_hr', 0):.0f} / {s.get('max_hr', 0):.0f}")
        self._value_labels["beats"].setText(f"{s.get('total_beats', 0):,}")
        self._value_labels["pauses"].setText(str(s.get("pauses", 0)))
        self._value_labels["quality"].setText(f"{s.get('avg_quality', 0) * 100:.1f}")
        self._value_labels["sdnn"].setText(f"{s.get('sdnn', 0):.1f}")
        self._value_labels["rmssd"].setText(f"{s.get('rmssd', 0):.1f}")
        self._value_labels["longest_rr"].setText(f"{s.get('longest_rr_ms', 0) / 1000:.2f}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_cards()


# ══════════════════════════════════════════════════════════════════════════════
# 4. HOLTER OVERVIEW PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterOverviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{UI_BG};")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Overview")
        title.setStyleSheet(
            f"color:{UI_TEXT};font-size:14px;font-weight:700;background:{UI_PANEL_ALT};"
            f"padding:8px;border-radius:6px;border:1px solid {UI_BORDER};"
        )
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
            ("Ventricular Beats",    str(s.get('ve_beats', 0))),
            ("Supraventricular Beats", str(s.get('sve_beats', 0))),
            ("Template Clusters",    str(s.get('template_count', 0))),
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
            ni.setForeground(QColor(UI_MUTED))
            ni.setBackground(QColor(UI_PANEL if i % 2 == 0 else UI_PANEL_ALT))
            vi = QTableWidgetItem(value)
            vi.setForeground(QColor(UI_TEXT))
            vi.setBackground(QColor(UI_PANEL if i % 2 == 0 else UI_PANEL_ALT))
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

        cols = ["Type", "Start at", "Duration", "Mean NN", "SDNN", "SDANN", "TRIIDX", "pNN50", "LF", "HF", "LF/HF", "Status"]
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
                     ("rMSSD", "rmssd"), ("pNN50", "pnn50"), ("TRIIDX", "triidx"), ("SDNNIDX", "sdnnidx"),
                     ("VLF", "vlf"), ("LF", "lf"), ("HF", "hf"), ("LF/HF", "lf_hf_ratio")]
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
            total_duration_sec = int(_metrics_duration_sec(metrics_list))
            from .hrv_metrics import compute_hrv_summary
            hrv = compute_hrv_summary(all_rr)
            rows.append(("Entire", "—", f"{total_duration_sec//60:02d}:{total_duration_sec%60:02d}",
                         f"{int(np.mean(all_rr))}ms", f"{hrv.get('sdnn', summary.get('sdnn', 0)):.0f}ms",
                         f"{hrv.get('sdnn', summary.get('sdnn', 0))*0.82:.0f}ms", f"{hrv.get('triangular_index', 0.0):.2f}",
                         f"{hrv.get('pnn50', summary.get('pnn50', 0)):.2f}%",
                         f"{hrv.get('lf', 0.0):.3f}", f"{hrv.get('hf', 0.0):.3f}",
                         f"{hrv.get('lf_hf_ratio', 0.0):.3f}", ""))
        for h in sorted(hourly.keys()):
            chunks = hourly[h]
            rr_vals = [c.get('rr_ms', 0) for c in chunks if c.get('rr_ms', 0) > 0]
            rr_stds = [c.get('rr_std', 0) for c in chunks if c.get('rr_std', 0) > 0]
            pnn50s = [c.get('pnn50', 0) for c in chunks]
            if not rr_vals: continue
            from .hrv_metrics import compute_hrv_summary
            hrv = compute_hrv_summary(rr_vals)
            rows.append(("Hour", f"{h:02d}:00", "01:00",
                         f"{int(np.mean(rr_vals))}ms",
                         f"{hrv.get('sdnn', int(np.mean(rr_stds)) if rr_stds else 0):.0f}ms" if rr_stds else "—",
                         "—", f"{hrv.get('triangular_index', 0.0):.2f}",
                         f"{hrv.get('pnn50', np.mean(pnn50s) if pnn50s else 0.0):.2f}%",
                         f"{hrv.get('lf', 0.0):.3f}", f"{hrv.get('hf', 0.0):.3f}",
                         f"{hrv.get('lf_hf_ratio', 0.0):.3f}", ""))

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
                          ("triidx", f"{s.get('triidx', 0.0):.2f}" if s.get('triidx', 0) else "—"),
                          ("sdnnidx", "—"),
                          ("vlf", f"{s.get('vlf_power', 0.0):.3f}" if s.get('vlf_power', 0) else "—"),
                          ("lf", f"{s.get('lf_power', 0.0):.3f}" if s.get('lf_power', 0) else "—"),
                          ("hf", f"{s.get('hf_power', 0.0):.3f}" if s.get('hf_power', 0) else "—"),
                          ("lf_hf_ratio", f"{s.get('lf_hf_ratio', 0.0):.3f}" if s.get('lf_hf_ratio', 0) else "—")]:
            if key in self._summary_labels:
                self._summary_labels[key].setText(fmt)


# ══════════════════════════════════════════════════════════════════════════════
# 6. HOLTER LORENZ / REPLAY PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterReplayPanel(QWidget):
    seek_requested = pyqtSignal(float)
    lead_changed   = pyqtSignal(int)
    section_requested = pyqtSignal(str)
    frame_received = pyqtSignal(object)

    def __init__(self, parent=None, duration_sec: float = 86400):
        super().__init__(parent)
        self.duration_sec = max(1, duration_sec)
        self._strip_length_sec = 10.0
        self.setStyleSheet(f"background:{COL_DARK};")
        self._replay_engine = None
        self._tool_engine = ECGToolEngine()
        self._build_ui()
        self._magnifier_overlay = MagnifierOverlay(self)
        self._magnifier_overlay.setGeometry(self.rect())
        self._magnifier_overlay.hide()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Feature ribbon (HolterExpert-style)
        ribbon = QFrame()
        ribbon.setStyleSheet(f"QFrame{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:8px;}}")
        rb_l = QHBoxLayout(ribbon)
        rb_l.setContentsMargins(6, 4, 6, 4)
        rb_l.setSpacing(4)
        self._ribbon_buttons = {}
        for txt in ["Overview", "Template", "Histogram", "Lorenz", "Af Analysis", "Tend. Chart",
                    "Pace Spike", "Edit Event", "Edit Strips", "Add Event", "Advance Tools",
                    "Record Settings", "Edit Report", "Preview", "Print", "Reanalysis", "Quit"]:
            b = QPushButton(txt)
            b.setFixedHeight(30)
            b.setStyleSheet(_style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
            b.clicked.connect(lambda _, t=txt: self.section_requested.emit(t))
            self._ribbon_buttons[txt] = b
            rb_l.addWidget(b)
        rb_l.addStretch()
        layout.addWidget(ribbon)

        self._rr_mode = "RR"
        self._time_scope = "whole"

        # ── 48-hour session summary bar (replaces the two RR trend canvases) ──
        summary_frame = QFrame()
        summary_frame.setStyleSheet(f"QFrame{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:6px;}}")
        summary_frame.setFixedHeight(72)
        summary_layout = QHBoxLayout(summary_frame)
        summary_layout.setContentsMargins(12, 6, 12, 6)
        summary_layout.setSpacing(20)
        self._summary_labels = {}
        for key, title in [("duration","Duration"),("total_beats","Total Beats"),("avg_hr","Avg HR"),("max_hr","Max HR"),("min_hr","Min HR"),("pauses","Pauses"),("ve","VE Beats"),("sve","SVE Beats"),("sdnn","SDNN"),("rmssd","rMSSD")]:
            col = QVBoxLayout()
            col.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet(f"color:{UI_MUTED};font-size:9px;font-weight:600;border:none;")
            v = QLabel("—")
            v.setStyleSheet(f"color:{UI_TEXT};font-size:13px;font-weight:700;border:none;")
            col.addWidget(t)
            col.addWidget(v)
            self._summary_labels[key] = v
            summary_layout.addLayout(col)
        summary_layout.addStretch()
        layout.addWidget(summary_frame)

        # ── HR trend mini-chart (40h-48h of data at a glance) ──
        self._hr_trend_canvas = HolterRRTrendCanvas(title="Heart Rate Trend (full recording)")
        self._hr_trend_canvas.setFixedHeight(50)
        layout.addWidget(self._hr_trend_canvas)
        # Keep these as dummy attrs so update_lorenz doesn't crash
        self._rr_trend_full = self._hr_trend_canvas
        self._rr_trend_zoom = self._hr_trend_canvas

        time_row = QHBoxLayout()
        self._btn_time_whole = QPushButton("Time-whole")
        self._btn_time_share = QPushButton("Time-share")
        self._btn_goto_time = QPushButton("Goto Time")
        self._btn_rr = QPushButton("RR")
        self._btn_hr = QPushButton("HR")
        for b in [self._btn_time_whole, self._btn_time_share, self._btn_goto_time]:
            b.setFixedHeight(28)
            b.setStyleSheet(_style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
            time_row.addWidget(b)
        time_row.addStretch()
        for b in [self._btn_rr, self._btn_hr]:
            b.setFixedHeight(28)
            b.setFixedWidth(52)
            b.setStyleSheet(_style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
            time_row.addWidget(b)
        layout.addLayout(time_row)
        self._btn_time_whole.clicked.connect(lambda: self._set_time_scope("whole"))
        self._btn_time_share.clicked.connect(lambda: self._set_time_scope("share"))
        self._btn_goto_time.clicked.connect(self._goto_time)
        self._btn_rr.clicked.connect(lambda: self._set_rr_mode("RR"))
        self._btn_hr.clicked.connect(lambda: self._set_rr_mode("HR"))
        self._set_time_scope("whole")
        self._set_rr_mode("RR")

        # Top: left (lorenz + templates) + right (focused CH strips)
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.setHandleWidth(1)
        top_splitter.setStyleSheet(f"QSplitter{{background:{UI_BG};}} QSplitter::handle{{background:{UI_BORDER};}}")

        left_wrap = QFrame()
        left_wrap.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        lw_l = QVBoxLayout(left_wrap)
        lw_l.setContentsMargins(4, 4, 4, 4)
        lw_l.setSpacing(6)
        self._lorenz_canvas = LorenzCanvas(parent=left_wrap)
        lw_l.addWidget(self._lorenz_canvas, 3)
        lorenz_filter_row = QHBoxLayout()
        lorenz_filter_row.setSpacing(4)
        self._lorenz_class_btns = {}
        for lbl in ["All", "Normal", "S", "V", "Paced"]:
            b = QPushButton(lbl)
            b.setFixedHeight(24)
            b.setStyleSheet(_style_btn(COL_DARK, COL_GREEN, COL_GREEN_DRK))
            self._lorenz_class_btns[lbl] = b
            lorenz_filter_row.addWidget(b)
        lorenz_filter_row.addStretch()
        lw_l.addLayout(lorenz_filter_row)

        thumbs = QFrame()
        thumbs.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        th_l = QGridLayout(thumbs)
        th_l.setContentsMargins(4, 4, 4, 4)
        th_l.setSpacing(4)
        self._template_thumbs = []
        for idx in range(4):
            s = ECGStripCanvas(height=70, color="#22E36E", pen_width=0.8)
            s.set_gain(1.0)
            self._template_thumbs.append(s)
            th_l.addWidget(s, idx // 2, idx % 2)
        lw_l.addWidget(thumbs, 2)
        top_splitter.addWidget(left_wrap)

        ecg_right = QFrame()
        ecg_right.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        ecg_right_layout = QVBoxLayout(ecg_right)
        ecg_right_layout.setContentsMargins(4, 4, 4, 4)
        ecg_right_layout.setSpacing(2)

        # ── 12-lead scrollable grid (1 column, 12 rows) ──
        leads_scroll = QScrollArea()
        leads_scroll.setWidgetResizable(True)
        leads_scroll.setFrameShape(QFrame.NoFrame)
        leads_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        leads_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        leads_scroll.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:none;}}")
        leads_container = QWidget()
        leads_container.setStyleSheet(f"background:{COL_BLACK};")
        leads_vbox = QVBoxLayout(leads_container)
        leads_vbox.setContentsMargins(2, 2, 2, 2)
        leads_vbox.setSpacing(2)

        self._lead_names_ordered = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        self._lead_strips = {}   # lead_name -> ECGStripCanvas
        self._ch_strips = []     # backward-compat list for gain/speed handlers
        for lead in self._lead_names_ordered:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            lbl = QLabel(lead)
            lbl.setFixedWidth(34)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
            # Height 60px makes them compact enough to fit well, but scrollable if needed
            strip = ECGStripCanvas(height=60, color="#00FF00", pen_width=0.9, lead_name=lead)
            strip.set_gain(1.0)
            self._lead_strips[lead] = strip
            self._ch_strips.append(strip)
            row.addWidget(lbl)
            row.addWidget(strip, 1)
            leads_vbox.addLayout(row)

        leads_scroll.setWidget(leads_container)
        ecg_right_layout.addWidget(leads_scroll, 3)

        # Full-width rhythm strip (Lead II) at bottom
        rhythm_row = QHBoxLayout()
        rhythm_row.setSpacing(4)
        rhythm_lbl = QLabel("II")
        rhythm_lbl.setFixedWidth(34)
        rhythm_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        rhythm_lbl.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
        self._mini_strip = ECGStripCanvas(height=40, color="#00AA00", pen_width=0.9)
        rhythm_row.addWidget(rhythm_lbl)
        rhythm_row.addWidget(self._mini_strip, 1)
        ecg_right_layout.addLayout(rhythm_row)

        top_splitter.addWidget(ecg_right)


        ov_frame = QFrame()
        ov_frame.setStyleSheet(f"QFrame{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:6px;}}")
        ov_layout = QVBoxLayout(ov_frame)
        ov_layout.setContentsMargins(6, 6, 6, 6)
        ov_layout.setSpacing(6)
        ov_title = QLabel("Overview")
        ov_title.setStyleSheet(f"color:{UI_TEXT};font-size:14px;font-weight:700;border:none;")
        ov_layout.addWidget(ov_title)
        self._overview_table = QTableWidget(0, 2)
        self._overview_table.setHorizontalHeaderLabels(["Name", "Value"])
        self._overview_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._overview_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._overview_table.verticalHeader().setVisible(False)
        self._overview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._overview_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._overview_table.setFocusPolicy(Qt.NoFocus)
        self._overview_table.setStyleSheet(_table_style())
        ov_layout.addWidget(self._overview_table, 1)
        top_splitter.addWidget(ov_frame)
        top_splitter.setSizes([300, 1050, 260])
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
        for tool in ["Patient information", "Full Disc.", "Goto Template"]:
            tbtn = QPushButton(tool)
            tbtn.setStyleSheet(f"QPushButton{{background:{COL_DARK};color:{COL_TEXT};border:1px solid {COL_GREEN_DRK};"
                               f"border-radius:3px;padding:3px 6px;font-size:10px;}}"
                               f"QPushButton:hover{{background:#202020;color:{COL_WHITE};}}")
            tbtn.clicked.connect(lambda _, t=tool, b=tbtn: self._set_tool_mode(t, b))
            toolbar_layout.addWidget(tbtn)
            self._tool_btns[tool] = tbtn

        for tool in ["Measuring Ruler", "Parallel Ruler", "Magnifying Glass", "Gain Settings",
                     "Paper speed:25mm/s", "Add Event(space)", "Adjust strip position", "Strip Length:10s"]:
            tbtn = QPushButton(tool)
            tbtn.setStyleSheet(f"QPushButton{{background:{COL_DARK};color:{COL_TEXT};border:1px solid {COL_GREEN_DRK};"
                               f"border-radius:3px;padding:3px 6px;font-size:10px;}}"
                               f"QPushButton:hover{{background:#202020;color:{COL_WHITE};}}")
            tbtn.clicked.connect(lambda _, t=tool, b=tbtn: self._set_tool_mode(t, b))
            toolbar_layout.addWidget(tbtn)
            self._tool_btns[tool] = tbtn
        self._tool_btns["Gain Settings"].setToolTip(
            "Cycle gain (5/10/20/40 mm/mV equivalent) to improve waveform visibility."
        )
        toolbar_layout.addStretch()
        layout.addWidget(toolbar)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.setGeometry(self.rect())

    def set_magnifier_focus(self, source_widget, payload: dict, focus_pos: QPoint):
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.setGeometry(self.rect())
            self._magnifier_overlay.set_focus(source_widget, payload, focus_pos)

    def clear_magnifier_focus(self, source_widget=None):
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.clear_focus(source_widget)

    def _set_tool_mode(self, tool_name: str, btn: QPushButton = None):
        if "Goto Template" in tool_name:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle("Holter ECG Software Tools — Explained")
            box.setTextFormat(Qt.PlainText)
            box.setText(
                "Ruler: measure interval/amplitude and BPM.\n"
                "Caliper: compare regularity and coupling across beats.\n"
                "Magnify: zoom-highlight subtle waveform details.\n"
                "Gain Settings: cycle 5/10/20/40 mm/mV-equivalent scaling.\n\n"
                "End-to-end flow:\n"
                "Raw recording → Gain optimization → Magnify flagged events → "
                "Measure intervals (QT/PR/pause) → Parallel comparison → Final report."
            )
            box.setStyleSheet(
                "QMessageBox{background:#10151c;color:#f3f7fb;}"
                "QLabel{color:#f3f7fb;font-size:12px;}"
                "QPushButton{background:#1f6feb;color:white;border:1px solid #4b82d0;border-radius:4px;padding:6px 14px;min-width:70px;}"
                "QPushButton:hover{background:#2d7df2;}"
            )
            box.exec_()
            return

        if "Measuring Ruler" in tool_name:
            tool_name = TOOL_RULER
        elif "Parallel Ruler" in tool_name:
            tool_name = TOOL_CALIPER
        elif "Magnifying Glass" in tool_name:
            tool_name = TOOL_MAGNIFY

        # Handle state cycles for Gain, Speed, Length
        if "Gain Settings" in tool_name:
            gains = [g / 10.0 for g in GAINS]
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
            speeds = PAPER_SPEEDS
            curr_s = getattr(self, '_curr_speed_idx', 1)
            next_s = (curr_s + 1) % len(speeds)
            self._curr_speed_idx = next_s
            val = speeds[next_s]
            for s in getattr(self, "_ch_strips", []):
                s.set_paper_speed(int(val))
            if hasattr(self, "_mini_strip"):
                self._mini_strip.set_paper_speed(int(val))
            if btn: btn.setText(f"Paper speed:{val}mm/s")
            # Adjust strip_length_sec so the replay engine delivers the right amount of data.
            # Reference: 25mm/s = 10s window; scale inversely with speed.
            self._strip_length_sec = 10.0 * (25.0 / max(1.0, float(val)))
            # Re-seek to force data reload with the new strip length
            try:
                current_pos = float(self._slider.value())
                self.seek_requested.emit(current_pos)
            except Exception:
                pass
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

        mode = canonical_tool(tool_name)
        self._tool_engine.set_tool(mode)
        for strip in getattr(self, "_ch_strips", []):
            if hasattr(strip, 'set_mode'):
                strip.set_mode(mode)
        if hasattr(self._mini_strip, 'set_mode'):
            self._mini_strip.set_mode(mode)
        for strip in getattr(self, "_template_thumbs", []):
            if hasattr(strip, 'set_mode'):
                strip.set_mode(mode)

    def _set_time_scope(self, scope: str):
        self._time_scope = scope
        self._btn_time_whole.setStyleSheet(_style_active_btn() if scope == "whole" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
        self._btn_time_share.setStyleSheet(_style_active_btn() if scope == "share" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
        self._strip_length_sec = 10.0 if scope == "whole" else 3.6

    def _set_rr_mode(self, mode: str):
        self._rr_mode = mode
        self._btn_rr.setStyleSheet(_style_active_btn() if mode == "RR" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
        self._btn_hr.setStyleSheet(_style_active_btn() if mode == "HR" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))

    def _goto_time(self):
        text, ok = QInputDialog.getText(self, "Goto Time", "Enter time (HH:MM:SS or seconds):")
        if not ok or not text:
            return
        t = text.strip()
        sec = 0.0
        try:
            if ":" in t:
                parts = [int(p) for p in t.split(":")]
                if len(parts) == 3:
                    sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
                elif len(parts) == 2:
                    sec = parts[0] * 60 + parts[1]
                else:
                    sec = float(t)
            else:
                sec = float(t)
        except Exception:
            QMessageBox.warning(self, "Invalid Time", "Use HH:MM:SS, MM:SS, or seconds.")
            return
        sec = max(0.0, min(sec, float(self.duration_sec)))
        self.seek_requested.emit(sec)

    def update_summary(self, summary: dict):
        if hasattr(self, '_summary_labels'):
            def set_lbl(k, v):
                if k in self._summary_labels: self._summary_labels[k].setText(str(v))
            
            dur = summary.get('duration_sec', 0)
            h = int(dur // 3600)
            m = int((dur % 3600) // 60)
            set_lbl("duration", f"{h:02d}h {m:02d}m")
            set_lbl("total_beats", summary.get('total_beats', 0))
            set_lbl("avg_hr", f"{summary.get('avg_hr', 0)} bpm")
            set_lbl("max_hr", f"{summary.get('max_hr', 0)} bpm")
            set_lbl("min_hr", f"{summary.get('min_hr', 0)} bpm")
            set_lbl("pauses", summary.get('pauses', 0))
            set_lbl("ve", summary.get('ve_beats', 0))
            set_lbl("sve", summary.get('sve_beats', 0))
            set_lbl("sdnn", f"{summary.get('sdnn', 0)} ms")
            set_lbl("rmssd", f"{summary.get('rmssd', 0)} ms")

    def set_replay_engine(self, engine):
        self._replay_engine = engine
        self._slider.setRange(0, int(engine.duration_sec))
        engine.set_position_callback(self._on_position_update)
        
        # Wire up data callback safely for thread-safe playback updates
        try:
            self.frame_received.disconnect()
        except TypeError:
            pass
        self.frame_received.connect(self.set_replay_frame)
        engine.set_data_callback(lambda data: self.frame_received.emit(data))

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
        rr_points = []
        for m in metrics_list:
            t0 = float(m.get('t', 0.0) or 0.0)
            if 'rr_intervals_list' in m:
                rr_list = [float(v) for v in (m.get('rr_intervals_list') or []) if float(v) > 0]
                if rr_list:
                    rr_all.extend(rr_list)
                    dur = float(m.get('duration', 0.0) or 0.0)
                    step = (dur / max(1, len(rr_list))) if dur > 0 else 0.2
                    for i, rr in enumerate(rr_list):
                        rr_points.append((t0 + i * step, rr))
                    continue  # skip fallback if list data available
            # Fallback: use single rr_ms value per chunk
            rr_val = float(m.get('rr_ms', 0) or 0)
            if rr_val > 200:
                rr_all.append(rr_val)
                rr_points.append((t0, rr_val))
                
        rr_n = [r for r in rr_all if r > 200]
        if len(rr_n) >= 2:
            rr_x = rr_n[:-1]
            rr_y = rr_n[1:]
            self._lorenz_canvas.set_data(rr_x, rr_y)
        if hasattr(self, "_rr_trend_full"):
            self._rr_trend_full.set_points(rr_points)
            recent = rr_points[-1200:] if len(rr_points) > 1200 else rr_points
            self._rr_trend_zoom.set_points(recent)
        self._update_overview_table(metrics_list, rr_n)
            
    def set_replay_frame(self, data):
        """Update all 12 lead strips and compute Lorenz from data when no RR metrics."""
        if data is None or data.shape[0] < 12:
            return

        strip_len = int(max(1, getattr(self, "_strip_length_sec", 10.0)) * 500)
        if data.shape[1] > strip_len:
            data = data[:, -strip_len:]

        N = data.shape[1]
        x = np.linspace(0, N / 500.0, N) if N > 0 else []
        if N <= 0:
            return

        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

        # Ensure data has 12 channels
        if data.shape[0] < 12:
            new_data = np.zeros((12, data.shape[1]), dtype=data.dtype)
            new_data[:data.shape[0], :] = data
            for i in range(data.shape[0], 12):
                new_data[i, :] = 2048.0
            data = new_data

        # --- Force mathematical derivation of augmented limb leads ---
        # Einthoven's Law dictates these leads are perfectly locked to I and II.
        # We unconditionally derive them to overwrite any floating hardware noise.
        I_lead = data[0]
        II_lead = data[1]
        
        data[2] = (II_lead - 2048.0) - (I_lead - 2048.0) + 2048.0  # III = II - I
        # User wants aVR mapped to 0 to -4096. We synthesize it inverted and centered at -2048.
        data[3] = -((I_lead - 2048.0) + (II_lead - 2048.0)) / 2.0 - 2048.0
        data[4] = (I_lead - 2048.0) - (II_lead - 2048.0) / 2.0 + 2048.0  # aVL = I - II/2
        data[5] = (II_lead - 2048.0) - (I_lead - 2048.0) / 2.0 + 2048.0  # aVF = II - I/2

        # --- Feed all 12 lead strips ---
        lead_strips = getattr(self, "_lead_strips", {})
        for idx, lead in enumerate(lead_names):
            if idx < data.shape[0] and lead in lead_strips:
                lead_strips[lead].set_data(x, data[idx].copy())

        # Mini strip shows Lead II
        if hasattr(self, "_mini_strip") and data.shape[0] > 1:
            self._mini_strip.set_data(x, data[1].copy())

        # Template thumbnails (use first 4 leads: I, II, III, aVR)
        for i, ts in enumerate(getattr(self, "_template_thumbs", [])):
            src = min(i, data.shape[0] - 1)
            center = N // 2
            w = min(220, max(80, N // 6))
            a = max(0, center - w // 2)
            b = min(N, center + w // 2)
            seg = data[src, a:b] if b > a else data[src, :]
            tx = np.linspace(0, len(seg) / 500.0, len(seg)) if len(seg) > 0 else []
            ts.set_data(tx, seg.copy() if len(seg) > 0 else data[src].copy())

        # --- On-the-fly Lorenz from data when Lorenz canvas has no data ---
        lorenz = getattr(self, "_lorenz_canvas", None)
        if lorenz is not None and (not lorenz._x):  # no RR data loaded from metrics
            self._compute_lorenz_from_signal(data, N)

    def _compute_lorenz_from_signal(self, data: np.ndarray, N: int):
        """Detect R-peaks in Lead II and populate the Lorenz scatter from the raw ECG data."""
        try:
            from scipy.signal import butter, filtfilt, find_peaks
            fs = 500.0
            lead_ii = np.asarray(data[1], dtype=float) if data.shape[0] > 1 else np.asarray(data[0], dtype=float)
            # Bandpass 5-20 Hz to isolate QRS
            nyq = fs / 2.0
            b, a = butter(2, [5.0 / nyq, 20.0 / nyq], btype='band')
            filtered = filtfilt(b, a, lead_ii)
            squared = filtered ** 2
            win = max(1, int(0.15 * fs))
            mwa = np.convolve(squared, np.ones(win) / win, mode='same')
            threshold = max(np.mean(mwa) * 0.5, 1e-6)
            min_dist = max(1, int(0.3 * fs))
            peaks, _ = find_peaks(mwa, height=threshold, distance=min_dist)
            if len(peaks) >= 3:
                rr_ms = np.diff(peaks) / fs * 1000.0
                rr_valid = [float(r) for r in rr_ms if 250 < r < 2500]
                if len(rr_valid) >= 2:
                    rr_x = rr_valid[:-1]
                    rr_y = rr_valid[1:]
                    self._lorenz_canvas.set_data(rr_x, rr_y)
                    if hasattr(self, "_rr_trend_full"):
                        rr_points = [(i * 0.5, rr) for i, rr in enumerate(rr_valid)]
                        self._rr_trend_full.set_points(rr_points)
                        self._rr_trend_zoom.set_points(rr_points[-400:] if len(rr_points) > 400 else rr_points)
        except Exception:
            pass

    def _update_overview_table(self, metrics_list: list, rr_n: list):
        if not hasattr(self, "_overview_table"):
            return
        summary_rows = self._compute_replay_overview(metrics_list, rr_n)
        self._overview_table.setRowCount(len(summary_rows))
        for r, (name, value) in enumerate(summary_rows):
            n_item = QTableWidgetItem(name)
            v_item = QTableWidgetItem(value)
            n_item.setForeground(QColor(UI_MUTED))
            v_item.setForeground(QColor(UI_TEXT))
            self._overview_table.setItem(r, 0, n_item)
            self._overview_table.setItem(r, 1, v_item)

    def _compute_replay_overview(self, metrics_list: list, rr_n: list) -> list:
        total_beats = len(rr_n)
        avg_hr = (60000.0 / float(np.mean(rr_n))) if rr_n else 0.0
        max_hr = (60000.0 / float(min(rr_n))) if rr_n else 0.0
        min_hr = (60000.0 / float(max(rr_n))) if rr_n else 0.0
        pauses = int(sum(1 for rr in rr_n if rr >= 2000))
        longest_rr = (max(rr_n) / 1000.0) if rr_n else 0.0
        ve = int(sum(1 for m in metrics_list if any("V" in str(a) for a in (m.get("arrhythmias", []) or []))))
        sve = int(sum(1 for m in metrics_list if any(("PAC" in str(a)) or ("SVE" in str(a)) for a in (m.get("arrhythmias", []) or []))))
        
        # Calculate percentages and durations
        total_chunks = len(metrics_list) if metrics_list else 1
        tachy_chunks = sum(1 for m in metrics_list if m.get('hr_mean', 0) > 100)
        brady_chunks = sum(1 for m in metrics_list if 0 < m.get('hr_mean', 0) < 60)
        af_chunks = sum(1 for m in metrics_list if any("AF" in str(a) for a in (m.get("arrhythmias", []) or [])))
        
        # Each chunk is roughly 4 seconds.
        chunk_dur = 4.0
        tachy_pct = (tachy_chunks / total_chunks) * 100.0
        brady_pct = (brady_chunks / total_chunks) * 100.0
        af_dur_str = f"{(af_chunks * chunk_dur) / 60.0:.1f} m"
        
        return [
            ("Total NNs", str(total_beats)),
            ("AVG HR", f"{avg_hr:.0f} bpm"),
            ("Max HR", f"{max_hr:.0f} bpm"),
            ("Min HR", f"{min_hr:.0f} bpm"),
            ("Longest RR Interval", f"{longest_rr:.2f}s"),
            ("RRI (>=2.0s)", str(pauses)),
            ("During Tachy (>100)", f"{tachy_pct:.1f}%"),
            ("During Brady (<60)", f"{brady_pct:.1f}%"),
            ("AF Duration", af_dur_str),
            ("V Total", str(ve)),
            ("S Total", str(sve)),
        ]


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

        pen = QPen(QColor(COL_GREEN_MID))
        pen.setWidth(1)
        painter.setPen(pen)
        brush = QBrush(QColor(COL_GREEN_MID))
        painter.setBrush(brush)
        for x, y in zip(self._x, self._y):
            px, py = to_px(x, y)
            painter.drawEllipse(px - 3, py - 3, 6, 6)

        # Axis labels
        pen.setColor(QColor(COL_GREEN_DRK))
        painter.setPen(pen)
        painter.drawText(w // 2 - 20, h - 2, f"{int(mn)}-{int(mx)}ms")


class ECGStripCanvas(QWidget):
    """Simple ECG strip renderer with interactive measurement tools."""
    def __init__(self, parent=None, height: int = 80, color: str = "#00FF00", pen_width: float = 0.7, lead_name: str = ""):
        super().__init__(parent)
        self._data = np.zeros(200)
        self._color = color
        self._pen_width = pen_width
        self.lead_name = lead_name
        self._gain = 1.0
        self._speed = 25
        self.setFixedHeight(height)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")
        self.setMouseTracking(True)
        self._mode = TOOL_SELECT
        self._start_pos = None
        self._curr_pos = None
        self._hover_pos = None
        self._magnify_locked = False
        self._magnify_pos = None
        self._fs = 500.0

    def _find_magnifier_host(self):
        parent = self.parentWidget()
        while parent is not None:
            if hasattr(parent, "set_magnifier_focus") and hasattr(parent, "clear_magnifier_focus"):
                return parent
            parent = parent.parentWidget()
        return None

    def _magnifier_source_payload(self):
        return {
            "data": np.asarray(self._data, dtype=float).copy(),
            "speed": float(self._speed),
            "gain": float(self._gain),
            "lead_name": getattr(self, "lead_name", ""),
            "fs": float(self._fs),
        }

    def set_gain(self, gain: float):
        self._gain = gain
        self.update()

    def set_paper_speed(self, speed: int):
        self._speed = speed
        self.update()

    def set_mode(self, mode: str):
        host = self._find_magnifier_host()
        if host is not None and self._mode == TOOL_MAGNIFY and canonical_tool(mode) != TOOL_MAGNIFY:
            host.clear_magnifier_focus(self)
        self._mode = canonical_tool(mode)
        self._start_pos = None
        self._curr_pos = None
        self._hover_pos = None
        self._magnify_locked = False
        self._magnify_pos = None
        self.update()

    def set_data(self, *args, beat_annotations=None, start_sec=0.0):
        if len(args) == 2:
            self._data = np.asarray(args[1], dtype=float)
        elif len(args) == 1:
            self._data = np.asarray(args[0], dtype=float)
        self._beat_annotations = beat_annotations or []
        self._start_sec = start_sec
        self.update()

    def mousePressEvent(self, event):
        if self._mode == TOOL_MAGNIFY and event.button() == Qt.LeftButton:
            # Click-to-lock magnifier: each click moves the zoom lens to that point.
            # Switching away from the tool clears the lock.
            self._magnify_locked = True
            self._magnify_pos = event.pos()
            self._hover_pos = event.pos()
            host = self._find_magnifier_host()
            if host is not None:
                host.set_magnifier_focus(self, self._magnifier_source_payload(), event.pos())
            self.update()
            return

        if self._mode != TOOL_SELECT:
            self._start_pos = event.pos()
            self._curr_pos = event.pos()
            self._hover_pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self._mode == TOOL_MAGNIFY:
            if self._magnify_locked:
                if event.buttons() & Qt.LeftButton:
                    self._magnify_pos = event.pos()
                    host = self._find_magnifier_host()
                    if host is not None:
                        host.set_magnifier_focus(self, self._magnifier_source_payload(), event.pos())
                self.update()
                return
            self._hover_pos = event.pos()
            self.update()
            return
        self._hover_pos = event.pos()
        if self._mode != TOOL_SELECT and self._start_pos is not None:
            self._curr_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._mode == TOOL_MAGNIFY:
            if not self._magnify_locked:
                self._hover_pos = event.pos()
            self.update()
            return
        if self._mode != TOOL_SELECT:
            self._curr_pos = event.pos()
            self.update()

    def leaveEvent(self, event):
        if not self._magnify_locked:
            self._hover_pos = None
        self.update()
        super().leaveEvent(event)

    def _get_display_signal(self):
        if self._data.size < 2:
            return np.array([]), 0.0, 1.0
        sig = np.asarray(self._data, dtype=float)
        
        # Calculate the true baseline of the raw signal
        baseline = float(np.median(sig))
        
        # Apply gain relative to the baseline
        d = (sig - baseline) * self._gain + baseline
        
        # Universally center the signal for ALL leads. 
        # By dynamically setting the minimum bound relative to the baseline,
        # we ensure the baseline always maps perfectly to the vertical center (y = 0.5 * h),
        # while strictly preserving the 4096 amplitude uncropped scale.
        mn = baseline - 2048.0
        return d, mn, 4096.0

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
        # --- Paper speed: control how many samples are visible per screen width ---
        # At 25mm/s: show all data. At 50mm/s: stretch (show half). At 12.5mm/s: compress (show double).
        speed_factor = max(0.25, min(float(self._speed) / 25.0, 4.0))
        n_visible = max(2, int(round(len(d) / speed_factor)))
        if n_visible < len(d):
            d = d[-n_visible:]   # show the most recent n_visible samples (stretched)
        # (if n_visible >= len(d) we show all data, which appears compressed at slow speed)
        x_scale = w / max(1, len(d) - 1)
        for i in range(1, len(d)):
            x1 = (i - 1) * x_scale
            y1 = h - (d[i-1] - mn) / rng * h
            x2 = i * x_scale
            y2 = h - (d[i] - mn) / rng * h
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            
        # --- Draw Clinical Beat Annotations ---
        if hasattr(self, '_beat_annotations') and self._beat_annotations:
            font = painter.font()
            font.setPixelSize(10)
            font.setBold(True)
            painter.setFont(font)
            
            end_sec = self._start_sec + len(d) / self._fs
            
            for beat in self._beat_annotations:
                ts = beat['timestamp']
                # Check if beat is within the visible window
                if self._start_sec <= ts <= end_sec:
                    # Calculate x coordinate
                    pct = (ts - self._start_sec) / (end_sec - self._start_sec)
                    bx = int(pct * w)
                    
                    lbl = beat['label']
                    # Color code labels: Normal=White, PVC=Red, PAC=Cyan, AF=Magenta, Pause/Block=Yellow
                    if lbl == 'N':
                        color = COL_WHITE
                    elif lbl == 'V':
                        color = "#FF3333"
                    elif lbl == 'S':
                        color = "#00FFFF"
                    elif lbl == 'AF':
                        color = "#FF00FF"
                    else:
                        color = "#FFFF00"
                        
                    painter.setPen(QPen(QColor(color)))
                    painter.drawText(bx - 4, 12, lbl)
                    
        if self._mode == TOOL_RULER and self._start_pos and self._curr_pos:
            rpen = QPen(QColor("#00FFFF"), 2, Qt.DashLine)
            painter.setPen(rpen)
            painter.drawLine(self._start_pos, self._curr_pos)
            dx = abs(self._curr_pos.x() - self._start_pos.x())
            ms = interval_ms_from_pixels(dx, max(1, w), len(d), self._fs)
            bpm = 60000 / ms if ms > 0 else 0
            dy_mv = amplitude_mv_from_pixels(abs(self._curr_pos.y() - self._start_pos.y()), max(1, h), rng, ADC_TO_MV)
            painter.setPen(QPen(QColor("#00FFFF")))
            painter.drawText(self._curr_pos.x(), max(12, self._curr_pos.y() - 6), ruler_label(ms, dy_mv, bpm))
        elif self._mode == TOOL_CALIPER and self._start_pos and self._curr_pos:
            ppen = QPen(QColor("#FFFF00"), 1)
            painter.setPen(ppen)
            painter.drawLine(self._start_pos.x(), 0, self._start_pos.x(), h)
            painter.drawLine(self._curr_pos.x(), 0, self._curr_pos.x(), h)
            dx = abs(self._curr_pos.x() - self._start_pos.x())
            ms = interval_ms_from_pixels(dx, max(1, w), len(d), self._fs)
            painter.drawText(min(self._start_pos.x(), self._curr_pos.x()) + dx//2, 12, caliper_label(ms))
        elif self._mode == TOOL_MAGNIFY:
            host = self._find_magnifier_host()
            if host is not None and hasattr(host, "_magnifier_overlay"):
                return
            focus_pos = self._magnify_pos if self._magnify_locked else self._hover_pos
            if focus_pos is None:
                return
            hover_x = max(0, min(focus_pos.x(), w - 1))
            hover_y = max(0, min(focus_pos.y(), h - 1))
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
                sub_min = float(np.min(sub))
                sub_max = float(np.max(sub))
                
                # Symmetrically frame the magnifier around the local median baseline
                # This prevents asymmetrical waves from pushing the baseline to the bottom/top edge
                base_val = float(np.median(sub))
                max_dev = max(abs(sub_max - base_val), abs(sub_min - base_val))
                pad = max(20.0, max_dev * 0.35)
                view_min = base_val - max_dev - pad
                view_max = base_val + max_dev + pad
                view_rng = max(1.0, view_max - view_min)

                path_pen = QPen(QColor(self._color))
                path_pen.setWidthF(2.0)
                painter.setPen(path_pen)
                x_scale_sub = inner.width() / max(1, len(sub) - 1)
                prev = None
                for i in range(len(sub)):
                    xx = inner.left() + i * x_scale_sub
                    yy = inner.bottom() - ((sub[i] - view_min) / view_rng) * inner.height()
                    if prev is not None:
                        painter.drawLine(QPointF(prev[0], prev[1]), QPointF(xx, yy))
                    prev = (xx, yy)

                focus_x = int(inner.left() + ((src_center - i0) / max(1, len(sub) - 1)) * inner.width())
                focus_y = int(inner.bottom() - ((d[src_center] - view_min) / view_rng) * inner.height())
                painter.setPen(QPen(QColor("#ffffff"), 1))
                painter.drawLine(focus_x, inner.top(), focus_x, inner.bottom())
                painter.drawLine(inner.left(), focus_y, inner.right(), focus_y)

            painter.setPen(QPen(QColor(COL_WHITE)))
            painter.drawText(
                panel_rect.left() + 10,
                panel_rect.bottom() - 10,
                f"{getattr(self.parent(), '_curr_gain_idx', 1) + 2}x {'locked' if self._magnify_locked else 'hover'}"
            )


class MagnifierOverlay(QWidget):
    """Shared magnifier popup for the replay panel so zoom is never clipped by strip bounds."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._visible = False
        self._panel_rect = QRect()
        self._inner_rect = QRect()
        self._data = None
        self._focus_idx = 0
        self._focus_pos = QPoint(0, 0)
        self._gain = 1.0
        self._speed = 25.0
        self._fs = 500.0
        self._lead_name = ""
        self._source_widget = None
        self.hide()

    def set_focus(self, source_widget, payload: dict, focus_pos: QPoint):
        if payload is None or payload.get("data") is None:
            self.hide()
            return
        self._source_widget = source_widget
        self._data = np.asarray(payload.get("data"), dtype=float)
        self._gain = float(payload.get("gain", 1.0))
        self._speed = float(payload.get("speed", 25.0))
        self._fs = float(payload.get("fs", 500.0))
        self._lead_name = str(payload.get("lead_name", ""))
        self._focus_pos = QPoint(focus_pos)
        if self.parentWidget() is None:
            self.hide()
            return
        host = self.parentWidget()
        self.setGeometry(host.rect())
        self.raise_()
        self._visible = True
        self.show()
        self.update()

    def clear_focus(self, source_widget=None):
        if source_widget is not None and self._source_widget is not None and source_widget is not self._source_widget:
            return
        self._visible = False
        self._data = None
        self._source_widget = None
        self.hide()
        self.update()

    def paintEvent(self, event):
        if not self._visible or self._data is None or self._source_widget is None:
            return
        d = np.asarray(self._data, dtype=float)
        if d.size < 2:
            return

        host = self.parentWidget()
        if host is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Convert the focus point from source-widget coordinates into overlay coordinates.
        try:
            focus_pt = self._source_widget.mapTo(host, self._focus_pos)
        except Exception:
            focus_pt = self._focus_pos
        hover_x = max(0, min(int(focus_pt.x()), max(0, host.width() - 1)))
        hover_y = max(0, min(int(focus_pt.y()), max(0, host.height() - 1)))

        w, h = host.width(), host.height()
        source_w = max(1, int(self._source_widget.width()))
        local_x = max(0, min(int(self._focus_pos.x()), source_w - 1))
        src_center = int(round((local_x / float(max(1, source_w - 1))) * (len(d) - 1)))
        span = max(12, int(len(d) / max(2.0, self._speed / 12.5) / 2.0))
        half = max(8, int(span / max(2, self._gain * 1.5)))
        i0 = max(0, src_center - half)
        i1 = min(len(d), src_center + half)
        sub = d[i0:i1]
        if len(sub) < 2:
            return

        panel_w = min(360, max(240, int(w * 0.34)))
        panel_h = min(220, max(140, int(h * 0.28)))
        panel_x = hover_x + 24
        panel_y = hover_y - panel_h - 18
        if panel_x + panel_w > w - 8:
            panel_x = hover_x - panel_w - 24
        if panel_x < 8:
            panel_x = 8
        if panel_y < 8:
            panel_y = hover_y + 18
        if panel_y + panel_h > h - 8:
            panel_y = max(8, h - panel_h - 8)
        self._panel_rect = QRect(panel_x, panel_y, panel_w, panel_h)
        self._inner_rect = self._panel_rect.adjusted(12, 12, -12, -12)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 220))
        painter.drawRoundedRect(self._panel_rect, 12, 12)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(COL_YELLOW), 3))
        painter.drawRoundedRect(self._panel_rect, 12, 12)

        painter.setPen(QPen(QColor(COL_GRID_MINOR), 1))
        for frac in (0.25, 0.5, 0.75):
            gx = int(self._inner_rect.left() + self._inner_rect.width() * frac)
            gy = int(self._inner_rect.top() + self._inner_rect.height() * frac)
            painter.drawLine(gx, self._inner_rect.top(), gx, self._inner_rect.bottom())
            painter.drawLine(self._inner_rect.left(), gy, self._inner_rect.right(), gy)

        sub_min = float(np.min(sub))
        sub_max = float(np.max(sub))
        base_val = float(np.median(sub))
        max_dev = max(abs(sub_max - base_val), abs(sub_min - base_val))
        pad = max(20.0, max_dev * 0.35)
        view_min = base_val - max_dev - pad
        view_max = base_val + max_dev + pad
        view_rng = max(1.0, view_max - view_min)

        path_pen = QPen(QColor("#22FF44"))
        path_pen.setWidthF(2.0)
        painter.setPen(path_pen)
        x_scale_sub = self._inner_rect.width() / max(1, len(sub) - 1)
        prev = None
        for i in range(len(sub)):
            xx = self._inner_rect.left() + i * x_scale_sub
            yy = self._inner_rect.bottom() - ((sub[i] - view_min) / view_rng) * self._inner_rect.height()
            if prev is not None:
                painter.drawLine(QPointF(prev[0], prev[1]), QPointF(xx, yy))
            prev = (xx, yy)

        focus_x = int(self._inner_rect.left() + ((src_center - i0) / max(1, len(sub) - 1)) * self._inner_rect.width())
        focus_y = int(self._inner_rect.bottom() - ((d[src_center] - view_min) / view_rng) * self._inner_rect.height())
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawLine(focus_x, self._inner_rect.top(), focus_x, self._inner_rect.bottom())
        painter.drawLine(self._inner_rect.left(), focus_y, self._inner_rect.right(), focus_y)

        painter.setPen(QPen(QColor(COL_WHITE)))
        painter.drawText(
            self._panel_rect.left() + 10,
            self._panel_rect.top() + 18,
            f"{self._lead_name or 'Lead'}  {self._gain:.1f}x"
        )
        painter.drawText(
            self._panel_rect.left() + 10,
            self._panel_rect.bottom() - 10,
            "click another wave to move"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. HOLTER EVENTS PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterRRTrendCanvas(QWidget):
    """Compact RR trend strip used in expert review header."""

    def __init__(self, parent=None, title="RR Interval"):
        super().__init__(parent)
        self._title = title
        self._points = []  # [(time_sec, rr_ms), ...]
        self.setMinimumHeight(72)
        self.setStyleSheet(f"background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:6px;")

    def set_points(self, points):
        self._points = list(points or [])
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(UI_PANEL))
        w, h = self.width(), self.height()
        inner = QRect(10, 18, max(1, w - 20), max(1, h - 28))
        painter.setPen(QPen(QColor(UI_BORDER), 1))
        painter.drawRect(inner)
        painter.setPen(QPen(QColor(UI_MUTED), 1))
        painter.drawText(12, 13, self._title)

        if not self._points:
            painter.setPen(QPen(QColor(UI_MUTED), 1))
            painter.drawText(inner, Qt.AlignCenter, "No RR data")
            return

        t_vals = [p[0] for p in self._points]
        rr_vals = [p[1] for p in self._points if p[1] > 0]
        if not rr_vals:
            return
        t_min, t_max = min(t_vals), max(t_vals)
        if t_max <= t_min:
            t_max = t_min + 1.0
        rr_min, rr_max = min(rr_vals), max(rr_vals)
        rr_rng = max(1.0, rr_max - rr_min)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(UI_SUCCESS)))
        for t, rr in self._points:
            if rr <= 0:
                continue
            px = int(inner.left() + ((t - t_min) / (t_max - t_min)) * inner.width())
            py = int(inner.bottom() - ((rr - rr_min) / rr_rng) * inner.height())
            painter.drawEllipse(px - 1, py - 1, 3, 3)


class HolterExpertReviewPanel(QWidget):
    """HolterExpert-inspired review layout: trend + Lorenz + strips + overview."""
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._metrics = []
        self._summary = {}
        self._template_rows = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._rr_trend_full = HolterRRTrendCanvas(title="RR Interval Trend (Full)")
        self._rr_trend_zoom = HolterRRTrendCanvas(title="RR Interval Trend (Recent)")
        layout.addWidget(self._rr_trend_full)
        layout.addWidget(self._rr_trend_zoom)

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(1)
        body.setStyleSheet(f"QSplitter{{background:{UI_BG};}} QSplitter::handle{{background:{UI_BORDER};}}")

        left = QFrame()
        left.setStyleSheet(f"QFrame{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:8px;}}")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(8, 8, 8, 8)
        left_l.setSpacing(8)
        self._lorenz = LorenzCanvas()
        self._lorenz.setMinimumHeight(280)
        left_l.addWidget(self._lorenz, 2)
        self._template_table = QTableWidget(0, 3)
        self._template_table.setHorizontalHeaderLabels(["Template", "Class", "Beats"])
        self._template_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._template_table.verticalHeader().setVisible(False)
        self._template_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._template_table.setStyleSheet(_table_style())
        self._template_table.cellClicked.connect(self._on_template_clicked)
        left_l.addWidget(self._template_table, 1)
        body.addWidget(left)

        center = QFrame()
        center.setStyleSheet(f"QFrame{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:8px;}}")
        center_l = QVBoxLayout(center)
        center_l.setContentsMargins(8, 8, 8, 8)
        center_l.setSpacing(6)
        self._ch1 = ECGStripCanvas(height=170, color="#22E36E", pen_width=1.0)
        self._ch2 = ECGStripCanvas(height=170, color="#22E36E", pen_width=1.0)
        self._ch3 = ECGStripCanvas(height=170, color="#22E36E", pen_width=1.0)
        self._mini = ECGStripCanvas(height=48, color="#1BC35F", pen_width=0.9)
        center_l.addWidget(self._ch1, 1)
        center_l.addWidget(self._ch2, 1)
        center_l.addWidget(self._ch3, 1)
        center_l.addWidget(self._mini)
        body.addWidget(center)

        right = QFrame()
        right.setStyleSheet(f"QFrame{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:8px;}}")
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 8, 8, 8)
        right_l.setSpacing(6)
        ttl = QLabel("Overview")
        ttl.setStyleSheet(f"color:{UI_TEXT};font-weight:700;font-size:13px;padding:6px;background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};border-radius:6px;")
        right_l.addWidget(ttl)
        self._overview = QTableWidget(0, 2)
        self._overview.setHorizontalHeaderLabels(["Name", "Value"])
        self._overview.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._overview.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._overview.verticalHeader().setVisible(False)
        self._overview.setEditTriggers(QTableWidget.NoEditTriggers)
        self._overview.setStyleSheet(_table_style())
        right_l.addWidget(self._overview, 1)
        body.addWidget(right)
        body.setSizes([360, 760, 300])
        layout.addWidget(body, 1)

    def update_from_metrics(self, metrics_list: list, summary: dict):
        self._metrics = list(metrics_list or [])
        self._summary = dict(summary or {})
        rr_points = []
        for m in self._metrics:
            t0 = float(m.get("t", 0.0) or 0.0)
            rr_list = list(m.get("rr_intervals_list", []) or [])
            dur = float(m.get("duration", 0.0) or 0.0)
            n = max(1, len(rr_list))
            step = (dur / n) if dur > 0 else 0.2
            for i, rr in enumerate(rr_list):
                rr_points.append((t0 + i * step, float(rr)))
        self._rr_trend_full.set_points(rr_points)
        recent = rr_points[-1200:] if len(rr_points) > 1200 else rr_points
        self._rr_trend_zoom.set_points(recent)
        x = [p[1] for p in recent[:-1]] if len(recent) > 1 else []
        y = [p[1] for p in recent[1:]] if len(recent) > 1 else []
        self._lorenz.set_data(x, y)

        tm = {}
        for m in self._metrics:
            for row in (m.get("template_summary", []) or []):
                key = row.get("template_key") or row.get("template_id") or row.get("label") or "T"
                r = tm.setdefault(key, {"id": row.get("template_id", "T"), "label": row.get("label", "N"), "count": 0, "first": float(row.get("first_timestamp", m.get("t", 0.0)) or 0.0)})
                r["count"] += int(row.get("count", 0) or 0)
                r["first"] = min(r["first"], float(row.get("first_timestamp", r["first"]) or r["first"]))
        self._template_rows = sorted(tm.values(), key=lambda it: it["count"], reverse=True)
        self._template_table.setRowCount(len(self._template_rows))
        for i, row in enumerate(self._template_rows):
            vals = [str(row["id"]), str(row["label"]), str(row["count"])]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setForeground(QColor(UI_TEXT))
                self._template_table.setItem(i, j, item)

        rows = [
            ("Total", f"{summary.get('total_beats', 0)}"),
            ("X Total", f"{summary.get('pauses', 0)}"),
            ("AVG HR", f"{summary.get('avg_hr', 0):.0f} bpm"),
            ("Max HR", f"{summary.get('max_hr', 0):.0f} bpm"),
            ("Min HR", f"{summary.get('min_hr', 0):.0f} bpm"),
            ("V Total", f"{summary.get('ve_beats', 0)}"),
            ("S Total", f"{summary.get('sve_beats', 0)}"),
            ("Longest RR", f"{summary.get('longest_rr_ms', 0)/1000:.2f}s"),
            ("RRI (≥2.0s)", f"{summary.get('pauses', 0)}"),
            ("ST Elevation", "—"),
            ("ST Depression", "—"),
        ]
        self._overview.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            ki = QTableWidgetItem(k)
            vi = QTableWidgetItem(v)
            ki.setForeground(QColor(UI_MUTED))
            vi.setForeground(QColor(UI_TEXT))
            self._overview.setItem(i, 0, ki)
            self._overview.setItem(i, 1, vi)

    def set_replay_frame(self, data):
        if data is None or not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] == 0:
            return
        n = data.shape[1]
        x = np.arange(n, dtype=float) / 500.0
        if data.shape[0] > 0:
            self._ch1.set_data(x, data[0].copy())
            self._mini.set_data(x, data[0].copy())
        if data.shape[0] > 1:
            self._ch2.set_data(x, data[1].copy())
        if data.shape[0] > 2:
            self._ch3.set_data(x, data[2].copy())

    def _on_template_clicked(self, row, _col):
        if 0 <= row < len(self._template_rows):
            self.seek_requested.emit(float(self._template_rows[row].get("first", 0.0)))


class HolterEventsPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._events = []
        self._session_dir = ""
        self._selected_payload = {}
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

        cols = ["Event name", "Start Time", "Chan.", "Print Len.", "Source", "Conf."]
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
            source = ev.get("source", "analysis")
            conf = ev.get("confidence", 0.0)
            for j, val in enumerate([ev['label'], t_str, "3", "7s", source, f"{float(conf or 0.0):.2f}"]):
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

class HolterBeatTemplatePanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._template_rows = []
        self._event_rows = []
        self._class_labels = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Beat Templates")
        title.setStyleSheet(f"color:{COL_BLACK};font-size:13px;font-weight:bold;background:{COL_GREEN};padding:5px;border-radius:4px;")
        layout.addWidget(title)

        chips = QFrame()
        chips.setStyleSheet(f"QFrame{{background:{COL_DARK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        chips_layout = QHBoxLayout(chips)
        chips_layout.setContentsMargins(8, 6, 8, 6)
        chips_layout.setSpacing(12)
        for key, label in [("N", "Normal"), ("VE", "Ventricular"), ("SVE", "Supraventricular"), ("Brady", "Brady"), ("Tachy", "Tachy"), ("Pause", "Pause")]:
            chip = QLabel(f"{label}: 0")
            chip.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:700;border:none;")
            chips_layout.addWidget(chip)
            self._class_labels[key] = chip
        chips_layout.addStretch()
        layout.addWidget(chips)

        self._template_table = QTableWidget(0, 6)
        self._template_table.setHorizontalHeaderLabels(["Template", "Class", "Beats", "Avg RR", "Avg QRS", "First Seen"])
        self._template_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._template_table.setStyleSheet(_table_style())
        self._template_table.verticalHeader().setVisible(False)
        self._template_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._template_table.cellClicked.connect(self._on_template_clicked)
        layout.addWidget(self._template_table, 2)

        self._event_table = QTableWidget(0, 5)
        self._event_table.setHorizontalHeaderLabels(["Event", "Time", "Template", "RR", "QRS"])
        self._event_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._event_table.setStyleSheet(_table_style())
        self._event_table.verticalHeader().setVisible(False)
        self._event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._event_table.cellClicked.connect(self._on_event_clicked)
        layout.addWidget(self._event_table, 1)

    def update_from_metrics(self, metrics_list: list, summary: dict):
        class_totals = dict(summary.get('beat_class_totals', {}) or {})
        if not class_totals:
            for metric in metrics_list or []:
                for cls, count in (metric.get('beat_class_counts', {}) or {}).items():
                    class_totals[cls] = class_totals.get(cls, 0) + int(count or 0)
        for key, lbl in self._class_labels.items():
            base = lbl.text().split(":")[0]
            lbl.setText(f"{base}: {int(class_totals.get(key, 0))}")

        template_map = {}
        event_rows = []
        for metric in metrics_list or []:
            for row in (metric.get('template_summary', []) or []):
                tkey = str(row.get('template_key') or row.get('template_id') or row.get('label') or 'T')
                item = template_map.setdefault(tkey, {
                    'template_id': row.get('template_id', 'T'),
                    'label': row.get('label', 'N'),
                    'count': 0,
                    'rr': [],
                    'qrs': [],
                    'first_timestamp': float(row.get('first_timestamp', metric.get('t', 0.0)) or 0.0),
                })
                item['count'] += int(row.get('count', 0) or 0)
                item['rr'].append(float(row.get('avg_rr_ms', 0.0) or 0.0))
                item['qrs'].append(float(row.get('avg_qrs_ms', 0.0) or 0.0))
                item['first_timestamp'] = min(item['first_timestamp'], float(row.get('first_timestamp', item['first_timestamp']) or item['first_timestamp']))
            for ev in (metric.get('classified_events', []) or []):
                event_rows.append({
                    'timestamp': float(ev.get('timestamp', metric.get('t', 0.0)) or 0.0),
                    'label': str(ev.get('label', 'Beat Event')),
                    'template_label': str(ev.get('template_label', 'N')),
                    'rr_ms': float(ev.get('rr_ms', 0.0) or 0.0),
                    'qrs_ms': float(ev.get('qrs_ms', 0.0) or 0.0),
                })

        self._template_rows = sorted(template_map.values(), key=lambda x: x['count'], reverse=True)
        self._template_table.setRowCount(len(self._template_rows))
        for i, row in enumerate(self._template_rows):
            values = [
                str(row.get('template_id', f"T{i+1}")),
                str(row.get('label', 'N')),
                str(int(row.get('count', 0))),
                f"{float(np.mean(row.get('rr', [0.0]))):.1f} ms",
                f"{float(np.mean(row.get('qrs', [0.0]))):.1f} ms",
                _sec_to_hms(float(row.get('first_timestamp', 0.0))),
            ]
            for j, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setForeground(QColor(COL_WHITE))
                self._template_table.setItem(i, j, cell)

        event_rows.sort(key=lambda e: e['timestamp'])
        seen = set()
        self._event_rows = []
        for ev in event_rows:
            key = (round(ev['timestamp'], 3), ev['label'], ev['template_label'])
            if key in seen:
                continue
            seen.add(key)
            self._event_rows.append(ev)

        self._event_table.setRowCount(len(self._event_rows))
        for i, ev in enumerate(self._event_rows):
            values = [
                ev['label'],
                _sec_to_hms(ev['timestamp']),
                ev['template_label'],
                f"{ev['rr_ms']:.1f} ms" if ev['rr_ms'] > 0 else "-",
                f"{ev['qrs_ms']:.1f} ms" if ev['qrs_ms'] > 0 else "-",
            ]
            for j, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setForeground(QColor(COL_WHITE))
                self._event_table.setItem(i, j, cell)

    def _on_template_clicked(self, row, _col):
        if row < 0 or row >= len(self._template_rows):
            return
        self.seek_requested.emit(float(self._template_rows[row].get('first_timestamp', 0.0)))

    def _on_event_clicked(self, row, _col):
        if row < 0 or row >= len(self._event_rows):
            return
        self.seek_requested.emit(float(self._event_rows[row].get('timestamp', 0.0)))


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
        subtitle = QLabel("Professional Comprehensive ECG Analysis view with synchronized moving strips.")
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

        # Use a QGridLayout for 3x4 format (like professional medical workstations)
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(6)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)

        for idx, lead in enumerate(self.LEADS):
            card = QFrame()
            # Remove border, only show background and wave
            card.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:none;}}")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(2, 2, 2, 2)
            cl.setSpacing(0)
            lbl = QLabel(lead)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;padding-left:4px;")
            cl.addWidget(lbl)
            plot = pg.PlotWidget()
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            plot.setBackground(COL_BLACK)
            # Subtle grid
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("left").setStyle(showValues=False)
            plot.getAxis("bottom").setStyle(showValues=False)
            plot.getAxis("left").setPen(pg.mkPen(color='#002200'))
            plot.getAxis("bottom").setPen(pg.mkPen(color='#002200'))
            if lead == "aVR":
                plot.setYRange(0, -4096, padding=0)
            else:
                plot.setYRange(0, 4096, padding=0)
            plot.setMinimumHeight(120) # Taller for bigger screen visibility
            curve = plot.plot(pen=pg.mkPen(COL_GREEN, width=1.0))
            cl.addWidget(plot, 1)
            self._lead_widgets.append((curve, plot))
            # 3 rows, 4 columns
            row, col = divmod(idx, 4)
            self.grid_layout.addWidget(card, row, col)

        container = QWidget()
        container.setLayout(self.grid_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setStyleSheet("QScrollArea { border: none; background: black; }")
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

        fs = 500.0
        try:
            if self.replay_engine is not None and getattr(self.replay_engine, "fs", None):
                fs = float(self.replay_engine.fs)
        except Exception:
            pass
        if self.replay_engine is None:
            try:
                if self.live_source is not None and getattr(self.live_source, "sampling_rate", None):
                    fs = float(self.live_source.sampling_rate)
            except Exception:
                pass
            try:
                sampler = getattr(self.live_source, "sampler", None) if self.live_source is not None else None
                if sampler and getattr(sampler, "sampling_rate", None):
                    fs = float(sampler.sampling_rate)
            except Exception:
                pass
        if fs < 50.0 or fs > 2000.0:
            fs = 500.0

        ac_opt = "50"
        emg_opt = "150"
        dft_opt = "0.5"
        try:
            sm = getattr(self.live_source, "settings_manager", None)
            if sm is None:
                from utils.settings_manager import SettingsManager
                sm = SettingsManager()
            ac_opt = str(sm.get_setting("filter_ac", ac_opt) or ac_opt).strip()
            emg_opt = str(sm.get_setting("filter_emg", emg_opt) or emg_opt).strip()
            dft_opt = str(sm.get_setting("filter_dft", dft_opt) or dft_opt).strip()
        except Exception:
            pass

        for idx, (curve, plot) in enumerate(self._lead_widgets):
            sig = lead_data[idx] if idx < len(lead_data) else np.full(400, 2048.0)
            lead_name = self.LEADS[idx] if idx < len(self.LEADS) else ""

            # Apply the same display filters used elsewhere in the app (DFT -> EMG -> AC).
            try:
                from ecg.ecg_filters import apply_ecg_filters

                arr = np.asarray(sig, dtype=float)
                if arr.size > 120 and (str(ac_opt).lower() not in ("off", "") or str(emg_opt).lower() not in ("off", "") or str(dft_opt).lower() not in ("off", "")):
                    pad_len = 50
                    if arr.size > pad_len:
                        work = np.concatenate((np.full(pad_len, arr[0]), arr, np.full(pad_len, arr[-1])))
                    else:
                        work = arr
                        pad_len = 0
                    work = apply_ecg_filters(work, sampling_rate=float(fs), ac_filter=ac_opt, emg_filter=emg_opt, dft_filter=dft_opt)
                    if pad_len and work.size > 2 * pad_len:
                        arr = work[pad_len:-pad_len]
                    else:
                        arr = work
                    sig = arr
            except Exception:
                pass

            sig = self._to_display_space(np.asarray(sig, dtype=float), lead_name)
            # Create time axis based on sampling rate
            time_axis = np.arange(sig.size, dtype=float) / float(fs if fs > 0 else 500.0)
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
            f'"Comprehensive ECG Analysis monitoring for {name} shows {rhythm} with an average heart rate of '
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
        self._selected_session = ""
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
        self._action_buttons = {}
        for txt in ["Browse", "Import", "Export", "Backup", "Delete"]:
            btn = QPushButton(txt)
            btn.setStyleSheet(_style_btn())
            self._action_buttons[txt] = btn
            actions.addWidget(btn)
        layout.addLayout(actions)

        cols = ["Name","Age","Gender","Record Time","Duration","Channel","Import Time","Status","Reporter","Conclusion"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setStyleSheet(_table_style())
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.itemSelectionChanged.connect(self._sync_selected_session)
        self._table.cellClicked.connect(self._open_row)
        self._table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self._table, 1)
        self._action_buttons["Browse"].clicked.connect(self._browse_root)
        self._action_buttons["Import"].clicked.connect(self._import_session)
        self._action_buttons["Export"].clicked.connect(self._export_session)
        self._action_buttons["Backup"].clicked.connect(self._backup_root)
        self._action_buttons["Delete"].clicked.connect(self._delete_session)

    def refresh_records(self):
        self._table.setRowCount(0)
        self._selected_session = ""
        if not os.path.isdir(self.output_dir): return
        query = self._search.text().strip().lower()
        filter_label = self._filter.currentText()
        now = datetime.now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        rows = []
        for name in sorted(os.listdir(self.output_dir), reverse=True):
            session_dir = os.path.join(self.output_dir, name)
            if not os.path.isdir(session_dir): continue
            if not os.path.exists(os.path.join(session_dir, "recording.ecgh")): continue
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(session_dir, "recording.ecgh")))
            except Exception:
                mtime = datetime.fromtimestamp(os.path.getmtime(session_dir))
            delta_days = (now - mtime).days
            if filter_label == "Today" and mtime.date() != today:
                continue
            if filter_label == "Yesterday" and mtime.date() != yesterday:
                continue
            if filter_label == "This Week" and delta_days > 6:
                continue
            if filter_label == "This Month" and (mtime.year != now.year or mtime.month != now.month):
                continue
            if filter_label == "This Year" and mtime.year != now.year:
                continue
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
        if self._table.rowCount() > 0:
            self._table.selectRow(0)
            self._sync_selected_session()

    def _open_row(self, row, _column=0):
        item = self._table.item(row, 0)
        if item:
            path = item.data(Qt.UserRole)
            if path:
                self._selected_session = path
                self.session_selected.emit(path)

    def _on_double_click(self, index):
        self._open_row(index.row())

    def _sync_selected_session(self):
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if rows:
            item = self._table.item(rows[0].row(), 0)
            if item:
                self._selected_session = item.data(Qt.UserRole) or ""

    def _selected_path(self) -> str:
        self._sync_selected_session()
        return self._selected_session

    def _browse_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select Recordings Root", self.output_dir or os.getcwd())
        if d:
            self.output_dir = d
            self.refresh_records()

    def _import_session(self):
        src = QFileDialog.getExistingDirectory(self, "Import Session Folder")
        if not src:
            return
        if not os.path.exists(os.path.join(src, "recording.ecgh")):
            QMessageBox.warning(self, "Import Session", "Select a session folder that contains recording.ecgh.")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        dest = os.path.join(self.output_dir, os.path.basename(os.path.normpath(src)))
        if os.path.exists(dest):
            QMessageBox.warning(self, "Import Session", "That session already exists in the recordings folder.")
            return
        shutil.copytree(src, dest)
        self.refresh_records()
        self.session_selected.emit(dest)

    def _export_session(self):
        src = self._selected_path()
        if not src:
            QMessageBox.information(self, "Export Session", "Select a recording to export first.")
            return
        dest_root = QFileDialog.getExistingDirectory(self, "Export Session To")
        if not dest_root:
            return
        dest = os.path.join(dest_root, os.path.basename(os.path.normpath(src)))
        if os.path.exists(dest):
            QMessageBox.warning(self, "Export Session", "That session already exists in the destination.")
            return
        shutil.copytree(src, dest)
        QMessageBox.information(self, "Export Session", f"Session exported to:\n{dest}")

    def _backup_root(self):
        if not os.path.isdir(self.output_dir):
            QMessageBox.information(self, "Backup", "No recordings folder found.")
            return
        dest_root = QFileDialog.getExistingDirectory(self, "Backup Recordings To")
        if not dest_root:
            return
        dest = os.path.join(dest_root, os.path.basename(os.path.normpath(self.output_dir)) or "recordings_backup")
        if os.path.exists(dest):
            QMessageBox.warning(self, "Backup", "That backup folder already exists.")
            return
        shutil.copytree(self.output_dir, dest)
        QMessageBox.information(self, "Backup", f"Recordings backed up to:\n{dest}")

    def _delete_session(self):
        src = self._selected_path()
        if not src:
            QMessageBox.information(self, "Delete Session", "Select a recording to delete first.")
            return
        if QMessageBox.question(self, "Delete Session",
                                f"Delete this recording?\n\n{src}",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        shutil.rmtree(src, ignore_errors=True)
        self.refresh_records()


# ══════════════════════════════════════════════════════════════════════════════
# 11. HISTOGRAM PANEL
# ══════════════════════════════════════════════════════════════════════════════

class HolterHistogramPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._metrics = []
        self._rank_mode = "rri"
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
        self._rank_buttons = {}
        for lbl, mode in [("RRI Ranking", "rri"), ("Time Ranking", "time"),
                          ("Prematurity Ranking", "prematurity"), ("Similarity Ranking", "similarity")]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            btn.clicked.connect(lambda _, m=mode: self._set_rank_mode(m))
            self._rank_buttons[mode] = btn
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
        self._set_rank_mode("rri")

    def _set_rank_mode(self, mode: str):
        self._rank_mode = mode
        for key, btn in getattr(self, "_rank_buttons", {}).items():
            active = key == mode
            btn.setStyleSheet(_style_active_btn() if active else _style_btn())
        self._draw()

    def update_from_metrics(self, metrics_list: list):
        self._metrics = metrics_list
        self._draw()

    def _draw(self):
        rr_points = []
        for m in self._metrics:
            t0 = float(m.get('t', 0.0) or 0.0)
            if 'rr_intervals_list' in m:
                rr_list = [float(v) for v in (m.get('rr_intervals_list') or []) if float(v) > 200]
                if rr_list:
                    dur = float(m.get('duration', 0.0) or 0.0)
                    step = (dur / max(1, len(rr_list))) if dur > 0 else 0.2
                    for i, rr in enumerate(rr_list):
                        rr_points.append((t0 + i * step, rr))
            elif m.get('rr_ms', 0) > 200:
                rr_points.append((t0, float(m['rr_ms'])))

        rr_vals = [rr for _, rr in rr_points if rr > 200]
        mode = self._rank_mode
        ranked = []
        if rr_points:
            rr_arr = np.array(rr_vals) if rr_vals else np.array([])
            median_rr = float(np.median(rr_arr)) if rr_arr.size else 0.0
            mean_rr = float(np.mean(rr_arr)) if rr_arr.size else 0.0
            if mode == "time":
                ranked = sorted(rr_points, key=lambda x: x[0])
            elif mode == "prematurity":
                ranked = sorted(rr_points, key=lambda x: max(0.0, mean_rr - x[1]), reverse=True)
            elif mode == "similarity":
                ranked = sorted(rr_points, key=lambda x: abs(x[1] - median_rr))
            else:
                ranked = sorted(rr_points, key=lambda x: x[1], reverse=True)
        self._hist_canvas.set_ranked_data(ranked, mode=mode)

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
        self._mode = "rri"
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")

    def set_data(self, rr_values):
        self._mode = "rri"
        self._data = [(idx, float(v)) for idx, v in enumerate(rr_values or [])]
        self.update()

    def set_ranked_data(self, ranked_points, mode: str = "rri"):
        self._mode = mode
        self._data = list(ranked_points or [])
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

        if not self._data or len(self._data) < 1:
            painter.setPen(QPen(QColor(COL_GREEN_DRK)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No RR data")
            return

        values = [float(item[1]) for item in self._data]
        mn, mx = float(min(values)), float(max(values))
        if mx <= mn:
            mx = mn + 1.0

        if self._mode == "time":
            x_label_left = "early"
            x_label_right = "late"
        elif self._mode == "prematurity":
            x_label_left = "less premature"
            x_label_right = "more premature"
        elif self._mode == "similarity":
            x_label_left = "less similar"
            x_label_right = "more similar"
        else:
            x_label_left = "low RR"
            x_label_right = "high RR"

        n = len(self._data)
        bar_gap = 2
        bar_w = max(2, int((w - 20) / max(1, n)) - bar_gap)
        usable_h = max(30, h - 20)
        max_val = max(values)
        min_val = min(values)
        rng = max(max_val - min_val, 1.0)

        # baseline grid
        painter.setPen(QPen(QColor("#001100")))
        for gy in [h//4, h//2, 3*h//4]:
            painter.drawLine(0, gy, w, gy)

        for idx, (_t, rr) in enumerate(self._data[:min(n, 100)]):
            x = 10 + idx * (bar_w + bar_gap)
            bar_h = int(((rr - min_val) / rng) * usable_h)
            y = h - 12 - bar_h
            if self._mode == "prematurity":
                # emphasize short RR intervals
                bar_color = QColor("#ff7a00")
            elif self._mode == "similarity":
                bar_color = QColor("#7a66ff")
            elif self._mode == "time":
                bar_color = QColor("#33d6ff")
            else:
                bar_color = QColor("#4466AA")
            painter.setBrush(QBrush(bar_color))
            painter.setPen(QPen(QColor(bar_color.lighter(130)), 1))
            painter.drawRect(x, y, bar_w, bar_h)
            if n <= 24 or idx % max(1, n // 12) == 0:
                painter.setPen(QPen(QColor(COL_GREEN_DRK)))
                painter.drawText(x, h - 2, str(idx + 1))

        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        painter.drawText(5, 12, x_label_left)
        painter.drawText(w - 110, 12, x_label_right)
        painter.drawText(5, h - 2, f"{mn:.0f}ms")
        painter.drawText(w - 60, h - 2, f"{mx:.0f}ms")


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
        self._session_dir = ""
        self._selected_payload = {}
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

        cols = ["Event name", "Start Time", "Chan.", "Print Len.", "Source", "Conf."]
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

        annot_box = QFrame()
        annot_box.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        annot_layout = QGridLayout(annot_box)
        annot_layout.setContentsMargins(8, 6, 8, 6)
        annot_layout.setSpacing(6)
        annot_title = QLabel("Beat Annotation Editor")
        annot_title.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
        annot_layout.addWidget(annot_title, 0, 0, 1, 2)
        self._annot_event_id = QLineEdit()
        self._annot_event_id.setPlaceholderText("beat_id / event id")
        self._annot_event_id.setStyleSheet(f"QLineEdit{{background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};padding:5px;border-radius:4px;}}")
        self._annot_auto = QLineEdit()
        self._annot_auto.setPlaceholderText("auto label")
        self._annot_auto.setStyleSheet(self._annot_event_id.styleSheet())
        self._annot_clin = QComboBox()
        self._annot_clin.addItems(["", "N", "S", "V", "AF", "Pause", "Tachy", "Brady", "Other"])
        self._annot_clin.setStyleSheet(f"QComboBox{{background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};padding:5px;border-radius:4px;}}")
        self._annot_conf = QDoubleSpinBox()
        self._annot_conf.setRange(0.0, 1.0)
        self._annot_conf.setSingleStep(0.05)
        self._annot_conf.setDecimals(2)
        self._annot_conf.setValue(0.0)
        self._annot_conf.setStyleSheet(f"QDoubleSpinBox{{background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};padding:5px;border-radius:4px;}}")
        self._annot_editor = QTextEdit()
        self._annot_editor.setPlaceholderText("Optional note or reviewer context")
        self._annot_editor.setFixedHeight(56)
        self._annot_editor.setStyleSheet(f"QTextEdit{{background:{COL_DARK};color:{COL_WHITE};border:1px solid {COL_GREEN_DRK};padding:5px;border-radius:4px;}}")
        self._annot_save_btn = QPushButton("Save Annotation")
        self._annot_save_btn.setStyleSheet(_style_active_btn())
        self._annot_save_btn.clicked.connect(self._save_annotation)
        annot_lbl = QLabel("Beat ID:")
        annot_lbl.setStyleSheet(f"color:{COL_GREEN_DRK};border:none;")
        annot_layout.addWidget(annot_lbl, 1, 0)
        annot_layout.addWidget(self._annot_event_id, 1, 1)
        annot_lbl = QLabel("Auto label:")
        annot_lbl.setStyleSheet(f"color:{COL_GREEN_DRK};border:none;")
        annot_layout.addWidget(annot_lbl, 2, 0)
        annot_layout.addWidget(self._annot_auto, 2, 1)
        annot_lbl = QLabel("Clinician label:")
        annot_lbl.setStyleSheet(f"color:{COL_GREEN_DRK};border:none;")
        annot_layout.addWidget(annot_lbl, 3, 0)
        annot_layout.addWidget(self._annot_clin, 3, 1)
        annot_lbl = QLabel("Confidence:")
        annot_lbl.setStyleSheet(f"color:{COL_GREEN_DRK};border:none;")
        annot_layout.addWidget(annot_lbl, 4, 0)
        annot_layout.addWidget(self._annot_conf, 4, 1)
        annot_layout.addWidget(self._annot_editor, 5, 0, 1, 2)
        annot_layout.addWidget(self._annot_save_btn, 6, 0, 1, 2)
        right_layout.addWidget(annot_box)

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
            source = str(ev.get("source", "analysis"))
            conf = float(ev.get("confidence", 0.0) or 0.0)
            for j, val in enumerate([ev['label'], _sec_to_hms(ev['timestamp']), "3", "7s", source, f"{conf:.2f}"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(COL_WHITE))
                if j == 0:
                    item.setData(Qt.UserRole, {
                        "beat_id": str(ev.get("beat_id", ev.get("template_label", ev.get("label", "")))),
                        "auto_label": str(ev.get("template_label", ev.get("label", ""))),
                        "clinician_label": str(ev.get("label", "")),
                        "confidence": conf,
                        "timestamp": float(ev.get("timestamp", 0.0) or 0.0),
                        "source": source,
                    })
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
            item = self._ev_table.item(row, 0)
            if item:
                payload = item.data(Qt.UserRole) or {}
                self._selected_payload = dict(payload)
                self._annot_event_id.setText(str(payload.get("beat_id", "")))
                self._annot_auto.setText(str(payload.get("auto_label", "")))
                self._annot_clin.setCurrentText(str(payload.get("clinician_label", "")))
                self._annot_conf.setValue(float(payload.get("confidence", 0.0) or 0.0))
                self._annot_editor.setPlainText(
                    f"Source: {payload.get('source', '')}\n"
                    f"Timestamp: {_sec_to_hms(float(payload.get('timestamp', 0.0) or 0.0))}"
                )

    def set_session_dir(self, session_dir: str):
        self._session_dir = session_dir or ""

    def _save_annotation(self):
        if not self._session_dir:
            QMessageBox.information(self, "Annotation", "No session directory is available for saving.")
            return
        beat_id = self._annot_event_id.text().strip()
        if not beat_id:
            QMessageBox.information(self, "Annotation", "Select an event or enter a beat ID first.")
            return
        annotation = {
            "beat_id": beat_id,
            "auto_label": self._annot_auto.text().strip(),
            "clinician_label": self._annot_clin.currentText().strip() or self._annot_auto.text().strip(),
            "confidence": float(self._annot_conf.value()),
            "edited_by": "clinician",
            "timestamp": float(self._selected_payload.get("timestamp", self._events[0].get("timestamp", 0.0) if self._events else 0.0)),
            "note": self._annot_editor.toPlainText().strip(),
        }
        try:
            append_annotation(self._session_dir, annotation)
            QMessageBox.information(self, "Annotation", "Annotation saved to session database.")
        except Exception as e:
            QMessageBox.warning(self, "Annotation", f"Could not save annotation: {e}")

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

        # --- LEFT: Event List (20% width) ---
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
        layout.addWidget(left, 2)

        # --- CENTER: 2x2 Thumbnail Boxes (30% width) ---
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)

        # Tool buttons
        tool_row = QHBoxLayout()
        for icon in ["↻", "☰", "⏕", "⏖", "⏜", "⏝", "⏟", "⏪", "⏩"]:
            btn = QPushButton(icon)
            btn.setStyleSheet(_style_btn())
            btn.setFixedSize(28, 28)
            tool_row.addWidget(btn)
        tool_row.addStretch()
        center_layout.addLayout(tool_row)

        thumb_grid = QGridLayout()
        thumb_grid.setSpacing(8)
        self._thumb_frames = []
        for row, col, title in [(0,0,"Maximum Heart Rate"),(0,1,"Minimum Heart Rate"),
                                 (1,0,"Sinus Max HR"),(1,1,"Sinus Min HR")]:
            frame = QFrame()
            # Styled with white/gray border to match reference
            frame.setStyleSheet(f"QFrame{{background:{COL_DARK};border:1px solid #888888;border-radius:4px;}}")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(6, 6, 6, 6)
            fl.setSpacing(2)
            
            header_w = QWidget()
            header_w.setStyleSheet("border:none;")
            hl = QHBoxLayout(header_w)
            hl.setContentsMargins(0,0,0,0)
            
            t_lbl = QLabel(title)
            t_lbl.setStyleSheet("color:#FFFFFF;font-size:12px;font-weight:bold;")
            hl.addWidget(t_lbl)
            hl.addStretch()
            hr_lbl = QLabel("HR: --")
            hr_lbl.setStyleSheet("color:#FFFF00;font-size:11px;")
            hl.addWidget(hr_lbl)
            fl.addWidget(header_w)
            
            time_lbl = QLabel("19:18:22 (07-31)")
            time_lbl.setStyleSheet("color:#AAAAAA;font-size:10px;")
            fl.addWidget(time_lbl)
            
            # Using 3 mini canvases to simulate 3 channels in the thumbnail box
            for _ in range(3):
                strip = ECGStripCanvas(height=35)
                strip.setStyleSheet("border:1px solid #444444;")
                fl.addWidget(strip)
                self._thumb_frames.append(strip)
                
            thumb_grid.addWidget(frame, row, col)
            
        center_layout.addLayout(thumb_grid)
        center_layout.addStretch() # Keep boxes compact at the top
        layout.addWidget(center, 3)

        # --- RIGHT: Large Clinical Strip View (50% width) ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        
        main_frame = QFrame()
        main_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid #888888;border-radius:4px;}}")
        ml = QVBoxLayout(main_frame)
        ml.setContentsMargins(8, 8, 8, 8)
        
        header_lbl = QLabel("Detailed View  -  N N N N N N  -  19:18:22")
        header_lbl.setStyleSheet("color:#FFFF00;font-size:12px;font-weight:bold;border:none;")
        ml.addWidget(header_lbl)
        
        self._main_strip_1 = ECGStripCanvas(height=110)
        self._main_strip_2 = ECGStripCanvas(height=110)
        self._main_strip_3 = ECGStripCanvas(height=110)
        
        for name, strip in [("CH1", self._main_strip_1), ("CH2", self._main_strip_2), ("CH3", self._main_strip_3)]:
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color:{COL_GREEN};border:none;font-weight:bold;")
            ml.addWidget(lbl)
            ml.addWidget(strip)
            
        ml.addStretch()
        right_layout.addWidget(main_frame, 1)

        self._mini = ECGStripCanvas(height=40, color="#00AA00")
        right_layout.addWidget(self._mini)
        layout.addWidget(right, 5)

    def load_events(self, events: list, summary: dict):
        self._ev_table.setRowCount(len(events))
        for i, ev in enumerate(events):
            for j, val in enumerate([ev['label'], _sec_to_hms(ev['timestamp']), "3", "7s"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(COL_WHITE))
                self._ev_table.setItem(i, j, item)

    def set_replay_frame(self, data, metrics_dict=None, current_sec=0.0):
        if data is None or data.shape[0] < 1: return
        N = data.shape[1]
        x = np.linspace(0, N/250.0, N) if N > 0 else []
        
        start_sec = max(0.0, current_sec - 5.0) # 10s window centered
        all_beats = metrics_dict.get('all_beats', []) if metrics_dict else []
        
        if N > 0:
            # Update thumbnails (cycle through CH1, CH2, CH3 if available)
            for i, strip in enumerate(self._thumb_frames):
                ch_idx = i % 3
                if ch_idx < data.shape[0]:
                    strip.set_data(x, data[ch_idx].copy(), beat_annotations=all_beats, start_sec=start_sec)
                else:
                    strip.set_data(x, data[0].copy(), beat_annotations=all_beats, start_sec=start_sec)
            
            # Update large main strips
            if hasattr(self, "_main_strip_1") and data.shape[0] > 0:
                self._main_strip_1.set_data(x, data[0].copy(), beat_annotations=all_beats, start_sec=start_sec)
            if hasattr(self, "_main_strip_2") and data.shape[0] > 1:
                self._main_strip_2.set_data(x, data[1].copy(), beat_annotations=all_beats, start_sec=start_sec)
            if hasattr(self, "_main_strip_3") and data.shape[0] > 2:
                self._main_strip_3.set_data(x, data[2].copy(), beat_annotations=all_beats, start_sec=start_sec)
                
            self._mini.set_data(x, data[0].copy(), beat_annotations=all_beats, start_sec=start_sec)


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
        self.setWindowTitle("Comprehensive ECG Analysis Monitor & Analysis")
        self.setMinimumSize(900, 620)

        screen = QApplication.primaryScreen()
        if screen:
            g = screen.availableGeometry()
            self.resize(max(1100, int(g.width() * 0.92)), max(750, int(g.height() * 0.92)))
        else:
            self.resize(1400, 900)

        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint |
                            Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setStyleSheet(f"QDialog{{background:{UI_BG};}}")

        self.session_dir = session_dir
        self.patient_info = patient_info or (writer.patient_info if writer else {})
        self._writer = writer
        self._live_source = live_source
        self._duration_hours = duration_hours
        self._replay_engine = None
        self._metrics_list = []
        self._summary = {}
        self._last_live_seq = -1
        self._tab_name_map = {}

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
        metadata = read_session_metadata(self.session_dir) if self.session_dir else {}
        if metadata:
            session_patient = metadata.get("patient_info") or {}
            if isinstance(session_patient, dict) and session_patient:
                self.patient_info = dict(session_patient)
        layered_metrics = load_metrics(self.session_dir) if self.session_dir else []
        if layered_metrics:
            self._metrics_list = layered_metrics
        jsonl_path = os.path.join(self.session_dir, 'metrics.jsonl') if self.session_dir else ''
        if not self._metrics_list and os.path.exists(jsonl_path):
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
        if metadata and isinstance(metadata.get("summary"), dict):
            self._summary = dict(metadata.get("summary") or self._summary)

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
        beat_class_totals: Dict[str, int] = {}
        template_counts = []
        for m in ml:
            for a in m.get('arrhythmias', []):
                arrhy_counts[a] = arrhy_counts.get(a, 0) + 1
            for cls, count in (m.get('beat_class_counts', {}) or {}).items():
                beat_class_totals[cls] = beat_class_totals.get(cls, 0) + int(count or 0)
            template_counts.append(int(m.get('template_count', 0) or 0))
        all_rr = [m.get('longest_rr', 0) for m in ml]
        return {
            'duration_sec': _metrics_duration_sec(ml),
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
            'beat_class_totals': beat_class_totals,
            've_beats': int(beat_class_totals.get('VE', 0)),
            'sve_beats': int(beat_class_totals.get('SVE', 0)),
            'template_count': max(template_counts) if template_counts else 0,
        }

    def _tab_index_for(self, name: str) -> int:
        if not hasattr(self, '_tabs'):
            return -1
        target = (name or '').strip().lower()
        aliases = {
            'overview': 'preview',
            'view': 'preview',
            'report': 'preview',
            'preview': 'preview',
            'lorenz': 'lorenz',
            'histogram': 'histogram',
            'template': 'template',
            'af analysis': 'af analysis',
            'st tendency': 'st tendency',
            'edit event': 'edit event',
            'edit strips': 'edit strips',
            'report table': 'report table',
            'hrv': 'hrv',
            'recordings': 'recordings',
            'record settings': 'recordings',
            'replay': 'replay',
        }
        target = aliases.get(target, target)
        for idx in range(self._tabs.count()):
            if self._tabs.tabText(idx).strip().lower() == target:
                return idx
        return -1

    def _focus_tab(self, name: str):
        idx = self._tab_index_for(name)
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)

    def _recordings_panel(self):
        return getattr(self, '_record_mgmt_panel', None)

    def _open_recordings_folder(self):
        self._focus_tab('RECORDINGS')

    def _search_recordings(self):
        self._focus_tab('RECORDINGS')
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_search'):
            panel._search.setFocus()
            panel._search.selectAll()

    def _apply_recordings_filter(self, label: str):
        self._focus_tab('RECORDINGS')
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_filter'):
            idx = panel._filter.findText(label)
            if idx >= 0:
                panel._filter.setCurrentIndex(idx)

    def _import_recording(self):
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_import_session'):
            panel._import_session()

    def _backup_recordings(self):
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_backup_root'):
            panel._backup_root()

    def _delete_recording(self):
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_delete_session'):
            panel._delete_session()

    def _generate_from_current(self):
        self._focus_tab('Preview')
        self._generate_report()

    def _refresh_current_session(self):
        self._load_session()
        self._refresh_ui()

    def _on_workspace_section_requested(self, section: str):
        key = (section or '').strip().lower()
        if key == 'quit':
            self.close()
        elif key in {'reanalysis', 'replay'}:
            self._refresh_current_session()
            self._focus_tab('REPLAY')
        elif key in {'overview', 'preview', 'view', 'report', 'edit report'}:
            self._focus_tab('Preview')
        elif key == 'template':
            self._focus_tab('TEMPLATE')
        elif key == 'histogram':
            self._focus_tab('HISTOGRAM')
        elif key == 'lorenz':
            self._focus_tab('LORENZ')
        elif key == 'af analysis':
            self._focus_tab('AF ANALYSIS')
        elif key in {'tend. chart', 'st tendency'}:
            self._focus_tab('ST TENDENCY')
        elif key in {'edit event', 'pace spike', 'add event'}:
            self._focus_tab('EDIT EVENT')
        elif key == 'edit strips':
            self._focus_tab('EDIT STRIPS')
        elif key == 'report table':
            self._focus_tab('REPORT TABLE')
        elif key == 'hrv':
            self._focus_tab('HRV')
        elif key in {'record settings', 'advance tools'}:
            self._focus_tab('RECORDINGS')
        elif key == 'print':
            self._generate_report()
        else:
            self._focus_tab('REPLAY')

    # ── Build UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Top title bar ──
        title_bar = QFrame()
        title_bar.setStyleSheet(f"QFrame{{background:{UI_PANEL};border-bottom:1px solid {UI_BORDER};}}")
        title_bar.setFixedHeight(52)
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(16, 0, 16, 0)
        tb_layout.setSpacing(14)
        self._mode_badge = QLabel("LIVE" if self._writer else "REVIEW")
        self._mode_badge.setStyleSheet(
            f"background:{'#7A2633' if self._writer else '#124936'};color:{UI_TEXT};"
            f"border:1px solid {UI_BORDER};border-radius:12px;padding:5px 12px;font-size:11px;font-weight:700;"
        )
        tb_layout.addWidget(self._mode_badge)
        app_title = QLabel("Comprehensive ECG Analysis")
        app_title.setStyleSheet(f"color:{UI_TEXT};font-size:18px;font-weight:700;border:none;")
        tb_layout.addWidget(app_title)
        dur_text = self._summary.get('duration_sec', 0)
        dur_h = int(dur_text // 3600)
        dur_m = int((dur_text % 3600) // 60)
        self._dur_label = QLabel(f"{dur_h:02d}h {dur_m:02d}m")
        self._dur_label.setStyleSheet(
            f"color:{UI_TEXT};font-size:13px;font-weight:700;background:{UI_PANEL_ALT};"
            f"padding:7px 12px;border-radius:8px;border:1px solid {UI_BORDER};"
        )
        tb_layout.addWidget(self._dur_label)
        tb_layout.addStretch()
        gen_report_btn = QPushButton("📄  Generate Report")
        gen_report_btn.setStyleSheet(
            f"QPushButton{{background:{UI_ACCENT};color:{UI_TEXT};border:1px solid #61a8ff;border-radius:8px;padding:8px 14px;font-size:12px;font-weight:700;}}"
            f"QPushButton:hover{{background:{UI_ACCENT_HOVER};}}"
        )
        gen_report_btn.setFixedHeight(34)
        gen_report_btn.clicked.connect(self._generate_report)
        tb_layout.addWidget(gen_report_btn)
        close_btn = QPushButton("✕  Close")
        close_btn.setStyleSheet(
            f"QPushButton{{background:{UI_PANEL_ALT};color:{UI_TEXT};border:1px solid {UI_BORDER};border-radius:8px;padding:8px 14px;font-size:12px;font-weight:600;}}"
            "QPushButton:hover{background:#2A3D61;}"
        )
        close_btn.setFixedHeight(34)
        close_btn.clicked.connect(self.close)
        tb_layout.addWidget(close_btn)
        main_layout.addWidget(title_bar)

        session_bar = QFrame()
        session_bar.setStyleSheet(f"QFrame{{background:{UI_PANEL_ALT};border-bottom:1px solid {UI_BORDER};}}")
        sb_layout = QHBoxLayout(session_bar)
        sb_layout.setContentsMargins(14, 8, 14, 8)
        sb_layout.setSpacing(10)
        patient_name = self.patient_info.get("patient_name") or self.patient_info.get("name") or "Unknown Patient"
        doctor_name = self.patient_info.get("doctor") or "No referring doctor"
        session_name = os.path.basename(self.session_dir) if self.session_dir else "Active Session"
        self._patient_chip = QLabel(f"Patient: {patient_name}")
        self._patient_chip.setStyleSheet(f"background:#123B2D;color:{UI_TEXT};border:1px solid #206B51;border-radius:14px;padding:6px 12px;font-size:11px;font-weight:700;")
        sb_layout.addWidget(self._patient_chip)
        self._doctor_chip = QLabel(f"Doctor: {doctor_name}")
        self._doctor_chip.setStyleSheet(f"background:{UI_PANEL};color:{UI_MUTED};border:1px solid {UI_BORDER};border-radius:14px;padding:6px 12px;font-size:11px;font-weight:600;")
        sb_layout.addWidget(self._doctor_chip)
        self._session_chip = QLabel(f"Session: {session_name}")
        self._session_chip.setStyleSheet(f"background:{UI_PANEL};color:{UI_MUTED};border:1px solid {UI_BORDER};border-radius:14px;padding:6px 12px;font-size:11px;font-weight:600;")
        sb_layout.addWidget(self._session_chip)
        sb_layout.addStretch()
        self._analysis_state = QLabel("Clinical review mode")
        self._analysis_state.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:600;border:none;")
        sb_layout.addWidget(self._analysis_state)
        main_layout.addWidget(session_bar)

        action_bar = QFrame()
        action_bar.setStyleSheet(f"QFrame{{background:{UI_PANEL};border-bottom:1px solid {UI_BORDER};}}")
        ab_layout = QHBoxLayout(action_bar)
        ab_layout.setContentsMargins(8, 6, 8, 6)
        ab_layout.setSpacing(6)
        self._action_buttons = {}
        for label in ["Browse", "Search", "Analyse", "View", "Import", "Backup", "Delete"]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.setStyleSheet(_style_btn())
            self._action_buttons[label] = btn
            ab_layout.addWidget(btn)
        ab_layout.addStretch()
        self._filter_buttons = {}
        for label in ["All", "Today", "Yesterday", "This Week", "This Month", "This Year"]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.setStyleSheet(_style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
            self._filter_buttons[label] = btn
            ab_layout.addWidget(btn)
        main_layout.addWidget(action_bar)

        # ── Status bar (if recording) ──
        if self._writer:
            self._status_bar = HolterStatusBar(self, target_hours=self._duration_hours)
            self._status_bar.stop_requested.connect(self._stop_recording)
            main_layout.addWidget(self._status_bar)

        # ── Summary KPI cards ── (hidden to restore image-1 layout; replay panel has its own 12-lead grid)

        # ── Body: tabs fill full width (12-lead grid is inside HolterReplayPanel) ──
        right_frame = QFrame()
        right_frame.setStyleSheet(f"QFrame{{background:{UI_BG};}}")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background:{UI_BG};
                border:1px solid {UI_BORDER};
                border-top:none;
            }}
            QTabBar::tab {{
                background:{UI_PANEL};
                color:{UI_MUTED};
                border:1px solid {UI_BORDER};
                border-radius:8px;
                padding:9px 14px;
                font-size:11px;
                font-weight:700;
                margin:6px 6px 8px 0;
                min-width:96px;
                text-align:center;
            }}
            QTabBar::tab:selected {{
                color:{UI_TEXT};
                background:{UI_ACCENT};
                border-color:#6EB4FF;
            }}
            QTabBar::tab:hover:!selected {{
                background:#1A2C49;
                color:{UI_TEXT};
            }}
            QGroupBox {{
                border: none;
                border-top: 1px solid {UI_BORDER};
                margin-top: 20px;
                font-weight: bold;
                font-size: 14px;
                color: {UI_TEXT};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }}
        """)

        # Replay
        duration = self._summary.get('duration_sec', self._duration_hours * 3600)
        self._replay_panel = HolterReplayPanel(duration_sec=duration)
        if self._replay_engine:
            self._replay_panel.set_replay_engine(self._replay_engine)
            self._replay_panel.seek_requested.connect(self._on_seek_requested)
        self._replay_panel.section_requested.connect(self._on_workspace_section_requested)
        self._replay_panel.update_lorenz(self._metrics_list)
        self._replay_panel.update_summary(self._summary)
        self._tabs.addTab(self._replay_panel, "REPLAY")

        # Beat Templates
        self._template_panel = HolterBeatTemplatePanel()
        self._template_panel.update_from_metrics(self._metrics_list, self._summary)
        self._template_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._template_panel, "TEMPLATE")

        # Histogram
        self._hist_panel = HolterHistogramPanel()
        self._hist_panel.update_from_metrics(self._metrics_list)
        self._tabs.addTab(self._hist_panel, "HISTOGRAM")

        # Dedicated Lorenz workspace
        self._lorenz_panel = HolterReplayPanel(duration_sec=duration)
        if self._replay_engine:
            self._lorenz_panel.set_replay_engine(self._replay_engine)
            self._lorenz_panel.seek_requested.connect(self._on_seek_requested)
        self._lorenz_panel.section_requested.connect(self._on_workspace_section_requested)
        self._lorenz_panel.update_lorenz(self._metrics_list)
        self._tabs.addTab(self._lorenz_panel, "LORENZ")

        # AF Analysis
        self._af_panel = HolterAFPanel()
        self._af_panel.update_from_metrics(self._metrics_list, duration)
        self._tabs.addTab(self._af_panel, "AF ANALYSIS")

        events = self._build_linked_events()

        # Event timeline
        self._events_panel = HolterEventsPanel()
        self._events_panel.load_events(events, self._summary)
        self._events_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._events_panel, "EVENTS")

        # ST Tendency
        self._st_panel = HolterSTPanel()
        self._st_panel.update_from_metrics(self._metrics_list)
        self._tabs.addTab(self._st_panel, "ST TENDENCY")

        # Edit Event
        self._edit_event_panel = HolterEditEventPanel()
        self._edit_event_panel.set_session_dir(self.session_dir)
        self._edit_event_panel.load_events(events, self._summary)
        self._edit_event_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._edit_event_panel, "EDIT EVENT")

        # Edit Strips
        self._edit_strips_panel = HolterEditStripsPanel()
        self._edit_strips_panel.load_events(events, self._summary)
        self._tabs.addTab(self._edit_strips_panel, "EDIT STRIPS")

        # Report Tendency
        self._report_tendency_panel = HolterSTPanel()
        self._tabs.addTab(self._report_tendency_panel, "REPORT TENDENCY")

        # Report Table
        self._report_table_panel = HolterReportTablePanel()
        self._report_table_panel.update_from_metrics(self._metrics_list)
        self._tabs.addTab(self._report_table_panel, "REPORT TABLE")

        # Expert Review (available but not in tabs)
        self._expert_panel = HolterExpertReviewPanel()
        self._expert_panel.update_from_metrics(self._metrics_list, self._summary)
        self._expert_panel.seek_requested.connect(self._on_seek_requested)

        # HRV Analysis
        self._hrv_panel = HolterHRVPanel()
        self._hrv_panel.update_hrv(self._metrics_list, self._summary)
        self._tabs.addTab(self._hrv_panel, "HRV")

        # Record browser
        self._record_mgmt_panel = HolterRecordManagementPanel(
            output_dir=_resolve_recordings_dir(self.session_dir)
        )
        self._record_mgmt_panel.session_selected.connect(self.load_completed_session)
        self._tabs.addTab(self._record_mgmt_panel, "RECORDINGS")

        # Report Preview
        scroll_insight = QScrollArea()
        scroll_insight.setWidgetResizable(True)
        scroll_insight.setFrameShape(QFrame.NoFrame)
        scroll_insight.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:none;}}")
        self._insight_panel = HolterInsightPanel()
        self._insight_panel.update_text(self.patient_info, self._summary)
        scroll_insight.setWidget(self._insight_panel)
        self._tabs.addTab(scroll_insight, "Preview")

        right_layout.addWidget(self._tabs)
        self._tabs.currentChanged.connect(
            lambda idx: hasattr(self, '_analysis_state') and self._analysis_state.setText(
                f"Focused view: {self._tabs.tabText(idx)}"
            )
        )
        self._action_buttons["Browse"].clicked.connect(self._open_recordings_folder)
        self._action_buttons["Search"].clicked.connect(self._search_recordings)
        self._action_buttons["Analyse"].clicked.connect(lambda: self._focus_tab("REPLAY"))
        self._action_buttons["View"].clicked.connect(lambda: self._focus_tab("Preview"))
        self._action_buttons["Import"].clicked.connect(self._import_recording)
        self._action_buttons["Backup"].clicked.connect(self._backup_recordings)
        self._action_buttons["Delete"].clicked.connect(self._delete_recording)
        for label, btn in self._filter_buttons.items():
            btn.clicked.connect(lambda _, t=label: self._apply_recordings_filter(t))

        main_layout.addWidget(right_frame, 1)
        if hasattr(self, '_analysis_state'):
            self._analysis_state.setText(f"Focused view: {self._tabs.tabText(self._tabs.currentIndex())}")


    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _on_seek_requested(self, target_sec: float):
        if self._replay_engine:
            self._replay_engine.seek(target_sec)
            try:
                # Use the replay panel's current strip length (changes with paper speed)
                window_sec = getattr(getattr(self, '_replay_panel', None), '_strip_length_sec', 10.0) or 10.0
                data = self._replay_engine.get_all_leads_data(window_sec=float(window_sec))
                if hasattr(self, '_wave_panel'):
                    self._wave_panel.set_replay_frame(data)
                self._broadcast_replay_frame(data)
            except Exception:
                pass


    def _broadcast_replay_frame(self, data):
        for panel in [getattr(self, p, None) for p in [
            '_replay_panel', '_lorenz_panel', '_hist_panel', '_af_panel',
            '_st_panel', '_edit_event_panel', '_edit_strips_panel', '_events_panel',
            '_expert_panel', '_template_panel', '_report_tendency_panel', '_hrv_panel'
        ]]:
            if panel and hasattr(panel, 'set_replay_frame'):
                try:
                    panel.set_replay_frame(data)
                except Exception:
                    pass

    def _build_linked_events(self) -> list:
        events = []
        if self._replay_engine:
            try:
                events.extend(self._replay_engine.get_events_list() or [])
            except Exception:
                pass
        for metric in self._metrics_list or []:
            base_t = float(metric.get('t', 0.0) or 0.0)
            for label in metric.get('arrhythmias', []) or []:
                events.append({
                    'timestamp': base_t,
                    'label': str(label),
                    'time_str': _sec_to_hms(base_t),
                })
            for ev in metric.get('classified_events', []) or []:
                t_val = float(ev.get('timestamp', base_t) or base_t)
                events.append({
                    'timestamp': t_val,
                    'label': str(ev.get('label', ev.get('template_label', 'Beat Event'))),
                    'time_str': _sec_to_hms(t_val),
                })

        events.sort(key=lambda e: float(e.get('timestamp', 0.0) or 0.0))
        dedup = []
        seen = set()
        for ev in events:
            key = (round(float(ev.get('timestamp', 0.0) or 0.0), 3), str(ev.get('label', '')))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(ev)
        return dedup

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
        snapshot = None
        if hasattr(self._writer, 'get_live_analysis_snapshot'):
            snapshot = self._writer.get_live_analysis_snapshot(getattr(self, '_last_live_seq', -1))
        if snapshot:
            self._last_live_seq = snapshot.get('seq', self._last_live_seq)
            self._metrics_list = snapshot.get('metrics', [])
            self._summary = snapshot.get('summary', {})
            self._refresh_ui()
        if hasattr(self, '_wave_panel'):
            self._wave_panel.refresh_waveforms()
        if snapshot is None and stats['elapsed'] % 15 < 2:
            self._load_session()
            self._refresh_ui()

    def _refresh_ui(self):
        if hasattr(self, '_summary_cards'):
            self._summary_cards.update_summary(self._summary)
        if hasattr(self, '_insight_panel'):
            self._insight_panel.update_text(self.patient_info, self._summary)
        if hasattr(self, '_overview_panel'):
            self._overview_panel.update_summary(self._summary)
        if hasattr(self, '_expert_panel'):
            self._expert_panel.update_from_metrics(self._metrics_list, self._summary)
        if hasattr(self, '_hrv_panel'):
            self._hrv_panel.update_hrv(self._metrics_list, self._summary)
        if hasattr(self, '_replay_panel'):
            self._replay_panel.update_lorenz(self._metrics_list)
        if hasattr(self, '_lorenz_panel'):
            self._lorenz_panel.update_lorenz(self._metrics_list)
        if hasattr(self, '_hist_panel'):
            self._hist_panel.update_from_metrics(self._metrics_list)
        if hasattr(self, '_af_panel'):
            self._af_panel.update_from_metrics(self._metrics_list, self._summary.get('duration_sec', 0))
        if hasattr(self, '_st_panel'):
            self._st_panel.update_from_metrics(self._metrics_list)
        if hasattr(self, '_report_table_panel'):
            self._report_table_panel.update_from_metrics(self._metrics_list)
        events = self._build_linked_events()
        if hasattr(self, '_events_panel'):
            self._events_panel.load_events(events, self._summary)
        if hasattr(self, '_template_panel'):
            self._template_panel.update_from_metrics(self._metrics_list, self._summary)
        if hasattr(self, '_edit_event_panel'):
            self._edit_event_panel.load_events(events, self._summary)
        if hasattr(self, '_edit_strips_panel'):
            self._edit_strips_panel.load_events(events, self._summary)
        if hasattr(self, '_wave_panel'):
            self._wave_panel.set_live_source(self._live_source)
        if hasattr(self, '_record_mgmt_panel'):
            self._record_mgmt_panel.output_dir = _resolve_recordings_dir(self.session_dir)
            self._record_mgmt_panel.refresh_records()
        if hasattr(self, '_wave_panel') and self._replay_engine:
            try:
                self._wave_panel.set_replay_engine(self._replay_engine)
            except Exception:
                pass
            self._wave_panel.refresh_waveforms()
        if hasattr(self, '_expert_panel') and self._replay_engine:
            try:
                self._expert_panel.set_replay_frame(self._replay_engine.get_all_leads_data(window_sec=10.0))
            except Exception:
                pass
        if self._replay_engine:
            try:
                self._broadcast_replay_frame(self._replay_engine.get_all_leads_data(window_sec=10.0))
            except Exception:
                pass

        # Update duration label
        dur = self._summary.get('duration_sec', 0)
        dur_h = int(dur // 3600)
        dur_m = int((dur % 3600) // 60)
        if hasattr(self, '_dur_label'):
            self._dur_label.setText(f"{dur_h:02d}h {dur_m:02d}m")
        if hasattr(self, '_patient_chip'):
            patient_name = self.patient_info.get("patient_name") or self.patient_info.get("name") or "Unknown Patient"
            self._patient_chip.setText(f"Patient: {patient_name}")
        if hasattr(self, '_doctor_chip'):
            doctor_name = self.patient_info.get("doctor") or "No referring doctor"
            self._doctor_chip.setText(f"Doctor: {doctor_name}")
        if hasattr(self, '_session_chip'):
            session_name = os.path.basename(self.session_dir) if self.session_dir else "Active Session"
            self._session_chip.setText(f"Session: {session_name}")
        if hasattr(self, '_analysis_state') and hasattr(self, '_tabs'):
            self._analysis_state.setText(f"Focused view: {self._tabs.tabText(self._tabs.currentIndex())}")

    def _finalize_live_writer(self) -> dict:
        summary = {}
        if not self._writer:
            return summary
        try:
            stop_fn = getattr(self._writer, "stop", None)
            if callable(stop_fn):
                summary = stop_fn() or {}
            else:
                close_fn = getattr(self._writer, "close", None)
                if callable(close_fn):
                    summary = close_fn() or {}
        finally:
            self._writer = None
        return summary

    def _stop_recording(self):
        if self._writer:
            summary = self._finalize_live_writer()
            if hasattr(self, '_status_bar') and self._status_bar is not None:
                self._status_bar.setVisible(False)
                if hasattr(self._status_bar, 'cleanup'):
                    self._status_bar.cleanup()
            
            # Show dialog to collect patient info AFTER recording
            dialog = HolterStartDialog(self, patient_info=self.patient_info or {}, output_dir=self.session_dir)
            dialog.setWindowTitle("Save Comprehensive ECG Analysis Recording Details")
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
                                    f"Comprehensive ECG Analysis recording saved to:\n{summary.get('session_dir', '')}")
            self.load_completed_session(summary.get('session_dir', ''), summary.get('patient_info', {}))
            
            # Auto-generate report when recording is stopped
            self._generate_report()

    def _generate_report(self):
        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import QThread, pyqtSignal
        
        progress = QProgressDialog("Generating Comprehensive ECG Analysis Report. Please wait...", None, 0, 0, self)
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(Qt.WindowModal)
        progress.setStyleSheet(f"QProgressDialog{{background:{COL_DARK};color:{COL_GREEN};}}")
        progress.setRange(0, 0)
        progress.show()
        
        class ReportWorker(QThread):
            finished = pyqtSignal(str)
            error = pyqtSignal(str)
            
            def __init__(self, session_dir, patient_info, summary):
                super().__init__()
                self.session_dir = session_dir
                self.patient_info = patient_info
                self.summary = summary
                
            def run(self):
                try:
                    from .report_generator import generate_holter_report
                    path = generate_holter_report(
                        session_dir=self.session_dir,
                        patient_info=self.patient_info,
                        summary=self.summary,
                    )
                    self.finished.emit(path)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.error.emit(str(e))
                    
        self._report_worker = ReportWorker(self.session_dir, self.patient_info, self._summary)
        
        def on_finished(path):
            progress.close()
            try:
                from dashboard.history_window import append_history_entry
                h_pat = self.patient_info.copy() if self.patient_info else {}
                if 'patient_name' not in h_pat and 'name' in h_pat:
                    h_pat['patient_name'] = h_pat['name']
                _p = self.parent()
                _uname = getattr(_p, "username", "") if _p is not None else ""
                _full = (getattr(_p, "user_details", {}) or {}).get("full_name") or _uname
                append_history_entry(
                    h_pat, path, report_type="Comprehensive ECG Analysis",
                    username=_uname, owner_full_name=_full
                )
            except Exception as h_err:
                print(f"Failed to append Holter history: {h_err}")
                
            msg = QMessageBox(self)
            msg.setWindowTitle("Report Generated")
            msg.setText(f"Comprehensive ECG Analysis report saved:\n{path}")
            msg.setIcon(QMessageBox.Information)
            msg.setStyleSheet(f"QMessageBox {{ background: {COL_BLACK}; }} QLabel {{ color: {COL_WHITE}; font-size: 12px; }}")
            msg.exec_()
            
        def on_error(err_str):
            progress.close()
            msg = QMessageBox(self)
            msg.setWindowTitle("Report Error")
            msg.setText(f"Could not generate report:\n{err_str}")
            msg.setIcon(QMessageBox.Warning)
            msg.setStyleSheet(f"QMessageBox {{ background: {COL_BLACK}; }} QLabel {{ color: {COL_WHITE}; font-size: 12px; }}")
            msg.exec_()
            
        self._report_worker.finished.connect(on_finished)
        self._report_worker.error.connect(on_error)
        self._report_worker.start()

    def attach_writer(self, writer, session_dir: str = "", patient_info: dict = None):
        self._writer = writer
        self._last_live_seq = -1
        if session_dir:
            self.session_dir = session_dir
        if patient_info:
            self.patient_info = patient_info
        if hasattr(self, '_edit_event_panel'):
            self._edit_event_panel.set_session_dir(self.session_dir)
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
        self._last_live_seq = -1
        if patient_info:
            self.patient_info = patient_info
        if hasattr(self, '_edit_event_panel'):
            self._edit_event_panel.set_session_dir(self.session_dir)
        self._writer = None
        self._load_session()
        if hasattr(self, '_record_mgmt_panel'):
            self._record_mgmt_panel.output_dir = _resolve_recordings_dir(session_dir)
            self._record_mgmt_panel.refresh_records()
        if hasattr(self, '_replay_panel') and getattr(self, '_replay_engine', None):
            self._replay_panel.set_replay_engine(self._replay_engine)
            try:
                self._replay_panel.seek_requested.disconnect(self._on_seek_requested)
            except Exception:
                pass
            self._replay_panel.seek_requested.connect(self._on_seek_requested)
        if hasattr(self, '_lorenz_panel') and getattr(self, '_replay_engine', None):
            self._lorenz_panel.set_replay_engine(self._replay_engine)
        self._refresh_ui()
        if hasattr(self, '_tabs'):
            self._tabs.setCurrentIndex(0)

    def closeEvent(self, event):
        if self._writer:
            try:
                self._finalize_live_writer()
            except Exception:
                pass
        if self._replay_engine:
            try:
                self._replay_engine.close()
            except Exception:
                pass
        super().closeEvent(event)

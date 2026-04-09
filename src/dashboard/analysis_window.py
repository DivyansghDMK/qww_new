"""
ECG Analysis Window — Professional Clinical Edition
====================================================
Enhanced with:
  • Interactive Tool Modes:  Select | Ruler | Caliper | Magnifier | Annotate
  • Measurement ruler (drag → shows Δt ms, Δ amplitude mV)
  • Dual-caliper (PP/RR interval measurement like a real ECG machine)
  • Crosshair cursor + live readout (time, amplitude)
  • Floating magnifier lens (2–5× zoom, follows cursor)
  • Right-click context-menu annotation on any lead
  • Click any lead → dedicated expanded analysis popup
  • All original JSON-load, API-fetch, frame-nav, PDF-gen features kept intact
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QPoint, QPointF, QRect, QRectF, pyqtSignal
from PyQt5.QtGui import (QFont, QPixmap, QCursor, QPainter, QPen,
                         QColor, QBrush, QRadialGradient, QFontMetrics, QImage)
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QFrame, QMessageBox,
    QSizePolicy, QComboBox, QFileDialog, QTextEdit, QSlider,
    QLineEdit, QAction, QMenu, QApplication, QButtonGroup,
    QToolButton, QWidget, QSplitter
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrow
import matplotlib.patches as mpatches

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from ecg.arrhythmia_detector import ArrhythmiaDetector, get_interpretation
    from ecg.expanded_lead_view import PQRSTAnalyzer
except ImportError:
    try:
        from src.ecg.arrhythmia_detector import ArrhythmiaDetector, get_interpretation
        from src.ecg.expanded_lead_view import PQRSTAnalyzer
    except ImportError:
        try:
            from ..ecg.arrhythmia_detector import ArrhythmiaDetector, get_interpretation
            from ..ecg.expanded_lead_view import PQRSTAnalyzer
        except (ImportError, ValueError):
            ArrhythmiaDetector = None
            get_interpretation = None
            PQRSTAnalyzer = None


try:
    from ecg.holter.theme import ADC_TO_MV, COL_CROSSHAIR, TOOL_ANNOTATE, TOOL_CALIPER, TOOL_MAGNIFY, TOOL_RULER, TOOL_SELECT
    from ecg.holter.tool_engine import caliper_label, cursor as tool_cursor, hint as tool_hint, ruler_label, tool_specs
except ImportError:
    try:
        from src.ecg.holter.theme import ADC_TO_MV, COL_CROSSHAIR, TOOL_ANNOTATE, TOOL_CALIPER, TOOL_MAGNIFY, TOOL_RULER, TOOL_SELECT
        from src.ecg.holter.tool_engine import caliper_label, cursor as tool_cursor, hint as tool_hint, ruler_label, tool_specs
    except ImportError:
        from ..ecg.holter.theme import ADC_TO_MV, COL_CROSSHAIR, TOOL_ANNOTATE, TOOL_CALIPER, TOOL_MAGNIFY, TOOL_RULER, TOOL_SELECT
        from ..ecg.holter.tool_engine import caliper_label, cursor as tool_cursor, hint as tool_hint, ruler_label, tool_specs


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE CANVAS  (one per lead)
# ─────────────────────────────────────────────────────────────────────────────
class InteractiveLeadCanvas(FigureCanvas):
    """
    A matplotlib canvas that supports professional ECG interaction:
      - Crosshair + live readout
      - Ruler measurement (drag)
      - Caliper (two vertical lines)
      - Magnifier lens (overlay widget)
      - Right-click annotation menu
      - Double-click to expand
    """

    # Signals
    ruler_measured    = pyqtSignal(str)          # human-readable measurement string
    caliper_measured  = pyqtSignal(str)
    annotation_req    = pyqtSignal(float, float, str)  # start_sec, end_sec, lead_name
    expand_requested  = pyqtSignal(str)           # lead name
    def __init__(self, fig, ax, lead_name, parent_window, parent=None):
        super().__init__(fig)
        self.ax = ax
        self.lead_name = lead_name
        self.parent_window = parent_window

        # Interaction state
        self._drag_start = None          # (x_data, y_data) in data coordinates
        self._drag_end   = None
        self._caliper_x  = [None, None]  # two vertical positions (data x)
        self._mag_pos    = None          # cursor pos for magnifier (widget coords)
        # Overlay items (matplotlib artists we redraw)
        self._ruler_patch    = None
        self._caliper_lines  = [None, None]
        self._crosshair_v    = None
        self._crosshair_h    = None
        self._readout_text   = None

        # Mouse tracking
        self.setMouseTracking(True)

        # Connect matplotlib events
        self.mpl_connect('motion_notify_event',  self._on_mouse_move)
        self.mpl_connect('button_press_event',   self._on_mouse_press)
        self.mpl_connect('button_release_event', self._on_mouse_release)

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def tool(self):
        return self.parent_window.current_tool

    def _data_coords(self, event):
        """Return (x_sec, y_adc) from a matplotlib mouse event, or None."""
        if event.inaxes != self.ax:
            return None, None
        return event.xdata, event.ydata

    def _widget_to_data(self, qx, qy):
        """Convert Qt widget pixel pos → matplotlib data coordinates."""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return None, None
        # matplotlib uses bottom-left origin
        mpl_x = qx
        mpl_y = h - qy
        try:
            inv = self.ax.transData.inverted()
            xd, yd = inv.transform((mpl_x, mpl_y))
            return xd, yd
        except Exception:
            return None, None

    def _clear_overlay(self, *names):
        for name in names:
            obj = getattr(self, name, None)
            if obj is not None:
                if isinstance(obj, list):
                    # Handle list of artists (like _caliper_lines)
                    for item in obj:
                        if item is not None:
                            try:
                                item.remove()
                            except Exception:
                                pass
                    # Reset the list to its original state [None, None] or []
                    if name == '_caliper_lines':
                        setattr(self, name, [None, None])
                    else:
                        setattr(self, name, [])
                else:
                    # Handle single artist
                    try:
                        obj.remove()
                    except Exception:
                        pass
                    setattr(self, name, None)

    # ── matplotlib event handlers ─────────────────────────────────────────────
    def _on_mouse_move(self, event):
        tool = self.tool
        xd, yd = self._data_coords(event)
        if xd is None:
            self._clear_overlay('_crosshair_v', '_crosshair_h', '_readout_text')
            self.draw_idle()
            return

        if not self.parent_window.lead_has_visible_data(self.lead_name):
            self._clear_overlay('_crosshair_v', '_crosshair_h', '_readout_text')
            self.draw_idle()
            return

        # Always show crosshair in ruler/caliper modes, and in select mode
        if tool in (TOOL_SELECT, TOOL_RULER, TOOL_CALIPER, TOOL_ANNOTATE):
            self._draw_crosshair(xd, yd)

        # Ruler: drag to measure
        if tool == TOOL_RULER and self._drag_start is not None:
            self._draw_ruler(self._drag_start[0], self._drag_start[1], xd, yd)

        # Caliper: drag second line
        if tool == TOOL_CALIPER and self._caliper_x[0] is not None and self._caliper_x[1] is None:
            self._draw_caliper_preview(self._caliper_x[0], xd)

        if tool == TOOL_MAGNIFY:
            self._mag_pos = QPoint(int(event.x), self.height() - int(event.y))
            self.update()

        self.draw_idle()

    def mouseMoveEvent(self, event):
        if self.tool == TOOL_MAGNIFY:
            self._mag_pos = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._mag_pos = None
        self.update()
        super().leaveEvent(event)

    def _on_mouse_press(self, event):
        tool = self.tool
        xd, yd = self._data_coords(event)
        if xd is None:
            return

        if event.button == 1:  # Left click
            if tool == TOOL_SELECT:
                self.expand_requested.emit(self.lead_name)
                return

            if tool == TOOL_RULER:
                self._drag_start = (xd, yd)
                self._drag_end   = None

            elif tool == TOOL_CALIPER:
                if self._caliper_x[0] is None:
                    self._caliper_x[0] = xd
                elif self._caliper_x[1] is None:
                    self._caliper_x[1] = xd
                    self._finalise_caliper()
                else:
                    # Reset
                    self._caliper_x = [xd, None]
                    self._clear_overlay('_caliper_lines')
                self.draw_idle()

            elif tool == TOOL_ANNOTATE:
                if self._drag_start is None:
                    self._drag_start = (xd, yd)
                else:
                    end_x = xd
                    start_x = self._drag_start[0]
                    self._drag_start = None
                    self.annotation_req.emit(
                        min(start_x, end_x),
                        max(start_x, end_x),
                        self.lead_name
                    )

        elif event.button == 3:  # Right click — context menu
            self._show_context_menu(xd, yd, event)

    def _on_mouse_release(self, event):
        tool = self.tool
        xd, yd = self._data_coords(event)
        if xd is None:
            return

        if event.button == 1 and tool == TOOL_RULER and self._drag_start is not None:
            self._drag_end = (xd, yd)
            self._finalise_ruler()
            self._drag_start = None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.expand_requested.emit(self.lead_name)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # ── drawing helpers ───────────────────────────────────────────────────────
    def _draw_crosshair(self, xd, yd):
        self._clear_overlay('_crosshair_v', '_crosshair_h', '_readout_text')
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self._crosshair_v, = self.ax.plot(
            [xd, xd], ylim, color=COL_CROSSHAIR, linewidth=0.7,
            linestyle='--', alpha=0.85, zorder=10)
        self._crosshair_h, = self.ax.plot(
            xlim, [yd, yd], color=COL_CROSSHAIR, linewidth=0.7,
            linestyle='--', alpha=0.85, zorder=10)

        # Convert ADC to mV for display
        mv = (yd - 2048) * ADC_TO_MV
        txt = f" t={xd*1000:.1f}ms  {mv:+.2f}mV"
        self._readout_text = self.ax.text(
            xd, ylim[1] * 0.97, txt, fontsize=7, color=COL_CROSSHAIR,
            va='top', ha='left', zorder=11,
            bbox=dict(boxstyle='round,pad=0.15', fc='#000000', alpha=0.55, ec='none'))

    def _draw_ruler(self, x0, y0, x1, y1):
        self._clear_overlay('_ruler_patch')
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dt_ms = dx * 1000.0
        dv_mv = dy * ADC_TO_MV

        # Shaded rectangle
        lx = min(x0, x1); rx = max(x0, x1)
        ly = min(y0, y1); ry = max(y0, y1)
        rect = mpatches.FancyBboxPatch(
            (lx, ly), rx - lx, ry - ly,
            boxstyle="square,pad=0", linewidth=1.2,
            edgecolor='#ffdd00', facecolor='#ffdd0020', zorder=8)
        self.ax.add_patch(rect)
        self._ruler_patch = rect

        # Text
        mx = (lx + rx) / 2
        my = ry
        ylim = self.ax.get_ylim()
        label = ruler_label(dt_ms, dv_mv)
        self.ax.text(mx, min(my + (ylim[1] - ylim[0]) * 0.05, ylim[1] * 0.95),
                     label, fontsize=7.5, color='#ffdd00', ha='center',
                     va='bottom', fontweight='bold', zorder=12,
                     bbox=dict(boxstyle='round,pad=0.2', fc='#111111', alpha=0.7, ec='none'))
        self.ruler_measured.emit(label)

    def _finalise_ruler(self):
        if self._drag_start and self._drag_end:
            x0, y0 = self._drag_start
            x1, y1 = self._drag_end
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            dt_ms = dx * 1000.0
            dv_mv = dy * ADC_TO_MV
            label = ruler_label(dt_ms, dv_mv)
            self.ruler_measured.emit(label)
        self.draw_idle()

    def _draw_caliper_preview(self, x0, x1):
        # Draw first line solid, second dashed
        for i, (xl, style, col) in enumerate([(x0, '-', '#ff9900'), (x1, '--', '#ff9900')]):
            if self._caliper_lines[i] is not None:
                try:
                    self._caliper_lines[i].remove()
                except Exception:
                    pass
            ylim = self.ax.get_ylim()
            line, = self.ax.plot([xl, xl], ylim, color=col,
                                 linewidth=1.3, linestyle=style,
                                 alpha=0.9, zorder=9)
            self._caliper_lines[i] = line

    def _finalise_caliper(self):
        x0, x1 = sorted(self._caliper_x)
        dt_ms = abs(x1 - x0) * 1000.0
        bpm   = 60000.0 / dt_ms if dt_ms > 0 else 0
        label = caliper_label(dt_ms)
        # draw bracket
        self._draw_caliper_preview(x0, x1)
        # mid-span label
        ylim = self.ax.get_ylim()
        mid  = (x0 + x1) / 2
        self.ax.annotate(
            '', xy=(x1, (ylim[0]+ylim[1])/2), xytext=(x0, (ylim[0]+ylim[1])/2),
            arrowprops=dict(arrowstyle='<->', color='#ff9900', lw=1.3), zorder=9)
        self.ax.text(mid, (ylim[0]+ylim[1])/2 + (ylim[1]-ylim[0])*0.06,
                     label, fontsize=7.5, color='#ff9900', ha='center',
                     fontweight='bold', zorder=12,
                     bbox=dict(boxstyle='round,pad=0.2', fc='#111111', alpha=0.7, ec='none'))
        self.caliper_measured.emit(label)
        self.draw_idle()

    def _show_context_menu(self, xd, yd, event):
        """Right-click context menu for quick annotation."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background:#1e2235; color:#fff4e8; border:1px solid #6a4d24;
                    font-size:11px; border-radius:6px; }
            QMenu::item:selected { background:#ff8a1f; color:white; }
            QMenu::separator { height:1px; background:#6a4d24; margin:2px 10px; }
        """)
        menu.addAction(f"📍  Mark point at {xd*1000:.0f} ms")
        menu.addSeparator()
        types = ["Atrial Fibrillation", "PVC", "PAC", "SVT", "VT",
                 "Bradycardia", "Tachycardia", "2nd Degree Block", "LBBB", "RBBB"]
        for t in types:
            act = menu.addAction(f"⚡  Annotate: {t}")
            act.setData((xd, t))

        menu.addSeparator()
        clear_act = menu.addAction("🗑   Clear all overlays on this lead")

        chosen = menu.exec_(QCursor.pos())
        if chosen and chosen.data():
            xpos, ann_type = chosen.data()
            # Create a 0.5s annotation around click point
            self.annotation_req.emit(max(0, xpos - 0.25), xpos + 0.25, self.lead_name)
            # Also update the combo in the parent
            pw = self.parent_window
            idx = pw.arrhythmia_type_combo.findText(ann_type)
            if idx >= 0:
                pw.arrhythmia_type_combo.setCurrentIndex(idx)
        elif chosen == clear_act:
            self._clear_overlay('_ruler_patch', '_crosshair_v', '_crosshair_h',
                                '_readout_text')
            self._caliper_x = [None, None]
            self.draw_idle()

    # ── Magnifier paintEvent overlay ─────────────────────────────────────────
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.tool != TOOL_MAGNIFY or self._mag_pos is None:
            return
        if not self.parent_window.lead_has_visible_data(self.lead_name):
            return

        cx, cy = self._mag_pos.x(), self._mag_pos.y()
        panel_w = 280
        panel_h = 180
        zoom = self.parent_window.magnifier_zoom
        xd, yd = self._widget_to_data(cx, cy)
        if xd is None or yd is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        left = cx + 24
        top = cy - panel_h - 24
        if left + panel_w > self.width() - 8:
            left = cx - panel_w - 24
        if left < 8:
            left = 8
        if top < 8:
            top = cy + 24
        if top + panel_h > self.height() - 8:
            top = max(8, self.height() - panel_h - 8)

        dst_rect = QRect(int(left), int(top), panel_w, panel_h)
        inner_rect = dst_rect.adjusted(10, 10, -10, -10)

        painter.setBrush(QColor(7, 10, 18, 240))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(dst_rect, 12, 12)

        painter.setBrush(QColor(10, 14, 24, 28))
        painter.setPen(QPen(QColor(255, 176, 92), 3.2))
        painter.drawRoundedRect(dst_rect, 12, 12)

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        x_span = max(1e-6, (x_max - x_min) / max(zoom, 1))
        y_span = max(1.0, abs(y_max - y_min) / max(zoom, 1))

        view_x0 = max(x_min, xd - x_span / 2.0)
        view_x1 = min(x_max, xd + x_span / 2.0)
        if view_x1 <= view_x0:
            view_x0, view_x1 = x_min, x_max

        center_y = yd
        if self.lead_name == 'aVR':
            view_y0 = max(min(y_min, y_max), center_y - y_span / 2.0)
            view_y1 = min(max(y_min, y_max), center_y + y_span / 2.0)
        else:
            view_y0 = max(0.0, center_y - y_span / 2.0)
            view_y1 = min(4096.0, center_y + y_span / 2.0)
        if view_y1 <= view_y0:
            view_y0, view_y1 = min(y_min, y_max), max(y_min, y_max)

        grid_pen = QPen(QColor(36, 68, 36, 160), 1)
        painter.setPen(grid_pen)
        for frac in (0.25, 0.5, 0.75):
            gx = int(inner_rect.left() + inner_rect.width() * frac)
            gy = int(inner_rect.top() + inner_rect.height() * frac)
            painter.drawLine(gx, inner_rect.top(), gx, inner_rect.bottom())
            painter.drawLine(inner_rect.left(), gy, inner_rect.right(), gy)

        data = self.parent_window.lead_data.get(self.lead_name, np.array([]))
        if len(data) > 1 and self.parent_window.sampling_rate > 0:
            start_idx = max(0, int(np.floor(view_x0 * self.parent_window.sampling_rate)))
            end_idx = min(len(data), int(np.ceil(view_x1 * self.parent_window.sampling_rate)) + 1)
            segment = data[start_idx:end_idx]
            if len(segment) > 1:
                xs = np.arange(start_idx, end_idx, dtype=float) / self.parent_window.sampling_rate
                if self.lead_name == 'aVR':
                    seg_y = -segment
                else:
                    seg_y = segment

                points = []
                x_range = max(1e-6, view_x1 - view_x0)
                y_range = max(1e-6, view_y1 - view_y0)
                for px_t, py_v in zip(xs, seg_y):
                    nx = (px_t - view_x0) / x_range
                    ny = (py_v - view_y0) / y_range
                    qx = inner_rect.left() + nx * inner_rect.width()
                    qy = inner_rect.bottom() - ny * inner_rect.height()
                    points.append(QPointF(qx, qy))

                if len(points) > 1:
                    path = __import__('PyQt5.QtGui', fromlist=['QPainterPath']).QPainterPath()
                    path.moveTo(points[0])
                    for pt in points[1:]:
                        path.lineTo(pt)
                    painter.setClipRect(inner_rect)
                    painter.setPen(QPen(QColor(0, 255, 32), 2.2))
                    painter.drawPath(path)
                    painter.setClipping(False)

        focus_x = inner_rect.left() + ((xd - view_x0) / max(1e-6, view_x1 - view_x0)) * inner_rect.width()
        focus_y = inner_rect.bottom() - ((yd - view_y0) / max(1e-6, view_y1 - view_y0)) * inner_rect.height()

        painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
        painter.drawLine(int(focus_x), inner_rect.top(), int(focus_x), inner_rect.bottom())
        painter.drawLine(inner_rect.left(), int(focus_y), inner_rect.right(), int(focus_y))

        painter.setPen(QPen(QColor(255, 244, 232)))
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        painter.drawText(dst_rect.left() + 12, dst_rect.bottom() - 12, f"{zoom}x")
        painter.end()


class LeadExpandedPopup(QDialog):
    """Standalone expanded lead window with metrics, zoom/amplification, and rhythm interpretation."""

    def __init__(self, lead_name: str, lead_data: np.ndarray, sampling_rate: float, parent=None):
        super().__init__(parent)
        self.lead_name = str(lead_name)
        self.sampling_rate = float(sampling_rate or 500.0)
        self.raw_data = np.asarray(lead_data if lead_data is not None else [], dtype=float)
        self.time_zoom_factor = 1.0
        self.amplification = 1.0
        self.analysis = {}

        self.setWindowTitle(f"Lead {self.lead_name} Expanded Analysis")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint)

        self._build_ui()
        self._fit_to_screen()
        self.refresh_from_data(self.raw_data, self.sampling_rate)

    def _fit_to_screen(self):
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 760)
            return
        rect = screen.availableGeometry()
        width = int(rect.width() * 0.86)
        height = int(rect.height() * 0.82)
        x = rect.x() + (rect.width() - width) // 2
        y = rect.y() + (rect.height() - height) // 2
        self.setGeometry(x, y, width, height)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QLabel(f"Lead {self.lead_name} - Expanded Waveform")
        header.setStyleSheet("color:#fff4e8;font-size:14px;font-weight:bold;")
        root.addWidget(header)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("Time Zoom:"))
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["1x", "2x", "4x", "8x"])
        self.zoom_combo.setCurrentText("1x")
        self.zoom_combo.currentTextChanged.connect(self._on_zoom_changed)
        controls.addWidget(self.zoom_combo)

        controls.addWidget(QLabel("Amplification:"))
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(25, 400)
        self.amp_slider.setValue(100)
        self.amp_slider.valueChanged.connect(self._on_amp_changed)
        self.amp_label = QLabel("1.00x")
        controls.addWidget(self.amp_slider, 1)
        controls.addWidget(self.amp_label)

        controls.addWidget(QLabel("Position:"))
        self.pos_slider = QSlider(Qt.Horizontal)
        self.pos_slider.setRange(0, 1000)
        self.pos_slider.setValue(0)
        self.pos_slider.valueChanged.connect(self._render_plot)
        controls.addWidget(self.pos_slider, 2)
        root.addLayout(controls)

        fig = Figure(facecolor="#090b14")
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background:#090b14;border:1px solid #5b4525;")
        root.addWidget(self.canvas, stretch=1)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)
        self.rr_lbl = QLabel("RR: -- ms")
        self.pr_lbl = QLabel("PR: -- ms")
        self.qrs_lbl = QLabel("QRS: -- ms")
        self.p_lbl = QLabel("P: --")
        for lbl in (self.rr_lbl, self.pr_lbl, self.qrs_lbl, self.p_lbl):
            lbl.setStyleSheet(
                "color:#ffedda;background:#1b1f34;border:1px solid #6a532e;"
                "border-radius:6px;padding:6px 10px;font-weight:bold;"
            )
            metrics_row.addWidget(lbl)
        metrics_row.addStretch()
        root.addLayout(metrics_row)

        self.interpret_lbl = QLabel("Arrhythmia Interpretation: analyzing...")
        self.interpret_lbl.setWordWrap(True)
        self.interpret_lbl.setStyleSheet(
            "color:#e8f1ff;background:#15192b;border:1px solid #5b4525;border-radius:6px;padding:8px;"
        )
        root.addWidget(self.interpret_lbl)

    def refresh_from_data(self, lead_data: np.ndarray, sampling_rate: float):
        self.raw_data = np.asarray(lead_data if lead_data is not None else [], dtype=float)
        self.sampling_rate = float(sampling_rate or 500.0)
        self._run_analysis()
        self._render_plot()

    def _on_zoom_changed(self, text: str):
        try:
            self.time_zoom_factor = max(1.0, float(str(text).replace("x", "")))
        except Exception:
            self.time_zoom_factor = 1.0
        self._render_plot()

    def _on_amp_changed(self, value: int):
        self.amplification = max(0.25, float(value) / 100.0)
        self.amp_label.setText(f"{self.amplification:.2f}x")
        self._render_plot()

    def _run_analysis(self):
        if self.raw_data.size < 20:
            self.analysis = {}
            self.rr_lbl.setText("RR: -- ms")
            self.pr_lbl.setText("PR: -- ms")
            self.qrs_lbl.setText("QRS: -- ms")
            self.p_lbl.setText("P: No waveform")
            self.interpret_lbl.setText("Arrhythmia Interpretation: Not enough data.")
            return

        analysis = {}
        try:
            if PQRSTAnalyzer is not None:
                analyzer = PQRSTAnalyzer(self.sampling_rate)
                analysis = analyzer.analyze_signal(self.raw_data) or {}
        except Exception:
            analysis = {}

        r_peaks = np.asarray(analysis.get("r_peaks", []), dtype=int)
        p_peaks = np.asarray(analysis.get("p_peaks", []), dtype=int)
        q_peaks = np.asarray(analysis.get("q_peaks", []), dtype=int)
        s_peaks = np.asarray(analysis.get("s_peaks", []), dtype=int)

        rr_ms = 0.0
        if r_peaks.size >= 2:
            rr_ms = float(np.median(np.diff(r_peaks)) * 1000.0 / max(self.sampling_rate, 1.0))

        pr_values = []
        if p_peaks.size and r_peaks.size:
            for r in r_peaks:
                prior_p = p_peaks[p_peaks < r]
                if prior_p.size:
                    pr_values.append((int(r) - int(prior_p[-1])) * 1000.0 / max(self.sampling_rate, 1.0))
        pr_ms = float(np.median(pr_values)) if pr_values else 0.0

        qrs_values = []
        if q_peaks.size and s_peaks.size:
            for r in r_peaks:
                q_before = q_peaks[q_peaks < r]
                s_after = s_peaks[s_peaks > r]
                if q_before.size and s_after.size:
                    qrs_values.append((int(s_after[0]) - int(q_before[-1])) * 1000.0 / max(self.sampling_rate, 1.0))
        qrs_ms = float(np.median(qrs_values)) if qrs_values else 0.0

        p_status = "Present" if p_peaks.size > 0 else "Absent"
        self.rr_lbl.setText(f"RR: {rr_ms:.0f} ms" if rr_ms > 0 else "RR: -- ms")
        self.pr_lbl.setText(f"PR: {pr_ms:.0f} ms" if pr_ms > 0 else "PR: -- ms")
        self.qrs_lbl.setText(f"QRS: {qrs_ms:.0f} ms" if qrs_ms > 0 else "QRS: -- ms")
        self.p_lbl.setText(f"P: {p_status} ({int(p_peaks.size)})")

        interpretation_text = "No arrhythmia interpretation available."
        try:
            if ArrhythmiaDetector is not None:
                det = ArrhythmiaDetector(self.sampling_rate, counts_per_mv=500.0)
                arr = det.detect_arrhythmias(
                    self.raw_data,
                    analysis,
                    lead_signals={"II": self.raw_data}
                )
                summary_lines = []
                if get_interpretation is not None:
                    result_stub = {
                        "heart_rate_bpm": (60000.0 / rr_ms) if rr_ms > 0 else 0.0,
                        "rr_ms": rr_ms,
                        "pr_ms": pr_ms,
                        "qrs_ms": qrs_ms,
                        "qtc_bazett": 0.0,
                        "arrhythmias": arr or [],
                        "st_levels": {},
                        "is_nsr": any("Normal Sinus Rhythm" in str(x) for x in (arr or [])),
                        "nsr_failed_criteria": [],
                    }
                    summary_lines = [line for line in get_interpretation(result_stub) if line]
                if not summary_lines:
                    summary_lines = arr or ["Normal Sinus Rhythm"]
                interpretation_text = " | ".join(summary_lines)
        except Exception:
            pass

        self.interpret_lbl.setText(f"Arrhythmia Interpretation: {interpretation_text}")
        self.analysis = analysis

    def _render_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("#080e08")
        self.ax.grid(True, color="#1a3a1a", linewidth=0.35, linestyle="-", alpha=1.0)

        if self.raw_data.size < 2:
            self.ax.text(0.5, 0.5, "No data", transform=self.ax.transAxes,
                         ha="center", va="center", color="#97a78f")
            self.canvas.draw_idle()
            return

        fs = max(self.sampling_rate, 1.0)
        total_samples = int(self.raw_data.size)
        total_sec = total_samples / fs
        window_sec = max(1.0, total_sec / max(self.time_zoom_factor, 1.0))
        window_samples = max(2, int(round(window_sec * fs)))
        max_start = max(0, total_samples - window_samples)
        start = int(round((self.pos_slider.value() / 1000.0) * max_start))
        end = min(total_samples, start + window_samples)

        seg = self.raw_data[start:end]
        time_axis = np.arange(start, end) / fs
        baseline = float(np.mean(seg)) if seg.size else 0.0
        display = (seg - baseline) * self.amplification + baseline

        self.ax.plot(time_axis, display, color="#00d000", linewidth=1.0, antialiased=True)
        self.ax.set_xlim(time_axis[0], time_axis[-1] if len(time_axis) > 1 else time_axis[0] + 1.0)

        # Keep the classic ECG ADC range while respecting amplification.
        margin = 200.0
        y_min = float(np.min(display)) - margin
        y_max = float(np.max(display)) + margin
        if y_max <= y_min:
            y_min, y_max = baseline - 500.0, baseline + 500.0
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_title(
            f"Lead {self.lead_name} | Zoom {self.time_zoom_factor:.0f}x | Amp {self.amplification:.2f}x",
            color="#ffdcb5", fontsize=10
        )
        self.ax.tick_params(axis="x", colors="#b9c9b0", labelsize=8)
        self.ax.tick_params(axis="y", colors="#b9c9b0", labelsize=8)
        self.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class ECGAnalysisWindow(QDialog):
    """Professional ECG Analysis Window with clinical-grade doctor tools."""

    LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ECG Waveform Analysis — Clinical Edition")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        # ── Tool state ──────────────────────────────────────────────────
        self.current_tool  = TOOL_SELECT
        self.magnifier_zoom = 4
        # ── Data state ──────────────────────────────────────────────────
        self.reports             = []
        self.current_report      = None
        self.current_report_path = ""
        self.lead_data           = {lead: np.array([]) for lead in self.LEADS}
        self.sampling_rate       = 500.0

        self.window_seconds      = 10.0
        self.step_seconds        = 0.5
        self.frame_start_sample  = 0
        self.play_timer          = QTimer(self)
        self.play_timer.timeout.connect(self.next_frame)

        self.pending_mark_start_sec = None
        self.manual_annotations     = []
        self._expanded_lead         = None   # Deprecated: old in-grid expand path
        self._lead_popup_windows    = {}     # lead -> LeadExpandedPopup
        self._active_lead_popup     = None

        # Paths
        project_root = Path(__file__).resolve().parents[2]
        self.analysis_pdf_logo_path = project_root / "assets" / "DeckmountLogo.png"

        self._apply_stylesheet()
        self._build_ui()
        self.load_reports()
        QTimer.singleShot(0, self._fit_window_to_screen)

    # ─────────────────────────────────────────────────────────────────────────
    #  STYLESHEET
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QDialog {
                background: #0f1220;
                color: #f3e8dc;
            }
            QFrame#topbar {
                background: #15192b;
                border: none;
                border-bottom: 2px solid #6d4a1f;
            }
            QFrame#plotpanel {
                background: #0f1220;
                border: none;
            }
            QFrame#bottompanel {
                background: #15192b;
                border: 1px solid #5f4523;
                border-radius: 8px;
            }
            QFrame#leadbox {
                background: #090b14;
                border: 1px solid #342a23;
                border-radius: 4px;
            }
            QFrame#leadbox_expanded {
                background: #090b14;
                border: 2px solid #ff8a1f;
                border-radius: 6px;
            }
            QFrame#toolbar_frame {
                background: #171b2e;
                border: 1px solid #5b4325;
                border-radius: 8px;
            }
            QLabel {
                color: #e9dccd;
                font-size: 11px;
                background: transparent;
                border: none;
            }
            QLabel#leadlabel {
                color: #ff4444;
                font-size: 11px;
                font-weight: bold;
                font-family: 'Courier New';
                background: transparent;
                border: none;
            }
            QPushButton {
                background: #1b2036;
                color: #fff4e8;
                border: 1px solid #5e4827;
                border-radius: 5px;
                padding: 5px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #2a2230;
                border-color: #ff8a1f;
                color: #ffffff;
            }
            QPushButton#primary {
                background: #ff8a1f;
                color: #ffffff;
                border: 1px solid #ff8a1f;
                font-weight: bold;
            }
            QPushButton#primary:hover { background: #ff9d3d; }
            QPushButton#apibtn {
                background: #ff8a1f;
                color: #ffffff;
                border: none;
                border-radius: 5px;
            }
            QPushButton#apibtn:hover { background: #ff9d3d; }
            /* Tool buttons */
            QToolButton {
                background: #191d31;
                color: #fff4e8;
                border: 1px solid #5b4b2a;
                border-radius: 6px;
                padding: 6px;
                font-size: 13px;
                min-width: 36px;
                min-height: 36px;
            }
            QToolButton:hover {
                background: #2a2230;
                border-color: #ff8a1f;
                color: #ffffff;
            }
            QToolButton:checked {
                background: #ff8a1f;
                border: 2px solid #ffd9b0;
                color: #ffffff;
            }
            QComboBox {
                background: #1a1f34;
                color: #fff4e8;
                border: 1px solid #5b4525;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox::down-arrow { image: none; }
            QComboBox QAbstractItemView {
                background: #1a1f34;
                color: #fff4e8;
                selection-background-color: #ff8a1f;
                border: 1px solid #5b4525;
            }
            QLineEdit {
                background: #1a1f34;
                color: #fff4e8;
                border: 1px solid #5b4525;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QTextEdit {
                background: #090b14;
                color: #f0e1d1;
                border: 1px solid #5b4525;
                border-radius: 5px;
                padding: 4px;
                font-size: 10px;
            }
            QTableWidget {
                background: #090b14;
                color: #f0e1d1;
                border: 1px solid #5b4525;
                gridline-color: #46341f;
                selection-background-color: #ff8a1f;
                selection-color: #ffffff;
                border-radius: 4px;
                font-size: 10px;
            }
            QHeaderView::section {
                background: #1a1f34;
                color: #ffcb95;
                border: none;
                border-bottom: 1px solid #5b4525;
                padding: 5px;
                font-size: 10px;
                font-weight: bold;
            }
            QSlider::groove:horizontal {
                height: 3px;
                background: #5b4525;
                border-radius: 1px;
            }
            QSlider::sub-page:horizontal {
                background: #ff8a1f;
                border-radius: 1px;
            }
            QSlider::handle:horizontal {
                width: 12px; height: 12px;
                background: #ffb15a;
                margin: -4px 0;
                border-radius: 6px;
            }
            QScrollBar:vertical {
                background: #090b14;
                width: 6px;
            }
            QScrollBar::handle:vertical {
                background: #6b4a23;
                border-radius: 3px;
            }
        """)

    # ─────────────────────────────────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_top_bar())

        content = QHBoxLayout()
        content.setContentsMargins(6, 6, 6, 6)
        content.setSpacing(6)
        self._tool_sidebar = self._build_tool_sidebar()
        self._plot_panel = self._build_plot_panel()
        content.addWidget(self._tool_sidebar)
        content.addWidget(self._plot_panel, stretch=1)

        mid = QWidget()
        mid.setLayout(content)
        root.addWidget(mid, stretch=4)
        self._bottom_panel = self._build_bottom_panel()
        root.addWidget(self._bottom_panel, stretch=0)

    # ── TOP BAR ──────────────────────────────────────────────────────────────
    def _build_top_bar(self):
        frame = QFrame()
        frame.setObjectName("topbar")
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(10)

        self.back_btn = QPushButton("⬅ Dashboard")
        self.back_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;"
            "font-size:12px;padding:7px 14px;border:none;border-radius:5px;")
        self.back_btn.clicked.connect(self.close)
        lay.addWidget(self.back_btn)

        logo_label = QLabel()
        pixmap = QPixmap(str(self.analysis_pdf_logo_path))
        if not pixmap.isNull():
            logo_label.setPixmap(
                pixmap.scaled(100, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo_label.setText("◈ DECKMOUNT")
            logo_label.setStyleSheet(
                "color:#ffb15a;font-size:15px;font-weight:bold;font-family:'Courier New';")
        lay.addWidget(logo_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("background:#ff8a1f;max-width:2px;margin:4px 6px;border:none;")
        lay.addWidget(sep)

        pat_col = QVBoxLayout()
        pat_col.setSpacing(1)
        self.patient_lbl = QLabel("Patient: —")
        self.patient_lbl.setFont(QFont("Courier New", 11, QFont.Bold))
        self.patient_lbl.setStyleSheet("color:#fff5ea;font-weight:bold;")
        self.patient_meta_lbl = QLabel("ID: — | Age: — | Gender: —")
        self.patient_meta_lbl.setStyleSheet("color:#d8b28b;font-size:10px;")
        pat_col.addWidget(self.patient_lbl)
        pat_col.addWidget(self.patient_meta_lbl)
        lay.addLayout(pat_col)
        lay.addStretch()

        # Measurement readout bar (ruler/caliper results)
        self.measure_lbl = QLabel("")
        self.measure_lbl.setStyleSheet(
            "color:#ffe5bf;font-family:'Courier New';font-size:11px;"
            "background:#241808;border:1px solid #8a5a1f;border-radius:4px;padding:3px 10px;")
        self.measure_lbl.setMinimumWidth(280)
        self.measure_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.measure_lbl)
        lay.addStretch()

        lay.addWidget(QLabel("Report:"))
        self.report_combo = QComboBox()
        self.report_combo.currentIndexChanged.connect(self.load_selected_report)
        self.report_combo.setMinimumWidth(280)
        lay.addWidget(self.report_combo)

        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(32)
        self.refresh_btn.clicked.connect(self.load_reports)
        lay.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("⬇ JSON")
        self.export_btn.clicked.connect(self.export_report)
        lay.addWidget(self.export_btn)

        self.pdf_btn = QPushButton("📄 PDF Report")
        self.pdf_btn.setObjectName("primary")
        self.pdf_btn.clicked.connect(self.generate_pdf_report)
        lay.addWidget(self.pdf_btn)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("background:#5b4525;max-width:1px;margin:4px 4px;border:none;")
        lay.addWidget(sep2)

        lay.addWidget(QLabel("API ID:"))
        self.api_id_input = QLineEdit()
        self.api_id_input.setPlaceholderText("ID")
        self.api_id_input.setFixedWidth(60)
        self.api_fetch_btn = QPushButton("Fetch")
        self.api_fetch_btn.setObjectName("apibtn")
        self.api_fetch_btn.clicked.connect(self.fetch_api_report)
        lay.addWidget(self.api_id_input)
        lay.addWidget(self.api_fetch_btn)
        return frame

    # ── TOOL SIDEBAR ─────────────────────────────────────────────────────────
    def _build_tool_sidebar(self):
        frame = QFrame()
        frame.setObjectName("toolbar_frame")
        frame.setFixedWidth(88)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(6, 10, 6, 10)
        lay.setSpacing(6)

        tool_data = tool_specs(include_annotate=True)

        self._tool_btns = {}
        self._tool_btn_group = QButtonGroup(self)
        self._tool_btn_group.setExclusive(True)

        for tool_id, icon, tip in tool_data:
            btn = QToolButton()
            btn.setText(icon)
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setFont(QFont("Arial", 9, QFont.Bold))
            btn.setMinimumHeight(42)
            if tool_id == TOOL_SELECT:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, t=tool_id: self.set_tool(t))
            self._tool_btn_group.addButton(btn)
            self._tool_btns[tool_id] = btn
            lay.addWidget(btn)

        lay.addSpacing(10)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background:#5b4525;border:none;max-height:1px;")
        lay.addWidget(sep)

        lay.addSpacing(4)

        zoom_lbl = QLabel("Zoom")
        zoom_lbl.setAlignment(Qt.AlignCenter)
        zoom_lbl.setStyleSheet("color:#ffd7b0;font-size:9px;font-weight:bold;")
        lay.addWidget(zoom_lbl)

        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems([f"{level}x" for level in [2, 3, 4, 5, 6]])
        self.zoom_combo.setCurrentText("4x")
        self.zoom_combo.currentIndexChanged.connect(
            lambda i: setattr(self, 'magnifier_zoom', i + 2))
        self.zoom_combo.setFixedWidth(62)
        self.zoom_combo.setStyleSheet("""
            font-size:10px;
            padding:2px;
            color:#fff4e8;
            background:#1b1f34;
            border:1px solid #6a532e;
            border-radius:4px;
        """)
        lay.addWidget(self.zoom_combo)

        lay.addStretch()

        # Clear overlays button
        clr_btn = QToolButton()
        clr_btn.setText("X")
        clr_btn.setToolTip("Clear all measurements & overlays")
        clr_btn.setFont(QFont("Arial", 11, QFont.Bold))
        clr_btn.clicked.connect(self._clear_all_overlays)
        lay.addWidget(clr_btn)

        return frame

    def set_tool(self, tool_id):
        self.current_tool = tool_id
        cursor = tool_cursor(tool_id)
        for canvas in self._lead_canvases.values():
            canvas.setCursor(cursor)
        self.measure_lbl.setText(tool_hint(tool_id))

    def _clear_all_overlays(self):
        for canvas in self._lead_canvases.values():
            canvas._caliper_x     = [None, None]
            canvas._drag_start    = None
            canvas._clear_overlay('_ruler_patch', '_crosshair_v', '_crosshair_h',
                                   '_readout_text', '_caliper_lines')
            canvas.draw_idle()
        self.measure_lbl.setText("")

    # ── PLOT PANEL ───────────────────────────────────────────────────────────
    def _build_plot_panel(self):
        frame = QFrame()
        frame.setObjectName("plotpanel")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v = QVBoxLayout(frame)
        v.setContentsMargins(4, 4, 4, 2)
        v.setSpacing(4)

        # Controls bar
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet(
            "background:#15192b;border-radius:6px;border:1px solid #5b4525;")
        controls = QHBoxLayout(ctrl_frame)
        controls.setContentsMargins(10, 5, 10, 5)
        controls.setSpacing(8)

        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_frame)
        for b in (self.prev_btn, self.play_btn, self.next_btn):
            controls.addWidget(b)

        controls.addSpacing(8)
        controls.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["1.0 s","2.0 s","3.0 s","5.0 s","10.0 s"])
        self.window_combo.setCurrentText("10.0 s")
        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        self.window_combo.setFixedWidth(74)
        controls.addWidget(self.window_combo)

        controls.addWidget(QLabel("Step:"))
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.2 s","0.5 s","1.0 s"])
        self.step_combo.setCurrentText("0.5 s")
        self.step_combo.currentTextChanged.connect(self._on_step_changed)
        self.step_combo.setFixedWidth(68)
        controls.addWidget(self.step_combo)

        controls.addSpacing(12)
        self.frame_label = QLabel("Frame: 0.00s – 10.00s")
        self.frame_label.setStyleSheet(
            "color:#fff4e8;font-weight:bold;font-family:'Courier New';font-size:11px;")
        controls.addWidget(self.frame_label)
        controls.addStretch()

        # Expand/collapse hint
        expand_hint = QLabel("click lead = expanded popup  |  right-click = annotate menu")
        expand_hint.setStyleSheet("color:#c59768;font-size:9px;")
        controls.addWidget(expand_hint)

        v.addWidget(ctrl_frame)

        # Timeline
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setMinimum(0)
        self.timeline.setMaximum(0)
        self.timeline.valueChanged.connect(self._on_timeline_changed)
        self.timeline.setFixedHeight(14)
        v.addWidget(self.timeline)

        # Lead grid
        from PyQt5.QtWidgets import QGridLayout, QWidget as _QW
        self._grid_widget = _QW()
        self._grid_widget.setStyleSheet("background:#0f1220;")
        self._grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._lead_grid = QGridLayout(self._grid_widget)
        self._lead_grid.setContentsMargins(0, 0, 0, 0)
        self._lead_grid.setSpacing(4)

        self._lead_canvases = {}
        self._lead_axes     = {}
        self._lead_figs     = {}
        self._lead_frames   = {}

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
                self._lead_frames[lead] = cell
                cell_lay = QVBoxLayout(cell)
                cell_lay.setContentsMargins(0, 0, 0, 0)
                cell_lay.setSpacing(0)

                lbl = QLabel(lead)
                lbl.setObjectName("leadlabel")
                lbl.setAlignment(Qt.AlignLeft)
                lbl.setContentsMargins(5, 2, 0, 0)
                lbl.setFixedHeight(16)
                cell_lay.addWidget(lbl)

                fig = Figure(facecolor='#090b14')
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                ax  = fig.add_axes([0, 0, 1, 1], facecolor='#090b14')
                ax.set_axis_off()

                canvas = InteractiveLeadCanvas(fig, ax, lead, self)
                canvas.setStyleSheet("background:#090b14;border:none;")

                # Connect signals
                canvas.ruler_measured.connect(self.measure_lbl.setText)
                canvas.caliper_measured.connect(self.measure_lbl.setText)
                canvas.annotation_req.connect(self._on_canvas_annotation)
                canvas.expand_requested.connect(self._open_lead_expanded_popup)
                cell_lay.addWidget(canvas, stretch=1)
                self._lead_grid.addWidget(cell, row, col)

                self._lead_figs[lead]    = fig
                self._lead_canvases[lead] = canvas
                self._lead_axes[lead]    = ax

        # Legacy compat
        self.figure = list(self._lead_figs.values())[0]
        self.canvas = list(self._lead_canvases.values())[0]
        self.axes   = [self._lead_axes[l] for l in self.LEADS]

        v.addWidget(self._grid_widget, stretch=1)
        return frame

    def _on_canvas_annotation(self, start_sec, end_sec, lead_name):
        """Handle annotation request from canvas right-click or annotate tool."""
        arr_type = self.arrhythmia_type_combo.currentText().strip()
        if arr_type == 'Other':
            arr_type = self.manual_type_input.text().strip() or 'Other'

        ann = {
            'start_sec':  round(start_sec, 3),
            'end_sec':    round(end_sec, 3),
            'type':       arr_type,
            'lead':       lead_name,
            'notes':      self.notes_input.text().strip() or "Quick annotate",
            'created_at': datetime.now().isoformat(timespec='seconds')
        }
        self.manual_annotations.append(ann)
        self._refresh_annotation_table()
        self._persist_annotations_in_report()
        self._render_current_frame()
        self.mark_status_lbl.setText(f"✅ Quick annotation: {arr_type} on {lead_name}")
        self.mark_status_lbl.setStyleSheet(
            "color:#7adf7a;border:none;background:transparent;")

    def _toggle_lead_expand(self, lead_name):
        """Deprecated old in-grid expand behavior. Kept commented-out logically and not used."""
        if self._expanded_lead == lead_name:
            # Collapse — show all
            self._expanded_lead = None
            for l, cell in self._lead_frames.items():
                cell.setVisible(True)
            # restore grid layout
            lead_order = [
                ['I',   'aVR', 'V1', 'V4'],
                ['II',  'aVL', 'V2', 'V5'],
                ['III', 'aVF', 'V3', 'V6'],
            ]
            for row, row_leads in enumerate(lead_order):
                for col, lead in enumerate(row_leads):
                    self._lead_grid.addWidget(self._lead_frames[lead], row, col)
            self._lead_frames[lead_name].setObjectName("leadbox")
        else:
            # Expand
            self._expanded_lead = lead_name
            for l, cell in self._lead_frames.items():
                cell.setVisible(l == lead_name)
            # Span the whole grid
            self._lead_grid.addWidget(self._lead_frames[lead_name], 0, 0, 3, 4)
            self._lead_frames[lead_name].setObjectName("leadbox_expanded")
            # Force style refresh
            self._lead_frames[lead_name].style().unpolish(self._lead_frames[lead_name])
            self._lead_frames[lead_name].style().polish(self._lead_frames[lead_name])

        self._render_current_frame()

    def _open_lead_expanded_popup(self, lead_name):
        """Open dedicated expanded lead popup with metrics + rhythm interpretation."""
        data = np.asarray(self.lead_data.get(lead_name, np.array([])), dtype=float)
        if data.size < 20:
            QMessageBox.information(self, "No Data", f"No usable waveform data for lead {lead_name}.")
            return

        popup = self._lead_popup_windows.get(lead_name)
        if popup is None or not popup.isVisible():
            popup = LeadExpandedPopup(lead_name, data, self.sampling_rate, parent=self)
            self._lead_popup_windows[lead_name] = popup
        else:
            popup.refresh_from_data(data, self.sampling_rate)

        self._active_lead_popup = popup
        popup.show()
        popup.raise_()
        popup.activateWindow()

    # ── BOTTOM PANEL ─────────────────────────────────────────────────────────
    def _build_bottom_panel(self):
        frame = QFrame()
        frame.setObjectName("bottompanel")
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        h = QHBoxLayout(frame)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(14)

        # ── Manual marking ───────────────────────────────────────────────────
        mark_box = QFrame()
        mark_box.setStyleSheet("background:transparent;border:none;")
        mark_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        av = QVBoxLayout(mark_box)
        av.setSpacing(4)

        title_lbl = QLabel("▌ Manual Arrhythmia Marking")
        title_lbl.setStyleSheet(
            "color:#fff4e8;font-size:12px;font-weight:bold;font-family:'Courier New';")
        av.addWidget(title_lbl)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Type:"))
        self.arrhythmia_type_combo = QComboBox()
        self.arrhythmia_type_combo.addItems([
            "Atrial Fibrillation", "PVC", "PAC", "SVT", "VT",
            "Bradycardia", "Tachycardia", "2nd Degree Block",
            "LBBB", "RBBB", "Normal Sinus Rhythm", "Other"
        ])
        row1.addWidget(self.arrhythmia_type_combo, 2)
        row1.addWidget(QLabel("Lead:"))
        self.mark_lead_combo = QComboBox()
        self.mark_lead_combo.addItems(["All Leads"] + self.LEADS)
        row1.addWidget(self.mark_lead_combo, 1)
        av.addLayout(row1)

        row1b = QHBoxLayout()
        self.manual_type_input = QLineEdit()
        self.manual_type_input.setPlaceholderText("Custom type (if Other selected)")
        self.notes_input = QLineEdit()
        self.notes_input.setPlaceholderText("Clinical notes...")
        row1b.addWidget(self.manual_type_input)
        row1b.addWidget(self.notes_input)
        av.addLayout(row1b)

        row2 = QHBoxLayout()
        btn_s = ("background:#15192b;color:#fff4e8;border:2px solid #ff8a1f;"
                 "border-radius:5px;padding:5px 12px;font-weight:bold;")
        self.mark_start_btn = QPushButton("① Mark Start")
        self.mark_start_btn.setStyleSheet(btn_s)
        self.mark_start_btn.clicked.connect(self.mark_start)

        self.mark_end_btn = QPushButton("② Mark End + Save")
        self.mark_end_btn.setStyleSheet(btn_s)
        self.mark_end_btn.clicked.connect(self.mark_end_and_save)
        self.mark_end_btn.setEnabled(False)

        self.auto_detect_btn = QPushButton("🤖 Auto Detect")
        self.auto_detect_btn.setStyleSheet(btn_s)
        self.auto_detect_btn.clicked.connect(self.run_automatic_detection)

        self.delete_mark_btn = QPushButton("🗑 Delete")
        self.delete_mark_btn.setStyleSheet(
            "background:#15192b;color:#ff8b8b;border:2px solid #a63d2d;"
            "border-radius:5px;padding:5px 12px;font-weight:bold;")
        self.delete_mark_btn.clicked.connect(self.delete_selected_annotation)

        for b in (self.mark_start_btn, self.mark_end_btn,
                  self.auto_detect_btn, self.delete_mark_btn):
            row2.addWidget(b)
        av.addLayout(row2)

        self.mark_status_lbl = QLabel("No active mark")
        self.mark_status_lbl.setStyleSheet(
            "color:#d7b183;font-family:'Courier New';font-size:10px;")
        av.addWidget(self.mark_status_lbl)

        self.annotation_table = QTableWidget(0, 5)
        self.annotation_table.setHorizontalHeaderLabels(
            ["Start (s)", "End (s)", "Type", "Lead", "Notes"])
        self.annotation_table.horizontalHeader().setStretchLastSection(True)
        self.annotation_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        av.addWidget(self.annotation_table)

        h.addWidget(mark_box, stretch=2)

        # ── Right: Metrics + Findings ────────────────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(3)

        metrics_lbl = QLabel("▌ ECG Metrics")
        metrics_lbl.setStyleSheet(
            "color:#fff4e8;font-size:11px;font-weight:bold;font-family:'Courier New';")
        right_col.addWidget(metrics_lbl)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        right_col.addWidget(self.metrics_table)

        findings_lbl = QLabel("▌ Clinical Findings")
        findings_lbl.setStyleSheet(
            "color:#fff4e8;font-size:11px;font-weight:bold;font-family:'Courier New';")
        right_col.addWidget(findings_lbl)
        self.findings_text = QTextEdit()
        self.findings_text.setReadOnly(True)
        self.findings_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        right_col.addWidget(self.findings_text)

        # Hide per original user instruction (kept for PDF gen logic)
        for w in (metrics_lbl, self.metrics_table, findings_lbl, self.findings_text):
            w.setVisible(False)

        h.addLayout(right_col, stretch=1)
        return frame

    def _fit_window_to_screen(self):
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.setGeometry(available.adjusted(6, 6, -6, -6))
        self._apply_responsive_window_layout()

    def _apply_responsive_window_layout(self):
        size = self.size()
        screen_h = max(size.height(), 1)
        screen_w = max(size.width(), 1)

        if hasattr(self, "_tool_sidebar") and self._tool_sidebar is not None:
            sidebar_width = max(76, min(96, int(screen_w * 0.055)))
            self._tool_sidebar.setFixedWidth(sidebar_width)

        if hasattr(self, "_bottom_panel") and self._bottom_panel is not None:
            bottom_max = max(170, min(260, int(screen_h * 0.22)))
            self._bottom_panel.setMaximumHeight(bottom_max)

        if hasattr(self, "annotation_table") and self.annotation_table is not None:
            self.annotation_table.setMaximumHeight(max(72, min(110, int(screen_h * 0.10))))

        if hasattr(self, "metrics_table") and self.metrics_table is not None:
            self.metrics_table.setMaximumHeight(max(72, min(120, int(screen_h * 0.11))))

        if hasattr(self, "findings_text") and self.findings_text is not None:
            self.findings_text.setMaximumHeight(max(56, min(84, int(screen_h * 0.07))))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_window_layout()

    # ─────────────────────────────────────────────────────────────────────────
    #  KEYBOARD SHORTCUTS
    # ─────────────────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_S:
            self.set_tool(TOOL_SELECT)
            self._tool_btns[TOOL_SELECT].setChecked(True)
        elif key == Qt.Key_R:
            self.set_tool(TOOL_RULER)
            self._tool_btns[TOOL_RULER].setChecked(True)
        elif key == Qt.Key_C:
            self.set_tool(TOOL_CALIPER)
            self._tool_btns[TOOL_CALIPER].setChecked(True)
        elif key == Qt.Key_M:
            self.set_tool(TOOL_MAGNIFY)
            self._tool_btns[TOOL_MAGNIFY].setChecked(True)
        elif key == Qt.Key_A:
            self.set_tool(TOOL_ANNOTATE)
            self._tool_btns[TOOL_ANNOTATE].setChecked(True)
        elif key == Qt.Key_Left:
            self.prev_frame()
        elif key == Qt.Key_Right:
            self.next_frame()
        elif key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_Escape:
            if self._active_lead_popup is not None and self._active_lead_popup.isVisible():
                self._active_lead_popup.close()
        else:
            super().keyPressEvent(event)

    # ─────────────────────────────────────────────────────────────────────────
    #  DATA LOADING  (identical to original — preserved completely)
    # ─────────────────────────────────────────────────────────────────────────
    def load_reports(self):
        self.report_combo.blockSignals(True)
        self.report_combo.clear()
        self.reports = []
        try:
            base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            reports_dir = os.path.join(base_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            files = [f for f in os.listdir(reports_dir)
                     if f.endswith('.json') and not f.startswith('index')]
            files.sort(reverse=True)
            for filename in files:
                filepath = os.path.join(reports_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        report = json.load(f)
                    patient_name = self._extract_patient_name(report)
                    date_str     = self._extract_report_date(report)
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
        self.current_report      = self.reports[index]
        self.current_report_path = self.report_combo.itemData(index) or ""
        self._update_patient_info()
        self._load_lead_data()
        self._load_metrics_findings()
        self._load_manual_annotations()
        self.frame_start_sample     = 0
        self.pending_mark_start_sec = None
        self.mark_end_btn.setEnabled(False)
        self.mark_status_lbl.setText("No active mark")
        self.mark_status_lbl.setStyleSheet("color:#5a8a5a;")
        self._update_timeline_limits()
        self._render_current_frame()

    def _extract_patient_name(self, report):
        return (report.get('patient_details', {}).get('name')
                or report.get('patient_name')
                or report.get('patient', {}).get('name')
                or 'Unknown')

    def _extract_report_date(self, report):
        return (report.get('patient_details', {}).get('report_date')
                or report.get('report_date')
                or report.get('date')
                or 'Unknown Date')

    def _update_patient_info(self):
        if not self.current_report:
            self.patient_lbl.setText("Patient: —")
            self.patient_meta_lbl.setText("ID: — | Age: — | Gender: —")
            return
        pd        = self.current_report.get('patient_details', {})
        p_fallback = self.current_report.get('patient', {})
        name   = pd.get('name')   or self.current_report.get('patient_name') or p_fallback.get('name')   or 'Unknown'
        pid    = pd.get('report_id') or pd.get('user_id') or self.current_report.get('patient_id') or '—'
        age    = pd.get('age')    or self.current_report.get('age')    or p_fallback.get('age')    or '—'
        gender = pd.get('gender') or self.current_report.get('gender') or p_fallback.get('gender') or '—'
        self.patient_lbl.setText(f"Patient: {name}")
        self.patient_meta_lbl.setText(f"ID: {pid} | Age: {age} | Gender: {gender}")

    def _load_lead_data(self):
        self.lead_data = {lead: np.array([]) for lead in self.LEADS}
        rpt = self.current_report or {}
        self.sampling_rate = float(
            rpt.get('data_details', {}).get('sampling_rate')
            or rpt.get('sampling_rate')
            or rpt.get('ecg_data', {}).get('sampling_rate')
            or 500)
        ecg_data  = rpt.get('ecg_data', {}) if isinstance(rpt.get('ecg_data', {}), dict) else {}
        leads_data = ecg_data.get('leads_data') if isinstance(ecg_data.get('leads_data'), dict) else None
        if leads_data:
            for lead in self.LEADS:
                arr = leads_data.get(lead, [])
                self.lead_data[lead] = np.array(arr, dtype=float) if isinstance(arr, list) else np.array([])
            return
        if any(lead in ecg_data for lead in self.LEADS):
            for lead in self.LEADS:
                arr = ecg_data.get(lead, [])
                self.lead_data[lead] = np.array(arr, dtype=float) if isinstance(arr, list) else np.array([])
            return
        if any(lead in rpt for lead in self.LEADS):
            for lead in self.LEADS:
                arr = rpt.get(lead, [])
                self.lead_data[lead] = np.array(arr, dtype=float) if isinstance(arr, list) else np.array([])
            return
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
        rpt     = self.current_report or {}
        metrics = rpt.get('result_reading') or rpt.get('metrics') or {}
        self.metrics_table.setRowCount(0)
        rv5_sv1      = metrics.get('RV5_SV1',      metrics.get('rv5_sv1', 'N/A'))
        rv5_plus_sv1 = metrics.get('RV5_plus_SV1', metrics.get('rv5_plus_sv1', 'N/A'))
        if isinstance(rv5_sv1, (list, tuple)) and len(rv5_sv1) >= 2:
            try:
                rv5_sv1 = f"{float(rv5_sv1[0]):.3f}/{abs(float(rv5_sv1[1])):.3f}"
            except Exception:
                pass
        items = [
            ("HR",     metrics.get('HR_bpm',  metrics.get('heart_rate', metrics.get('HR', 'N/A'))),    "bpm"),
            ("RR",     metrics.get('RR_ms',   metrics.get('rr_interval', metrics.get('RR', 'N/A'))),   "ms"),
            ("PR",     metrics.get('PR_ms',   metrics.get('pr_interval', metrics.get('PR', 'N/A'))),   "ms"),
            ("QRS",    metrics.get('QRS_ms',  metrics.get('qrs_duration', metrics.get('QRS', 'N/A'))), "ms"),
            ("QT",     metrics.get('QT_ms',   metrics.get('qt_interval', metrics.get('QT', 'N/A'))),   "ms"),
            ("QTc",    metrics.get('QTc_ms',  metrics.get('qtc_interval', metrics.get('QTc', 'N/A'))), "ms"),
            ("RV5/SV1",  str(rv5_sv1).replace(' mV', ''),      "mV"),
            ("RV5+SV1",  str(rv5_plus_sv1).replace(' mV', ''), "mV"),
        ]
        self.metrics_table.setRowCount(len(items))
        for i, (k, v, unit) in enumerate(items):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(k))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(
                f"{v} {unit}" if v not in ('', None, 'N/A') else 'N/A'))

        findings_lines = []
        clinical = rpt.get('clinical_findings', {})
        if isinstance(clinical, dict):
            for key in ('conclusion', 'arrhythmia', 'hyperkalemia'):
                vals = clinical.get(key, [])
                if isinstance(vals, list) and vals:
                    findings_lines.append(f"{key.title()}: " + ', '.join(str(x) for x in vals))
        for key in ('conclusion', 'arrhythmia', 'hyperkalemia', 'findings', 'recommendations'):
            vals = rpt.get(key)
            if isinstance(vals, list) and vals:
                findings_lines.append(f"{key.title()}: " + ', '.join(str(x) for x in vals))
        self.findings_text.setPlainText('\n'.join(findings_lines) or "No backend findings.")

    # ─────────────────────────────────────────────────────────────────────────
    #  FRAME NAVIGATION  (identical to original)
    # ─────────────────────────────────────────────────────────────────────────
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
        self.frame_start_sample = min(
            self._max_start_sample(), self.frame_start_sample + self._step_samples())
        self.timeline.setValue(self.frame_start_sample)

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            self.play_timer.start(250)
            self.play_btn.setText("⏸ Pause")

    # ─────────────────────────────────────────────────────────────────────────
    #  RENDERING
    # ─────────────────────────────────────────────────────────────────────────
    def _render_current_frame(self):
        ws = self._window_samples()
        st = self.frame_start_sample
        en = min(self._total_samples(), st + ws)

        t          = np.arange(st, en) / self.sampling_rate if en > st else np.array([])
        start_sec  = st / self.sampling_rate if self.sampling_rate > 0 else 0.0
        end_sec    = en / self.sampling_rate if self.sampling_rate > 0 else 0.0
        self.frame_label.setText(f"Frame: {start_sec:.2f}s – {end_sec:.2f}s")

        ECG_COLOR   = '#00d000'   # Classic green trace
        GRID_COLOR  = '#1a3a1a'
        ANNOT_COLOR = '#ff3333'

        for lead in self.LEADS:
            ax     = self._lead_axes.get(lead)
            fig    = self._lead_figs.get(lead)
            canvas = self._lead_canvases.get(lead)
            if ax is None:
                continue

            ax.clear()
            ax.set_axis_off()
            ax.set_facecolor('#080e08')

            if lead == 'aVR':
                ax.set_ylim(-4096, 0)
            else:
                ax.set_ylim(0, 4096)
            ax.set_xlim(start_sec, max(end_sec, start_sec + 1))

            # ECG paper grid
            ax.grid(True, color=GRID_COLOR, linewidth=0.35, linestyle='-', alpha=1.0)

            data = self.lead_data.get(lead, np.array([]))
            if len(data) > 0 and en > st:
                seg = data[st:en]
                if lead == 'aVR':
                    ax.plot(t, -seg, color=ECG_COLOR, linewidth=0.9, antialiased=True)
                    ax.set_ylim(-4096, 0)
                else:
                    ax.plot(t, seg, color=ECG_COLOR, linewidth=0.9, antialiased=True)
                    ax.set_ylim(0, 4096)
                ax.set_xlim(start_sec, end_sec)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color='#334433', fontsize=9)

            # Annotation spans
            for ann in self.manual_annotations:
                if ann.get('lead', 'All Leads') not in ('All Leads', lead):
                    continue
                a0 = ann.get('start_sec', 0.0)
                a1 = ann.get('end_sec',   0.0)
                if a1 < start_sec or a0 > end_sec:
                    continue
                lft = max(a0, start_sec)
                rgt = min(a1, end_sec)
                if rgt > lft:
                    ax.axvspan(lft, rgt, color=ANNOT_COLOR, alpha=0.15)
                    # Label at top of span
                    ylim = ax.get_ylim()
                    ax.text((lft + rgt) / 2, ylim[1] * 0.92,
                             ann.get('type', '')[:12],
                             fontsize=5.5, color='#ff8888', ha='center',
                             va='top', zorder=10,
                             bbox=dict(boxstyle='round,pad=0.1',
                                       fc='#200000', alpha=0.6, ec='none'))

            fig.tight_layout(pad=0)
            canvas.draw_idle()

    def lead_has_visible_data(self, lead_name):
        data = self.lead_data.get(lead_name, np.array([]))
        if len(data) == 0:
            return False
        st = self.frame_start_sample
        en = min(len(data), st + self._window_samples())
        return en > st and len(data[st:en]) > 0

    # ─────────────────────────────────────────────────────────────────────────
    #  MANUAL ANNOTATIONS  (identical to original)
    # ─────────────────────────────────────────────────────────────────────────
    def mark_start(self):
        self.pending_mark_start_sec = self.frame_start_sample / max(self.sampling_rate, 1.0)
        self.mark_end_btn.setEnabled(True)
        self.mark_status_lbl.setText(
            f"✅ Start @ {self.pending_mark_start_sec:.2f}s  →  navigate to end → ② Mark End")
        self.mark_status_lbl.setStyleSheet("color:#f5c518;")

    def mark_end_and_save(self):
        if self.pending_mark_start_sec is None:
            QMessageBox.information(self, "Marking", "Click 'Mark Start' first.")
            return
        end_sec   = (self.frame_start_sample + self._window_samples()) / max(self.sampling_rate, 1.0)
        start_sec = min(self.pending_mark_start_sec, end_sec)
        end_sec   = max(self.pending_mark_start_sec, end_sec)
        arr_type  = self.arrhythmia_type_combo.currentText().strip()
        if arr_type == 'Other':
            arr_type = self.manual_type_input.text().strip() or 'Other'
        ann = {
            'start_sec':  round(start_sec, 3),
            'end_sec':    round(end_sec, 3),
            'type':       arr_type,
            'lead':       self.mark_lead_combo.currentText(),
            'notes':      self.notes_input.text().strip(),
            'created_at': datetime.now().isoformat(timespec='seconds')
        }
        self.manual_annotations.append(ann)
        self.pending_mark_start_sec = None
        self.mark_end_btn.setEnabled(False)
        self.mark_status_lbl.setText(f"✅ Saved: {arr_type}")
        self.mark_status_lbl.setStyleSheet("color:#7adf7a;")
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
            self.annotation_table.setItem(i, 0, QTableWidgetItem(f"{ann.get('start_sec',0):.3f}"))
            self.annotation_table.setItem(i, 1, QTableWidgetItem(f"{ann.get('end_sec',0):.3f}"))
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

    # ─────────────────────────────────────────────────────────────────────────
    #  AUTOMATIC DETECTION  (identical to original)
    # ─────────────────────────────────────────────────────────────────────────
    def run_automatic_detection(self):
        if not self.current_report:
            QMessageBox.warning(self, "Detection", "No report loaded.")
            return
        self.auto_detect_btn.setText("⏳ Analyzing…")
        self.auto_detect_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            lead_name = self.mark_lead_combo.currentText()
            if lead_name == "All Leads":
                lead_name = 'II'
            data = self.lead_data.get(lead_name, np.array([]))
            if len(data) == 0:
                for l in self.LEADS:
                    if len(self.lead_data.get(l, [])) > 0:
                        lead_name = l; data = self.lead_data[l]; break
            if len(data) == 0:
                QMessageBox.warning(self, "Detection", "No ECG data available."); return

            ws      = self._window_samples()
            st      = self.frame_start_sample
            en      = min(len(data), st + ws)
            segment = data[st:en]

            if len(segment) < self.sampling_rate * 1.5:
                QMessageBox.warning(self, "Detection",
                                    "Window too short for detection (need >1.5s)."); return
            if PQRSTAnalyzer is None or ArrhythmiaDetector is None:
                QMessageBox.critical(self, "Error",
                                     "ECG analysis modules not loaded."); return

            analyzer = PQRSTAnalyzer(self.sampling_rate)
            analysis = analyzer.analyze_signal(segment)
            detector = ArrhythmiaDetector(self.sampling_rate)
            results  = detector.detect_arrhythmias(segment, analysis)

            if not results or (len(results) == 1 and "Insufficient data" in results[0]):
                QMessageBox.information(self, "Detection",
                                        "No specific arrhythmia detected in this window.")
                return

            rhythm_text = ", ".join(results)
            is_normal   = "Normal Sinus Rhythm" in rhythm_text
            msg = (f"<b>Window Analysis (Lead {lead_name}):</b><br><br>"
                   f"Detected: <span style='color:{'#2ecc71' if is_normal else '#e74c3c'};"
                   f"font-weight:bold;'>{rhythm_text}</span><br><br>"
                   "Add these findings to report?")
            reply = QMessageBox.question(self, "Detection Result", msg,
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                start_sec = st / self.sampling_rate
                end_sec   = en / self.sampling_rate
                added = 0
                for arr in results:
                    if arr == "Normal Sinus Rhythm":
                        continue
                    ann = {
                        'start_sec':  round(start_sec, 3),
                        'end_sec':    round(end_sec, 3),
                        'type':       arr,
                        'lead':       lead_name,
                        'notes':      "Auto detected",
                        'created_at': datetime.now().isoformat(timespec='seconds')
                    }
                    self.manual_annotations.append(ann)
                    added += 1
                if added:
                    self._refresh_annotation_table()
                    self._persist_annotations_in_report()
                    self._render_current_frame()
                    QMessageBox.information(self, "Done",
                                            f"Added {added} arrhythmia annotation(s).")
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
        finally:
            self.auto_detect_btn.setText("🤖 Auto Detect")
            self.auto_detect_btn.setEnabled(True)

    # ─────────────────────────────────────────────────────────────────────────
    #  API FETCH  (identical to original)
    # ─────────────────────────────────────────────────────────────────────────
    def fetch_api_report(self):
        id_text = self.api_id_input.text().strip()
        if not id_text:
            return
        url = f"https://deckmount.in/ankur_bhaiya.php?id={id_text}"
        import requests
        from scipy.ndimage import gaussian_filter1d
        try:
            self.api_fetch_btn.setText("…")
            QApplication.processEvents()
            resp     = requests.get(url, timeout=10)
            data     = resp.json()
            if not data.get("status"):
                QMessageBox.warning(self, "API Error", "ID not found")
                self.api_fetch_btn.setText("Fetch"); return
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
                ["lead1_reading","lead_1_reading","lead1"],
                ["lead2_reading","lead_2_reading","lead2"],
                ["lead3_reading","lead_3_reading","lead3"],
                ["leadavr_reading","lead_avr_reading","leadavr"],
                ["leadavl_reading","lead_avl_reading","leadavl"],
                ["leadavf_reading","lead_avf_reading","leadavf"],
                ["leadv1_reading","lead_v1_reading","leadv1"],
                ["leadv2_reading","lead_v2_reading","leadv2"],
                ["leadv3_reading","lead_v3_reading","leadv3"],
                ["leadv4_reading","lead_v4_reading","leadv4"],
                ["leadv5_reading","lead_v5_reading","leadv5"],
                ["leadv6_reading","lead_v6_reading","leadv6"],
            ]
            lower_keys  = {k.lower(): k for k in api_data.keys()}
            leads_list  = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
            for i, variants in enumerate(possible_keys):
                leadstr = leads_list[i]
                for variant in variants:
                    actual_key = lower_keys.get(variant.lower())
                    if actual_key and actual_key in api_data:
                        val_str = str(api_data[actual_key]).strip()
                        if val_str:
                            if val_str.endswith(','): val_str = val_str[:-1]
                            arr_vals = np.array(
                                [float(x.strip()) for x in val_str.split(',') if x.strip()])
                            filt   = gaussian_filter1d(arr_vals, sigma=1.5)
                            c_mean = np.mean(filt)
                            if not np.isnan(c_mean):
                                filt = filt - c_mean
                            filt = filt + 2048
                            ecg_data[leadstr] = filt.tolist()
                        break

            new_report = {
                "patient_details": {
                    "name":        api_data.get("name", "Unknown"),
                    "age":         api_data.get("age", ""),
                    "gender":      api_data.get("gender", ""),
                    "report_id":   api_data.get("report_id", id_text),
                    "report_date": api_data.get("report_date", ""),
                },
                "result_reading": res_reading,
                "clinical_findings": {"conclusion": concl, "arrhythmia": arr},
                "ecg_data": ecg_data,
                "api_id": id_text,
            }
            self.reports.append(new_report)
            idx  = len(self.reports) - 1
            name = api_data.get("name", "Unknown API")
            self.report_combo.addItem(f"[API] {name} | ID:{id_text}", "")
            self.report_combo.setCurrentIndex(idx)
            self.api_fetch_btn.setText("Fetch")
        except Exception as e:
            self.api_fetch_btn.setText("Fetch")
            QMessageBox.critical(self, "API Error", f"Failed: {str(e)}")

    # ─────────────────────────────────────────────────────────────────────────
    #  EXPORT / PDF  (identical to original)
    # ─────────────────────────────────────────────────────────────────────────
    def export_report(self):
        if not self.current_report:
            QMessageBox.warning(self, "Export", "No report selected"); return
        default_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", default_name, "JSON (*.json)")
        if not path: return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.current_report, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Export", f"Exported:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def generate_pdf_report(self):
        if not self.current_report:
            QMessageBox.warning(self, "Export", "No report loaded."); return

        rpt = self.current_report
        pat = rpt.get('patient_details', {}) or {}

        raw_metrics = rpt.get('result_reading') or rpt.get('metrics') or {}
        if isinstance(raw_metrics, str):
            try: raw_metrics = json.loads(raw_metrics)
            except: raw_metrics = {}
        if not isinstance(raw_metrics, dict): raw_metrics = {}

        def _get(d, *keys):
            for k in keys:
                v = d.get(k)
                if v is not None and str(v).strip() not in ('', '--', 'N/A', 'null'):
                    return str(v)
            return None

        hr   = _get(raw_metrics, 'HR',  'heart_rate',   'HR_bpm')
        pr   = _get(raw_metrics, 'PR',  'pr_interval',  'PR_ms')
        qrs  = _get(raw_metrics, 'QRS', 'qrs_duration', 'QRS_ms')
        qt   = _get(raw_metrics, 'QT',  'qt_interval',  'QT_ms')
        qtc  = _get(raw_metrics, 'QTc', 'qtc_interval', 'QTc_ms')
        qtcf = _get(raw_metrics, 'QTcF','qtcf_interval','QTcF_ms')
        rr   = _get(raw_metrics, 'RR',  'rr_interval',  'RR_ms')
        rv5sv1  = _get(raw_metrics, 'RV5_SV1',     'rv5_sv1')
        rv5plus = _get(raw_metrics, 'RV5_plus_SV1','rv5_plus_sv1')
        if isinstance(rv5sv1, (list, tuple)) and len(rv5sv1) >= 2:
            try:
                rv5sv1 = f"{float(rv5sv1[0]):.3f}/{abs(float(rv5sv1[1])):.3f}"
            except Exception:
                pass
        axes_s  = _get(raw_metrics, 'axes','P/QRS/T','p_qrs_t')

        clinical   = rpt.get('clinical_findings') or {}
        conclusions = clinical.get('conclusion', []) if isinstance(clinical, dict) else []
        if isinstance(conclusions, str): conclusions = [conclusions]
        elif not isinstance(conclusions, list): conclusions = []
        if not conclusions:
            c2 = rpt.get('conclusion', [])
            conclusions = [c2] if isinstance(c2, str) else (c2 if isinstance(c2, list) else [])

        patient_name = pat.get('name', 'Unknown')
        timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
        project_root = Path(__file__).resolve().parents[2]
        reports_dir  = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ECG PDF",
            str(reports_dir / f"ECG_Analysis_{patient_name}_{timestamp}.pdf"),
            "PDF Files (*.pdf)")
        if not path: return

        from PyQt5.QtWidgets import (
            QDialog as _QD, QVBoxLayout as _VL, QButtonGroup,
            QRadioButton, QDialogButtonBox, QLabel as _QL
        )
        fmt_dlg = _QD(self)
        fmt_dlg.setWindowTitle("Report Format")
        fmt_dlg.setMinimumWidth(360)
        fmt_dlg.setStyleSheet(
            "QDialog{background:#12152a;color:white;} QLabel{color:#e0e0ff;font-size:12px;"
            "background:transparent;border:none;} QRadioButton{color:#e0e0ff;font-size:11px;"
            "background:transparent;padding:5px;} QPushButton{background:#0097e6;color:white;"
            "border:none;border-radius:5px;padding:7px 20px;font-weight:bold;}")
        fmt_lay = _VL(fmt_dlg)
        fmt_lay.setContentsMargins(22, 18, 22, 18)
        fmt_lay.addWidget(_QL("Choose ECG report layout:"))
        rb1 = QRadioButton("4:3  —  Standard 12-lead grid")
        rb2 = QRadioButton("12:1 —  Full rhythm strip roll")
        rb3 = QRadioButton("6:2  —  Compact comparative")
        rb1.setChecked(True)
        grp = QButtonGroup(fmt_dlg)
        for rb in (rb1, rb2, rb3):
            grp.addButton(rb); fmt_lay.addWidget(rb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(fmt_dlg.accept)
        bb.rejected.connect(fmt_dlg.reject)
        fmt_lay.addWidget(bb)
        if fmt_dlg.exec_() != _QD.Accepted: return
        pdf_format = "4_3" if rb1.isChecked() else ("12_1" if rb2.isChecked() else "6_2")

        try:
            self.pdf_btn.setText("Generating…")
            QApplication.processEvents()

            st = self.frame_start_sample
            leads_order = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
            snap_raw = []
            for l in leads_order:
                d = self.lead_data.get(l, np.array([]))
                snap_raw.append(d[st:] if len(d) > st else np.array([]))

            frozen = {
                'HR': int(float(hr) if hr else 0),
                'RR': int(float(rr) if rr else 0),
                'PR': int(float(pr) if pr else 0),
                'QRS': int(float(qrs) if qrs else 0),
                'QT': int(float(qt) if qt else 0),
                'QTc': int(float(qtc) if qtc else 0),
                'QTcF': int(float(qtcf) if qtcf else 0),
                'rv5': 0.0, 'sv1': 0.0,
                'p_axis': '--', 'QRS_axis': '--', 't_axis': '--',
                'lead_seq': 'Standard',
                'logo_path': str(self.analysis_pdf_logo_path),
            }
            try:
                if rv5sv1:
                    parts = str(rv5sv1).split('/')
                    frozen['rv5'] = float(parts[0].strip(' mV+'))
                    if len(parts) > 1:
                        frozen['sv1'] = float(parts[1].strip(' mV+'))
            except: pass
            try:
                if axes_s and len(str(axes_s).split('/')) == 3:
                    p = str(axes_s).split('/')
                    frozen['p_axis'] = p[0].strip()
                    frozen['QRS_axis'] = p[1].strip()
                    frozen['t_axis'] = p[2].strip()
            except: pass

            pat_mapped = {
                'first_name': pat.get('name', 'Unknown'), 'last_name': '',
                'age': pat.get('age', ''), 'gender': pat.get('gender', ''),
                'date_time': pat.get('report_date') or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'doctor_name': pat.get('doctor', ''),
                'org': pat.get('Org.', ''),
                'phone': pat.get('phone', '') or pat.get('doctor_mobile', ''),
            }
            extra_figs = []
            if self.manual_annotations:
                fig2 = self._generate_annotation_page()
                if fig2: extra_figs.append(fig2)

            from ecg.ecg_report_android import generate_report as _gen
            _gen(snap_raw=snap_raw, frozen=frozen, patient=pat_mapped,
                 filename=path, fmt=pdf_format, conc_list=conclusions,
                 fs=float(self.sampling_rate), extra_figs=extra_figs)

            try:
                from dashboard.history_window import append_history_entry
                append_history_entry({
                    "patient_name": pat.get('name','Unknown'),
                    "age": str(pat.get('age','')),
                    "gender": pat.get('gender',''),
                    "doctor": pat.get('doctor',''),
                    "Org.": pat.get('Org.',''),
                }, path, report_type="Analysis")
            except Exception as h_err:
                print(f"History append failed: {h_err}")

            self.pdf_btn.setText("📄 PDF Report")
            QMessageBox.information(self, "PDF Saved", f"Report saved:\n{path}")
        except Exception as e:
            self.pdf_btn.setText("📄 PDF Report")
            QMessageBox.critical(self, "PDF Error", f"Failed:\n{e}")

    def _generate_annotation_page(self):
        """Annotation summary + wave strips (identical to original)."""
        if not self.manual_annotations:
            return None
        import matplotlib.patches as mpa
        PAGE_W = 210.0; PAGE_H = 297.0
        ML = 15.0; MR = 15.0; MT = 15.0; MB = 15.0
        fig = Figure(figsize=(PAGE_W/25.4, PAGE_H/25.4), dpi=150, facecolor='white')
        ax  = fig.add_axes([0,0,1,1], facecolor='white')
        ax.set_xlim(0, PAGE_W); ax.set_ylim(PAGE_H, 0); ax.set_aspect('equal')
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False, which='both')
        ax.text(PAGE_W/2, MT+5, "ECG ARRHYTHMIA & FINDINGS REPORT",
                fontsize=12, fontweight='bold', ha='center', va='top', color='#0000cc')
        y_cursor = MT+20
        ax.text(ML, y_cursor, "Summary of Manual & Automatic Annotations:",
                fontsize=9, fontweight='bold', va='top')
        y_cursor += 8
        cols = [("Start (s)",20),("End (s)",20),("Type",50),("Lead",20),("Notes",60)]
        xc = ML
        for lbl, w in cols:
            ax.text(xc, y_cursor, lbl, fontsize=8, fontweight='bold', va='top'); xc += w
        y_cursor += 5
        ax.plot([ML, PAGE_W-MR], [y_cursor]*2, color='black', linewidth=0.5)
        y_cursor += 2
        for ann in self.manual_annotations[:12]:
            xc = ML
            ax.text(xc, y_cursor, f"{ann.get('start_sec',0):.2f}", fontsize=7, va='top'); xc += 20
            ax.text(xc, y_cursor, f"{ann.get('end_sec',0):.2f}", fontsize=7, va='top');   xc += 20
            ax.text(xc, y_cursor, ann.get('type',''), fontsize=7, fontweight='bold', va='top'); xc += 50
            ax.text(xc, y_cursor, ann.get('lead',''), fontsize=7, va='top');                    xc += 20
            ax.text(xc, y_cursor, ann.get('notes',''), fontsize=7, va='top')
            y_cursor += 5
            if y_cursor > PAGE_H/2-10: break
        y_cursor = max(y_cursor+10, PAGE_H/2-20)
        ax.text(ML, y_cursor, "Waveform Strips for Detected Events:",
                fontsize=9, fontweight='bold', va='top')
        y_cursor += 10
        important = [a for a in self.manual_annotations if "Rhythm" not in str(a.get('type',''))]
        if not important: important = self.manual_annotations
        ADC_PER_MM = 128.0
        MM_PER_SAMPLE = 25.0 / float(self.sampling_rate)
        for i, ann in enumerate(important[:3]):
            if y_cursor > PAGE_H-50: break
            lead_name = ann.get('lead','II')
            if lead_name == "All Leads": lead_name = 'II'
            data = self.lead_data.get(lead_name, np.array([]))
            if len(data) == 0: continue
            start_s = ann.get('start_sec', 0); end_s = ann.get('end_sec', 0)
            duration_s = end_s - start_s
            strip_w_mm = PAGE_W - ML - MR
            time_shown_s = min(10.0, max(3.0, duration_s*1.5))
            center_s  = (start_s+end_s)/2
            strip_start_s = max(0, center_s - time_shown_s/2)
            strip_end_s   = strip_start_s + time_shown_s
            st_idx = int(strip_start_s*self.sampling_rate)
            en_idx = int(strip_end_s*self.sampling_rate)
            segment = data[st_idx:en_idx]
            if len(segment) < 10: continue
            strip_h = 30.0
            rect = mpa.FancyBboxPatch((ML, y_cursor), strip_w_mm, strip_h,
                                       boxstyle="square,pad=0", linewidth=0.5,
                                       edgecolor='#e09696', facecolor='#fff5f5')
            ax.add_patch(rect)
            for gy in np.arange(y_cursor, y_cursor+strip_h, 5):
                ax.plot([ML, ML+strip_w_mm], [gy,gy], color='#f5d8d8', linewidth=0.2, zorder=1)
            for gx in np.arange(ML, ML+strip_w_mm, 5):
                ax.plot([gx,gx], [y_cursor, y_cursor+strip_h], color='#f5d8d8', linewidth=0.2, zorder=1)
            baseline = np.median(segment)
            seg_mm   = (segment - baseline) / ADC_PER_MM
            wx_mm    = ML + np.arange(len(segment))*MM_PER_SAMPLE
            mask     = wx_mm <= (ML+strip_w_mm)
            wx_mm    = wx_mm[mask]
            wy_mm    = y_cursor + strip_h/2 - seg_mm[:len(wx_mm)]
            ax.plot(wx_mm, wy_mm, color='black', linewidth=0.5, zorder=2)
            hl_start_mm = ML + (start_s-strip_start_s)*25.0
            hl_end_mm   = ML + (end_s  -strip_start_s)*25.0
            if hl_start_mm < ML+strip_w_mm and hl_end_mm > ML:
                hl_s = max(ML, hl_start_mm); hl_e = min(ML+strip_w_mm, hl_end_mm)
                ax.axvspan(hl_s, hl_e, color='#ff0000', alpha=0.1,
                           ymin=1-(y_cursor+strip_h)/PAGE_H, ymax=1-y_cursor/PAGE_H)
            ax.text(ML+2, y_cursor+3, f"Event {i+1}: {ann.get('type','')} (Lead {lead_name})",
                    fontsize=8, fontweight='bold', color='black', va='top', zorder=3)
            ax.text(ML+2, y_cursor+strip_h-2, f"Time: {start_s:.2f}s – {end_s:.2f}s",
                    fontsize=6, color='#555', va='bottom', zorder=3)
            y_cursor += strip_h + 10
        ax.text(PAGE_W/2, PAGE_H-MB+5,
                "Deckmount Electronics Pvt Ltd | RhythmPro ECG | Made in India",
                fontsize=6, ha='center', va='top', color='#333', zorder=9)
        return fig
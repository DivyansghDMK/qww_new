"""
Lorenz Plot (Poincaré Plot) for ECG RR Interval Analysis
=========================================================
Shows RR(n) vs RR(n+1) scatter plot — the clinical standard for
visualizing HRV and detecting arrhythmia patterns.

Interpretation:
  - Tight comet shape  → Normal sinus rhythm
  - Wide/round cloud   → High HRV or atrial fibrillation
  - Multiple clusters  → Ectopic beats (PVCs/PACs)
  - Vertical fan       → Bigeminy
  - Horizontal fan     → Trigeminy

SD1  = short-term HRV (beat-to-beat variability)
SD2  = long-term HRV  (overall variability)
SD1/SD2 ratio < 0.5 → regular rhythm
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QGridLayout
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# ── RR extraction ─────────────────────────────────────────────────────────────

def extract_rr_intervals(signal: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """Extract RR intervals (ms) from raw ECG signal."""
    if len(signal) < int(fs * 2):
        return np.array([])
    try:
        nyq = fs / 2.0
        b, a = butter(2, [0.5 / nyq, min(40.0 / nyq, 0.99)], btype="band")
        filt = filtfilt(b, a, signal.astype(float))
        mean_v, std_v = np.mean(filt), np.std(filt)
        peaks, _ = find_peaks(
            filt,
            height=mean_v + 0.3 * std_v,
            distance=int(0.25 * fs),
            prominence=std_v * 0.3,
        )
        if len(peaks) < 3:
            return np.array([])
        rr = np.diff(peaks) * (1000.0 / fs)
        return rr[(rr >= 200) & (rr <= 2000)]
    except Exception:
        return np.array([])


# ── Lorenz metrics ────────────────────────────────────────────────────────────

def compute_lorenz_metrics(rr: np.ndarray):
    """
    Compute SD1, SD2 and derived HRV metrics from RR intervals.

    Returns dict with keys: SD1, SD2, ratio, RMSSD, SDNN, mean_hr, n_beats
    """
    if len(rr) < 4:
        return None
    rr1 = rr[:-1]
    rr2 = rr[1:]
    diff = rr2 - rr1
    sd1 = float(np.std(diff) / np.sqrt(2))
    sd2 = float(np.sqrt(2 * np.var(rr) - np.var(diff) / 2))
    rmssd = float(np.sqrt(np.mean(diff ** 2)))
    sdnn = float(np.std(rr))
    mean_rr = float(np.mean(rr))
    mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0
    return {
        "SD1": round(sd1, 1),
        "SD2": round(sd2, 1),
        "ratio": round(sd1 / sd2, 3) if sd2 > 0 else 0.0,
        "RMSSD": round(rmssd, 1),
        "SDNN": round(sdnn, 1),
        "mean_hr": round(mean_hr, 1),
        "n_beats": len(rr),
    }


def _interpret(metrics: dict) -> str:
    if metrics is None:
        return "Insufficient data"
    sd1, sd2, ratio = metrics["SD1"], metrics["SD2"], metrics["ratio"]
    if sd1 < 5 and sd2 < 20:
        return "Very low HRV — possible stress or autonomic dysfunction"
    if ratio > 0.8:
        return "Irregular rhythm — possible Atrial Fibrillation"
    if sd1 > 50:
        return "High short-term variability — possible ectopic beats"
    if ratio < 0.25:
        return "Regular rhythm — normal sinus"
    return "Normal HRV pattern"


# ── Plot helper ───────────────────────────────────────────────────────────────

def _confidence_ellipse(ax, x, y, n_std=2.0, color="#e74c3c", alpha=0.25):
    """Draw a confidence ellipse (SD1/SD2 axes) on ax."""
    if len(x) < 4:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]) + 1e-9)
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=rx * 2,
        height=ry * 2,
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=1.5,
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    t = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(t + ax.transData)
    ax.add_patch(ellipse)


# ── Dialog ────────────────────────────────────────────────────────────────────

class LorenzPlotDialog(QDialog):
    """
    Lorenz (Poincaré) plot dialog.

    Usage:
        dlg = LorenzPlotDialog(ecg_signal, fs=500, lead_name="II", parent=self)
        dlg.exec_()
    """

    def __init__(self, ecg_signal: np.ndarray, fs: float = 500.0,
                 lead_name: str = "II", parent=None):
        super().__init__(parent)
        self.ecg_signal = np.asarray(ecg_signal, dtype=float)
        self.fs = fs
        self.lead_name = lead_name
        self.setWindowTitle(f"Lorenz Plot — Lead {lead_name}")
        self.setMinimumSize(820, 600)
        self.resize(960, 680)
        self.setStyleSheet("QDialog { background: #f0f2f5; }")
        self._build_ui()
        self._plot()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # Header
        hdr = QFrame()
        hdr.setFixedHeight(50)
        hdr.setStyleSheet("QFrame { background: white; border-radius: 8px; border: 1px solid #e0e0e0; }")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(15, 0, 10, 0)
        title = QLabel(f"Lorenz (Poincaré) Plot — Lead {self.lead_name}")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; border: none; background: transparent;")
        hdr_lay.addWidget(title)
        hdr_lay.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedSize(80, 32)
        close_btn.setStyleSheet(
            "QPushButton { background:#34495e; color:white; border-radius:5px; font-weight:bold; }"
            "QPushButton:hover { background:#5d6d7e; }"
        )
        close_btn.clicked.connect(self.close)
        hdr_lay.addWidget(close_btn)
        root.addWidget(hdr)

        # Body: plot + metrics side by side
        body = QHBoxLayout()
        body.setSpacing(8)

        # Plot canvas
        plot_frame = QFrame()
        plot_frame.setStyleSheet("QFrame { background: white; border-radius: 8px; border: 1px solid #e0e0e0; }")
        pf_lay = QVBoxLayout(plot_frame)
        pf_lay.setContentsMargins(6, 6, 6, 6)
        self.fig = Figure(figsize=(6, 5), facecolor="white", dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pf_lay.addWidget(self.canvas)
        body.addWidget(plot_frame, 7)

        # Metrics panel
        metrics_frame = QFrame()
        metrics_frame.setMinimumWidth(220)
        metrics_frame.setStyleSheet("QFrame { background: white; border-radius: 8px; border: 1px solid #e0e0e0; }")
        mf_lay = QVBoxLayout(metrics_frame)
        mf_lay.setContentsMargins(14, 14, 14, 14)
        mf_lay.setSpacing(10)

        m_title = QLabel("HRV Metrics")
        m_title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        m_title.setStyleSheet("color: #2c3e50;")
        mf_lay.addWidget(m_title)

        self._metric_labels: dict[str, QLabel] = {}
        for key, display in [
            ("SD1",      "SD1 (ms)"),
            ("SD2",      "SD2 (ms)"),
            ("ratio",    "SD1/SD2"),
            ("RMSSD",    "RMSSD (ms)"),
            ("SDNN",     "SDNN (ms)"),
            ("mean_hr",  "Mean HR (bpm)"),
            ("n_beats",  "Beats analyzed"),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(f"{display}:")
            lbl.setFont(QFont("Segoe UI", 10))
            lbl.setStyleSheet("color: #7f8c8d;")
            val = QLabel("—")
            val.setFont(QFont("Segoe UI", 11, QFont.Bold))
            val.setStyleSheet("color: #2c3e50;")
            val.setAlignment(Qt.AlignRight)
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            mf_lay.addLayout(row)
            self._metric_labels[key] = val

        mf_lay.addSpacing(10)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ecf0f1;")
        mf_lay.addWidget(sep)

        interp_title = QLabel("Interpretation")
        interp_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        interp_title.setStyleSheet("color: #2c3e50;")
        mf_lay.addWidget(interp_title)

        self.interp_label = QLabel("—")
        self.interp_label.setFont(QFont("Segoe UI", 10))
        self.interp_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.interp_label.setWordWrap(True)
        mf_lay.addWidget(self.interp_label)

        mf_lay.addStretch()

        # Legend
        legend_title = QLabel("Legend")
        legend_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        legend_title.setStyleSheet("color: #7f8c8d;")
        mf_lay.addWidget(legend_title)
        for color, text in [
            ("#3498db", "RR(n) vs RR(n+1)"),
            ("#e74c3c", "SD1 axis (short-term)"),
            ("#27ae60", "SD2 axis (long-term)"),
        ]:
            row2 = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            dot.setFixedWidth(18)
            txt = QLabel(text)
            txt.setFont(QFont("Segoe UI", 9))
            txt.setStyleSheet("color: #7f8c8d;")
            row2.addWidget(dot)
            row2.addWidget(txt)
            row2.addStretch()
            mf_lay.addLayout(row2)

        body.addWidget(metrics_frame, 3)
        root.addLayout(body, 1)

    # ── Plot ──────────────────────────────────────────────────────────────────

    def _plot(self):
        rr = extract_rr_intervals(self.ecg_signal, self.fs)
        metrics = compute_lorenz_metrics(rr)

        self.ax.clear()
        self.ax.set_facecolor("#fafafa")
        self.ax.set_xlabel("RR(n)  [ms]", fontsize=11, color="#34495e")
        self.ax.set_ylabel("RR(n+1)  [ms]", fontsize=11, color="#34495e")
        self.ax.set_title(
            f"Lorenz Plot — Lead {self.lead_name}",
            fontsize=13, fontweight="bold", color="#2c3e50"
        )
        self.ax.grid(True, linestyle="--", linewidth=0.5, color="#bdc3c7", alpha=0.7)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        if len(rr) < 4:
            self.ax.text(
                0.5, 0.5,
                "Not enough RR intervals\n(need ≥ 4 beats)",
                transform=self.ax.transAxes,
                ha="center", va="center",
                fontsize=13, color="#7f8c8d",
            )
            self.canvas.draw_idle()
            self.interp_label.setText("Insufficient data")
            return

        rr1, rr2 = rr[:-1], rr[1:]

        # Identity line (RR(n) = RR(n+1))
        lo, hi = min(rr.min(), rr.min()), max(rr.max(), rr.max())
        self.ax.plot([lo, hi], [lo, hi], color="#95a5a6", linewidth=1.0,
                     linestyle="--", alpha=0.6, label="Identity line")

        # Confidence ellipse
        _confidence_ellipse(self.ax, rr1, rr2, n_std=2.0,
                            color="#3498db", alpha=0.15)

        # Scatter
        self.ax.scatter(rr1, rr2, s=18, color="#3498db", alpha=0.65,
                        edgecolors="none", zorder=3, label="RR(n) vs RR(n+1)")

        # SD1 / SD2 axes through centroid
        if metrics:
            cx, cy = np.mean(rr1), np.mean(rr2)
            sd1, sd2 = metrics["SD1"], metrics["SD2"]
            # SD1 axis: perpendicular to identity (direction [-1,1]/√2)
            d1 = np.array([-1, 1]) / np.sqrt(2)
            self.ax.annotate(
                "", xy=(cx + d1[0] * sd1, cy + d1[1] * sd1),
                xytext=(cx - d1[0] * sd1, cy - d1[1] * sd1),
                arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=2.0),
            )
            self.ax.text(
                cx + d1[0] * sd1 * 1.1, cy + d1[1] * sd1 * 1.1,
                f"SD1={sd1:.0f}ms", color="#e74c3c", fontsize=9, fontweight="bold"
            )
            # SD2 axis: along identity (direction [1,1]/√2)
            d2 = np.array([1, 1]) / np.sqrt(2)
            self.ax.annotate(
                "", xy=(cx + d2[0] * sd2, cy + d2[1] * sd2),
                xytext=(cx - d2[0] * sd2, cy - d2[1] * sd2),
                arrowprops=dict(arrowstyle="<->", color="#27ae60", lw=2.0),
            )
            self.ax.text(
                cx + d2[0] * sd2 * 1.05, cy + d2[1] * sd2 * 1.05,
                f"SD2={sd2:.0f}ms", color="#27ae60", fontsize=9, fontweight="bold"
            )

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

        # Update metric labels
        if metrics:
            for key, lbl in self._metric_labels.items():
                val = metrics.get(key, "—")
                lbl.setText(str(val))
            interp = _interpret(metrics)
            self.interp_label.setText(interp)
            # Color code interpretation
            if "Fibrillation" in interp or "ectopic" in interp.lower():
                self.interp_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            elif "Normal" in interp or "regular" in interp.lower():
                self.interp_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            else:
                self.interp_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        else:
            self.interp_label.setText("Insufficient data")

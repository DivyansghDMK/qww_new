"""
HRV ECG Report Generator - Clean Rewrite
Matches Spandan/Deckmount style: pink ECG grid paper, landscape, 3-page report
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Table, TableStyle,
    Spacer, Image, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import (
    Drawing, Rect, Line, Path, String as GString, Group
)
from reportlab.pdfbase import pdfmetrics

# ─────────────────────────── CONSTANTS ────────────────────────────────────────
A4_L_W, A4_L_H = landscape(A4)          # 841.89 × 595.28 pt
MARGIN          = 18 * mm               # 18 mm margins
CONTENT_W       = A4_L_W - 2 * MARGIN   # usable width
CONTENT_H       = A4_L_H - 2 * MARGIN

ECG_PINK_BG     = colors.HexColor("#fff0f0")
ECG_MINOR_GRID  = colors.HexColor("#ffc8c8")
ECG_MAJOR_GRID  = colors.HexColor("#ff9999")
ECG_LINE_COLOR  = colors.HexColor("#000000")

DARK_HEADER_BG  = colors.HexColor("#1e2d40")   # deep navy
ACCENT_BLUE     = colors.HexColor("#6497b1")
GREY_CARD       = colors.HexColor("#f0f0f0")
BORDER_COLOR    = colors.HexColor("#cccccc")
TEXT_DARK       = colors.HexColor("#1a1a1a")

BOX_MM          = 5                            # 1 ECG small box = 5 mm
BOX_PT          = BOX_MM * mm                  # in points

# ─────────────────────────── HELPER: generate fake ECG for demo ───────────────
def _make_demo_ecg(n_samples=3000, fs=500, hr_bpm=60, noise=0.04):
    """Generate a realistic-looking synthetic ECG waveform."""
    t = np.linspace(0, n_samples / fs, n_samples)
    rr  = 60.0 / hr_bpm
    sig = np.zeros(n_samples)

    def _beat(t_rel):
        # P wave
        p = 0.12 * np.exp(-((t_rel - 0.10) ** 2) / (2 * 0.012 ** 2))
        # Q
        q = -0.05 * np.exp(-((t_rel - 0.17) ** 2) / (2 * 0.005 ** 2))
        # R
        r = 1.00 * np.exp(-((t_rel - 0.20) ** 2) / (2 * 0.006 ** 2))
        # S
        s = -0.15 * np.exp(-((t_rel - 0.23) ** 2) / (2 * 0.006 ** 2))
        # T
        t_w = 0.18 * np.exp(-((t_rel - 0.36) ** 2) / (2 * 0.020 ** 2))
        return p + q + r + s + t_w

    for i, ti in enumerate(t):
        t_rel = ti % rr
        if t_rel < 0.55:
            sig[i] = _beat(t_rel)

    sig += noise * np.random.randn(n_samples)
    # Convert to ADC-like values centred at 2000
    return 2000 + sig * 800

# ─────────────────────────── ECG GRID DRAWING ─────────────────────────────────
def make_ecg_strip(ecg_data, width_pt, height_pt,
                   minor_boxes_x=60, minor_boxes_y=20,
                   major_every=5,
                   wave_gain=10.0, adc_per_mv=800.0,
                   show_notch=True):
    """
    Return a ReportLab Drawing with a proper pink ECG paper grid + waveform.
    ecg_data: 1-D array of ADC values (centre ~2000) or None for grid-only.
    """
    d = Drawing(width_pt, height_pt)

    # ── Background ────────────────────────────────────────────────────────────
    d.add(Rect(0, 0, width_pt, height_pt,
               fillColor=ECG_PINK_BG, strokeColor=None))

    # ── Grid lines ────────────────────────────────────────────────────────────
    xs = minor_boxes_x
    ys = minor_boxes_y
    dx = width_pt  / xs
    dy = height_pt / ys

    # Minor
    for i in range(xs + 1):
        x = i * dx
        w = 0.3 if i % major_every else 0.0   # will be overdrawn by major
        d.add(Line(x, 0, x, height_pt,
                   strokeColor=ECG_MINOR_GRID, strokeWidth=0.35))
    for j in range(ys + 1):
        y = j * dy
        d.add(Line(0, y, width_pt, y,
                   strokeColor=ECG_MINOR_GRID, strokeWidth=0.35))

    # Major
    for i in range(0, xs + 1, major_every):
        x = i * dx
        d.add(Line(x, 0, x, height_pt,
                   strokeColor=ECG_MAJOR_GRID, strokeWidth=0.8))
    for j in range(0, ys + 1, major_every):
        y = j * dy
        d.add(Line(0, y, width_pt, y,
                   strokeColor=ECG_MAJOR_GRID, strokeWidth=0.8))

    cy = height_pt / 2.0   # centre baseline

    # ── Calibration notch (1 mV = 10 mm at 10mm/mV gain) ─────────────────────
    if show_notch:
        notch_w = 5 * mm         # 5 mm wide
        notch_h = wave_gain * mm  # 1 mV calibration
        nx = 3 * mm
        p = Path(fillColor=None, strokeColor=ECG_LINE_COLOR,
                 strokeWidth=0.8, strokeLineCap=1, strokeLineJoin=0)
        p.moveTo(nx, cy)
        p.lineTo(nx, cy + notch_h)
        p.lineTo(nx + notch_w, cy + notch_h)
        p.lineTo(nx + notch_w, cy)
        d.add(p)

    # ── ECG Waveform ──────────────────────────────────────────────────────────
    if ecg_data is not None and len(ecg_data) >= 2:
        adc   = np.array(ecg_data, dtype=float)
        adc   -= np.mean(adc)                      # remove baseline wander
        # scale: 1 mV (800 ADC) → wave_gain mm → wave_gain*mm points
        scale = (wave_gain * mm) / adc_per_mv      # pt per ADC unit
        y_pts = cy + adc * scale

        ecg_start_x = 10 * mm                      # leave room for notch
        tx = np.linspace(ecg_start_x, width_pt - 2 * mm, len(adc))

        # Clip to safe range
        y_pts = np.clip(y_pts, 2, height_pt - 2)

        # Draw as path (much faster than line-by-line)
        path = Path(fillColor=None, strokeColor=ECG_LINE_COLOR,
                    strokeWidth=0.5, strokeLineCap=1, strokeLineJoin=1)
        path.moveTo(tx[0], float(y_pts[0]))
        for i in range(1, len(tx)):
            path.lineTo(float(tx[i]), float(y_pts[i]))
        d.add(path)

    return d

# ─────────────────────────── CHART HELPERS ────────────────────────────────────
def _bar_chart_image(values, labels, title, ylabel, color='#6497b1',
                     figsize=(5, 3), dpi=120):
    """Return a BytesIO PNG of a bar chart."""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=color, edgecolor=color, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8, fontweight='bold')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    if len(values) > 0 and max(values) > 0:
        ax.set_ylim(0, max(values) * 1.18)
    # Value labels
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{int(round(v))}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf

def _radar_chart_image(values_norm, categories, figsize=(4, 4), dpi=120):
    """Return a BytesIO PNG of a radar (spider) chart, values 0-100."""
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals   = values_norm + values_norm[:1]
    angs   = angles + angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'),
                           facecolor='white')
    ax.plot(angs, vals, 'o-', linewidth=2, color='#228B22', markersize=4)
    ax.fill(angs, vals, alpha=0.25, color='#32CD32')
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.set_title('Radar chart', fontsize=9, fontweight='bold', pad=14)
    fig.tight_layout(pad=0.3)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf

def _psd_chart_image(lf, hf, figsize=(9, 2.8), dpi=120):
    """Return a BytesIO PNG of the frequency-domain PSD chart (LF/HF shaded)."""
    freqs = np.linspace(0, 0.5, 500)
    # Synthetic smooth PSD for display only
    lf_peak = 0.10
    hf_peak = 0.30
    psd = (
        lf * 18 * np.exp(-((freqs - lf_peak)**2) / (2*0.025**2)) +
        hf * 12 * np.exp(-((freqs - hf_peak)**2) / (2*0.040**2))
    )
    psd = np.maximum(psd, 0)

    fig, ax = plt.subplots(figsize=figsize, facecolor='#e8e8e8')
    ax.set_facecolor('#e8e8e8')
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    ax.fill_between(freqs, psd, where=lf_mask, color='#a8c4d8', alpha=0.85, label='LF')
    ax.fill_between(freqs, psd, where=hf_mask, color='#c8a8a8', alpha=0.85, label='HF')
    ax.plot(freqs, psd, color='#333333', linewidth=1.2)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, np.max(psd) * 1.25 if np.max(psd) > 0 else 1)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticklabels(['0Hz','0.1Hz','0.2Hz','0.3Hz','0.4Hz','0.5Hz'], fontsize=8)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.3)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='#e8e8e8')
    plt.close(fig)
    buf.seek(0)
    return buf

# ─────────────────────────── PAGE TEMPLATES ───────────────────────────────────
def _make_doc(filename):
    """Create BaseDocTemplate with landscape A4."""
    doc = BaseDocTemplate(
        filename,
        pagesize=landscape(A4),
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title="HRV ECG Report",
    )
    frame = Frame(MARGIN, MARGIN, CONTENT_W, CONTENT_H, id='main')
    doc.addPageTemplates([PageTemplate(id='main', frames=[frame])])
    return doc

# ─────────────────────────── HEADER DRAWING ───────────────────────────────────
def _header_drawing(info, logo_path=None, page_label="Page: 1 of 3"):
    """
    Deckmount/Spandan-style header bar:
    Left: big title + date  |  Centre: patient info table  |  Right: logo + page
    """
    H = 52          # header height pt
    W = CONTENT_W
    d = Drawing(W, H)

    # Background strip (dark navy)
    d.add(Rect(0, 0, W, H, fillColor=DARK_HEADER_BG, strokeColor=None))

    # Title text (white)
    d.add(GString(8, H - 14,
                  "HRV (Heart Rate Variability) Report",
                  fontName="Helvetica-Bold", fontSize=13,
                  fillColor=colors.white))
    date_str = info.get("date", datetime.now().strftime("%d %b %Y, %I:%M %p"))
    d.add(GString(8, H - 28,
                  f"Date:  {date_str}",
                  fontName="Helvetica", fontSize=8,
                  fillColor=colors.HexColor("#ccddee")))

    # Patient info columns
    fields = [
        ("NAME",      info.get("name", "—")),
        ("AGE",       info.get("age", "—")),
        ("GENDER",    info.get("gender", "—")),
        ("HEIGHT",    info.get("height", "—")),
        ("WEIGHT",    info.get("weight", "—")),
        ("REPORT ID", info.get("report_id", "—")),
    ]
    col_start = 210
    col_gap   = (W - col_start - 90) / len(fields)
    for i, (label, val) in enumerate(fields):
        cx = col_start + i * col_gap
        d.add(GString(cx, H - 14, label,
                      fontName="Helvetica", fontSize=7,
                      fillColor=colors.HexColor("#99bbcc")))
        d.add(GString(cx, H - 26, val,
                      fontName="Helvetica-Bold", fontSize=8,
                      fillColor=colors.white))

    # Page label (top-right)
    d.add(GString(W - 2, H - 10, page_label,
                  fontName="Helvetica", fontSize=7,
                  fillColor=colors.HexColor("#aaccdd"),
                  textAnchor='end'))

    return d

# ─────────────────────────── FOOTER DRAWING ───────────────────────────────────
def _footer_text(info):
    addr  = info.get("org_address", "Deckmount Electronics, Plot No. 683, Phase V, Udyog Vihar, Gurugram")
    return (f"<font size='6' color='#555555'>"
            f"<b>Disclaimer:</b> Medical advice accepted without physical examination is at patient's own risk. "
            f"Visit a doctor in case of emergency.&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;{addr}</font>")

# ─────────────────────────── SECTION CARD HELPER ──────────────────────────────
def _grey_card(content_rows, col_widths, heading=""):
    """Wrap rows in a grey-background bordered card table."""
    style = [
        ("BACKGROUND", (0, 0), (-1, -1), GREY_CARD),
        ("BOX",        (0, 0), (-1, -1), 0.8, BORDER_COLOR),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    return Table(content_rows, colWidths=col_widths,
                 style=TableStyle(style))

# ─────────────────────────── MAIN GENERATOR ───────────────────────────────────
def generate_hrv_report(
    filename,
    patient_info: dict,
    ecg_segments: list,      # list of 5 np.arrays (1 per minute), Lead II ADC values
    hrv_metrics: dict,        # SDANN, SDNN, RMSSD, NN50
    ecg_params: dict,         # PR, QRS, QT, QTc, QTCF, HR, RR
    conclusions: list,        # list of strings
    org_info: dict,
    logo_path: str = None,
    wave_gain_mm_mv: float = 10.0,
    adc_per_mv: float = 800.0,
    sampling_rate: float = 500.0,
):
    """
    Generate a 3-page landscape A4 HRV PDF report.

    Page 1 : Patient info table  +  5 one-minute ECG strips  +  Conclusion
    Page 2 : Time Domain Analysis (metrics header + RR bar + HR bar + Radar)
    Page 3 : Frequency Domain Analysis (LF/HF PSD chart)
    """
    styles = getSampleStyleSheet()

    # ── Derived data ──────────────────────────────────────────────────────────
    # Per-minute RR & HR from ecg_segments
    rr_per_min = []
    hr_per_min = []
    for seg in ecg_segments:
        if seg is not None and len(seg) > 100:
            from scipy.signal import find_peaks
            vals_n = (seg - np.mean(seg)) / (np.std(seg) + 1e-9)
            peaks, _ = find_peaks(vals_n, distance=int(0.3 * sampling_rate),
                                  height=0.3)
            if len(peaks) >= 2:
                rr_ms = np.diff(peaks) * (1000.0 / sampling_rate)
                rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
                if len(rr_ms) >= 1:
                    avg_rr = float(np.median(rr_ms))
                    rr_per_min.append(avg_rr)
                    hr_per_min.append(60000 / avg_rr)
                    continue
        # fallback
        hr_fallback = ecg_params.get("HR", 60)
        rr_per_min.append(60000 / hr_fallback if hr_fallback > 0 else 1000)
        hr_per_min.append(hr_fallback)

    sdann = hrv_metrics.get("SDANN", 0.0)
    sdnn  = hrv_metrics.get("SDNN",  0.0)
    rmssd = hrv_metrics.get("RMSSD", 0.0)
    nn50  = hrv_metrics.get("NN50",  0)
    lf    = hrv_metrics.get("LF",    0.0)
    hf    = hrv_metrics.get("HF",    0.0)
    lf_hf = (lf / hf) if hf > 0 else 0.0

    # Classification
    def _cls(val, low, mid, label_low="Low", label_mid="Borderline",
             label_ok="Normal", label_exc="Excellent", hi=None):
        if val >= (hi or mid * 2):
            return label_exc, "#1a7a1a"
        if val >= mid:
            return label_ok, "#2a8a2a"
        if val >= low:
            return label_mid, "#c87800"
        return label_low, "#c82020"

    sdnn_lbl,  sdnn_clr  = _cls(sdnn,  30, 50)
    rmssd_lbl, rmssd_clr = _cls(rmssd, 10, 20, hi=42)
    if sdnn >= 50 and rmssd >= 20:
        hrv_status, hrv_status_clr = "Healthy autonomic regulation", "#1a7a1a"
    elif sdnn >= 30 or rmssd >= 10:
        hrv_status, hrv_status_clr = "Mild autonomic imbalance – monitor", "#c87800"
    else:
        hrv_status, hrv_status_clr = "Autonomic dysfunction – clinical review advised", "#c82020"

    # ── Build PDF story ───────────────────────────────────────────────────────
    doc   = _make_doc(filename)
    story = []
    SS    = styles['Normal']

    def _h(txt, size=11, bold=True, color=TEXT_DARK, space_before=6, space_after=4):
        return Paragraph(
            f"<b>{txt}</b>" if bold else txt,
            ParagraphStyle('H', parent=SS, fontSize=size,
                           textColor=color,
                           spaceBefore=space_before, spaceAfter=space_after))

    def _p(txt, size=8, color=TEXT_DARK):
        return Paragraph(txt, ParagraphStyle('P', parent=SS,
                                             fontSize=size, textColor=color,
                                             leading=11))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — Patient Info + ECG Strips + Conclusion
    # ══════════════════════════════════════════════════════════════════════════

    # Header
    story.append(Image(BytesIO(_drawing_to_png(_header_drawing(
        patient_info, logo_path, "Page: 1 of 3"))),
        width=CONTENT_W, height=52))
    story.append(Spacer(1, 5))

    # ── ECG Parameters table + Result Details ─────────────────────────────────
    param_data = [
        [_p("<b>Parameter</b>"), _p("<b>Observed Values</b>"), _p("<b>Standard Range</b>")],
        [_p("PR Interval"),  _p(f"{ecg_params.get('PR',  0)} ms"),  _p("100 ms – 200 ms")],
        [_p("QRS Interval"), _p(f"{ecg_params.get('QRS', 0)} ms"),  _p("60 ms – 120 ms")],
        [_p("QT Interval"),  _p(f"{ecg_params.get('QT',  0)} ms"),  _p("300 ms – 450 ms")],
        [_p("QTc Interval"), _p(f"{ecg_params.get('QTc', 0)} ms"),  _p("300 ms – 450 ms")],
        [_p("Heart Rate"),   _p(f"{ecg_params.get('HR',  0)} bpm"), _p("60 bpm – 100 bpm")],
    ]
    param_tbl = Table(param_data, colWidths=[80, 80, 90],
                      style=TableStyle([
                          ("BACKGROUND", (0, 0), (-1, 0), DARK_HEADER_BG),
                          ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                          ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
                          ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
                          ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                           [colors.white, colors.HexColor("#f7f9fb")]),
                          ("GRID",       (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                          ("TOPPADDING",    (0, 0), (-1, -1), 4),
                          ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                          ("FONTSIZE",   (0, 0), (-1, -1), 7),
                          ("ROUNDEDCORNERS", [3]),
                      ]))

    # Result details column
    results_para = [
        _h("Result Details", 9),
        _p("<b>Heart Health Analysis</b>", 8),
        _p(patient_info.get("heart_health_analysis",
                            "LF and HF are within normal limits; Heart rate is within normal limits"), 7),
        Spacer(1, 4),
        _p("<b>Stress Coping Ability Analysis</b>", 8),
        _p(patient_info.get("stress_analysis", "Stress coping ability is normal"), 7),
        Spacer(1, 4),
        _p("<b>HRV Test Analysis</b>", 8),
        _p(patient_info.get("hrv_analysis", "AGE and SDNN within expected correlation"), 7),
        Spacer(1, 4),
        _p("<b>Heart Electrical Stability Test Analysis</b>", 8),
        _p(patient_info.get("electrical_analysis", "Normal autonomic function"), 7),
        Spacer(1, 4),
        _p("<b>Average Breath per minute</b>", 8),
        _p(patient_info.get("breath_analysis", "Normal Breath (average 7.8 breath(s) per minute)"), 7),
    ]

    info_row = Table(
        [[param_tbl, Table([[r] for r in results_para],
                           colWidths=[CONTENT_W - 275],
                           style=TableStyle([
                               ("TOPPADDING", (0,0),(-1,-1), 1),
                               ("BOTTOMPADDING", (0,0),(-1,-1), 1),
                           ]))]],
        colWidths=[265, CONTENT_W - 265],
        style=TableStyle([
            ("VALIGN", (0,0),(-1,-1), "TOP"),
            ("LEFTPADDING", (0,0),(-1,-1), 0),
            ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ])
    )
    story.append(info_row)
    story.append(Spacer(1, 8))

    # ── 5 One-Minute ECG Strips ───────────────────────────────────────────────
    minute_labels = [
        "Lead II (First minute)",  "Lead II (Second minute)", "Lead II (Third minute)",
        "Lead II (Fourth minute)", "Lead II (Fifth minute)",
    ]
    STRIP_H = 62        # height of each ECG strip
    LABEL_W = 10 * mm   # left label column

    for idx in range(5):
        seg = ecg_segments[idx] if idx < len(ecg_segments) else None

        strip_drawing = make_ecg_strip(
            seg, CONTENT_W, STRIP_H,
            minor_boxes_x=70, minor_boxes_y=16,
            wave_gain=wave_gain_mm_mv,
            adc_per_mv=adc_per_mv,
            show_notch=True,
        )

        # Label above strip (left-aligned)
        strip_buf = BytesIO(_drawing_to_png(strip_drawing,
                                            int(CONTENT_W), int(STRIP_H)))
        strip_img = Image(strip_buf, width=CONTENT_W, height=STRIP_H)

        label_para = _p(f"<b>{minute_labels[idx]}</b>", 7)
        story.append(label_para)
        story.append(strip_img)
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 6))

    # ── Conclusion box ────────────────────────────────────────────────────────
    concl_items = conclusions or ["Rhythm Analysis", "Normal heart rate",
                                  "Normal PR interval", "Normal QRS duration",
                                  "Normal QTc interval"]
    # Split into 2 columns
    mid = (len(concl_items) + 1) // 2
    col1 = concl_items[:mid]
    col2 = concl_items[mid:]
    rows = []
    for i in range(max(len(col1), len(col2))):
        c1 = f"{i+1}. {col1[i]}" if i < len(col1) else ""
        c2 = f"{i+mid+1}. {col2[i]}" if i < len(col2) else ""
        rows.append([_p(c1, 7), _p(c2, 7)])

    concl_tbl = Table(rows, colWidths=[CONTENT_W/2 - 4, CONTENT_W/2 - 4],
                      style=TableStyle([
                          ("TOPPADDING",    (0,0),(-1,-1), 2),
                          ("BOTTOMPADDING", (0,0),(-1,-1), 2),
                          ("LEFTPADDING",   (0,0),(-1,-1), 6),
                      ]))
    concl_card = Table(
        [[_h("✦ CONCLUSION ✦", 9, color=TEXT_DARK)],
         [concl_tbl]],
        colWidths=[CONTENT_W],
        style=TableStyle([
            ("BOX",        (0,0),(-1,-1), 0.8, colors.HexColor("#aaaaaa")),
            ("BACKGROUND", (0,0),(-1,-1), colors.white),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ])
    )
    story.append(concl_card)
    story.append(Spacer(1, 6))

    # Doctor row
    doc_name = org_info.get("doctor_name", "")
    story.append(_p(f"<b>Doctor Name:</b> {doc_name}   &nbsp;&nbsp;&nbsp;  <b>Doctor Sign:</b> ___________", 7))
    story.append(Spacer(1, 4))
    story.append(_p(_footer_text(org_info), 6))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — Time Domain Analysis
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())

    story.append(Image(BytesIO(_drawing_to_png(_header_drawing(
        patient_info, logo_path, "Page: 2 of 3"))),
        width=CONTENT_W, height=52))
    story.append(Spacer(1, 8))

    story.append(_h("Time Domain Analysis", 12))
    story.append(Spacer(1, 4))

    # Metrics header row
    def _metric_cell(label, value, suffix="", extra=""):
        txt = f"<b>{label}:</b> {value}{(' ' + suffix) if suffix else ''}"
        if extra:
            txt += f" <font color='{extra[1]}'><b>({extra[0]})</b></font>"
        return _p(txt, 8)

    metrics_header = [
        [_metric_cell("SDANN", f"{sdann:.2f}"),
         _metric_cell("SDNN",  f"{sdnn:.2f}",  extra=(sdnn_lbl, sdnn_clr)),
         _metric_cell("RMSSD", f"{rmssd:.2f}", extra=(rmssd_lbl, rmssd_clr)),
         _metric_cell("NN50",  str(nn50))],
        [Paragraph(
            f"<b>Interpretation:</b> <font color='{hrv_status_clr}'>{hrv_status}</font>"
            f"&nbsp;|&nbsp;<font size='7'>Reference: SDNN ≥50ms, RMSSD ≥20ms = Normal (ESC/AHA Task Force 1996)</font>",
            ParagraphStyle('interp', parent=SS, fontSize=7.5, leading=11)),
         "", "", ""],
    ]
    mh_tbl = Table(metrics_header,
                   colWidths=[CONTENT_W/4]*4,
                   style=TableStyle([
                       ("SPAN",          (0,1),(-1,1)),
                       ("BACKGROUND",    (0,0),(-1,-1), GREY_CARD),
                       ("BOX",           (0,0),(-1,-1), 0.8, BORDER_COLOR),
                       ("INNERGRID",     (0,0),(-1, 0), 0.3, BORDER_COLOR),
                       ("TOPPADDING",    (0,0),(-1,-1), 5),
                       ("BOTTOMPADDING", (0,0),(-1,-1), 5),
                       ("LEFTPADDING",   (0,0),(-1,-1), 8),
                       ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
                   ]))
    story.append(mh_tbl)
    story.append(Spacer(1, 10))

    # 3 charts: RR bar | HR bar | Radar
    min_labels = [f"Min {i+1}" for i in range(len(rr_per_min))]

    buf_rr = _bar_chart_image(rr_per_min, min_labels,
                              "Avg. RR Interval per minute", "Milliseconds")
    buf_hr = _bar_chart_image(hr_per_min, min_labels,
                              "Avg. Heart Rate per minute", "Beats per minute")

    # Radar
    radar_vals = [
        min(sdann / 2.0, 100),
        min(nn50, 100),
        min(rmssd / 2.0, 100),
        min(sdnn / 2.0, 100),
        min(ecg_params.get("HR", 60), 100),
    ]
    buf_rad = _radar_chart_image(radar_vals,
                                 ['NN50', 'RMSSD', 'SDNN', 'BPM', 'SDANN'])

    CHART_H = 190
    CW3 = CONTENT_W / 3

    charts_row = [[
        Image(buf_rr,  width=CW3 - 6, height=CHART_H),
        Image(buf_hr,  width=CW3 - 6, height=CHART_H),
        Image(buf_rad, width=CW3 - 6, height=CHART_H),
    ]]
    charts_tbl = Table(charts_row, colWidths=[CW3]*3,
                       style=TableStyle([
                           ("BACKGROUND", (0,0),(-1,-1), GREY_CARD),
                           ("BOX",        (0,0),(-1,-1), 0.8, BORDER_COLOR),
                           ("INNERGRID",  (0,0),(-1,-1), 0.3, BORDER_COLOR),
                           ("ALIGN",      (0,0),(-1,-1), "CENTER"),
                           ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
                           ("TOPPADDING",    (0,0),(-1,-1), 6),
                           ("BOTTOMPADDING", (0,0),(-1,-1), 6),
                       ]))
    story.append(charts_tbl)
    story.append(Spacer(1, 8))
    story.append(_p(_footer_text(org_info), 6))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — Frequency Domain Analysis
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())

    story.append(Image(BytesIO(_drawing_to_png(_header_drawing(
        patient_info, logo_path, "Page: 3 of 3"))),
        width=CONTENT_W, height=52))
    story.append(Spacer(1, 8))

    story.append(_h("Frequency Domain Analysis", 12))
    story.append(Spacer(1, 6))

    freq_metrics = [[
        _metric_cell("LF",    f"{lf:.2f}", "ms²"),
        _metric_cell("HF",    f"{hf:.2f}", "ms²"),
        _metric_cell("LF/HF", f"{lf_hf:.2f}"),
    ]]
    fm_tbl = Table(freq_metrics, colWidths=[CONTENT_W/3]*3,
                   style=TableStyle([
                       ("BACKGROUND", (0,0),(-1,-1), GREY_CARD),
                       ("BOX",        (0,0),(-1,-1), 0.8, BORDER_COLOR),
                       ("INNERGRID",  (0,0),(-1,-1), 0.3, BORDER_COLOR),
                       ("TOPPADDING",    (0,0),(-1,-1), 6),
                       ("BOTTOMPADDING", (0,0),(-1,-1), 6),
                       ("LEFTPADDING",   (0,0),(-1,-1), 12),
                       ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
                   ]))
    story.append(fm_tbl)
    story.append(Spacer(1, 8))

    buf_psd = _psd_chart_image(lf, hf, figsize=(10, 3.5), dpi=120)
    psd_img = Image(buf_psd, width=CONTENT_W, height=200)
    psd_card = Table([[psd_img]], colWidths=[CONTENT_W],
                     style=TableStyle([
                         ("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#e8e8e8")),
                         ("BOX",        (0,0),(-1,-1), 0.8, BORDER_COLOR),
                         ("TOPPADDING",    (0,0),(-1,-1), 4),
                         ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                         ("ALIGN",      (0,0),(-1,-1), "CENTER"),
                     ]))
    story.append(psd_card)
    story.append(Spacer(1, 8))
    story.append(_p(_footer_text(org_info), 6))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    print(f"✅ HRV Report generated: {filename}")
    return filename


# ─────────────────────────── UTILITY: Drawing → PNG bytes ─────────────────────
def _drawing_to_png(drawing, width=None, height=None, dpi=150):
    """Rasterise a ReportLab Drawing to PNG bytes via matplotlib."""
    from reportlab.graphics import renderSVG
    import io, re

    # Render to SVG string
    svg_buf = io.BytesIO()
    renderSVG.drawToFile(drawing, svg_buf)
    svg_buf.seek(0)
    svg_str = svg_buf.read().decode('utf-8')

    # Use cairosvg if available, else fall back to PIL+svglib
    try:
        import cairosvg
        png = cairosvg.svg2png(bytestring=svg_str.encode(),
                               output_width=width, output_height=height,
                               dpi=dpi)
        return png
    except ImportError:
        pass

    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        import io as _io
        buf2 = _io.BytesIO(svg_str.encode())
        rlg = svg2rlg(buf2)
        png_buf = _io.BytesIO()
        renderPM.drawToFile(rlg, png_buf, fmt='PNG', dpi=dpi)
        png_buf.seek(0)
        return png_buf.read()
    except Exception:
        pass

    # Ultimate fallback: render directly using matplotlib (convert drawing to fig)
    # This path handles the ECG strip drawings specifically
    return _drawing_to_png_via_mpl(drawing, width, height, dpi)


def _drawing_to_png_via_mpl(drawing, width=None, height=None, dpi=120):
    """Render a ReportLab Drawing to PNG using matplotlib canvas (fallback)."""
    W = width  or drawing.width
    H = height or drawing.height
    W_in = W / 72.0
    H_in = H / 72.0
    fig = plt.figure(figsize=(W_in, H_in), dpi=dpi)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    # Draw background pink
    from matplotlib.patches import Rectangle as MplRect
    ax.add_patch(MplRect((0, 0), W, H,
                          facecolor='#fff0f0', edgecolor='none', zorder=0))

    # Draw grid lines
    xs, ys = 70, 16
    dx = W / xs
    dy = H / ys
    maj = 5
    for i in range(xs + 1):
        x = i * dx
        clr = '#ff9999' if i % maj == 0 else '#ffc8c8'
        lw  = 0.8 if i % maj == 0 else 0.3
        ax.axvline(x=x, color=clr, lw=lw, zorder=1)
    for j in range(ys + 1):
        y = j * dy
        clr = '#ff9999' if j % maj == 0 else '#ffc8c8'
        lw  = 0.8 if j % maj == 0 else 0.3
        ax.axhline(y=y, color=clr, lw=lw, zorder=1)

    # Draw shapes from drawing
    for shape in drawing.contents:
        _mpl_draw_shape(ax, shape, H)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='#fff0f0',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _mpl_draw_shape(ax, shape, H):
    """Attempt to draw basic ReportLab shapes onto matplotlib axes."""
    from reportlab.graphics.shapes import Path, Line as RLLine, Rect as RLRect, String as RLString
    import matplotlib.patches as mpatches
    import matplotlib.lines  as mlines
    from matplotlib.path import Path as MPath

    if isinstance(shape, Path):
        ops  = shape.operators
        pts  = list(shape.points)
        pi   = 0
        verts, codes = [], []
        for op in ops:
            if op == 0:   # MOVETO
                verts.append((pts[pi], H - pts[pi+1])); pi += 2
                codes.append(MPath.MOVETO)
            elif op == 1: # LINETO
                verts.append((pts[pi], H - pts[pi+1])); pi += 2
                codes.append(MPath.LINETO)
            elif op == 4: # CLOSE
                codes.append(MPath.CLOSEPOLY)
                verts.append(verts[-1])
        if verts:
            mpath = MPath(verts, codes)
            sc = shape.strokeColor
            if sc:
                fc = (sc.red, sc.green, sc.blue)
                patch = mpatches.PathPatch(mpath, linewidth=shape.strokeWidth or 0.5,
                                           edgecolor=fc, facecolor='none', zorder=3)
                ax.add_patch(patch)

    elif isinstance(shape, RLLine):
        sc = shape.strokeColor
        if sc:
            ax.plot([shape.x1, shape.x2],
                    [H - shape.y1, H - shape.y2],
                    color=(sc.red, sc.green, sc.blue),
                    lw=shape.strokeWidth or 0.4, zorder=2)

    elif isinstance(shape, RLString):
        fc = shape.fillColor
        clr = (fc.red, fc.green, fc.blue) if fc else (0,0,0)
        ax.text(shape.x, H - shape.y, shape.text,
                fontsize=(shape.fontSize or 8) * 0.75,
                color=clr, va='top', zorder=4)


# ─────────────────────────── ECG STRIP PNG (direct mpl) ─────────────────────
def ecg_strip_to_png(ecg_data, width_px, height_px,
                     wave_gain=10.0, adc_per_mv=800.0, dpi=120):
    """
    Render a single ECG strip to PNG bytes completely via matplotlib.
    This avoids all SVG/ReportLab rendering issues for the waveform strips.
    """
    W_in = width_px  / dpi
    H_in = height_px / dpi
    fig = plt.figure(figsize=(W_in, H_in), dpi=dpi, facecolor='#fff0f0')
    ax  = fig.add_axes([0, 0, 1, 1], facecolor='#fff0f0')
    ax.set_xlim(0, width_px)
    ax.set_ylim(0, height_px)
    ax.axis('off')

    # Grid
    xs, ys = 70, 16
    dx = width_px  / xs
    dy = height_px / ys
    maj = 5
    for i in range(xs + 1):
        x = i * dx
        clr = '#ff9999' if i % maj == 0 else '#ffc8c8'
        lw  = 0.8 if i % maj == 0 else 0.3
        ax.axvline(x=x, color=clr, lw=lw, zorder=1)
    for j in range(ys + 1):
        y = j * dy
        clr = '#ff9999' if j % maj == 0 else '#ffc8c8'
        lw  = 0.8 if j % maj == 0 else 0.3
        ax.axhline(y=y, color=clr, lw=lw, zorder=1)

    cy = height_px / 2.0

    # Calibration notch
    nx = 3 * (dpi / 25.4) * 5  # 3mm in px
    nh = wave_gain * (dpi / 25.4) * 5  # gain*mm in px
    nw = 5 * (dpi / 25.4) * 5   # 5mm wide
    ax.plot([nx, nx, nx+nw, nx+nw], [cy, cy+nh, cy+nh, cy],
            color='black', lw=0.8, zorder=3)

    # ECG waveform
    if ecg_data is not None and len(ecg_data) >= 2:
        adc = np.array(ecg_data, dtype=float)
        adc -= np.mean(adc)
        scale = (wave_gain * (dpi/25.4) * 5) / adc_per_mv
        y_vals = cy + adc * scale
        x_start = nx + nw + 5
        tx = np.linspace(x_start, width_px - 5, len(adc))
        y_vals = np.clip(y_vals, 2, height_px - 2)
        ax.plot(tx, y_vals, color='black', lw=0.5, zorder=4)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='#fff0f0',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────── REVISED GENERATOR (using mpl strips) ─────────────
def generate_hrv_report_v2(
    filename,
    patient_info: dict,
    ecg_segments: list,
    hrv_metrics: dict,
    ecg_params: dict,
    conclusions: list,
    org_info: dict,
    logo_path: str = None,
    wave_gain_mm_mv: float = 10.0,
    adc_per_mv: float = 800.0,
    sampling_rate: float = 500.0,
):
    """
    Main entry point. Uses matplotlib for ECG strips (reliable PNG rendering).
    """
    styles = getSampleStyleSheet()
    SS     = styles['Normal']

    def _h(txt, size=10, bold=True, color=TEXT_DARK, sb=4, sa=4):
        s = ParagraphStyle('hh', parent=SS, fontSize=size, textColor=color,
                           spaceBefore=sb, spaceAfter=sa,
                           fontName='Helvetica-Bold' if bold else 'Helvetica')
        return Paragraph(txt, s)

    def _p(txt, size=7.5, color=TEXT_DARK, leading=10):
        s = ParagraphStyle('pp', parent=SS, fontSize=size, textColor=color,
                           leading=leading)
        return Paragraph(txt, s)

    # ── Compute per-minute RR/HR ───────────────────────────────────────────────
    from scipy.signal import find_peaks
    rr_per_min, hr_per_min = [], []
    for seg in ecg_segments:
        if seg is not None and len(seg) > 100:
            std_v = np.std(seg)
            if std_v < 1e-9:
                std_v = 1
            vals_n = (seg - np.mean(seg)) / std_v
            dist   = max(50, int(0.3 * sampling_rate))
            peaks, _ = find_peaks(vals_n, distance=dist, height=0.3)
            if len(peaks) >= 2:
                rr_ms = np.diff(peaks) * (1000.0 / sampling_rate)
                rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
                if len(rr_ms) >= 1:
                    avg_rr = float(np.median(rr_ms))
                    rr_per_min.append(avg_rr)
                    hr_per_min.append(60000 / avg_rr)
                    continue
        hr_fb = ecg_params.get("HR", 60)
        rr_per_min.append(60000 / max(hr_fb, 1))
        hr_per_min.append(hr_fb)

    sdann = hrv_metrics.get("SDANN", 0.0)
    sdnn  = hrv_metrics.get("SDNN",  0.0)
    rmssd = hrv_metrics.get("RMSSD", 0.0)
    nn50  = hrv_metrics.get("NN50",  0)
    lf    = hrv_metrics.get("LF",    0.03)
    hf    = hrv_metrics.get("HF",    0.21)
    lf_hf = (lf / hf) if hf > 0 else 0.0

    def _cls(val, lo, mid, hi=None):
        if hi and val >= hi: return "Excellent", "#1a7a1a"
        if val >= mid:       return "Normal",    "#2a8a2a"
        if val >= lo:        return "Borderline","#c87800"
        return "Low", "#c82020"

    sdnn_lbl,  sdnn_clr  = _cls(sdnn,  30, 50)
    rmssd_lbl, rmssd_clr = _cls(rmssd, 10, 20, 42)
    if sdnn >= 50 and rmssd >= 20:
        hrv_status, hrv_clr = "Healthy autonomic regulation", "#1a7a1a"
    elif sdnn >= 30 or rmssd >= 10:
        hrv_status, hrv_clr = "Mild autonomic imbalance – monitor", "#c87800"
    else:
        hrv_status, hrv_clr = "Autonomic dysfunction – clinical review advised", "#c82020"

    doc   = _make_doc(filename)
    story = []

    def _page_header(page_label):
        d = _header_drawing(patient_info, logo_path, page_label)
        return _drawing_as_rl_image(d, CONTENT_W, 52)

    def _footer():
        return _p(_footer_text(org_info), 6, color=colors.HexColor("#555555"))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1
    # ══════════════════════════════════════════════════════════════════════════
    story.append(_page_header("Page: 1 of 3"))
    story.append(Spacer(1, 6))

    # ECG Parameters table
    param_rows = [
        ["Parameter", "Observed Values", "Standard Range"],
        ["PR Interval",  f"{ecg_params.get('PR',0)} ms",   "100 ms – 200 ms"],
        ["QRS Interval", f"{ecg_params.get('QRS',0)} ms",  "60 ms – 120 ms"],
        ["QT Interval",  f"{ecg_params.get('QT',0)} ms",   "300 ms – 450 ms"],
        ["QTc Interval", f"{ecg_params.get('QTc',0)} ms",  "300 ms – 450 ms"],
        ["Heart Rate",   f"{ecg_params.get('HR',0)} bpm",  "60 bpm – 100 bpm"],
    ]
    p_style = [
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_HEADER_BG),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1,-1), 7.5),
        ("ALIGN",       (0, 0), (-1,-1), "CENTER"),
        ("VALIGN",      (0, 0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, colors.HexColor("#f5f8fb")]),
        ("GRID",        (0, 0), (-1,-1), 0.4, BORDER_COLOR),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]
    param_tbl = Table(param_rows, colWidths=[90, 85, 95],
                      style=TableStyle(p_style))

    # Result details
    def _result_row(title, body):
        return [_p(f"<b>{title}</b>", 7.5),
                _p(body, 7)]

    result_rows = [
        _result_row("Heart Health Analysis",
                    patient_info.get("heart_health_analysis",
                        "LF and HF are within normal limits. Heart rate is within normal limits.")),
        _result_row("Stress Coping Ability Analysis",
                    patient_info.get("stress_analysis", "Stress coping ability is normal.")),
        _result_row("HRV Test Analysis",
                    patient_info.get("hrv_analysis", "HRV parameters within expected range.")),
        _result_row("Heart Electrical Stability Test Analysis",
                    patient_info.get("electrical_analysis", "Normal autonomic function.")),
        _result_row("Average Breath per minute",
                    patient_info.get("breath_analysis",
                        "Normal Breath (average 7.8 breath(s) per minute)")),
    ]
    res_heading = _h("Result Details", 9, sb=0)
    res_tbl = Table([[res_heading]] +
                    [[Table([r], colWidths=[100, CONTENT_W - 290 - 100],
                            style=TableStyle([
                                ("TOPPADDING", (0,0),(-1,-1),2),
                                ("BOTTOMPADDING",(0,0),(-1,-1),2),
                            ]))] for r in result_rows],
                    colWidths=[CONTENT_W - 285],
                    style=TableStyle([
                        ("TOPPADDING", (0,0),(-1,-1),1),
                        ("BOTTOMPADDING", (0,0),(-1,-1),1),
                    ]))

    layout_row = Table([[param_tbl, res_tbl]],
                       colWidths=[280, CONTENT_W - 280],
                       style=TableStyle([
                           ("VALIGN", (0,0),(-1,-1), "TOP"),
                           ("LEFTPADDING", (0,0),(-1,-1), 0),
                           ("RIGHTPADDING",(0,0),(-1,-1), 0),
                       ]))
    story.append(layout_row)
    story.append(Spacer(1, 8))

    # ── ECG Strips ─────────────────────────────────────────────────────────────
    minute_labels = [
        "Lead II (First minute)",   "Lead II (Second minute)",
        "Lead II (Third minute)",   "Lead II (Fourth minute)",
        "Lead II (Fifth minute)",
    ]
    STRIP_H_PX = 88
    STRIP_W_PT = CONTENT_W

    for idx in range(5):
        seg = ecg_segments[idx] if idx < len(ecg_segments) else None
        story.append(_p(f"<b>{minute_labels[idx]}</b>", 7))
        png = ecg_strip_to_png(seg, int(STRIP_W_PT), STRIP_H_PX,
                               wave_gain_mm_mv, adc_per_mv)
        story.append(Image(BytesIO(png), width=STRIP_W_PT, height=STRIP_H_PX))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 4))

    # Conclusion + Doctor
    concl_items = conclusions or ["Rhythm Analysis", "Normal heart rate",
                                  "Normal PR interval", "Normal QRS duration",
                                  "Normal QTc interval"]
    mid = (len(concl_items) + 1) // 2
    c_rows = []
    for i in range(max(mid, len(concl_items) - mid)):
        l1 = f"{i+1}. {concl_items[i]}"   if i < len(concl_items[:mid]) else ""
        l2 = f"{i+mid+1}. {concl_items[mid+i]}" if (mid+i) < len(concl_items) else ""
        c_rows.append([_p(l1, 7), _p(l2, 7)])

    concl_inner = Table(c_rows, colWidths=[CONTENT_W/2 - 10]*2,
                        style=TableStyle([
                            ("TOPPADDING",(0,0),(-1,-1),2),
                            ("BOTTOMPADDING",(0,0),(-1,-1),2),
                        ]))
    concl_card = Table(
        [[_h("✦  CONCLUSION  ✦", 8, sb=2, sa=2)],
         [concl_inner]],
        colWidths=[CONTENT_W],
        style=TableStyle([
            ("BOX",       (0,0),(-1,-1), 0.8, colors.HexColor("#aaaaaa")),
            ("BACKGROUND",(0,0),(-1,-1), colors.white),
            ("LEFTPADDING",(0,0),(-1,-1), 8),
            ("TOPPADDING", (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ])
    )
    story.append(concl_card)
    story.append(Spacer(1, 5))

    # Doctor + footer
    doc_name  = org_info.get("doctor_name", "")
    doc_sign  = org_info.get("doctor_sign", "")
    story.append(Table(
        [[_p(f"<b>Doctor Name:</b>  {doc_name}", 7),
          _p(f"<b>Doctor Sign:</b>  {doc_sign}", 7)]],
        colWidths=[CONTENT_W/2]*2,
        style=TableStyle([("TOPPADDING",(0,0),(-1,-1),0)]),
    ))
    story.append(Spacer(1, 4))
    story.append(_footer())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — Time Domain Analysis
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(_page_header("Page: 2 of 3"))
    story.append(Spacer(1, 8))
    story.append(_h("Time Domain Analysis", 12, sb=0))
    story.append(Spacer(1, 5))

    # Metrics row
    sdann_s = f"{sdann:.2f}"
    sdnn_s  = f"{sdnn:.2f}"
    rmsd_s  = f"{rmssd:.2f}"
    nn50_s  = str(int(nn50))

    mrow1 = [
        _p(f"<b>SDANN:</b> {sdann_s}", 8),
        _p(f"<b>SDNN:</b> {sdnn_s} <font color='{sdnn_clr}'>({sdnn_lbl})</font>", 8),
        _p(f"<b>RMSSD:</b> {rmsd_s} <font color='{rmssd_clr}'>({rmssd_lbl})</font>", 8),
        _p(f"<b>NN50:</b> {nn50_s}", 8),
    ]
    mrow2_txt = (
        f"<b>Interpretation:</b> "
        f"<font color='{hrv_clr}'>{hrv_status}</font>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<font size='6.5'>Reference: SDNN ≥50ms, RMSSD ≥20ms = Normal "
        f"(ESC/AHA Task Force 1996)</font>"
    )
    metrics_tbl = Table(
        [mrow1, [Paragraph(mrow2_txt, ParagraphStyle(
            'interp', parent=SS, fontSize=7.5, leading=11)), "", "", ""]],
        colWidths=[CONTENT_W/4]*4,
        style=TableStyle([
            ("SPAN",          (0,1),(-1,1)),
            ("BACKGROUND",    (0,0),(-1,-1), GREY_CARD),
            ("BOX",           (0,0),(-1,-1), 0.8, BORDER_COLOR),
            ("LINEBELOW",     (0,0),(-1, 0), 0.4, BORDER_COLOR),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("TOPPADDING",    (0,0),(-1,-1),  5),
            ("BOTTOMPADDING", (0,0),(-1,-1),  5),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ])
    )
    story.append(metrics_tbl)
    story.append(Spacer(1, 10))

    # 3 charts
    min_labels = [f"Min {i+1}" for i in range(len(rr_per_min))]
    buf_rr  = _bar_chart_image(rr_per_min, min_labels,
                               "Avg. RR Interval per minute", "Milliseconds",
                               figsize=(5.5, 3.2))
    buf_hr  = _bar_chart_image(hr_per_min, min_labels,
                               "Avg. Heart Rate per minute",  "Beats per minute",
                               figsize=(5.5, 3.2))
    rv = [
        min(sdann / 2.0, 100),
        min(nn50,         100),
        min(rmssd / 2.0,  100),
        min(sdnn  / 2.0,  100),
        min(ecg_params.get("HR", 60), 100),
    ]
    buf_rad = _radar_chart_image(rv, ['NN50', 'RMSSD', 'SDNN', 'BPM', 'SDANN'],
                                 figsize=(3.8, 3.8))

    CW3 = CONTENT_W / 3
    CH  = 200
    charts_tbl = Table(
        [[Image(buf_rr,  width=CW3-8, height=CH),
          Image(buf_hr,  width=CW3-8, height=CH),
          Image(buf_rad, width=CW3-8, height=CH)]],
        colWidths=[CW3]*3,
        style=TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), GREY_CARD),
            ("BOX",        (0,0),(-1,-1), 0.8, BORDER_COLOR),
            ("INNERGRID",  (0,0),(-1,-1), 0.3, BORDER_COLOR),
            ("ALIGN",      (0,0),(-1,-1), "CENTER"),
            ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ])
    )
    story.append(charts_tbl)
    story.append(Spacer(1, 8))
    story.append(_footer())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — Frequency Domain Analysis
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(_page_header("Page: 3 of 3"))
    story.append(Spacer(1, 8))
    story.append(_h("Frequency Domain Analysis", 12, sb=0))
    story.append(Spacer(1, 6))

    freq_row = [
        _p(f"<b>LF:</b> {lf:.2f} ms²", 9),
        _p(f"<b>HF:</b> {hf:.2f} ms²", 9),
        _p(f"<b>LF/HF:</b> {lf_hf:.2f}", 9),
    ]
    freq_tbl = Table([freq_row], colWidths=[CONTENT_W/3]*3,
                     style=TableStyle([
                         ("BACKGROUND",  (0,0),(-1,-1), GREY_CARD),
                         ("BOX",         (0,0),(-1,-1), 0.8, BORDER_COLOR),
                         ("LEFTPADDING", (0,0),(-1,-1), 14),
                         ("TOPPADDING",  (0,0),(-1,-1),  7),
                         ("BOTTOMPADDING",(0,0),(-1,-1), 7),
                         ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
                     ]))
    story.append(freq_tbl)
    story.append(Spacer(1, 10))

    buf_psd = _psd_chart_image(lf, hf, figsize=(10, 3.8), dpi=120)
    psd_card = Table(
        [[Image(buf_psd, width=CONTENT_W-16, height=220)]],
        colWidths=[CONTENT_W],
        style=TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#e8e8e8")),
            ("BOX",        (0,0),(-1,-1), 0.8, BORDER_COLOR),
            ("ALIGN",      (0,0),(-1,-1), "CENTER"),
            ("TOPPADDING",    (0,0),(-1,-1), 6),
            ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ])
    )
    story.append(psd_card)
    story.append(Spacer(1, 10))
    story.append(_footer())

    doc.build(story)
    print(f"✅ HRV Report generated: {filename}")
    return filename


def _drawing_as_rl_image(drawing, w, h):
    """Convert a ReportLab Drawing to an Image platypus element via PNG."""
    from reportlab.graphics import renderPM
    buf = BytesIO()
    try:
        renderPM.drawToFile(drawing, buf, fmt='PNG', dpi=150)
        buf.seek(0)
    except Exception:
        # Fallback: minimal matplotlib header render
        return _header_via_mpl(drawing, w, h)
    return Image(buf, width=w, height=h)


def _header_via_mpl(drawing, w_pt, h_pt, dpi=150):
    """Render the header drawing via matplotlib as PNG fallback."""
    W_in = w_pt / 72.0
    H_in = h_pt / 72.0
    fig = plt.figure(figsize=(W_in, H_in), dpi=dpi, facecolor='#1e2d40')
    ax  = fig.add_axes([0, 0, 1, 1], facecolor='#1e2d40')
    ax.set_xlim(0, w_pt)
    ax.set_ylim(0, h_pt)
    ax.axis('off')
    for shape in drawing.contents:
        _mpl_draw_shape(ax, shape, h_pt)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='#1e2d40',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w_pt, height=h_pt)


# ─────────────────────────── DEMO RUNNER ──────────────────────────────────────
if __name__ == "__main__":
    OUT = "/mnt/user-data/outputs/HRV_Report_Clean.pdf"

    patient = {
        "name":       "Test Patient",
        "age":        "30 year(s)",
        "gender":     "Male",
        "height":     "170 cm",
        "weight":     "70 kg",
        "report_id":  "TEST20260401001",
        "date":       datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "heart_health_analysis":
            "LF and HF are within normal limits. Heart rate is within normal limits.",
        "stress_analysis":  "Stress coping ability is normal.",
        "hrv_analysis":     "AGE and SDNN within expected correlation.",
        "electrical_analysis": "Normal autonomic function.",
        "breath_analysis":  "Normal Breath (average 7.8 breath(s) per minute)",
    }

    org = {
        "org_name":    "Deckmount Electronics",
        "org_address": "Deckmount Electronics, Plot No. 683, Phase V, Udyog Vihar, Gurugram, Haryana 122016",
        "phone":       "0000000000",
        "doctor_name": "Dr. Example",
        "doctor_sign": "",
    }

    # Generate 5 one-minute demo ECG segments
    fs = 500
    segments = [_make_demo_ecg(n_samples=fs*60, fs=fs, hr_bpm=60+i*2)
                for i in range(5)]

    hrv = {"SDANN": 0.15, "SDNN": 0.30, "RMSSD": 0.24, "NN50": 0,
           "LF": 0.03, "HF": 0.21}

    params = {"PR": 182, "QRS": 99, "QT": 314, "QTc": 382, "QTCF": 314,
              "HR": 60, "RR": 1006}

    conclusions = [
        "Rhythm Analysis", "Normal heart rate",
        "Normal PR interval", "Normal QRS duration",
        "Normal QTc interval", "Good heart rate variability",
    ]

    generate_hrv_report_v2(
        OUT, patient, segments, hrv, params, conclusions, org,
        wave_gain_mm_mv=10.0, adc_per_mv=800.0, sampling_rate=fs,
    )
    print(f"Done → {OUT}")
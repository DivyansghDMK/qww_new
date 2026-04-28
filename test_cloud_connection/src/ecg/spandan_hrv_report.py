import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from datetime import datetime
from scipy.signal import find_peaks
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Table, TableStyle, Spacer, Image, PageBreak
)
from reportlab.graphics.shapes import Drawing, Rect, Line, Path, String as GStr

# ── Page geometry ──────────────────────────────────────────────────────────────
PW, PH = landscape(A4)          # 841.89 × 595.28 pt
M_SIDE  = 12 * mm               # left/right margin
M_TOP   = 10 * mm
M_BOT   = 8  * mm
CW      = PW - 2 * M_SIDE       # content width

# ── Colours ────────────────────────────────────────────────────────────────────
C_BG        = "#fff0f0"          # very light pink page background
C_MINOR     = "#ffc8c8"          # minor grid lines
C_MAJOR     = "#ff9999"          # major grid lines
C_ECG       = "#000000"          # ECG waveform
C_NAVY      = "#1b2a3b"          # header text / param table header
C_GREY_CARD = "#f0f0f0"
C_BORDER    = "#cccccc"
C_BLUE_BAR  = "#6497b1"

# ── ECG paper constants ────────────────────────────────────────────────────────
MINOR_MM = 1.0          # 1 mm per minor box
MAJOR_MM = 5.0          # 5 mm per major box
DPI      = 150

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: render full-page pink ECG grid as PNG (for page1 background)
# ══════════════════════════════════════════════════════════════════════════════
def full_page_ecg_grid_png(w_pt, h_pt, dpi=DPI):
    """Return PNG bytes of a full pink ECG grid page (no waveform)."""
    w_in = w_pt / 72.0
    h_in = h_pt / 72.0
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi, facecolor=C_BG)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
    ax.set_xlim(0, w_pt); ax.set_ylim(0, h_pt)
    ax.axis('off')

    minor_pt = MINOR_MM * mm
    major_pt = MAJOR_MM * mm

    x = 0.0
    while x <= w_pt + 0.1:
        is_major = abs(round(x / minor_pt) % 5) < 0.1
        ax.axvline(x, color=C_MAJOR if is_major else C_MINOR,
                   lw=0.8 if is_major else 0.3, zorder=1)
        x += minor_pt

    y = 0.0
    while y <= h_pt + 0.1:
        is_major = abs(round(y / minor_pt) % 5) < 0.1
        ax.axhline(y, color=C_MAJOR if is_major else C_MINOR,
                   lw=0.8 if is_major else 0.3, zorder=1)
        y += minor_pt

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor=C_BG,
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: render a single ECG strip as PNG (pink grid + waveform)
# ══════════════════════════════════════════════════════════════════════════════
def ecg_strip_png(ecg_data, w_pt, h_pt, wave_gain=10.0, adc_per_mv=800.0,
                  fs=500.0, dpi=DPI):
    w_in = w_pt / 72.0
    h_in = h_pt / 72.0
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi, facecolor=C_BG)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
    ax.set_xlim(0, w_pt); ax.set_ylim(0, h_pt)
    ax.axis('off')

    minor_pt = MINOR_MM * mm
    x = 0.0
    while x <= w_pt + 0.1:
        is_maj = (round(x / minor_pt) % 5 == 0)
        ax.axvline(x, color=C_MAJOR if is_maj else C_MINOR,
                   lw=0.8 if is_maj else 0.3, zorder=1)
        x += minor_pt
    y = 0.0
    while y <= h_pt + 0.1:
        is_maj = (round(y / minor_pt) % 5 == 0)
        ax.axhline(y, color=C_MAJOR if is_maj else C_MINOR,
                   lw=0.8 if is_maj else 0.3, zorder=1)
        y += minor_pt

    cy  = h_pt / 2.0
    notch_x   = 2.5 * mm
    notch_w   = 5.0 * mm
    notch_h   = wave_gain * mm
    ax.plot([notch_x, notch_x, notch_x + notch_w, notch_x + notch_w],
            [cy, cy + notch_h, cy + notch_h, cy],
            color=C_ECG, lw=0.9, zorder=4)

    ecg_start_x = notch_x + notch_w + 1.5 * mm
    if ecg_data is not None and len(ecg_data) >= 4:
        adc = np.array(ecg_data, dtype=float)
        adc -= np.mean(adc)
        scale = (wave_gain * mm) / adc_per_mv
        y_vals = cy + adc * scale
        tx = np.linspace(ecg_start_x, w_pt - 1 * mm, len(adc))
        y_vals = np.clip(y_vals, 1, h_pt - 1)
        ax.plot(tx, y_vals, color=C_ECG, lw=0.55, zorder=5)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor=C_BG,
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: bar chart
# ══════════════════════════════════════════════════════════════════════════════
def bar_chart_png(values, x_labels, title, ylabel, color=C_BLUE_BAR,
                  figsize=(5.5, 3.2), dpi=120):
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=color, edgecolor=color, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8, fontweight='bold')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    if values and max(values) > 0:
        ax.set_ylim(0, max(values) * 1.20)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (max(values) if values else 1) * 0.015,
                f'{int(round(v))}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')
    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: radar chart
# ══════════════════════════════════════════════════════════════════════════════
def radar_chart_png(norm_values, categories, figsize=(3.8, 3.8), dpi=120):
    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals   = list(norm_values) + [norm_values[0]]
    angs   = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'),
                           facecolor='white')
    ax.plot(angs, vals, 'o-', lw=2, color='#228B22', markersize=4)
    ax.fill(angs, vals, alpha=0.25, color='#32CD32')
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.grid(True, lw=0.5, alpha=0.6)
    ax.set_title('Radar chart', fontsize=9, fontweight='bold', pad=14)
    fig.tight_layout(pad=0.3)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: PSD frequency chart
# ══════════════════════════════════════════════════════════════════════════════
def psd_chart_png(lf, hf, figsize=(11, 3.8), dpi=120):
    freqs = np.linspace(0, 0.5, 600)
    base  = 0.35 * np.exp(-freqs * 8)
    lf_b  = lf  * 20 * np.exp(-((freqs - 0.09)**2) / (2 * 0.022**2))
    hf_b  = hf  * 14 * np.exp(-((freqs - 0.28)**2) / (2 * 0.045**2))
    psd   = np.maximum(base + lf_b + hf_b, 0)

    fig, ax = plt.subplots(figsize=figsize, facecolor='#e8e8e8')
    ax.set_facecolor('#e8e8e8')

    lf_m = (freqs >= 0.04) & (freqs < 0.15)
    hf_m = (freqs >= 0.15) & (freqs < 0.40)
    ax.fill_between(freqs, psd, where=lf_m, color='#a8c4d8', alpha=0.85)
    ax.fill_between(freqs, psd, where=hf_m, color='#c8a8a8', alpha=0.85)
    ax.plot(freqs, psd, color='#222222', lw=1.2)

    ax.set_xlim(0, 0.5)
    top = np.max(psd) * 1.3 if np.max(psd) > 0 else 1
    ax.set_ylim(0, top)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticklabels(['0Hz','0.1Hz','0.2Hz','0.3Hz','0.4Hz','0.5Hz'], fontsize=9)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.4)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='#e8e8e8')
    plt.close(fig)
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_hrv_report(
    output_path,
    patient:   dict,   
    org:       dict,   
    ecg_params: dict,  
    hrv_metrics: dict, 
    segments:  list,   
    conclusions: list, 
    wave_gain: float = 10.0,
    adc_per_mv: float = 800.0,
    fs: float = 500.0,
):
    SS = getSampleStyleSheet()

    def _p(txt, size=7.5, color='#111111', bold=False, leading=10):
        fn = 'Helvetica-Bold' if bold else 'Helvetica'
        return Paragraph(txt, ParagraphStyle('x', parent=SS['Normal'],
            fontSize=size, textColor=colors.HexColor(color),
            fontName=fn, leading=leading, spaceBefore=0, spaceAfter=0))

    def _h(txt, size=12, color='#1b2a3b'):
        return Paragraph(f'<b>{txt}</b>',
            ParagraphStyle('h', parent=SS['Normal'], fontSize=size,
                           textColor=colors.HexColor(color),
                           fontName='Helvetica-Bold', spaceBefore=4, spaceAfter=4))

    sdann = hrv_metrics.get('SDANN', 0.0)
    sdnn  = hrv_metrics.get('SDNN',  0.0)
    rmssd = hrv_metrics.get('RMSSD', 0.0)
    nn50  = hrv_metrics.get('NN50',  0)
    lf    = hrv_metrics.get('LF',    0.03)
    hf    = hrv_metrics.get('HF',    0.21)
    lf_hf = lf / hf if hf > 0 else 0.0

    def _cls(v, lo, mid, hi=None):
        if hi and v >= hi: return 'Excellent', '#1a7a1a'
        if v >= mid:       return 'Normal',    '#2a8a2a'
        if v >= lo:        return 'Borderline','#c87800'
        return 'Low', '#c82020'

    sdnn_lbl,  sdnn_clr  = _cls(sdnn,  30, 50)
    rmssd_lbl, rmssd_clr = _cls(rmssd, 10, 20, 42)
    if sdnn >= 50 and rmssd >= 20:
        hrv_status, hrv_clr = 'Healthy autonomic regulation', '#1a7a1a'
    elif sdnn >= 30 or rmssd >= 10:
        hrv_status, hrv_clr = 'Mild autonomic imbalance \u2013 monitor', '#c87800'
    else:
        hrv_status, hrv_clr = 'Autonomic dysfunction \u2013 clinical review advised', '#c82020'

    rr_per_min = hrv_metrics.get('rr_per_min', [])
    hr_per_min = hrv_metrics.get('hr_per_min', [])

    doc = BaseDocTemplate(output_path, pagesize=landscape(A4),
                          leftMargin=0, rightMargin=0,
                          topMargin=0, bottomMargin=0)

    fr_full  = Frame(0, 0, PW, PH, leftPadding=0, rightPadding=0,
                     topPadding=0, bottomPadding=0, id='full')
    fr_inner = Frame(M_SIDE, M_BOT, CW, PH - M_TOP - M_BOT,
                     leftPadding=0, rightPadding=0,
                     topPadding=0, bottomPadding=0, id='inner')

    def on_page1(canvas, doc):
        canvas.saveState()
        grid_png = full_page_ecg_grid_png(PW, PH, dpi=120)
        canvas.drawImage(
            __import__('reportlab.lib.utils', fromlist=['ImageReader']).ImageReader(
                BytesIO(grid_png)),
            0, 0, PW, PH)

        bar_h = 68
        canvas.setFillColor(colors.white)
        canvas.rect(0, PH - bar_h, PW, bar_h, fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor('#cccccc'))
        canvas.setLineWidth(0.5)
        canvas.line(0, PH - bar_h, PW, PH - bar_h)

        canvas.setFont('Helvetica-Bold', 8)
        canvas.setFillColor(colors.HexColor('#444444'))
        lx = M_SIDE
        canvas.drawString(lx, PH - 14, f"Name: {patient.get('name','')}")
        canvas.drawString(lx, PH - 26, f"Age: {patient.get('age','')}  "
                                        f"Gender: {patient.get('gender','')}  "
                                        f"Type: Standard")

        px = lx + 155
        params_left = [
            ('HR',  f"{ecg_params.get('HR',0)} bpm"),
            ('PR',  f"{ecg_params.get('PR',0)} ms"),
            ('QRS', f"{ecg_params.get('QRS',0)} ms"),
            ('RR',  f"{ecg_params.get('RR',0)} ms"),
            ('QT',  f"{ecg_params.get('QT',0)} ms"),
            ('QTc', f"{ecg_params.get('QTc',0)} ms"),
        ]
        canvas.setFont('Helvetica', 7.5)
        for i, (k, v) in enumerate(params_left):
            y = PH - 14 - i * 9
            canvas.setFont('Helvetica-Bold', 7.5)
            canvas.drawString(px, y, f"{k}")
            canvas.setFont('Helvetica', 7.5)
            canvas.drawString(px + 28, y, f":  {v}")

        mx = px + 150
        canvas.setFont('Helvetica', 7.5)
        canvas.drawString(mx, PH - 40,
                          f"QTCF   :  {ecg_params.get('QTCF',0)} ms")
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(mx, PH - 52,
                          "25.0 mm/s  0.5-150Hz  AC : 50Hz  10.0 mm/mV")

        lrx = PW - M_SIDE - 110
        canvas.setFont('Helvetica-Bold', 13)
        canvas.setFillColor(colors.HexColor(C_NAVY))
        canvas.drawString(lrx, PH - 20, "DECK")
        canvas.setFillColor(colors.HexColor('#e8b800'))
        canvas.drawString(lrx + 46, PH - 20, "/")
        canvas.setFillColor(colors.HexColor(C_NAVY))
        canvas.drawString(lrx + 56, PH - 20, "MOUNT")
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawString(lrx, PH - 32,
                          f"Date: {patient.get('date', datetime.now().strftime('%Y-%m-%d'))}")
        canvas.drawString(lrx, PH - 42,
                          f"Time: {patient.get('time', datetime.now().strftime('%H:%M:%S'))}")
        canvas.drawString(lrx, PH - 54,
                          f"Phone No: {org.get('phone','')}")
        canvas.drawString(lrx, PH - 64,
                          f"Org: {org.get('org_name','')}")

        foot_h = 16
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, PW, foot_h, fill=1, stroke=0)
        canvas.setFont('Helvetica', 6.5)
        canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawCentredString(PW/2, 5,
            org.get('org_address',
                    'Deckmount Electronics, Plot No. 683, Phase V, Udyog Vihar, Gurugram'))
        canvas.restoreState()

    def on_page23(canvas, doc):
        canvas.saveState()
        lrx = PW - M_SIDE - 110
        canvas.setFont('Helvetica-Bold', 12)
        canvas.setFillColor(colors.HexColor(C_NAVY))
        canvas.drawString(lrx, PH - M_TOP - 2, "DECK")
        canvas.setFillColor(colors.HexColor('#e8b800'))
        canvas.drawString(lrx + 42, PH - M_TOP - 2, "/")
        canvas.setFillColor(colors.HexColor(C_NAVY))
        canvas.drawString(lrx + 52, PH - M_TOP - 2, "MOUNT")
        canvas.setFont('Helvetica', 6.5)
        canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawCentredString(PW/2, 7,
            org.get('org_address','Deckmount Electronics, Plot No. 683, Gurugram'))
        canvas.restoreState()

    doc.addPageTemplates([
        PageTemplate(id='p1',   frames=[fr_full],  onPage=on_page1),
        PageTemplate(id='p23',  frames=[fr_inner], onPage=on_page23),
    ])

    story = []
    from reportlab.platypus import NextPageTemplate
    story.append(Spacer(1, 68))

    STRIP_H_PT = 76
    min_labels = [
        'Lead II (First minute)',  'Lead II (Second minute)',
        'Lead II (Third minute)',  'Lead II (Fourth minute)',
        'Lead II (Fifth minute)',
    ]
    STRIP_W_PT = PW

    for idx in range(5):
        seg = segments[idx] if idx < len(segments) else [0]*100
        lbl = Paragraph(
            f'<b>{min_labels[idx]}</b>',
            ParagraphStyle('lbl', parent=SS['Normal'], fontSize=7,
                           fontName='Helvetica-Bold', spaceBefore=0, spaceAfter=0,
                           leftIndent=M_SIDE))
        story.append(lbl)
        png = ecg_strip_png(seg, STRIP_W_PT, STRIP_H_PT,
                            wave_gain, adc_per_mv, fs, dpi=120)
        img = Image(BytesIO(png), width=STRIP_W_PT, height=STRIP_H_PT)
        story.append(img)

    story.append(Spacer(1, 4))

    concl = conclusions or ['Rhythm Analysis']
    mid = (len(concl) + 1) // 2
    c_rows = []
    for i in range(max(1, max(mid, len(concl) - mid))):
        l1 = f"{i+1}. {concl[i]}"       if i < len(concl[:mid]) else ''
        l2 = f"{i+mid+1}. {concl[mid+i]}" if (mid+i) < len(concl) else ''
        c_rows.append([_p(l1, 7.5), _p(l2, 7.5)])

    doc_name = org.get('doctor_name', '')
    doc_sign = org.get('doctor_sign', '')
    doc_block = Table(
        [[_p(f'<b>Doctor Name:</b>  {doc_name}', 7.5)],
         [_p('<b>Doctor Sign:</b>', 7.5)]],
        colWidths=[120],
        style=TableStyle([('TOPPADDING',(0,0),(-1,-1),2),
                          ('BOTTOMPADDING',(0,0),(-1,-1),2)]))

    concl_title = _p('<b>✦ CONCLUSION ✦</b>', 8, color='#1b2a3b')
    concl_inner = Table(c_rows,
                        colWidths=[(CW - 130)/2]*2,
                        style=TableStyle([
                            ('TOPPADDING',(0,0),(-1,-1),1),
                            ('BOTTOMPADDING',(0,0),(-1,-1),1),
                        ]))
    concl_box = Table(
        [[concl_title], [concl_inner]],
        colWidths=[CW - 130],
        style=TableStyle([
            ('BOX',        (0,0),(-1,-1), 0.8, colors.HexColor('#aaaaaa')),
            ('BACKGROUND', (0,0),(-1,-1), colors.white),
            ('LEFTPADDING',(0,0),(-1,-1), 8),
            ('TOPPADDING', (0,0),(-1,-1), 4),
            ('BOTTOMPADDING',(0,0),(-1,-1),4),
        ])
    )

    bottom_row = Table(
        [[doc_block, concl_box]],
        colWidths=[130, CW - 130],
        style=TableStyle([
            ('VALIGN',      (0,0),(-1,-1),'TOP'),
            ('LEFTPADDING', (0,0),(-1,-1), M_SIDE),
            ('RIGHTPADDING',(0,0),(-1,-1), M_SIDE),
        ])
    )
    story.append(bottom_row)

    story.append(NextPageTemplate('p23'))
    story.append(PageBreak())

    story.append(_h('Time Domain Analysis', 14))
    story.append(Spacer(1, 6))

    sdann_s = f'{sdann:.2f}'
    sdnn_s  = f'{sdnn:.2f}'
    rmsd_s  = f'{rmssd:.2f}'
    nn50_s  = str(int(nn50))

    mrow1 = [
        _p(f'<b>SDANN:</b> {sdann_s}', 8),
        _p(f'<b>SDNN:</b> {sdnn_s} <font color="{sdnn_clr}">({sdnn_lbl})</font>', 8),
        _p(f'<b>RMSSD:</b> {rmsd_s} <font color="{rmssd_clr}">({rmssd_lbl})</font>', 8),
        _p(f'<b>NN50:</b> {nn50_s}', 8),
    ]
    interp_txt = (f'<b>Interpretation:</b> <font color="{hrv_clr}">{hrv_status}</font>'
                  f'  |  <font size="6.5">Reference: SDNN ≥50ms, RMSSD ≥20ms'
                  f' = Normal (ESC/AHA Task Force 1996)</font>')
    mrow2 = [Paragraph(interp_txt,
                        ParagraphStyle('it', parent=SS['Normal'], fontSize=7.5,
                                       leading=11)),
             '', '', '']
    metrics_card = Table(
        [mrow1, mrow2],
        colWidths=[CW/4]*4,
        style=TableStyle([
            ('SPAN',           (0,1),(-1,1)),
            ('BACKGROUND',     (0,0),(-1,-1), colors.HexColor(C_GREY_CARD)),
            ('BOX',            (0,0),(-1,-1), 0.8, colors.HexColor(C_BORDER)),
            ('LINEBELOW',      (0,0),(-1, 0), 0.4, colors.HexColor(C_BORDER)),
            ('LEFTPADDING',    (0,0),(-1,-1), 10),
            ('TOPPADDING',     (0,0),(-1,-1),  6),
            ('BOTTOMPADDING',  (0,0),(-1,-1),  6),
            ('VALIGN',         (0,0),(-1,-1), 'MIDDLE'),
        ])
    )
    story.append(metrics_card)
    story.append(Spacer(1, 10))

    min_lbl = [f'Min {i+1}' for i in range(len(rr_per_min))]
    buf_rr  = bar_chart_png(rr_per_min, min_lbl, 'Avg. RR Interval per minute', 'Milliseconds')
    buf_hr  = bar_chart_png(hr_per_min, min_lbl, 'Avg. Heart Rate per minute',  'Beats per minute')
    rv = [
        min(sdann / 2.0, 100),
        min(nn50,         100),
        min(rmssd / 2.0,  100),
        min(sdnn  / 2.0,  100),
        min(ecg_params.get('HR', 60), 100),
    ]
    buf_rad = radar_chart_png(rv, ['SDANN', 'NN50', 'RMSSD', 'SDNN', 'BPM'])

    CW3 = CW / 3
    CH  = 205
    charts_card = Table(
        [[Image(buf_rr,  width=CW3-10, height=CH),
          Image(buf_hr,  width=CW3-10, height=CH),
          Image(buf_rad, width=CW3-10, height=CH)]],
        colWidths=[CW3]*3,
        style=TableStyle([
            ('BACKGROUND', (0,0),(-1,-1), colors.HexColor(C_GREY_CARD)),
            ('BOX',        (0,0),(-1,-1), 0.8, colors.HexColor(C_BORDER)),
            ('INNERGRID',  (0,0),(-1,-1), 0.4, colors.HexColor(C_BORDER)),
            ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
            ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
            ('TOPPADDING',    (0,0),(-1,-1), 8),
            ('BOTTOMPADDING', (0,0),(-1,-1), 8),
        ])
    )
    story.append(charts_card)

    story.append(PageBreak())

    story.append(_h('Frequency Domain Analysis', 14))
    story.append(Spacer(1, 6))

    freq_row = [
        _p(f'<b>LF:</b> {lf:.2f} ms²', 9),
        _p(f'<b>HF:</b> {hf:.2f} ms²', 9),
        _p(f'<b>LF/HF:</b> {lf_hf:.2f}', 9),
    ]
    freq_card = Table([freq_row], colWidths=[CW/3]*3,
                      style=TableStyle([
                          ('BACKGROUND',  (0,0),(-1,-1), colors.HexColor(C_GREY_CARD)),
                          ('BOX',         (0,0),(-1,-1), 0.8, colors.HexColor(C_BORDER)),
                          ('INNERGRID',   (0,0),(-1,-1), 0.4, colors.HexColor(C_BORDER)),
                          ('LEFTPADDING', (0,0),(-1,-1), 14),
                          ('TOPPADDING',  (0,0),(-1,-1),  8),
                          ('BOTTOMPADDING',(0,0),(-1,-1), 8),
                          ('VALIGN',      (0,0),(-1,-1), 'MIDDLE'),
                      ]))
    story.append(freq_card)
    story.append(Spacer(1, 10))

    buf_psd = psd_chart_png(lf, hf, figsize=(10.5, 4.0))
    psd_card = Table(
        [[Image(buf_psd, width=CW - 4, height=230)]],
        colWidths=[CW],
        style=TableStyle([
            ('BACKGROUND', (0,0),(-1,-1), colors.HexColor('#e8e8e8')),
            ('BOX',        (0,0),(-1,-1), 0.8, colors.HexColor(C_BORDER)),
            ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
            ('TOPPADDING',    (0,0),(-1,-1), 4),
            ('BOTTOMPADDING', (0,0),(-1,-1), 4),
        ])
    )
    story.append(psd_card)

    doc.build(story)
    return output_path

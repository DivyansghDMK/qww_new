"""
ecg_report_android.py  —  RhythmPro-matching ECG PDF generator
===============================================================
Formats : 12:1 (portrait)  |  6:2 (landscape)  |  4:3 (landscape)

Changes vs previous:
  ✅ Logo  : assets/DeckmountLogo.png  (no crop, aspect preserved)
  ✅ BG    : pure white — no pink tint
  ✅ Conc  : max 5 items, transparent box (grid shows through)
  ✅ NOTE  : removed ("automated analysis…" disclaimer gone)
  ✅ Org/Phone row : removed from specs section
  ✅ Type  : shows Lead Sequence setting (Standard / Cabrera)
  ✅ RV5/SV1 : real calculated values shown (not 0.000)
  ✅ 12:1  : no white gap — waves fill header→footer tightly
  ✅ PDF   : exact A4 size (no bbox_inches='tight' extra margin)
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# ─── Page constants (mm) ──────────────────────────────────────────────────────
A4_P_W = 210.0
A4_P_H = 297.0
A4_L_W = 297.0
A4_L_H = 210.0

MT = MB = ML = MR = 5.0

GRID_BOX   = 5.0
GRID_MINOR = 1.0

ECG_FS        = 500.0
FIXED_SPEED   = 25.0
FIXED_GAIN    = 10.0
MM_PER_SAMPLE = FIXED_SPEED / ECG_FS   # 0.05 mm/sample
# ADC_PER_MM: ADC counts per 1mm at standard 10mm/mV gain.
# Formula: (ADC_full_scale / mV_full_scale) / (mm_per_mV)
# = 6400 ADC / 50mV / 10mm*mV = 12800 / 1000 = 12.8? No -
# The hardware is 12-bit with mid=2000, span ~6400 ADC for 50mV swing.
# At 10mm/mV: 1mV = 10mm, 1mm = 0.1mV, 1mm = 6400/50*0.1 = 12.8 ADC? 
# Empirically from 4_3 code: adc_per_box = 6400/wave_gain, box=5mm
# So ADC_per_mm = 6400/(wave_gain*5) = 1280/wave_gain
# At wave_gain=10: ADC_per_mm = 128  ✓
ADC_PER_MM    = 128.0   # updated dynamically from wave_gain setting in generate_report()

COL_MINOR = '#f5dcdc'
COL_MAJOR = '#e69696'
COL_BG    = 'white'      # ← pure white

ALL_LEADS  = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# Logo — user's file at assets/DeckmountLogo.png
LOGO_FNAME = "DeckmountLogo.png"
# Fallback candidates if primary not found
LOGO_FALLBACKS = [
    "DeckmountLogo.png",
    "Deck Mount Electronics Logo (3).png",
    "Deckmountimg.png",
    "deck_mount_logo.png",
    "logo.png",
]


# ─── Public entry point ───────────────────────────────────────────────────────

def generate_report(snap_raw, frozen, patient, filename, fmt,
                    conc_list=None, fs=500.0, extra_figs=None):
    """
    snap_raw  : list[12] of numpy arrays — raw ADC per lead
    frozen    : dict with HR, PR, QRS, QT, QTc, QTcF, rv5, sv1,
                     p_axis, QRS_axis, t_axis, lead_seq, logo_path
    patient   : dict with first_name/name, last_name, age, gender,
                     doctor_name, date_time
    filename  : output PDF path
    fmt       : '12_1' | '6_2' | '4_3'
    conc_list : conclusion strings (max 5 shown)
    fs        : sampling rate Hz
    """
    global ECG_FS, MM_PER_SAMPLE, ADC_PER_MM
    ECG_FS        = float(fs)
    MM_PER_SAMPLE = FIXED_SPEED / ECG_FS

    conc_list   = (conc_list or [])[:5]   # ← max 5 conclusions
    is_portrait = (fmt == '12_1')
    PW = A4_P_W if is_portrait else A4_L_W
    PH = A4_P_H if is_portrait else A4_L_H

    # ── Read user filter settings ──────────────────────────────────────────
    _settings_mgr = None
    try:
        import sys as _sys, os as _os
        _here = _os.path.dirname(_os.path.abspath(__file__))
        _src  = _os.path.abspath(_os.path.join(_here, '..'))
        for _p in [_here, _src]:
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from utils.settings_manager import SettingsManager
        _settings_mgr = SettingsManager()
    except Exception as _e:
        print(f"[ecg_report_android] Could not load SettingsManager: {_e}")

    def _get_filter_setting(key, default):
        if _settings_mgr is None:
            return default
        try:
            val = _settings_mgr.get_setting(key, default)
            return str(val).strip() if val else default
        except Exception:
            return default

    ac_setting  = _get_filter_setting("filter_ac",  "50")   # "50", "60", or "off"
    emg_setting = _get_filter_setting("filter_emg", "150")  # "25".."150" or "off"
    dft_setting = _get_filter_setting("filter_dft", "0.5")  # "0.05", "0.5", or "off"
    print(f"[ecg_report_android] Filter settings — AC:{ac_setting}  EMG:{emg_setting}  DFT:{dft_setting}")

    # Read wave_gain from settings to set correct ADC→mm scaling
    wave_gain_val = 10.0
    try:
        wg = _get_filter_setting("wave_gain", "10")
        wave_gain_val = float(wg) if wg and wg not in ("", "off") else 10.0
    except Exception:
        wave_gain_val = 10.0
    # ADC_per_mm: at wave_gain=10mm/mV, 1mm = 128 ADC. Formula: 1280/wave_gain
    ADC_PER_MM = 1280.0 / max(wave_gain_val, 1.0)
    print(f"[ecg_report_android] wave_gain={wave_gain_val} mm/mV  →  ADC_PER_MM={ADC_PER_MM:.2f}")

    # ── Apply user-selected filters to all 12 leads ───────────────────────
    lead_mv = {}
    for i, lead in enumerate(ALL_LEADS):
        arr = np.asarray(snap_raw[i], dtype=float) if i < len(snap_raw) else np.array([])
        if len(arr) < 100:
            lead_mv[lead] = arr
            continue
        try:
            from ecg.ecg_filters import apply_ecg_filters, stabilize_report_edges
            ac_param  = ac_setting  if ac_setting  not in ("off", "") else None
            emg_param = emg_setting if emg_setting not in ("off", "") else None
            dft_param = dft_setting if dft_setting not in ("off", "") else None
            filtered = apply_ecg_filters(
                arr,
                sampling_rate=float(ECG_FS),
                ac_filter=ac_param,
                emg_filter=emg_param,
                dft_filter=dft_param,
            )
            # Gentle edge stabilisation (removes filter transients at strip edges)
            filtered = stabilize_report_edges(filtered, float(ECG_FS))
            lead_mv[lead] = filtered
        except Exception as _fe:
            print(f"[ecg_report_android] Filter failed for {lead}: {_fe} — using median-centred raw")
            lead_mv[lead] = arr - float(np.median(arr))

    # ── Figure — exact A4, white background ───────────────────────────────
    fig = Figure(figsize=(PW/25.4, PH/25.4), dpi=150, facecolor='white')
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=COL_BG)
    ax.set_xlim(0, PW)
    ax.set_ylim(PH, 0)
    ax.set_aspect('equal')
    ax.axis('off')

    _draw_grid(ax, 0, 0, PW, PH)
    _draw_header(ax, frozen, patient, PW, fmt)

    if fmt == '12_1':
        _draw_1x12(ax, lead_mv, PW, PH, target_samples=3500)
    elif fmt == '6_2':
        _draw_2x6(ax, lead_mv, PW, PH, target_samples=2500)
    else:
        _draw_3x4(ax, lead_mv, PW, PH, target_samples=1600)

    _draw_footer(ax, frozen, patient, conc_list, PW, PH, is_portrait)

    # ── Save — exact A4, no extra white margins ────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches=None)
        if extra_figs:
            for extra_fig in extra_figs:
                pdf.savefig(extra_fig, bbox_inches='tight')
    import gc; gc.collect()


# ─── Grid ─────────────────────────────────────────────────────────────────────

def _draw_grid(ax, x0, y0, w, h):
    for xi in np.arange(x0, x0+w+GRID_MINOR, GRID_MINOR):
        c  = COL_MAJOR if round((xi-x0) % GRID_BOX, 6) < 1e-3 else COL_MINOR
        lw = 0.55 if c == COL_MAJOR else 0.22
        ax.plot([xi,xi], [y0, y0+h], color=c, linewidth=lw, zorder=1)
    for yi in np.arange(y0, y0+h+GRID_MINOR, GRID_MINOR):
        c  = COL_MAJOR if round((yi-y0) % GRID_BOX, 6) < 1e-3 else COL_MINOR
        lw = 0.55 if c == COL_MAJOR else 0.22
        ax.plot([x0, x0+w], [yi,yi], color=c, linewidth=lw, zorder=1)


# ─── Header ───────────────────────────────────────────────────────────────────

def _find_logo():
    """Search for logo file in assets/ folder."""
    # Walk up from this file to find the project root
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [
        os.path.abspath(os.path.join(here, '..', '..')),   # qww root
        os.path.abspath(os.path.join(here, '..')),
        here,
    ]
    for root in roots:
        for fname in LOGO_FALLBACKS:
            for subdir in ['assets', '.', 'src', 'src/assets']:
                candidate = os.path.join(root, subdir, fname)
                if os.path.exists(candidate):
                    return candidate
    return None


def _draw_header(ax, frozen, patient, PW, fmt):
    is_portrait = (fmt == '12_1')
    yb = MT          # 5mm from top
    lh = 5.0         # line height mm
    x  = 10.0 if is_portrait else (ML + 15.0)

    # Patient info
    fn   = (patient.get('first_name','') or patient.get('name','') or '').strip()
    ln   = (patient.get('last_name','') or '').strip()
    full = (fn + ' ' + ln).strip() or 'Unknown'
    age    = patient.get('age', '--')
    gender = patient.get('gender', '--')

    # Lead sequence (Standard / Cabrera) from settings
    lead_seq = frozen.get('lead_seq', 'Standard') or 'Standard'

    # Col 1: patient
    _t(ax, f"Name: {full}",       x, yb,       9)
    _t(ax, f"Age: {age}",         x, yb+lh,    9)
    _t(ax, f"Gender: {gender}",   x, yb+lh*2,  9)
    _t(ax, f"Type: {lead_seq}",   x, yb+lh*4,  9)   # ← shows setting value

    # Col 2: ECG measurements
    x += 50.0 if is_portrait else (12*5+2*5+10)
    _t(ax, f"HR  : {frozen.get('HR',0)} bpm", x, yb,       9, bold=True)
    _t(ax, f"RR  : {_rr_ms(frozen)} ms",       x, yb+lh,   9, bold=True)
    _t(ax, f"PR  : {frozen.get('PR',0)} ms",   x, yb+lh*2, 9, bold=True)
    _t(ax, f"QRS : {frozen.get('QRS',0)} ms",  x, yb+lh*3, 9, bold=True)
    _t(ax, f"QT  : {frozen.get('QT',0)} ms",   x, yb+lh*4, 9, bold=True)

    # Col 3: extra measurements
    x += 35.0 if is_portrait else (5*5+2*5)
    rv5 = float(frozen.get('rv5', 0.0) or 0.0)
    sv1 = float(frozen.get('sv1', 0.0) or 0.0)
    idx_val = rv5 - abs(sv1)
    idx_str = f"{idx_val:.3f} mV" + (" *" if idx_val >= 3.5 else "")
    p_ax = frozen.get('p_axis','--')
    q_ax = frozen.get('QRS_axis','--')
    t_ax = frozen.get('t_axis','--')

    _t(ax, f"QTc : {frozen.get('QTc',0)} ms",            x, yb,       9, bold=True)
    _t(ax, f"QTcF: {frozen.get('QTcF',0)} ms",           x, yb+lh,    9, bold=True)
    _t(ax, f"RV5/SV1: {rv5:.3f}/{sv1:.3f} mV",          x, yb+lh*2,  9, bold=True)
    _t(ax, f"RV5+SV1: {idx_str}",                        x, yb+lh*3,  9, bold=True)
    _t(ax, f"P/QRS/T: {p_ax}/{q_ax}/{t_ax}\u00b0",       x, yb+lh*4,  9, bold=True)

    # ── Logo top-right — no crop, full aspect ──────────────────────────────
    logo_w = 55.0   # mm width budget
    logo_h = 18.0   # mm height budget
    logo_x = PW - 7.0 - logo_w
    logo_y = yb

    # Find logo file
    logo_path = frozen.get('logo_path', '') or ''
    if not logo_path or not os.path.exists(logo_path):
        logo_path = _find_logo() or ''

    logo_placed = False
    if logo_path and os.path.exists(logo_path):
        try:
            from matplotlib.image import imread as _imread
            img = _imread(logo_path)

            # Compute actual pixel aspect ratio
            img_h_px, img_w_px = img.shape[:2]
            aspect = img_w_px / img_h_px   # width/height

            # Fit inside budget keeping aspect ratio (no crop/stretch)
            if logo_w / logo_h >= aspect:
                # height is the limiting dimension
                actual_h = logo_h
                actual_w = logo_h * aspect
            else:
                # width is the limiting dimension
                actual_w = logo_w
                actual_h = logo_w / aspect

            # Centre in allocated box
            x_off = (logo_w - actual_w) / 2.0
            y_off = (logo_h - actual_h) / 2.0
            x0 = logo_x + x_off
            y0 = logo_y + y_off
            x1 = x0 + actual_w
            y1 = y0 + actual_h

            ax.imshow(img,
                      extent=[x0, x1, y1, y0],   # [xmin,xmax,ymax,ymin]
                      aspect='auto', zorder=10, interpolation='bilinear')
            logo_placed = True
        except Exception as e:
            print(f"Logo load failed: {e}")

    if not logo_placed:
        _t(ax, "DECK\u26a1MOUNT", logo_x + logo_w/2, logo_y+4, 11,
           bold=True, color='#0000cc', ha='center')

    # Specs row — technical specs + date
    dt        = patient.get('date_time','') or ''
    date_part = dt[:10] if len(dt) >= 10 else ''
    time_part = dt[11:19] if len(dt) >= 19 else ''

    spec_y = logo_y + logo_h + 2.0
    spec   = f"{FIXED_SPEED:.1f} mm/s   0.5-25Hz   AC:50Hz   {FIXED_GAIN:.1f} mm/mV"
    _t(ax, spec,                                      logo_x, spec_y,     7)
    _t(ax, f"Date & Time: {date_part} {time_part}",  logo_x, spec_y+4.0, 7)


# ─── 12:1 Portrait — waves fill header→footer, no white gap ──────────────────

def _draw_1x12(ax, lead_mv, PW, PH, target_samples=None):
    HEADER_H  = 28.0
    FOOTER_H  = 25.0
    top_offset = MT + HEADER_H                   # 33mm
    bot_limit  = PH - MB - FOOTER_H              # 267mm
    usable_h   = bot_limit - top_offset          # 234mm
    cell_h     = usable_h / 12.0                 # 19.5mm per lead

    wave_w = PW - ML - MR - 15.0   # 185mm
    # half_clip: allow peaks to use up to 90% of half cell height
    # (8.775mm per side) → prevents flat-topping of V4/V5 R-waves.
    # Clinical ECGs allow peaks to extend to adjacent rows; we clip at
    # the full cell height (both sides) to allow true amplitude display.
    half_clip = cell_h * 0.90   # ~17.5mm at 10mm/mV → allows ~1.75mV peaks

    for i, lead in enumerate(ALL_LEADS):
        mid_y   = top_offset + i * cell_h + cell_h / 2.0
        label_y = mid_y - 8.0
        _draw_calibration(ax, ML, mid_y, FIXED_GAIN)
        _t(ax, lead, ML+11, label_y, 8.5, bold=True)
        _draw_waveform(ax, lead_mv.get(lead, np.array([])),
                       ML+13, mid_y, wave_w, half_clip, target_samples=target_samples)


# ─── 6:2 Landscape ────────────────────────────────────────────────────────────

def _draw_2x6(ax, lead_mv, PW, PH, target_samples=None):
    HEADER_H  = 25.0
    FOOTER_H  = 20.0
    start_y   = MT + HEADER_H
    bot_limit = PH - MB - FOOTER_H
    usable_h  = bot_limit - start_y
    row_h     = min(usable_h / 7.0, 22.0)   # 6 rows + rhythm strip

    left_margin = ML + 8.0
    lead_w      = 123.0
    div_pad     = 5.0

    pair_map = [("I","V1"),("II","V2"),("III","V3"),
                ("aVR","V4"),("aVF","V5"),("aVL","V6")]

    for r, (l1, l2) in enumerate(pair_map):
        mid_y   = start_y + r*row_h + row_h/2.0
        label_y = mid_y - 9.0
        # half_clip: 90% of row half-height to show true peak amplitude
        half_clip = row_h * 0.90

        _draw_calibration_pad(ax, left_margin-4, mid_y, FIXED_GAIN)

        _t(ax, l1, left_margin+9, label_y, 10, bold=True)
        _draw_waveform(ax, lead_mv.get(l1, np.array([])),
                       left_margin+14, mid_y, lead_w, half_clip, target_samples=target_samples)

        div_x = left_margin + 14 + lead_w + div_pad
        # Removed vertical dashes completely

        right_x = div_x + div_pad
        _t(ax, l2, right_x, label_y, 10, bold=True)
        _draw_waveform(ax, lead_mv.get(l2, np.array([])),
                       right_x+5, mid_y, lead_w, half_clip, target_samples=target_samples)

    rhythm_mid = start_y + 6*row_h + row_h/2.0
    _draw_calibration_pad(ax, left_margin-4, rhythm_mid, FIXED_GAIN)
    _t(ax, "II", left_margin+10, rhythm_mid-9, 12, bold=True)
    _draw_waveform(ax, lead_mv.get("II", np.array([])),
                   left_margin+14, rhythm_mid,
                   PW - left_margin - MR - 25, row_h*0.90, target_samples=5000)


# ─── 4:3 Landscape ────────────────────────────────────────────────────────────

def _draw_3x4(ax, lead_mv, PW, PH, target_samples=None):
    HEADER_H  = 25.0
    FOOTER_H  = 20.0
    start_y   = MT + HEADER_H
    bot_limit = PH - MB - FOOTER_H
    usable_h  = bot_limit - start_y
    row_h     = min(usable_h / 5.0, 30.0)   # 4 rows + rhythm strip

    left_margin = ML + 8.0
    left_pad    = left_margin + 10.0
    lead_w      = 80.0
    div_pad     = 5.0

    lead_groups = [
        ["I","II","III"],
        ["aVR","aVL","aVF"],
        ["V1","V2","V3"],
        ["V4","V5","V6"],
    ]

    # Draw full-height dashed column divider lines (matches reference image)
    col_dividers = []
    usable_top = start_y
    usable_bot = start_y + 4*min(usable_h / 5.0, 30.0) + min(usable_h / 5.0, 30.0)

    for r, group in enumerate(lead_groups):
        mid_y   = start_y + r*row_h + row_h/2.0
        label_y = mid_y - 9.0
        # half_clip: 90% of row half-height to show true peak amplitude
        half_clip = row_h * 0.90
        _draw_calibration_pad(ax, left_margin-4, mid_y, FIXED_GAIN)
        for c, lead in enumerate(group):
            x_start = left_pad if c == 0 else left_pad + c*(lead_w+div_pad+div_pad)
            _t(ax, lead, x_start, label_y, 10.5, bold=True)
            _draw_waveform(ax, lead_mv.get(lead, np.array([])),
                           x_start, mid_y, lead_w, half_clip, target_samples=target_samples)
            if c < 2:
                div_x = x_start + lead_w + div_pad
                if div_x not in col_dividers:
                    col_dividers.append(div_x)

    # Draw full-column-height dashed dividers in one pass
    strip_top = start_y
    strip_bot = start_y + 5*row_h
    for div_x in col_dividers:
        ax.plot([div_x, div_x], [strip_top, strip_bot],
                color='black', linewidth=0.6,
                linestyle=(0, (4, 4)), zorder=4)

    rhythm_mid = start_y + 4*row_h + row_h/2.0
    _draw_calibration_pad(ax, left_margin-4, rhythm_mid, FIXED_GAIN)
    _t(ax, "II", left_margin+10, rhythm_mid-9, 12.5, bold=True)
    _draw_waveform(ax, lead_mv.get("II", np.array([])),
                   left_margin+14, rhythm_mid,
                   PW - left_margin - MR - 25, row_h*0.90, target_samples=5000)


# ─── Footer ───────────────────────────────────────────────────────────────────

def _draw_footer(ax, frozen, patient, conc_list, PW, PH, is_portrait):
    if is_portrait:
        _draw_footer_portrait(ax, frozen, patient, conc_list, PW, PH)
    else:
        _draw_footer_landscape(ax, frozen, patient, conc_list, PW, PH)


def _draw_footer_portrait(ax, frozen, patient, conc_list, PW, PH):
    footer_y = PH - MB - 25.0   # 267mm

    # Doctor sign section (left)
    doc_name = patient.get('doctor_name','') or ''
    _t(ax, "Reference Report Confirmed by:",
       ML, footer_y+8, 8)
    _t(ax, f"Doctor Name: {doc_name}",
       ML, footer_y+14, 8)
    _t(ax, "Doctor Sign:",
       ML, footer_y+20, 8)

    # Conclusion box — TRANSPARENT (grid shows through)
    box_x = 95.0
    box_y = footer_y + 5.0
    box_w = PW - box_x - MR - 5.0
    box_h = 18.0
    ax.add_patch(Rectangle((box_x, box_y), box_w, box_h,
                            linewidth=0.8, edgecolor='black',
                            facecolor='none', zorder=8))

    _t(ax, "CONCLUSION",
       box_x+box_w/2, box_y+1, 7, bold=True, ha='center', zorder=9)

    # Max 5 conclusions, 3 columns
    items = conc_list[:5]
    cols  = 3
    col_w = (box_w - 4.0) / cols
    row_h = 3.5
    sx    = box_x + 2.0
    sy    = box_y + 6.0
    for i, line in enumerate(items):
        row = i // cols; col = i % cols
        tx  = sx + col*col_w
        ty  = sy + row*row_h
        if ty + row_h > box_y + box_h: break
        _t(ax, f"{i+1}. {line}", tx, ty, 6, zorder=9)

    # Brand line
    brand = "Deckmount Electronics Pvt Ltd  |  Rhythm Ultra ECG  |  IEC 60601  |  Made in India"
    _t(ax, brand, PW/2, PH-MB+1.5, 5.5, ha='center', zorder=9)


def _draw_footer_landscape(ax, frozen, patient, conc_list, PW, PH):
    footer_top_y = PH - MB - 15.0

    # Doctor sign section (left)
    doc_name = patient.get('doctor_name','') or ''
    _t(ax, "Reference Report Confirmed by:",
       ML+5, footer_top_y, 8)
    _t(ax, f"Doctor Name: {doc_name}",
       ML+5, footer_top_y+5, 8)
    _t(ax, "Doctor Sign:",
       ML+5, footer_top_y+10, 8)

    # Conclusion box — TRANSPARENT
    box_w = 155.0; box_h = 22.0
    box_x = PW - box_w - MR - 7.0
    box_y = footer_top_y - 5.0
    ax.add_patch(Rectangle((box_x, box_y), box_w, box_h,
                            linewidth=0.8, edgecolor='black',
                            facecolor='none', zorder=8))

    _t(ax, "CONCLUSION",
       box_x+box_w/2, box_y+2, 8, bold=True, ha='center', zorder=9)

    # Max 5 conclusions, 3 columns
    items   = conc_list[:5]
    cols    = 3; col_gap = 5.0
    col_w   = (box_w - 10.0 - col_gap*2) / cols
    sx      = box_x+5.0; sy = box_y+7.0; row_gap=5.0
    for i, txt in enumerate(items):
        row=i//cols; col=i%cols
        tx = sx + col*(col_w+col_gap)
        ty = sy + row*row_gap
        if ty+row_gap > box_y+box_h-1: break
        _t(ax, f"{i+1}. {txt}", tx, ty, 6, zorder=9)

    # Brand line
    brand = "Deckmount Electronics Pvt Ltd  |  Rhythm Ultra ECG  |  IEC 60601  |  Made in India"
    _t(ax, brand, PW/2, PH-MB+1.5, 5.5, ha='center', zorder=9)


# ─── Calibration pulse ────────────────────────────────────────────────────────

def _draw_calibration(ax, x_mm, y_mm, gain_mm):
    pts = [(x_mm,   y_mm), (x_mm+2, y_mm),
           (x_mm+2, y_mm-gain_mm), (x_mm+7, y_mm-gain_mm),
           (x_mm+7, y_mm), (x_mm+9, y_mm)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            color='black', linewidth=0.8,
            solid_capstyle='projecting', zorder=6)

def _draw_calibration_pad(ax, x_mm, y_mm, gain_mm):
    _draw_calibration(ax, x_mm+4, y_mm, gain_mm)


# ─── Waveform ─────────────────────────────────────────────────────────────────

def _draw_waveform(ax, samples, x0_mm, y0_mm, width_mm, half_cell_mm=10.0, target_samples=None):
    arr = np.asarray(samples, dtype=float)
    if len(arr) < 2:
        return
    if target_samples is not None:
        if len(arr) > target_samples:
            arr = arr[-target_samples:]
    else:
        n_max = int(width_mm / MM_PER_SAMPLE) + 1
        arr = arr[:n_max]
        
    if len(arr) < 2:
        return

    # Apply Gaussian smoothing to match live display (SMOOTH_SIGMA = 0.8)
    try:
        from scipy.ndimage import gaussian_filter1d
        arr = gaussian_filter1d(arr, sigma=0.8)
    except Exception:
        pass

    # X coordinate strictly respects the scale
    xs = x0_mm + np.arange(len(arr)) * MM_PER_SAMPLE
    
    # Y axis: ADC units → mm using dynamic ADC_PER_MM (set from wave_gain in generate_report)
    ys = y0_mm - arr / ADC_PER_MM
    # Clip at the full cell boundary (half_cell_mm from caller already accounts for
    # full allowed height). Do NOT use the old max(..., 8.0) floor — that was
    # the root cause of V4/V5 flat-topped peaks.
    ys = np.clip(ys, y0_mm - half_cell_mm, y0_mm + half_cell_mm)

    ax.plot(xs, ys, color='black', linewidth=0.5,
            solid_joinstyle='round', solid_capstyle='round', zorder=5)


# ─── Text helper ──────────────────────────────────────────────────────────────

def _t(ax, text, x_mm, y_mm, pt_size,
       bold=False, italic=False, color='black',
       ha='left', zorder=7):
    ax.text(x_mm, y_mm, text,
            fontsize=pt_size,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            color=color, va='top', ha=ha, zorder=zorder)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _rr_ms(frozen):
    hr = frozen.get('HR', 0)
    if hr and hr > 0:
        return int(round(60000.0 / hr))
    return frozen.get('RR', 0)

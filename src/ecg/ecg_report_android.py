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
    ac_freq = f"{ac_setting}Hz" if ac_setting in ("50", "60") else "Off"
    if dft_setting not in ("off", "") and emg_setting not in ("off", ""):
        filter_band = f"{dft_setting}-{emg_setting}Hz"
    elif dft_setting not in ("off", ""):
        filter_band = f"HP:{dft_setting}Hz"
    elif emg_setting not in ("off", ""):
        filter_band = f"LP:{emg_setting}Hz"
    else:
        filter_band = "Filter: Off"
    frozen = dict(frozen or {})
    frozen["filter_band"] = filter_band
    frozen["ac_frequency"] = ac_freq
    frozen["speed_text"] = f"{FIXED_SPEED:.1f} mm/s"
    frozen["gain_text"] = f"{FIXED_GAIN:.1f} mm/mV"
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

    if is_portrait:
        dt = patient.get('date_time', '') or ''
        date_part = dt[:10] if len(dt) >= 10 else ''
        time_part = dt[11:19] if len(dt) >= 19 else ''
        org   = (patient.get('org', '') or patient.get('Org.', '') or '').strip()
        phone = (patient.get('phone', '') or patient.get('doctor_mobile', '') or '').strip()
        filter_band = frozen.get('filter_band', '0.5-150Hz')
        ac_freq     = frozen.get('ac_frequency', '50Hz')
        speed_text  = frozen.get('speed_text', f"{FIXED_SPEED:.1f} mm/s")
        gain_text   = frozen.get('gain_text', f"{FIXED_GAIN:.1f} mm/mV")

        hr   = frozen.get('HR',  0) or 0
        rr   = _rr_ms(frozen)
        pr   = frozen.get('PR',  0) or 0
        qrs  = frozen.get('QRS', 0) or 0
        qt   = frozen.get('QT',  0) or 0
        qtc  = frozen.get('QTc', 0) or 0
        qtcf = frozen.get('QTcF',0) or 0

        rv5     = float(frozen.get('rv5', 0.0) or 0.0)
        sv1     = float(frozen.get('sv1', 0.0) or 0.0)
        idx_val = rv5 - abs(sv1)
        p_ax    = frozen.get('p_axis',   '--')
        q_ax    = frozen.get('QRS_axis', '--')
        t_ax    = frozen.get('t_axis',   '--')

        # ── 4-COLUMN layout exactly matching reference image ─────────────────
        #
        #  COL1 (x=10):  Patient info — 6 rows at 7pt
        #                At 7pt, "Date & Time: YYYY-MM-DD HH:MM:SS" ≈ 43mm wide
        #                → ends at x≈53, safely before COL2 at x=60
        #
        #  COL2 (x=60):  HR / RR / PR / QRS / QT  (5 rows, bold 8.5pt)
        #
        #  COL3 (x=100): QTc / QTcF / RV5/SV1 / RV5+SV1 / P/QRS/T (5 rows, bold 8.5pt)
        #
        #  COL4 (x=152): Org name / address / phone  (up to 3 rows, bold 8.5pt)
        #                This is the "hospital info" block visible top-right in reference
        #
        left_x  = 5.0
        col2_x  = 72.0
        col3_x  = 115.0
        col4_x  = 160.0
        row_gap = 4.8   # 5 metric rows × 4.8mm = 24mm; 6 left rows = 28.8mm

        # COL1: patient details (6 rows)
        # Use 7pt so long "Date & Time" line stays within ~50mm
        left_lines = [
            f"Name: {full}",
            f"Age: {age}",
            f"Gender: {gender}",
            f"ECG Type: {lead_seq}",
            f"Date & Time: {date_part} {time_part}".strip(),
            f"{speed_text}  {filter_band}  AC:{ac_freq}  {gain_text}",
        ]

        # COL2: primary ECG metrics (5 rows — NO QTc here)
        col2_lines = [
            f"HR: {hr} bpm",
            f"RR: {rr} ms",
            f"PR: {pr} ms",
            f"QRS: {qrs} ms",
            f"QT: {qt} ms",
        ]

        # COL3: secondary ECG metrics (5 rows)
        col3_lines = [
            f"QTc: {qtc} ms",
            f"QTcF: {qtcf} ms",
            f"RV5/SV1: {rv5:.3f}/{sv1:.3f} mV",
            f"RV5+SV1: {idx_val:.3f} mV",
            f"P/QRS/T: {p_ax}/{q_ax}/{t_ax}\u00b0",
        ]

        # COL4: hospital / org info — displayed bold, right side (rows 0-2)
        col4_lines = []
        if org:
            col4_lines.append(org)
        if phone:
            col4_lines.append(phone)

        # Draw COL1 — 9pt
        for i, text in enumerate(left_lines):
            fs = 9.0
            _t(ax, text, left_x, yb + i * row_gap, fs)

        # Draw COL2 — 9pt
        for i, text in enumerate(col2_lines):
            _t(ax, text, col2_x, yb + i * row_gap, 9.0, bold=False)

        # Draw COL3 — 9pt
        for i, text in enumerate(col3_lines):
            _t(ax, text, col3_x, yb + i * row_gap, 9.0, bold=False)

        # Draw COL4 — bold, 9pt (hospital info top-right)
        for i, text in enumerate(col4_lines[:3]):
            _t(ax, text, col4_x, yb + i * row_gap, 9.0, bold=True)

        return

    # ── LANDSCAPE 4-COLUMN layout — same structure as 12:1 portrait ─────────
    #
    #  COL1 (x=10):   Patient info — 6 rows at 7pt
    #  COL2 (x=95):   HR / RR / PR / QRS / QT  (5 rows, bold)
    #  COL3 (x=163):  QTc / QTcF / RV5/SV1 / RV5+SV1 / P/QRS/T (5 rows, bold)
    #  COL4 (x=228):  Org name / phone  (bold, top-right of header)
    #
    #  Landscape page is 297mm wide — columns spread proportionally.
    #  At 7pt, "Date & Time: YYYY-MM-DD HH:MM:SS" ≈ 43mm wide →
    #  ends at x≈53mm, safely before COL2 at x=95.
    #
    dt         = patient.get('date_time', '') or ''
    date_part  = dt[:10] if len(dt) >= 10 else ''
    time_part  = dt[11:19] if len(dt) >= 19 else ''
    org        = (patient.get('org', '') or patient.get('Org.', '') or '').strip()
    phone      = (patient.get('phone', '') or patient.get('doctor_mobile', '') or '').strip()
    filter_band = frozen.get('filter_band', '0.5-150Hz')
    ac_freq     = frozen.get('ac_frequency', '50Hz')
    speed_text  = frozen.get('speed_text', f'{FIXED_SPEED:.1f} mm/s')
    gain_text   = frozen.get('gain_text',  f'{FIXED_GAIN:.1f} mm/mV')

    hr   = frozen.get('HR',  0) or 0
    rr   = _rr_ms(frozen)
    pr   = frozen.get('PR',  0) or 0
    qrs  = frozen.get('QRS', 0) or 0
    qt   = frozen.get('QT',  0) or 0
    qtc  = frozen.get('QTc', 0) or 0
    qtcf = frozen.get('QTcF',0) or 0
    rv5     = float(frozen.get('rv5', 0.0) or 0.0)
    sv1     = float(frozen.get('sv1', 0.0) or 0.0)
    idx_val = rv5 - abs(sv1)
    p_ax    = frozen.get('p_axis',   '--')
    q_ax    = frozen.get('QRS_axis', '--')
    t_ax    = frozen.get('t_axis',   '--')

    # Column X positions calibrated against reference images 1 & 3:
    # Page = 297mm wide. Proportions from image measurement:
    #   COL1 ends ≈ x53mm  → COL2 starts at 88mm (32% of 270mm usable)
    #   COL3 starts ≈ 155mm (58% of 270mm usable)
    #   COL4 starts ≈ 213mm (79% of 270mm usable)
    L_left  = 10.0
    L_col2  = 88.0
    L_col3  = 155.0
    L_col4  = 213.0
    L_gap   = 4.5    # 6 rows × 4.5mm = 27mm → fits in HEADER_H=30mm

    ls_left = [
        f'Name: {full}',
        f'Age: {age}',
        f'Gender: {gender}',
        f'ECG Type: {lead_seq}',
        f'Date & Time: {date_part} {time_part}'.strip(),
        f'{speed_text}  {filter_band}  AC:{ac_freq}  {gain_text}',
    ]
    ls_col2 = [
        f'HR: {hr} bpm',
        f'RR: {rr} ms',
        f'PR: {pr} ms',
        f'QRS: {qrs} ms',
        f'QT: {qt} ms',
    ]
    ls_col3 = [
        f'QTc: {qtc} ms',
        f'QTcF: {qtcf} ms',
        f'RV5/SV1: {rv5:.3f}/{sv1:.3f} mV',
        f'RV5+SV1: {idx_val:.3f} mV',
        f'P/QRS/T: {p_ax}/{q_ax}/{t_ax}\u00b0',
    ]
    ls_col4 = [t for t in [org, phone] if t]

    for i, txt in enumerate(ls_left):
        fs = 9.0
        _t(ax, txt, L_left, yb + i * L_gap, fs)
    for i, txt in enumerate(ls_col2):
        _t(ax, txt, L_col2, yb + i * L_gap, 9.0, bold=False)
    for i, txt in enumerate(ls_col3):
        _t(ax, txt, L_col3, yb + i * L_gap, 9.0, bold=False)
    for i, txt in enumerate(ls_col4[:3]):
        _t(ax, txt, L_col4, yb + i * L_gap, 9.0, bold=True)


# ─── 12:1 Portrait — waves fill header→footer, no white gap ──────────────────

def _draw_1x12(ax, lead_mv, PW, PH, target_samples=None):
    HEADER_H  = 35.0   # increased from 28 → fits 6 header rows at 4.8mm each
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
    HEADER_H  = 30.0   # increased: 6 header rows × 4.5mm = 27mm fits here
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

    # Dashed vertical column divider — stops at TOP of rhythm strip (not through it)
    # Image 2 reference: dividers end cleanly where the full-width II row begins
    div_x   = left_margin + 14 + lead_w + div_pad
    right_x = div_x + div_pad
    rhythm_top = start_y + 6 * row_h   # top edge of rhythm strip row
    ax.plot([div_x, div_x], [start_y, rhythm_top],
            color='black', linewidth=0.6,
            linestyle=(0, (4, 4)), zorder=6)

    for r, (l1, l2) in enumerate(pair_map):
        mid_y   = start_y + r*row_h + row_h/2.0
        label_y = mid_y - 9.0
        half_clip = row_h * 0.90

        _draw_calibration_pad(ax, left_margin-4, mid_y, FIXED_GAIN)

        _t(ax, l1, left_margin+9, label_y, 10, bold=True)
        _draw_waveform(ax, lead_mv.get(l1, np.array([])),
                       left_margin+14, mid_y, lead_w, half_clip, target_samples=target_samples)

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
    HEADER_H  = 30.0   # increased: 6 header rows × 4.5mm = 27mm fits here
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

    rhythm_mid = start_y + 4*row_h + row_h/2.0
    _draw_calibration_pad(ax, left_margin-4, rhythm_mid, FIXED_GAIN)
    _t(ax, "II", left_margin+10, rhythm_mid-9, 12.5, bold=True)
    _draw_waveform(ax, lead_mv.get("II", np.array([])),
                   left_margin+14, rhythm_mid,
                   PW - left_margin - MR - 25, row_h*0.90, target_samples=5000)

    # Draw column dividers AFTER all waveforms, stopping at TOP of rhythm strip
    # (rhythm strip is full-width — no dividers should cross through it)
    strip_top = start_y
    strip_bot = start_y + 4*row_h   # stop at top edge of II rhythm strip row
    for div_x in col_dividers:
        ax.plot([div_x, div_x], [strip_top, strip_bot],
                color='black', linewidth=0.6,
                linestyle=(0, (4, 4)), zorder=6)


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

    # Draw conclusions in a single column to avoid overlapping
    items = conc_list[:5]
    row_h = 4.0
    sx    = box_x + 2.0
    sy    = box_y + 6.0
    for i, line in enumerate(items):
        ty  = sy + i*row_h
        # User request: If getting cropped (exceeds box height), don't put in this.
        if ty + row_h > box_y + box_h - 1.0: break 
        _t(ax, f"{i+1}. {line}", sx, ty, 9, zorder=9)

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
       box_x+box_w/2, box_y+2, 9, bold=True, ha='center', zorder=9)

    # Draw conclusions in a single column to avoid overlapping
    items   = conc_list[:5]
    sx      = box_x+5.0; sy = box_y+8.0; row_gap=5.0
    for i, txt in enumerate(items):
        ty = sy + i*row_gap
        # User request: If getting cropped (exceeds box height), don't put in this.
        if ty+row_gap > box_y+box_h-1.5: break
        _t(ax, f"{i+1}. {txt}", sx, ty, 9, zorder=9)

    # Brand line
    brand = "Deckmount Electronics Pvt Ltd  |  Rhythm Ultra ECG  |  IEC 60601  |  Made in India"
    _t(ax, brand, PW/2, PH-MB+1.5, 9, ha='center', zorder=9)


# ─── Calibration pulse ────────────────────────────────────────────────────────

def _draw_calibration(ax, x_mm, y_mm, gain_mm):
    pts = [(x_mm,   y_mm), (x_mm+2, y_mm),
           (x_mm+2, y_mm-gain_mm), (x_mm+7, y_mm-gain_mm),
           (x_mm+7, y_mm), (x_mm+9, y_mm)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            color='black', linewidth=0.8,
            solid_capstyle='butt', zorder=6)   # 'butt' = no cap beyond endpoints → no grey dot

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

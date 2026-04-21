"""
report_generator.py  (Holter)
================================
Generates a Holter report PDF that matches the 12-lead ECG report style:
  - Pink ECG-grid paper background
  - DECK⚡MOUNT logo / brand
  - Patient / vitals header (same layout as analysis_window.py)
  - 12 stacked lead strips with calibration pulse
  - HRV / summary / arrhythmia table below strips
  - Conclusion box + footer

Falls back to a plain-text report when matplotlib / reportlab are unavailable.
"""

import os
import sys
import json
import traceback
import numpy as np
from datetime import datetime
from typing import Optional
from scipy.signal import butter, filtfilt, iirnotch


# ── resource helper ──────────────────────────────────────────────────────────

def _res(relative_path: str) -> str:
    """Return absolute path; handles both dev mode and PyInstaller."""
    base = getattr(sys, '_MEIPASS',
                   os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    return os.path.join(base, relative_path)


# ── public entry point ───────────────────────────────────────────────────────

def generate_holter_report(session_dir: str,
                            patient_info: dict,
                            summary: dict,
                            settings_manager=None) -> str:
    """
    Generate holter_report.pdf in session_dir.
    Returns path to generated file (PDF or TXT fallback).
    """
    output_path = os.path.join(session_dir, 'holter_report.pdf')
    try:
        return _generate_pdf_report(session_dir, patient_info, summary, output_path)
    except Exception as e:
        print(f"[HolterReport] PDF error: {e}")
        traceback.print_exc()
        txt_path = output_path.replace('.pdf', '.txt')
        return _generate_text_report(session_dir, patient_info, summary, txt_path)


# ── PDF generator ─────────────────────────────────────────────────────────────

def _generate_pdf_report(session_dir: str,
                          patient_info: dict,
                          summary: dict,
                          output_path: str) -> str:
    """
    Render a full A4 PDF that matches the ECG 12-lead report style exactly.
    Uses matplotlib (same engine as analysis_window.py).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle as MRect

    # ── page constants (mm) ──────────────────────────────────────────────────
    PAGE_W  = 210.0;  PAGE_H  = 297.0
    ML = 10.0;  MR = 10.0;  MT = 10.0;  MB = 10.0
    HEADER_H = 30.0      # mm reserved for patient header
    FOOTER_H = 60.0      # mm for summary tables + conclusion + footer brand
    STRIP_TOP  = MT + HEADER_H
    STRIP_H    = PAGE_H - STRIP_TOP - FOOTER_H - MB
    CELL_H     = STRIP_H / 12.0

    # ECG scale constants (25 mm/s, 10 mm/mV — same as analysis_window)
    FS          = 500.0              # assumed sampling rate
    MM_PER_SAMPLE = 25.0 / FS       # horizontal mm per sample
    ADC_PER_MM  = 128.0             # ADC units per mm (same as analysis_window)
    CALIB_MM    = 10.0              # calibration pulse height mm
    HALF_CELL   = CELL_H / 2.0 - 1.0

    # ── build figure ─────────────────────────────────────────────────────────
    fig = Figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), dpi=150, facecolor='white')
    ax  = fig.add_axes([0, 0, 1, 1], facecolor='#fff5f5')
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(PAGE_H, 0)    # y increases downward (screen-like)
    ax.set_aspect('equal')

    # ── pink ECG grid ────────────────────────────────────────────────────────
    ax.set_xticks(np.arange(0, PAGE_W + 1, 5))
    ax.set_xticks(np.arange(0, PAGE_W + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, PAGE_H + 1, 5))
    ax.set_yticks(np.arange(0, PAGE_H + 1, 1), minor=True)
    ax.grid(True, which='major', color='#e09696', linewidth=0.55, zorder=1)
    ax.grid(True, which='minor', color='#f5d8d8', linewidth=0.22, zorder=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticklabels([]);  ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False, which='both')

    # ── patient header (left col) ─────────────────────────────────────────────
    yb  = MT;  lh = 5.0
    x1  = ML
    pat = patient_info or {}
    pname  = pat.get('name') or pat.get('patient_name') or '—'
    age    = pat.get('age', '—')
    gender = pat.get('gender') or pat.get('sex') or '—'
    org    = pat.get('Org.') or pat.get('org') or '—'
    phone  = pat.get('phone') or pat.get('doctor_mobile') or '—'

    ax.text(x1, yb,       f"Name: {pname}",     fontsize=9, va='top')
    ax.text(x1, yb+lh,    f"Age: {age}",         fontsize=9, va='top')
    ax.text(x1, yb+lh*2,  f"Gender: {gender}",   fontsize=9, va='top')
    ax.text(x1, yb+lh*3,  f"Org: {org}",         fontsize=9, fontweight='bold', va='top')
    ax.text(x1, yb+lh*4,  f"Phone: {phone}",     fontsize=9, fontweight='bold', va='top')

    # ── vitals (middle col) ──────────────────────────────────────────────────
    x2 = x1 + 55
    avg_hr  = summary.get('avg_hr', 0)
    min_hr  = summary.get('min_hr', 0)
    max_hr  = summary.get('max_hr', 0)
    sdnn    = summary.get('sdnn', 0)
    rmssd   = summary.get('rmssd', 0)
    pnn50   = summary.get('pnn50', 0)
    dur_sec = summary.get('duration_sec', 0)
    dur_h   = int(dur_sec // 3600)
    dur_m   = int((dur_sec % 3600) // 60)

    vitals = [
        f"Avg HR : {avg_hr:.0f} bpm",
        f"Min HR : {min_hr:.0f} bpm",
        f"Max HR : {max_hr:.0f} bpm",
        f"Duration: {dur_h}h {dur_m}m",
        f"SDNN: {sdnn:.1f} ms  rMSSD: {rmssd:.1f} ms",
    ]
    for i, v in enumerate(vitals):
        ax.text(x2, yb + i * lh, v, fontsize=9, va='top')

    # ── DECK⚡MOUNT brand (top-right) ────────────────────────────────────────
    ax.text(PAGE_W - MR, yb,       "DECK\u26a1MOUNT",
            fontsize=10, fontweight='bold', color='#0000cc', ha='right', va='top')
    ax.text(PAGE_W - MR, yb+lh*2, "HOLTER ECG ANALYSIS REPORT",
            fontsize=9, ha='right', va='top', color='#333', fontweight='bold')
    ax.text(PAGE_W - MR, yb+lh*3, "25.0mm/s  0.5–40Hz  AC:50Hz  10.0mm/mV",
            fontsize=9, ha='right', va='top', color='#555')
    ax.text(PAGE_W - MR, yb+lh*4,
            f"Date & Time: {datetime.now().strftime('%d/%m/%Y  %H:%M')}",
            fontsize=9, ha='right', va='top', color='#555')

    # ── 12-lead ECG strips ───────────────────────────────────────────────────
    LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

    # Try to get lead data from replay engine / session
    lead_data = _load_lead_data(session_dir)

    for i, lead in enumerate(LEADS):
        mid_y = STRIP_TOP + i * CELL_H + CELL_H / 2.0
        lbl_y = mid_y - CELL_H * 0.4

        # Calibration square pulse (1 mV → 10 mm)
        cx, cy, cg = ML, mid_y, CALIB_MM
        ax.plot([cx, cx+2, cx+2, cx+7, cx+7, cx+9],
                [cy, cy,  cy-cg, cy-cg, cy,  cy],
                color='black', linewidth=0.9, zorder=6)

        # Lead label
        ax.text(ML + 11, lbl_y, lead,
                fontsize=6.5, fontweight='bold', color='black', va='top', zorder=7)

        # Waveform (if available)
        data_arr = lead_data.get(lead, np.array([]))
        if len(data_arr) > 10:
            seg      = np.asarray(data_arr, dtype=float)
            
            seg = _apply_ecg_filters(seg, fs=FS)
            
            baseline = float(np.median(seg))
            seg_mm   = (seg - baseline) / ADC_PER_MM       # ADC → mm
            wx0      = ML + 13.0
            wx_mm    = wx0 + np.arange(len(seg)) * MM_PER_SAMPLE
            mask     = wx_mm <= (PAGE_W - MR)
            wx_mm    = wx_mm[mask]
            wy_mm    = mid_y - seg_mm[:len(wx_mm)]         # upward = smaller y
            ax.plot(wx_mm, wy_mm, color='black', linewidth=0.5, zorder=5)

    # ── Summary / HRV / Arrhythmia tables (footer zone) ─────────────────────
    ft_top = PAGE_H - MB - FOOTER_H + 4.0   # top of footer area
    lh2 = 4.2

    # --- HRV block ---
    ax.text(ML, ft_top, "HRV SUMMARY",
            fontsize=6.5, fontweight='bold', va='top', color='#0000cc')
    hrv_lines = [
        f"SDNN: {sdnn:.1f} ms",
        f"rMSSD: {rmssd:.1f} ms",
        f"pNN50: {pnn50:.2f}%",
        f"Total Beats: {summary.get('total_beats', 0):,}",
        f"Pauses: {summary.get('pauses', 0)}",
    ]
    for j, line in enumerate(hrv_lines):
        ax.text(ML, ft_top + 5 + j * lh2, line, fontsize=6, va='top')

    # --- Arrhythmia block ---
    ax2_x = ML + 45
    ax.text(ax2_x, ft_top, "ARRHYTHMIA SUMMARY",
            fontsize=6.5, fontweight='bold', va='top', color='#0000cc')
    arrhy = summary.get('arrhythmia_counts', {})
    if arrhy:
        for j, (label, count) in enumerate(
                sorted(arrhy.items(), key=lambda x: -x[1])[:8]):
            ax.text(ax2_x, ft_top + 5 + j * lh2,
                    f"• {label}: {count} episodes", fontsize=6, va='top')
    else:
        ax.text(ax2_x, ft_top + 5, "No significant arrhythmias detected.",
                fontsize=6, va='top')

    # --- Doctor signature block ---
    sig_x = PAGE_W - MR - 70
    ax.text(sig_x, ft_top - 7,  "Reference Report Confirmed by:",
            fontsize=8, va='top', color='black')
    ax.text(sig_x, ft_top,      "Doctor Name: _______________________",
            fontsize=8, va='top', color='black')
    ax.text(sig_x, ft_top + 7,  "Doctor Sign:  _______________________",
            fontsize=8, va='top', color='black')

    # ── Conclusion box ────────────────────────────────────────────────────────
    conc_y  = ft_top + 28.0
    bx, by  = ML + 44, conc_y
    bw, bh  = PAGE_W - bx - MR, 18.0
    rect = MRect((bx, by), bw, bh,
                 linewidth=0.8, edgecolor='black', facecolor='white', zorder=8)
    ax.add_patch(rect)
    ax.text(bx + bw / 2, by + 1.0, "CONCLUSION",
            fontsize=7, fontweight='bold', ha='center', va='top', zorder=9)

    conclusions = _build_conclusions(summary)
    cols2 = 3;  col_w2 = (bw - 4.0) / cols2;  rh2b = 4.0
    for idx, line in enumerate(conclusions):
        tx = bx + 2.0
        ty = by + 6.0 + idx * rh2b
        # User request: If getting cropped (exceeds box height), don't put in this.
        if ty + rh2b > by + bh - 1.0:
            break
        ax.text(tx, ty, f"{idx+1}. {line}", fontsize=9, va='top', zorder=9)

    # ── Footer brand line ────────────────────────────────────────────────────
    brand = "Deckmount Electronics Pvt Ltd | RhythmPro ECG | IEC 60601 | Made in India"
    ax.text(PAGE_W / 2, PAGE_H - MB + 1, brand,
            fontsize=6, ha='center', va='top', color='#333', zorder=9)

    # ── Save PDF ────────────────────────────────────────────────────────────
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    print(f"[HolterReport] PDF saved: {output_path}")
    return output_path


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_lead_data(session_dir: str) -> dict:
    """
    Try to load representative ECG data for all 12 leads from the session.
    Returns dict:  lead_name -> np.ndarray (ADC units).
    Falls back to empty arrays if unavailable.
    """
    leads = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

    # 1) Try .ecgh replay engine (same file the analysis window uses)
    ecgh_path = os.path.join(session_dir, 'recording.ecgh')
    if os.path.exists(ecgh_path):
        try:
            # HolterReplayEngine lives at replay_engine.py next to us
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from replay_engine import HolterReplayEngine
            engine = HolterReplayEngine(ecgh_path)
            window_sec = min(10.0, engine.duration_sec)
            all_data = engine.get_all_leads_data(window_sec=window_sec)
            # all_data is a list of 12 arrays; convert to ADC-like units
            result = {}
            for i, lead in enumerate(leads):
                if i < len(all_data):
                    arr = np.asarray(all_data[i], dtype=float)
                    # Normalise to ADC scale so existing ADC_PER_MM scaling works
                    p = float(np.percentile(np.abs(arr - np.median(arr)), 95)) or 1.0
                    result[lead] = (arr / p) * 640.0   # 1 mV ≈ 640 ADC
                else:
                    result[lead] = np.array([])
            return result
        except Exception as e:
            print(f"[HolterReport] Could not load ecgh: {e}")

    # 2) Try a saved ECG JSON (ecg_data_*.json) in the session dir
    for fname in sorted(os.listdir(session_dir), reverse=True):
        if fname.endswith('.json') and 'ecg' in fname.lower():
            fpath = os.path.join(session_dir, fname)
            try:
                with open(fpath) as f:
                    d = json.load(f)
                if 'leads' in d:
                    return {k: np.asarray(v) for k, v in d['leads'].items()}
            except Exception:
                pass

    # 3) Return empty arrays — page still renders pink grid + header
    return {lead: np.array([]) for lead in leads}
def _apply_ecg_filters(sig: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """
    Same filters as smooth_display.py IncrementalFilter:
      - Bandpass 0.5-40 Hz  (baseline wander + EMG removal)
      - Notch 50 Hz         (powerline noise removal)
    Uses filtfilt (zero-phase) — ideal for static report rendering.
    """
    if len(sig) < 30:
        return sig
    nyq = fs / 2.0

    # Notch filter — 50 Hz
    b_notch, a_notch = iirnotch(50.0 / nyq, Q=35.0)
    sig = filtfilt(b_notch, a_notch, sig)

    # Bandpass filter — 0.5 to 40 Hz
    b_bp, a_bp = butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
    sig = filtfilt(b_bp, a_bp, sig)

    return sig


def _build_conclusions(summary: dict) -> list:
    """
    Build conclusion lines exactly as the user specified.
    - 1. Primary Rhythm
    - 2. Morphology additions
    """
    report_lines = []

    arrhythmias = list(summary.get('secondary_findings') or summary.get('arrhythmias') or [])
    primary = summary.get('primary_rhythm') or summary.get('primary_diagnosis')
    if primary and primary not in arrhythmias:
        arrhythmias.insert(0, primary)

    priority = [
        "Ventricular Fibrillation",
        "Atrial Fibrillation",
        "Atrial Flutter",
        "Supraventricular Tachycardia",
        "AV Block"
    ]

    diagnosis = "Normal Sinus Rhythm"

    for p in priority:
        for a in arrhythmias:
            if p in str(a):
                diagnosis = str(a)
                break
        if diagnosis != "Normal Sinus Rhythm":
            break

    report_lines.append(diagnosis)
    
    for a in arrhythmias:
        sanitized = str(a)
        if sanitized and sanitized not in report_lines and sanitized != diagnosis:
            # Fix 6 sub-patch: Strip possible
            if "Possible" not in sanitized:
                report_lines.append(sanitized)

    # Fallback if somehow completely empty
    if not report_lines:
        report_lines.append("Unknown Rhythm")

    return report_lines


def _generate_text_report(session_dir, patient_info, summary, output_path) -> str:
    """Fallback plain-text report when matplotlib is unavailable."""
    pat = patient_info or {}
    lines = [
        "=" * 60,
        "HOLTER ECG REPORT — DECK\u26a1MOUNT",
        "=" * 60,
        f"Patient : {pat.get('name', 'Unknown')}",
        f"Age/Sex : {pat.get('age', '—')} / {pat.get('gender', '—')}",
        f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "RECORDING SUMMARY",
        f"  Duration : {summary.get('duration_sec', 0) / 3600:.1f} h",
        f"  Avg HR   : {summary.get('avg_hr', 0):.0f} bpm",
        f"  Min HR   : {summary.get('min_hr', 0):.0f} bpm",
        f"  Max HR   : {summary.get('max_hr', 0):.0f} bpm",
        f"  Beats    : {summary.get('total_beats', 0):,}",
        "",
        "HRV",
        f"  SDNN  : {summary.get('sdnn', 0):.1f} ms",
        f"  rMSSD : {summary.get('rmssd', 0):.1f} ms",
        f"  pNN50 : {summary.get('pnn50', 0):.2f}%",
        "",
        "ARRHYTHMIAS",
    ]
    arrhy = summary.get('arrhythmia_counts', {})
    if arrhy:
        for label, count in arrhy.items():
            lines.append(f"  {label}: {count}")
    else:
        lines.append("  None detected")

    lines += [
        "",
        "=" * 60,
        "Doctor Signature: _______________",
        "Date: _______________",
        "",
        "Deckmount Electronics Pvt Ltd | RhythmPro ECG | Made in India",
    ]
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    return output_path

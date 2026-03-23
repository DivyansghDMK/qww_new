"""
ecg/holter/report_generator.py
================================
Generates a clinical Holter report PDF from a completed recording session.

Reads:  metrics.jsonl (produced by HolterAnalysisWorker)
        recording.ecgh (for representative ECG strip)
Writes: holter_report.pdf

Reuses existing infrastructure:
  - ecg_report_generator.py utilities (patient info formatting, PDF setup)
  - matplotlib for charts (already a dependency)
"""

import os
import sys
import json
import time
import traceback
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

# Add project root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def generate_holter_report(session_dir: str,
                            patient_info: dict,
                            summary: dict,
                            settings_manager=None) -> str:
    """
    Main entry point. Generates holter_report.pdf in session_dir.
    Returns path to generated PDF.
    """
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.lib.units import mm, cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image, PageBreak,
                                         HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        _has_reportlab = True
    except ImportError:
        _has_reportlab = False

    output_path = os.path.join(session_dir, 'holter_report.pdf')

    if not _has_reportlab:
        # Fallback: text report
        _generate_text_report(session_dir, patient_info, summary, output_path.replace('.pdf', '.txt'))
        return output_path.replace('.pdf', '.txt')

    try:
        return _generate_pdf_report(session_dir, patient_info, summary,
                                     output_path, settings_manager)
    except Exception as e:
        print(f"[HolterReport] PDF generation error: {e}")
        traceback.print_exc()
        return _generate_text_report(session_dir, patient_info, summary,
                                      output_path.replace('.pdf', '.txt'))


# ── PDF Report ─────────────────────────────────────────────────────────────────

def _generate_pdf_report(session_dir, patient_info, summary, output_path, settings_manager):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image, PageBreak,
                                     HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    PAGE_W, PAGE_H = A4
    ORANGE = colors.HexColor('#E65100')
    DARK   = colors.HexColor('#1A1A2E')
    LIGHT  = colors.HexColor('#FFF8F0')
    GRAY   = colors.HexColor('#F5F5F5')
    GREEN  = colors.HexColor('#2E7D32')
    RED    = colors.HexColor('#B71C1C')
    BLUE   = colors.HexColor('#1565C0')

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
        title="Holter ECG Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Normal'],
                                  fontSize=20, textColor=ORANGE, bold=True,
                                  alignment=TA_CENTER, spaceAfter=4*mm)
    h1_style = ParagraphStyle('H1', parent=styles['Normal'],
                               fontSize=13, textColor=DARK, bold=True,
                               spaceBefore=6*mm, spaceAfter=2*mm,
                               borderPad=(0, 0, 2, 0))
    h2_style = ParagraphStyle('H2', parent=styles['Normal'],
                               fontSize=11, textColor=BLUE, bold=True,
                               spaceBefore=4*mm, spaceAfter=1*mm)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=9, textColor=DARK, spaceAfter=1*mm)
    small_style = ParagraphStyle('Small', parent=styles['Normal'],
                                  fontSize=8, textColor=colors.gray)

    story = []

    # ── Cover / Header ─────────────────────────────────────────────────────────
    story.append(Paragraph("HOLTER ECG REPORT", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))

    # Patient info table
    dur_h = int(summary.get('duration_sec', 0) // 3600)
    dur_m = int((summary.get('duration_sec', 0) % 3600) // 60)
    pname = patient_info.get('name', patient_info.get('patient_name', 'Unknown'))
    pinfo_data = [
        ['Patient Name', pname,              'Recording Duration', f"{dur_h}h {dur_m}m"],
        ['Age / Gender', f"{patient_info.get('age','—')} / {patient_info.get('gender','—')}",
         'Report Date', datetime.now().strftime('%Y-%m-%d %H:%M')],
        ['Doctor',       patient_info.get('doctor', '—'),
         'Organisation', patient_info.get('Org.', patient_info.get('org', '—'))],
        ['Email',        patient_info.get('email', '—'),
         'Phone',        patient_info.get('phone', patient_info.get('doctor_mobile', '—'))],
    ]

    pinfo_table = Table(pinfo_data, colWidths=[35*mm, 55*mm, 45*mm, 45*mm])
    pinfo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), GRAY),
        ('BACKGROUND', (2, 0), (2, -1), GRAY),
        ('TEXTCOLOR', (0, 0), (-1, -1), DARK),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, LIGHT]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(pinfo_table)
    story.append(Spacer(1, 4*mm))

    # ── Section 1: Overall Summary ─────────────────────────────────────────────
    story.append(Paragraph("1. RECORDING SUMMARY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    total_beats = summary.get('total_beats', 0)
    avg_hr      = summary.get('avg_hr', 0)
    max_hr      = summary.get('max_hr', 0)
    min_hr      = summary.get('min_hr', 0)
    pauses      = summary.get('pauses', 0)
    longest_rr  = summary.get('longest_rr_ms', 0)

    stats_data = [
        ['Parameter', 'Value', 'Parameter', 'Value'],
        ['Total Beats', f"{total_beats:,}",      'Avg Heart Rate',    f"{avg_hr:.0f} bpm"],
        ['Max Heart Rate', f"{max_hr:.0f} bpm",  'Min Heart Rate',    f"{min_hr:.0f} bpm"],
        ['Longest RR', f"{longest_rr:.0f} ms",   'Pauses (RR>2s)',    str(pauses)],
        ['Avg Quality', f"{summary.get('avg_quality',1)*100:.0f}%",
         'Chunks Analyzed', str(summary.get('chunks_analyzed', 0))],
    ]
    stats_table = Table(stats_data, colWidths=[45*mm, 35*mm, 45*mm, 35*mm])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ORANGE),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',   (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',   (2, 1), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(stats_table)

    story.append(Paragraph("Clinical Impression", h2_style))
    arrhy_counts = summary.get('arrhythmia_counts', {})
    top_events = ", ".join(f"{label} ({count})" for label, count in sorted(arrhy_counts.items(), key=lambda item: -item[1])[:4]) or "No significant arrhythmias detected"
    avg_quality = summary.get('avg_quality', 0) * 100
    impression_text = (
        f"This Holter study for <b>{pname}</b> covers <b>{dur_h}h {dur_m}m</b> with an average heart rate of "
        f"<b>{avg_hr:.0f} bpm</b> (minimum <b>{min_hr:.0f} bpm</b>, maximum <b>{max_hr:.0f} bpm</b>). "
        f"Overall signal quality was <b>{avg_quality:.1f}%</b>. The automated event summary shows: <b>{top_events}</b>."
    )
    story.append(Paragraph(impression_text, body_style))

    # ── Section 2: HRV Analysis ────────────────────────────────────────────────
    story.append(Paragraph("2. HEART RATE VARIABILITY (HRV)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    sdnn  = summary.get('sdnn', 0)
    rmssd = summary.get('rmssd', 0)
    pnn50 = summary.get('pnn50', 0)

    def hrv_status(sdnn):
        if sdnn > 100: return 'Normal', GREEN
        if sdnn > 50:  return 'Borderline', colors.orange
        return 'Reduced', RED

    hrv_label, hrv_color = hrv_status(sdnn)

    hrv_data = [
        ['Metric', 'Value', 'Reference', 'Status'],
        ['SDNN',   f"{sdnn:.1f} ms",  '>100 ms',  hrv_label],
        ['rMSSD',  f"{rmssd:.1f} ms", '>42 ms',   'Normal' if rmssd > 42 else 'Low'],
        ['pNN50',  f"{pnn50:.2f}%",   '>20%',     'Normal' if pnn50 > 20 else 'Low'],
    ]
    hrv_table = Table(hrv_data, colWidths=[40*mm, 35*mm, 35*mm, 30*mm])
    hrv_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',   (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('TEXTCOLOR', (3, 1), (3, 1), hrv_color),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(hrv_table)

    # ── Section 3: Arrhythmia Summary ─────────────────────────────────────────
    story.append(Paragraph("3. ARRHYTHMIA SUMMARY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    if arrhy_counts:
        arrhy_data = [['Arrhythmia Type', 'Episodes', 'Burden']]
        total_chunks = max(1, summary.get('chunks_analyzed', 1))
        for label, count in sorted(arrhy_counts.items(), key=lambda x: -x[1]):
            burden = f"{count / total_chunks * 100:.1f}%"
            arrhy_data.append([label, str(count), burden])

        arrhy_table = Table(arrhy_data, colWidths=[90*mm, 30*mm, 30*mm])
        arrhy_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), RED),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FFEBEE')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(arrhy_table)
    else:
        story.append(Paragraph("No significant arrhythmias detected during this recording.", body_style))

    # ── Section 4: Hourly HR Chart ─────────────────────────────────────────────
    story.append(Paragraph("4. HOURLY HEART RATE TREND", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    hourly_chart_path = _generate_hourly_hr_chart(session_dir, summary.get('hourly_hr', {}))
    if hourly_chart_path and os.path.exists(hourly_chart_path):
        story.append(Image(hourly_chart_path, width=170*mm, height=55*mm))
    else:
        story.append(Paragraph("Hourly HR chart not available.", small_style))

    # ── Section 5: Interval Statistics ────────────────────────────────────────
    jsonl_path = os.path.join(session_dir, 'metrics.jsonl')
    interval_stats = _compute_interval_stats(jsonl_path)

    story.append(Paragraph("5. ECG INTERVAL STATISTICS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    int_data = [['Interval', 'Mean', 'Std Dev', 'Min', 'Max', 'Normal Range']]
    for label, key, ref in [
        ('PR Interval',  'pr_ms',  '120–200 ms'),
        ('QRS Duration', 'qrs_ms', '60–120 ms'),
        ('QT Interval',  'qt_ms',  '350–450 ms'),
        ('QTc Interval', 'qtc_ms', '<440 ms'),
    ]:
        vals = interval_stats.get(key, [])
        if vals:
            int_data.append([label,
                             f"{np.mean(vals):.0f} ms",
                             f"{np.std(vals):.0f} ms",
                             f"{np.min(vals):.0f} ms",
                             f"{np.max(vals):.0f} ms",
                             ref])
        else:
            int_data.append([label, '—', '—', '—', '—', ref])

    int_table = Table(int_data, colWidths=[35*mm, 22*mm, 22*mm, 22*mm, 22*mm, 32*mm])
    int_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GREEN),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',   (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(int_table)

    # ── Section 6: Conclusion ──────────────────────────────────────────────────
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("6. PHYSICIAN INTERPRETATION", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    auto_conclusion = _auto_conclusion(summary)
    story.append(Paragraph(auto_conclusion, body_style))
    story.append(Spacer(1, 10*mm))

    # Signature box
    sig_data = [
        ['Physician Signature', 'Date', 'Stamp'],
        ['', datetime.now().strftime('%Y-%m-%d'), ''],
    ]
    sig_table = Table(sig_data, colWidths=[70*mm, 50*mm, 60*mm])
    sig_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 0), (-1, 0), [GRAY]),
        ('MINROWHEIGHT', (0, 1), (-1, 1), 20*mm),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(sig_table)

    doc.build(story)
    print(f"[HolterReport] PDF saved: {output_path}")
    return output_path


# ── Helper functions ────────────────────────────────────────────────────────────

def _generate_hourly_hr_chart(session_dir: str, hourly_hr: dict) -> str:
    """Generate bar chart of hourly mean HR, save as PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not hourly_hr:
            return ""

        hours = sorted(hourly_hr.keys())
        values = [hourly_hr[h] for h in hours]

        fig, ax = plt.subplots(figsize=(10, 3))
        bars = ax.bar(hours, values, color='#1565C0', alpha=0.8, width=0.7)
        ax.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='60 bpm')
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100 bpm')
        ax.set_xlabel('Hour of Recording', fontsize=9)
        ax.set_ylabel('Mean HR (bpm)', fontsize=9)
        ax.set_title('Hourly Heart Rate Trend', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(hours)
        ax.tick_params(labelsize=8)
        plt.tight_layout()

        chart_path = os.path.join(session_dir, 'hourly_hr_chart.png')
        plt.savefig(chart_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return chart_path
    except Exception as e:
        print(f"[HolterReport] Chart error: {e}")
        return ""


def _compute_interval_stats(jsonl_path: str) -> dict:
    """Load all interval values from JSONL for statistics."""
    stats = {'pr_ms': [], 'qrs_ms': [], 'qt_ms': [], 'qtc_ms': []}
    if not os.path.exists(jsonl_path):
        return stats
    try:
        with open(jsonl_path) as f:
            for line in f:
                m = json.loads(line.strip())
                for key in stats:
                    val = m.get(key, 0)
                    if val > 0:
                        stats[key].append(val)
    except Exception:
        pass
    return stats


def _auto_conclusion(summary: dict) -> str:
    """Generate an auto-summary conclusion text."""
    lines = []
    avg_hr = summary.get('avg_hr', 0)
    max_hr = summary.get('max_hr', 0)
    min_hr = summary.get('min_hr', 0)
    sdnn   = summary.get('sdnn', 0)
    arrhy  = summary.get('arrhythmia_counts', {})
    pauses = summary.get('pauses', 0)

    dur_h = int(summary.get('duration_sec', 0) // 3600)
    dur_m = int((summary.get('duration_sec', 0) % 3600) // 60)
    lines.append(f"Recording duration: {dur_h}h {dur_m}m. "
                 f"Average HR: {avg_hr:.0f} bpm (Max: {max_hr:.0f}, Min: {min_hr:.0f} bpm).")

    if sdnn > 100:
        lines.append("Heart rate variability (SDNN) is within normal limits.")
    elif sdnn > 50:
        lines.append("Heart rate variability (SDNN) is borderline reduced.")
    else:
        lines.append("Heart rate variability (SDNN) is significantly reduced — clinical correlation advised.")

    if not arrhy:
        lines.append("No significant arrhythmias detected during this recording.")
    else:
        arrhy_list = ', '.join(f"{k} ({v} episode{'s' if v>1 else ''})" for k, v in arrhy.items())
        lines.append(f"Arrhythmias detected: {arrhy_list}.")

    if pauses > 0:
        lines.append(f"Pauses (RR interval >2.0s): {pauses} episode(s) detected.")

    lines.append("\n[Physician to review and add clinical interpretation above.]")
    return " ".join(lines)


def _generate_text_report(session_dir, patient_info, summary, output_path) -> str:
    """Fallback plain-text report when reportlab is unavailable."""
    lines = [
        "=" * 60,
        "HOLTER ECG REPORT",
        "=" * 60,
        f"Patient: {patient_info.get('name', 'Unknown')}",
        f"Doctor: {patient_info.get('doctor', '—')}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "SUMMARY",
        f"  Duration: {summary.get('duration_sec',0)/3600:.1f} hours",
        f"  Total Beats: {summary.get('total_beats',0):,}",
        f"  Avg HR: {summary.get('avg_hr',0):.0f} bpm",
        f"  Max HR: {summary.get('max_hr',0):.0f} bpm",
        f"  Min HR: {summary.get('min_hr',0):.0f} bpm",
        "",
        "HRV",
        f"  SDNN:  {summary.get('sdnn',0):.1f} ms",
        f"  rMSSD: {summary.get('rmssd',0):.1f} ms",
        f"  pNN50: {summary.get('pnn50',0):.2f}%",
        "",
        "ARRHYTHMIAS",
    ]
    arrhy = summary.get('arrhythmia_counts', {})
    if arrhy:
        for label, count in arrhy.items():
            lines.append(f"  {label}: {count} episode(s)")
    else:
        lines.append("  None detected")

    lines += ["", "=" * 60, "Physician Signature: _______________", "Date: _______________"]

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path

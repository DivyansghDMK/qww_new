"""
ecg_pipeline.py - Production ECG analysis pipeline.

This version separates the measurement path from the display path so interval
measurements come from the clinical measurement signal while QRS detection stays
stable on the monitor/display signal.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def analyze_ecg_pro(
    signal: np.ndarray,
    fs: float = 500.0,
    leads: Optional[Dict[str, np.ndarray]] = None,
    ac_hz: float = 50.0,
) -> Dict:
    """
    Production-grade ECG analysis pipeline.

    Args:
        signal: Raw single-lead ECG used for rhythm and interval analysis.
        fs: Sampling rate in Hz.
        leads: Optional dict of multi-lead waveforms, ideally including V1/V6.
        ac_hz: Included for API compatibility with older callers.

    Returns:
        Dict containing diagnosis, secondary findings, and measured intervals.
    """
    del ac_hz  # Measurement/display filters already encapsulate their behavior.

    from ecg.clinical_measurements import build_median_beat
    from ecg.ecg_calculations import calculate_qtc_bazett
    from ecg.beat_classifier import classify_beats
    from ecg.intervals_pro import (
        detect_pr_interval,
        detect_qrs_width,
        detect_qt_interval,
    )
    from ecg.qrs_detector import compute_rr_ms, detect_qrs
    from ecg.rhythm_pro import final_diagnosis
    from ecg.signal_paths import measurement_filter
    from ecg.signal_processing import process_ecg

    sig = np.asarray(signal, dtype=float)
    if sig.size == 0:
        return {
            "Diagnosis": "Insufficient Data",
            "Additional Findings": [],
            "Primary Diagnosis": "Insufficient Data",
            "Secondary Findings": [],
            "Findings": [],
            "Confidence": 0.0,
            "Beat Classification": [],
            "beat_class_counts": {},
        }

    measurement_signal = measurement_filter(sig, fs)
    display_signal = process_ecg(sig, fs=fs)

    clean_leads: Dict[str, np.ndarray] = {}
    if leads:
        for name, lead_signal in leads.items():
            lead_arr = np.asarray(lead_signal, dtype=float)
            if lead_arr.size:
                clean_leads[name] = measurement_filter(lead_arr, fs)

    peaks = detect_qrs(display_signal, fs)
    if len(peaks) < 3:
        return {
            "Diagnosis": "Insufficient Data",
            "Additional Findings": [],
            "Primary Diagnosis": "Insufficient Data",
            "Secondary Findings": [],
            "Findings": [],
            "Confidence": 0.0,
            "Beat Classification": [],
            "beat_class_counts": {},
            "HR": 0,
            "PR": 0,
            "QRS": 0,
            "QT": 0,
            "QTc": 0,
            "QT/QTc": "0/0",
            "hr": 0.0,
            "qrs_count": int(len(peaks)),
            "rr_ms": np.array([], dtype=float),
            "mean_pr_ms": 0.0,
            "qrs_ms": 0.0,
            "confidence_note": "Fewer than 3 QRS complexes detected - rhythm classification unavailable",
        }

    rr_ms = compute_rr_ms(peaks, fs=fs)
    mean_rr_ms = float(np.mean(rr_ms)) if rr_ms.size else 0.0
    hr = float(60000.0 / mean_rr_ms) if mean_rr_ms > 0 else 0.0

    min_beats = max(3, min(8, len(peaks) - 2))
    _, median_beat = build_median_beat(
        measurement_signal,
        peaks,
        fs,
        min_beats=min_beats,
        already_filtered=True,
    )
    if median_beat is None:
        return {
            "Diagnosis": "Low Quality Signal",
            "Additional Findings": [],
            "Primary Diagnosis": "Low Quality Signal",
            "Secondary Findings": [],
            "Findings": [],
            "Confidence": 0.0,
            "Beat Classification": [],
            "beat_class_counts": {},
            "HR": int(round(hr)) if hr else 0,
            "PR": 0,
            "QRS": 0,
            "QT": 0,
            "QTc": 0,
            "QT/QTc": "0/0",
            "hr": hr,
            "qrs_count": int(len(peaks)),
            "rr_ms": rr_ms,
            "mean_pr_ms": 0.0,
            "qrs_ms": 0.0,
            "confidence_note": "Median beat could not be constructed from clean beats",
        }

    qrs_ms = float(detect_qrs_width(median_beat, fs))
    pr_ms = float(detect_pr_interval(median_beat, fs))
    qt_ms = float(detect_qt_interval(median_beat, fs))
    qtc_ms = float(calculate_qtc_bazett(qt_ms, mean_rr_ms)) if mean_rr_ms > 0 else 0.0

    beats = classify_beats(measurement_signal, peaks, fs)
    fusion = final_diagnosis(
        hr=hr,
        rr=rr_ms,
        pr=pr_ms,
        qrs=qrs_ms,
        qtc=qtc_ms,
        beats=beats,
        leads=clean_leads,
        qrs_count=len(peaks),
        signal=measurement_signal,
    )
    diagnosis = fusion["Diagnosis"]
    secondary = list(fusion["Findings"])
    confidence = float(fusion["Confidence"])
    beat_class_counts: Dict[str, int] = {}
    for beat in beats:
        beat_type = str(beat.get("type", "UNKNOWN"))
        beat_class_counts[beat_type] = beat_class_counts.get(beat_type, 0) + 1

    result = {
        "Diagnosis": diagnosis,
        "Additional Findings": secondary,
        "Primary Diagnosis": diagnosis,
        "Secondary Findings": secondary,
        "Findings": secondary,
        "Confidence": confidence,
        "HR": int(round(hr)) if hr else 0,
        "PR": int(round(pr_ms)) if pr_ms else 0,
        "QRS": int(round(qrs_ms)) if qrs_ms else 0,
        "QT": int(round(qt_ms)) if qt_ms else 0,
        "QTc": int(round(qtc_ms)) if qtc_ms else 0,
        "QT/QTc": f"{int(round(qt_ms)) if qt_ms else 0}/{int(round(qtc_ms)) if qtc_ms else 0}",
        "hr": hr,
        "qrs_count": int(len(peaks)),
        "rr_ms": rr_ms,
        "mean_pr_ms": pr_ms,
        "qrs_ms": qrs_ms,
        "Beat Classification": beats,
        "beat_class_counts": beat_class_counts,
    }

    return result


def analyze(
    signal: np.ndarray,
    fs: float = 500.0,
    leads: Optional[Dict[str, np.ndarray]] = None,
    ac_hz: float = 50.0,
) -> Dict:
    """Backward-compatible entry point used by the app."""
    return analyze_ecg_pro(signal=signal, fs=fs, leads=leads, ac_hz=ac_hz)


__all__ = ["analyze", "analyze_ecg_pro"]

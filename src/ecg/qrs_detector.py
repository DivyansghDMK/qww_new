"""
qrs_detector.py — Layer 2a: QRS / R-peak detection.

Wraps the existing Pan-Tompkins implementation (pan_tompkins.py) and
the improved detector in arrhythmia_detector.py with a single, clean API.

    from ecg.qrs_detector import detect_qrs, compute_rr, qrs_metrics

Design principle: Input is a CLEAN (already filtered) ECG signal.
Output is beat-level information only — no rhythm interpretation here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Use the existing robust Pan-Tompkins implementation
try:
    from ecg.pan_tompkins import pan_tompkins as _pan_tompkins
    _PT_AVAILABLE = True
except ImportError:
    _PT_AVAILABLE = False

# Also bring in the improved detector from arrhythmia_detector
try:
    from ecg.arrhythmia_detector import (
        detect_r_peaks_pan_tompkins as _detect_r_peaks_improved,
        measure_beat as _measure_beat,
    )
    _AD_AVAILABLE = True
except ImportError:
    _AD_AVAILABLE = False

DEFAULT_FS = 500.0


def detect_qrs(signal: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """
    Detect QRS R-peaks in a CLEAN ECG signal using Pan-Tompkins algorithm.

    Uses the improved adaptive-threshold Pan-Tompkins from pan_tompkins.py,
    with fallback to the one in arrhythmia_detector.py.

    Args:
        signal: 1-D clean ECG signal (already filtered)
        fs:     Sampling rate (Hz)

    Returns:
        Sorted array of R-peak sample indices (dtype int)
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size < int(fs * 1.5):
        return np.array([], dtype=int)

    if _PT_AVAILABLE:
        peaks = _pan_tompkins(sig, fs=int(fs))
        if len(peaks) >= 2:
            return np.asarray(peaks, dtype=int)

    if _AD_AVAILABLE:
        peaks = _detect_r_peaks_improved(sig, fs=fs)
        return np.asarray(peaks, dtype=int)

    # Minimal fallback: simple threshold peak detection
    from scipy.signal import find_peaks
    min_distance = max(1, int(0.25 * fs))  # 250 ms minimum between beats
    threshold = float(np.std(sig)) * 0.8
    peaks, _ = find_peaks(sig, height=threshold, distance=min_distance)
    return np.asarray(peaks, dtype=int)


def compute_rr(peaks: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """
    Compute RR intervals in SECONDS from R-peak sample indices.

    Args:
        peaks: Array of R-peak sample indices
        fs:    Sampling rate (Hz)

    Returns:
        Array of RR intervals in seconds (length = len(peaks) - 1)
    """
    peaks = np.asarray(peaks, dtype=float)
    if peaks.size < 2:
        return np.array([], dtype=float)
    return np.diff(peaks) / float(fs)


def compute_rr_ms(peaks: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """
    Compute RR intervals in MILLISECONDS from R-peak sample indices.

    Args:
        peaks: Array of R-peak sample indices
        fs:    Sampling rate (Hz)

    Returns:
        Array of RR intervals in milliseconds
    """
    return compute_rr(peaks, fs) * 1000.0


def beat_metrics(signal: np.ndarray, peaks: np.ndarray,
                 fs: float = DEFAULT_FS) -> List[Dict]:
    """
    Measure per-beat interval features (QRS width, PR, QT, ST, P presence).

    Uses the detailed beat measurer from arrhythmia_detector.py.

    Args:
        signal: Clean Lead-II ECG (or best available lead)
        peaks:  R-peak sample indices
        fs:     Sampling rate (Hz)

    Returns:
        List of beat dicts with keys:
            r_peak, qrs_ms, p_present, pr_ms, qt_ms, st_level_mv,
            p_amplitude, q_onset, j_point, noisy, …
    """
    if not _AD_AVAILABLE:
        return []
    sig = np.asarray(signal, dtype=float)
    result = []
    for r in peaks:
        b = _measure_beat(sig, int(r), fs)
        if b is not None:
            result.append(b)
    return result


def calculate_qrs_width(signal: np.ndarray, qrs_peak: int, fs: float = 250.0) -> float:
    """
    Exact QRS width calculation via adaptive amplitude boundaries.
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size < 3 or fs <= 0:
        return 0.0

    qrs_peak = int(max(0, min(sig.size - 1, qrs_peak)))
    start = max(0, qrs_peak - int(0.25 * fs))
    end = min(sig.size, qrs_peak + int(0.40 * fs) + 1)
    seg = sig[start:end]
    if seg.size < 3:
        return 0.0

    baseline = float(np.median(seg))
    centered = seg - baseline
    peak = int(np.argmax(np.abs(centered)))
    amp = float(np.max(np.abs(centered)))
    if amp <= 1e-9:
        return 0.0

    # ── Amplitude bounds ──────────────────────────────────────────
    amp_threshold = 0.10 * amp
    left_amp = peak
    while left_amp > 0 and abs(centered[left_amp]) > amp_threshold:
        left_amp -= 1

    right_amp = peak
    while right_amp < len(centered)-1 and abs(centered[right_amp]) > amp_threshold:
        right_amp += 1
    
    qrs_amp = (right_amp - left_amp) / float(fs) * 1000.0

    # ── Gradient bounds ───────────────────────────────────────────
    grad = np.abs(np.diff(centered))
    if grad.size == 0:
        return qrs_amp
        
    grad_peak = min(peak, grad.size - 1)
    max_grad = np.max(grad)
    grad_threshold = 0.05 * max_grad

    left_grad = grad_peak
    while left_grad > 0 and grad[left_grad] > grad_threshold:
        left_grad -= 1

    right_grad = grad_peak
    while right_grad < len(grad)-1 and grad[right_grad] > grad_threshold:
        right_grad += 1

    qrs_grad = (right_grad - left_grad) / float(fs) * 1000.0

    qrs_width = max(qrs_amp, qrs_grad)
    return qrs_width if 40.0 <= qrs_width <= 300.0 else 0.0


def qrs_metrics(signal: np.ndarray, fs: float = DEFAULT_FS) -> Dict:
    """
    One-call convenience: detect QRS, compute RR, measure beats.

    Returns a dict with:
        peaks       - R-peak indices
        rr_sec      - RR intervals in seconds
        rr_ms       - RR intervals in milliseconds
        hr          - Heart rate (bpm)
        qrs_count   - Number of QRS complexes detected
        qrs_ms      - Mean QRS width in ms
    """
    peaks = detect_qrs(signal, fs)
    rr = compute_rr(peaks, fs)
    rr_ms = rr * 1000.0
    hr = float(60.0 / np.mean(rr)) if rr.size else 0.0
    
    qrs_widths = [calculate_qrs_width(signal, p, fs) for p in peaks]
    mean_qrs_ms = float(np.mean(qrs_widths)) if qrs_widths else 0.0

    return {
        "peaks": peaks,
        "rr_sec": rr,
        "rr_ms": rr_ms,
        "hr": hr,
        "qrs_count": int(peaks.size),
        "qrs_ms": mean_qrs_ms,
    }


__all__ = [
    "detect_qrs",
    "compute_rr",
    "compute_rr_ms",
    "calculate_qrs_width",
    "beat_metrics",
    "qrs_metrics",
]

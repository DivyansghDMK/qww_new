"""
p_wave_detector.py — Layer 2b: P-wave detection.

Reliable P-wave detection is the foundation of:
  - AF detection (P absent + irregular)
  - AV block classification (P present but dissociated from QRS)
  - Sinus rhythm labeling

    from ecg.p_wave_detector import (
        detect_p_waves,
        compute_pr_intervals,
        p_wave_metrics,
    )

Design principle: Takes a CLEAN signal + known R-peaks as input.
Returns P-wave positions and PR intervals — no rhythm interpretation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

DEFAULT_FS = 500.0

# Search window: 80–200 ms before QRS onset (which is ~20–30 ms before R-peak)
# So relative to the R-peak: 50–230 ms before
_P_SEARCH_START_MS = 50.0    # ms before R-peak (end of P-wave search)
_P_SEARCH_END_MS   = 250.0   # ms before R-peak (start of P-wave search)
_P_MIN_AMPLITUDE   = 0.04    # mV — below this = noise, not P-wave


def detect_p_waves(signal: np.ndarray, qrs_peaks: np.ndarray,
                   fs: float = DEFAULT_FS) -> List[Optional[int]]:
    """
    Detect P-wave peaks, one per QRS complex.
    """
    sig = np.asarray(signal, dtype=float)
    p_positions = []

    for qrs in qrs_peaks:
        start = int(qrs - 0.25 * fs)
        end = int(qrs - 0.08 * fs)

        if start < 0:
            p_positions.append(None)
            continue

        segment = sig[start:end]

        if len(segment) == 0:
            p_positions.append(None)
            continue

        peak = np.max(segment)

        if peak > 0.06:  # 🔥 IMPORTANT threshold
            p_positions.append(int(start + np.argmax(segment)))
        else:
            p_positions.append(None)

    return p_positions


def compute_pr_intervals(p_positions: List[Optional[int]],
                         qrs_peaks: np.ndarray,
                         fs: float = DEFAULT_FS) -> List[Optional[float]]:
    """
    Compute PR interval in milliseconds for each beat.

    PR interval = time from P-wave onset to QRS onset.
    Approximated here as P-peak to R-peak (clinical monitors use P-onset,
    but P-peak is more robustly detectable).

    Args:
        p_positions: List of P-peak indices (or None) from detect_p_waves
        qrs_peaks:   R-peak sample indices
        fs:          Sampling rate (Hz)

    Returns:
        List of PR interval values in ms (None where P-wave was not detected),
        length = min(len(p_positions), len(qrs_peaks)).
    """
    results: List[Optional[float]] = []
    for p_idx, r_idx in zip(p_positions, qrs_peaks):
        if p_idx is None:
            results.append(None)
        else:
            pr_ms = float((int(r_idx) - int(p_idx))) * 1000.0 / float(fs)
            # Physiologically valid PR: 60–400 ms
            if 60.0 <= pr_ms <= 400.0:
                results.append(pr_ms)
            else:
                results.append(None)
    return results


def p_wave_metrics(signal: np.ndarray, qrs_peaks: np.ndarray,
                   fs: float = DEFAULT_FS) -> Dict:
    """
    One-call convenience: detect P-waves and compute PR intervals.

    Args:
        signal:    Clean Lead-II ECG
        qrs_peaks: R-peak sample indices

    Returns dict with:
        p_positions  - List of P-peak indices (None = not detected)
        pr_ms        - List of PR intervals in ms (None where absent)
        p_count      - Number of beats with a detected P-wave
        p_ratio      - Fraction of beats with P-waves (0 → AF suspect)
        p_present    - True if majority of beats have P-waves
        mean_pr_ms   - Mean PR interval (ms), None if no P-waves
        pr_std_ms    - Std-dev of PR intervals (ms), None if < 2 values
    """
    p_pos = detect_p_waves(signal, qrs_peaks, fs)
    pr    = compute_pr_intervals(p_pos, qrs_peaks, fs)

    p_count = sum(1 for p in p_pos if p is not None)
    p_ratio = p_count / max(len(qrs_peaks), 1)
    pr_valid = [v for v in pr if v is not None]

    return {
        "p_positions": p_pos,
        "pr_ms":       pr,
        "p_count":     p_count,
        "p_ratio":     p_ratio,
        "p_present":   p_ratio > 0.5,
        "mean_pr_ms":  float(np.mean(pr_valid)) if pr_valid else None,
        "pr_std_ms":   float(np.std(pr_valid))  if len(pr_valid) >= 2 else None,
    }


__all__ = [
    "detect_p_waves",
    "compute_pr_intervals",
    "p_wave_metrics",
]

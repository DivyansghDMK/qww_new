from __future__ import annotations

import numpy as np


def _safe_window(start: int, end: int, limit: int) -> tuple[int, int]:
    return max(0, start), min(limit, end)


def detect_qrs_width(beat: np.ndarray, fs: float) -> float:
    beat = np.asarray(beat, dtype=float)
    if beat.size < 3 or fs <= 0:
        return 0.0

    baseline = float(np.median(beat))
    signal = beat - baseline
    peak = int(np.argmax(np.abs(signal)))
    amp = float(np.max(np.abs(signal)))
    if amp <= 1e-9:
        return 0.0

    # Amplitude boundaries are essential for LBBB: the QRS can be broad and
    # slurred with low slope, so a gradient-only detector truncates it.
    amp_threshold = 0.10 * amp
    left = peak
    while left > 0 and abs(signal[left]) > amp_threshold:
        left -= 1

    right = peak
    while right < signal.size - 1 and abs(signal[right]) > amp_threshold:
        right += 1

    amp_width_ms = float((right - left) * 1000.0 / fs)

    grad = np.gradient(beat)
    max_grad = float(np.max(np.abs(grad))) if grad.size else 0.0
    grad_width_ms = 0.0
    if max_grad > 0:
        threshold = 0.2 * max_grad
        active = np.flatnonzero(np.abs(grad) >= threshold)
        if active.size >= 2:
            grad_width_ms = float((active[-1] - active[0]) * 1000.0 / fs)

    qrs_ms = max(amp_width_ms, grad_width_ms)
    return qrs_ms if 40.0 <= qrs_ms <= 300.0 else 0.0


def detect_pr_interval(beat: np.ndarray, fs: float) -> float:
    beat = np.asarray(beat, dtype=float)
    if beat.size < 3 or fs <= 0:
        return 0.0

    peak = int(np.argmax(np.abs(beat)))
    start, end = _safe_window(peak - int(0.30 * fs), peak, beat.size)
    search = beat[start:end]
    if search.size == 0:
        return 0.0

    p_idx = int(np.argmax(np.abs(search))) + start
    pr_samples = peak - p_idx
    return float(max(pr_samples, 0) * 1000.0 / fs)


def detect_qt_interval(beat: np.ndarray, fs: float) -> float:
    beat = np.asarray(beat, dtype=float)
    if beat.size < 3 or fs <= 0:
        return 0.0

    peak = int(np.argmax(np.abs(beat)))
    qt_samples = min(beat.size - 1, peak + int(0.40 * fs)) - peak
    return float(max(qt_samples, 0) * 1000.0 / fs)

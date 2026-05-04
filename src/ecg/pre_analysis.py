"""
Pre-Analysis Filter Layer
=========================
Runs BEFORE any metric calculation (HR, PR, QRS, QT, QTc).

Pipeline:
  1. Flatline / electrode-off detection
  2. Saturation / ADC clipping detection
  3. Baseline wander removal  (median + mean, Holter gold standard)
  4. Powerline noise removal  (50 Hz notch)
  5. High-frequency EMG / motion artifact rejection
  6. Segment quality scoring  (0.0 – 1.0)
  7. Return: (clean_signal, quality_score, reject_reason | None)

Usage:
    from ecg.pre_analysis import pre_analyze
    clean, score, reason = pre_analyze(raw_signal, fs=500)
    if score < 0.5:
        # skip metric calculation for this window
        ...
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, medfilt, welch
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Optional


# ── Thresholds ────────────────────────────────────────────────────────────────
_FLATLINE_STD_THRESHOLD   = 50.0    # ADC counts — below this = electrode off
_SATURATION_RATIO         = 0.02    # >2 % of samples clipped = saturated
_BASELINE_POWER_RATIO     = 0.35    # >35 % of power in <0.5 Hz = excessive wander
_HF_NOISE_RATIO           = 0.45    # >45 % of power above 100 Hz = EMG artifact
_POWERLINE_RATIO          = 10.0    # 50 Hz PSD / 10 Hz PSD > 10 = powerline noise
_MIN_SAMPLES              = 500     # ~1 s at 500 Hz — minimum usable window


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIGNAL CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def _check_flatline(signal: np.ndarray) -> Optional[str]:
    if np.std(signal) < _FLATLINE_STD_THRESHOLD:
        return "Flatline — check electrode connection"
    return None


def _check_saturation(signal: np.ndarray) -> Optional[str]:
    """Detect ADC clipping: samples stuck at min/max rail."""
    sig_min, sig_max = np.min(signal), np.max(signal)
    amp = sig_max - sig_min
    if amp < 1e-6:
        return "Flatline"
    # Count samples within 1 % of the rails
    rail_lo = sig_min + 0.01 * amp
    rail_hi = sig_max - 0.01 * amp
    clipped = np.sum((signal <= rail_lo) | (signal >= rail_hi))
    if clipped / len(signal) > _SATURATION_RATIO:
        return "Signal saturated — electrode or gain issue"
    return None


def _check_baseline_wander(signal: np.ndarray, fs: float) -> Tuple[Optional[str], float]:
    """Return (reason | None, baseline_power_ratio)."""
    try:
        nyq = fs / 2.0
        b, a = butter(2, 0.5 / nyq, btype="low")
        baseline = filtfilt(b, a, signal)
        bp = float(np.var(baseline))
        tp = float(np.var(signal))
        ratio = bp / tp if tp > 0 else 0.0
        if ratio > _BASELINE_POWER_RATIO:
            return "Excessive baseline wander — motion or breathing artifact", ratio
        return None, ratio
    except Exception:
        return None, 0.0


def _check_hf_noise(signal: np.ndarray, fs: float) -> Optional[str]:
    """Detect high-frequency EMG / motion noise (>100 Hz)."""
    if fs <= 200:
        return None  # can't assess above Nyquist
    try:
        nyq = fs / 2.0
        b, a = butter(2, 100.0 / nyq, btype="high")
        hf = filtfilt(b, a, signal)
        hf_power = float(np.var(hf))
        total    = float(np.var(signal))
        if total > 0 and hf_power / total > _HF_NOISE_RATIO:
            return "High-frequency noise — patient movement or EMG artifact"
    except Exception:
        pass
    return None


def _check_powerline(signal: np.ndarray, fs: float) -> Optional[str]:
    """Detect 50 Hz powerline interference via Welch PSD."""
    try:
        nperseg = min(256, len(signal) // 2)
        if nperseg < 32:
            return None
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        idx_50  = int(np.argmin(np.abs(freqs - 50.0)))
        idx_ref = int(np.argmin(np.abs(freqs - 10.0)))
        if psd[idx_ref] > 0 and psd[idx_50] / psd[idx_ref] > _POWERLINE_RATIO:
            return "50 Hz powerline interference detected"
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEANING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def remove_baseline_wander(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Holter gold-standard baseline removal:
      1. Median filter (200 ms) — removes QRS influence
      2. Moving average (600 ms) — smooths slow drift
      3. Subtract estimated baseline
    """
    if len(signal) < 50:
        return signal - np.mean(signal)
    try:
        med_win = max(3, int(0.20 * fs))
        if med_win % 2 == 0:
            med_win += 1
        med_win = min(med_win, len(signal) // 2 | 1)

        b1 = medfilt(signal, kernel_size=med_win)

        avg_win = max(10, int(0.60 * fs))
        avg_win = min(avg_win, len(b1) // 2)
        baseline = uniform_filter1d(b1.astype(float), size=avg_win, mode="nearest")

        return signal - baseline
    except Exception:
        return signal - np.mean(signal)


def remove_powerline_noise(signal: np.ndarray, fs: float,
                            freq: float = 50.0) -> np.ndarray:
    """Zero-phase notch filter at `freq` Hz."""
    try:
        nyq = fs / 2.0
        w0  = freq / nyq
        if not (0 < w0 < 1):
            return signal
        bw  = w0 / 30.0          # Q ≈ 30
        b, a = butter(2, [w0 - bw, w0 + bw], btype="bandstop")
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def remove_hf_noise(signal: np.ndarray, fs: float,
                    cutoff_hz: float = 40.0) -> np.ndarray:
    """Low-pass filter to suppress EMG / motion noise."""
    try:
        nyq = fs / 2.0
        cut = min(cutoff_hz / nyq, 0.99)
        if cut <= 0:
            return signal
        b, a = butter(4, cut, btype="low")
        return filtfilt(b, a, signal)
    except Exception:
        return signal


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUALITY SCORE
# ══════════════════════════════════════════════════════════════════════════════

def compute_quality_score(signal: np.ndarray, fs: float) -> float:
    """
    Composite signal quality score 0.0 – 1.0.
    Combines: flatline, saturation, baseline wander, HF noise, powerline.
    """
    score = 1.0

    if np.std(signal) < _FLATLINE_STD_THRESHOLD:
        return 0.0

    sat = _check_saturation(signal)
    if sat:
        score -= 0.4

    _, bw_ratio = _check_baseline_wander(signal, fs)
    score -= min(0.3, bw_ratio * 0.5)

    if _check_hf_noise(signal, fs):
        score -= 0.2

    if _check_powerline(signal, fs):
        score -= 0.1

    return max(0.0, min(1.0, score))


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def pre_analyze(
    raw_signal: np.ndarray,
    fs: float = 500.0,
    apply_baseline: bool = True,
    apply_notch: bool = True,
    apply_lpf: bool = True,
) -> Tuple[np.ndarray, float, Optional[str]]:
    """
    Pre-analysis filter layer — call this BEFORE any metric calculation.

    Args:
        raw_signal:       Raw ECG signal (ADC counts or mV).
        fs:               Sampling rate in Hz.
        apply_baseline:   Remove baseline wander (default True).
        apply_notch:      Remove 50 Hz powerline noise (default True).
        apply_lpf:        Apply 40 Hz low-pass to suppress EMG (default True).

    Returns:
        (clean_signal, quality_score, reject_reason)
        - clean_signal:   Cleaned signal ready for metric calculation.
        - quality_score:  0.0 (unusable) – 1.0 (excellent).
        - reject_reason:  Human-readable string if quality < 0.5, else None.
    """
    sig = np.asarray(raw_signal, dtype=float)

    # ── Hard reject: too short ────────────────────────────────────────────────
    if len(sig) < _MIN_SAMPLES:
        return sig, 0.0, "Signal too short for analysis"

    # ── Hard reject: flatline ─────────────────────────────────────────────────
    reason = _check_flatline(sig)
    if reason:
        return sig, 0.0, reason

    # ── Hard reject: saturation ───────────────────────────────────────────────
    reason = _check_saturation(sig)
    if reason:
        return sig, 0.2, reason

    # ── Cleaning pipeline ─────────────────────────────────────────────────────
    clean = sig.copy()

    if apply_baseline:
        clean = remove_baseline_wander(clean, fs)

    if apply_notch:
        clean = remove_powerline_noise(clean, fs, freq=50.0)

    if apply_lpf:
        clean = remove_hf_noise(clean, fs, cutoff_hz=40.0)

    # ── Quality score on cleaned signal ───────────────────────────────────────
    score = compute_quality_score(clean, fs)

    reject_reason: Optional[str] = None
    if score < 0.5:
        # Collect the most relevant reason
        for check_fn in (
            lambda s: _check_baseline_wander(s, fs)[0],
            lambda s: _check_hf_noise(s, fs),
            lambda s: _check_powerline(s, fs),
        ):
            r = check_fn(clean)
            if r:
                reject_reason = r
                break
        if not reject_reason:
            reject_reason = f"Low signal quality (score={score:.2f})"

    return clean, score, reject_reason


def should_analyze(quality_score: float, threshold: float = 0.5) -> bool:
    """Return True if signal quality is good enough for metric calculation."""
    return quality_score >= threshold


def quality_label(score: float) -> str:
    """Human-readable quality label."""
    if score >= 0.85:
        return "Excellent"
    if score >= 0.70:
        return "Good"
    if score >= 0.50:
        return "Fair"
    return "Poor"

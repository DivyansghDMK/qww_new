"""
signal_processing.py — Layer 1: Signal-level ECG processing.

Wraps the existing medical-grade filter chain from ecg_filters.py with
a clean public interface:

    from ecg.signal_processing import (
        bandpass_filter,
        notch_filter,
        process_ecg,
    )

Design principle: This layer knows nothing about rhythm or beats —
it only cleans the raw waveform.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

# Re-export the monitor-grade pipeline from ecg_filters.py
try:
    from ecg.ecg_filters import (
        apply_ecg_filters as _apply_ecg_filters,
        apply_baseline_wander_median_mean as _baseline_wander,
        notch_filter_butterworth as _notch,
        normalize_adc_signal,
    )
    _FILTERS_AVAILABLE = True
except ImportError:
    _FILTERS_AVAILABLE = False


DEFAULT_FS = 500.0


def bandpass_filter(signal: np.ndarray, fs: float = DEFAULT_FS,
                    low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """
    Bandpass filter (default 0.5–40 Hz) for general-purpose ECG cleaning.

    Removes:
      - Baseline wander (< 0.5 Hz)
      - High-frequency noise and EMG (> 40 Hz)

    Args:
        signal: 1-D ECG signal array
        fs:     Sampling rate in Hz
        low:    Lower cutoff (Hz)
        high:   Upper cutoff (Hz)

    Returns:
        Filtered signal
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size < 10:
        return signal
    nyq = 0.5 * float(fs)
    lo = max(low / nyq, 0.001)
    hi = min(high / nyq, 0.999)
    if lo >= hi:
        return signal
    b, a = butter(2, [lo, hi], btype="band")
    padlen = min(signal.size - 1, max(len(a), len(b)) * 3)
    if padlen <= 0:
        return signal
    return filtfilt(b, a, signal, padlen=padlen)


def notch_filter(signal: np.ndarray, fs: float = DEFAULT_FS,
                 freq: float = 50.0) -> np.ndarray:
    """
    Notch (band-stop) filter to remove power-line interference.

    Args:
        signal: 1-D ECG signal
        fs:     Sampling rate (Hz)
        freq:   Power-line frequency to suppress (50 or 60 Hz)

    Returns:
        Filtered signal
    """
    if _FILTERS_AVAILABLE:
        return _notch(np.asarray(signal, dtype=float), fs, freq=freq, q=25.0)

    # Fallback: simple Butterworth band-stop
    signal = np.asarray(signal, dtype=float)
    if signal.size < 10:
        return signal
    nyq = 0.5 * float(fs)
    w0 = freq / nyq
    if w0 <= 0 or w0 >= 1:
        return signal
    bw = w0 / 25.0
    b, a = butter(2, [w0 - bw, w0 + bw], btype="bandstop")
    return filtfilt(b, a, signal)


def remove_baseline_wander(signal: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """
    Remove baseline wander using the gold-standard median + mean filter approach.

    Args:
        signal: 1-D ECG signal
        fs:     Sampling rate (Hz)

    Returns:
        Signal with baseline drift removed
    """
    if _FILTERS_AVAILABLE:
        return _baseline_wander(np.asarray(signal, dtype=float), float(fs))
    # Simple fallback: subtract moving median
    signal = np.asarray(signal, dtype=float)
    win = max(3, int(0.6 * fs) | 1)  # ~600 ms, must be odd
    from scipy.signal import medfilt
    baseline = medfilt(signal, kernel_size=min(win, len(signal) | 1))
    return signal - baseline


def process_ecg(signal: np.ndarray, fs: float = DEFAULT_FS,
                ac_hz: float = 50.0) -> np.ndarray:
    """
    Full signal-processing pipeline — the entry point for raw ECG.

    Stages (in order):
      1. Notch filter (removes power-line noise)
      2. Baseline wander removal (gold-standard median+mean)
      3. Bandpass 0.5–40 Hz (EMG + residual noise)

    Args:
        signal: Raw 1-D ECG array (ADC counts or mV)
        fs:     Sampling rate (Hz)
        ac_hz:  Power-line frequency to notch (50 or 60)

    Returns:
        Clean ECG array, same length as input
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size < 10:
        return sig

    if _FILTERS_AVAILABLE:
        # Use monitor-grade chain: notch → baseline → EMG LP
        return _apply_ecg_filters(
            sig, sampling_rate=float(fs),
            ac_filter=str(int(ac_hz)),
            emg_filter="40",
            dft_filter="0.5",
        )

    # Fallback: manual chain
    sig = notch_filter(sig, fs, freq=ac_hz)
    sig = remove_baseline_wander(sig, fs)
    sig = bandpass_filter(sig, fs, low=0.5, high=40.0)
    return sig


__all__ = [
    "bandpass_filter",
    "notch_filter",
    "remove_baseline_wander",
    "process_ecg",
]

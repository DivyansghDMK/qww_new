import numpy as np
from scipy.signal import butter, filtfilt, welch, find_peaks


def assess_signal_quality(signal, fs: int = 500):
    """Return (quality_score 0.0–1.0, reason) for an ECG window.

    Score < 0.5 → do not analyze this window for arrhythmia.
    Expects `signal` in millivolts (mV), already calibrated from ADC counts.
    """
    # Require at least ~2 seconds of data
    if len(signal) < fs * 2:
        return 0.0, "Too short"

    sig = np.asarray(signal, dtype=float)

    # 1. Flatline check — electrode off or disconnected
    if np.std(sig) < 0.05:
        return 0.0, "Flatline — check electrodes"

    # 2. Saturation check — ADC clipping (±4.5 mV is well beyond physiologic)
    clipped = np.sum((sig > 4.5) | (sig < -4.5))
    if clipped / len(sig) > 0.01:
        return 0.2, "Signal saturated — electrode issue"

    # 3. Baseline wander — excessive low‑frequency power (<0.5 Hz)
    try:
        b_low, a_low = butter(2, 0.5 / (fs / 2), btype="low")
        baseline = filtfilt(b_low, a_low, sig)
        baseline_power = np.var(baseline)
        total_power = np.var(sig)
        if total_power > 0 and baseline_power / total_power > 0.30:
            return 0.4, "Excessive baseline wander — patient breathing or movement"
    except Exception:
        total_power = np.var(sig)

    # 4. High‑frequency noise (EMG / motion artifact) — >100 Hz band
    try:
        if fs > 200:
            b_high, a_high = butter(2, 100 / (fs / 2), btype="high")
            hf_signal = filtfilt(b_high, a_high, sig)
            hf_power = np.var(hf_signal)
            if total_power > 0 and hf_power / total_power > 0.40:
                return 0.4, "High frequency noise — patient movement or EMG"
    except Exception:
        pass

    # 5. 50 Hz powerline interference (India standard)
    try:
        freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig) // 2))
        if len(freqs) > 0:
            idx_50hz = np.argmin(np.abs(freqs - 50.0))
            idx_base = np.argmin(np.abs(freqs - 10.0))
            if psd[idx_base] > 0 and psd[idx_50hz] / psd[idx_base] > 10:
                return 0.5, "50 Hz powerline interference"
    except Exception:
        pass

    # 6. R‑peak regularity sanity check — extreme irregularity before arrhythmia logic → likely artifact
    try:
        diff = np.diff(sig)
        sq = diff ** 2
        win = int(0.15 * fs)
        win = max(win, 3)
        mwa = np.convolve(sq, np.ones(win) / win, mode="same")
        peaks, _ = find_peaks(
            mwa,
            height=np.mean(mwa) + np.std(mwa),
            distance=int(0.3 * fs),
        )
        if len(peaks) < 2:
            return 0.3, "No clear heartbeat detected"
        rr = np.diff(peaks)
        rr_cv = np.std(rr) / np.mean(rr) if np.mean(rr) > 0 else 1.0
        if rr_cv > 0.8:
            return 0.4, "Irregular signal — possible noise or artifact"
    except Exception:
        pass

    return 1.0, "Clean"

"""Signal Quality Index (SQI) Module

This module provides signal quality assessment for ECG signals.
SQI is used to gate metric calculations - preventing display of unreliable values
when signal quality is poor (motion artifact, noise, electrode issues).

WARNING FIX #9: SQI Gating
"""

import numpy as np
from typing import Tuple, Optional


def calculate_signal_quality_index(signal: np.ndarray, r_peaks: np.ndarray, sampling_rate: float = 500) -> float:
    """
    Calculate Signal Quality Index (SQI) for ECG signal
    
    SQI is a composite score (0-1) based on:
    1. R-peak regularity (coefficient of variation)
    2. Signal-to-noise ratio
    3. Baseline stability
    
    Args:
        signal: ECG signal array
        r_peaks: R-peak indices
        sampling_rate: Sampling rate in Hz (default: 500)
    
    Returns:
        float: SQI score 0-1 (0=poor, 1=excellent)
    """
    sqi_scores = []
    
    # Component 1: R-peak regularity (coefficient of variation)
    if len(r_peaks) >= 3:
        rr_intervals = np.diff(r_peaks)
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        
        if mean_rr > 0:
            cv = std_rr / mean_rr  # Coefficient of variation
            regularity_score = max(0, 1 - cv)  # Lower CV = better quality
            sqi_scores.append(regularity_score)
    
    # Component 2: Signal-to-noise ratio
    if len(signal) > 0:
        signal_power = np.var(signal)
        if signal_power > 0:
            # Estimate noise from high-frequency content (derivative variance)
            noise_est = np.var(np.diff(signal))
            snr = signal_power / max(noise_est, 1e-6)
            snr_score = min(1.0, snr / 100)  # Normalize (SNR=100 → score=1.0)
            sqi_scores.append(snr_score)
    
    # Component 3: Baseline stability
    if len(signal) >= 100:
        # Compare first half vs second half mean
        mid = len(signal) // 2
        baseline_drift = np.abs(np.mean(signal[:mid]) - np.mean(signal[mid:]))
        drift_score = max(0, 1 - baseline_drift / 100)  # Normalize
        sqi_scores.append(drift_score)
    
    # Overall SQI = average of component scores
    if len(sqi_scores) > 0:
        return np.mean(sqi_scores)
    else:
        return 0.0


def is_signal_quality_acceptable(sqi: float, threshold: float = 0.6) -> bool:
    """
    Check if signal quality is acceptable for metric calculation
    
    Args:
        sqi: Signal quality index (0-1)
        threshold: Minimum acceptable SQI (default: 0.6)
    
    Returns:
        bool: True if quality is acceptable, False otherwise
    """
    return sqi >= threshold


def get_quality_label(sqi: float) -> str:
    """
    Get human-readable quality label
    
    Args:
        sqi: Signal quality index (0-1)
    
    Returns:
        str: Quality label (Excellent/Good/Fair/Poor)
    """
    if sqi >= 0.8:
        return "Excellent"
    elif sqi >= 0.6:
        return "Good"
    elif sqi >= 0.4:
        return "Fair"
    else:
        return "Poor"


def calculate_sqi_with_details(signal: np.ndarray, r_peaks: np.ndarray, sampling_rate: float = 500) -> Tuple[float, dict]:
    """
    Calculate SQI with detailed component scores
    
    Args:
        signal: ECG signal array
        r_peaks: R-peak indices
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Tuple of (overall_sqi, component_scores_dict)
    """
    components = {}
    
    # R-peak regularity
    if len(r_peaks) >= 3:
        rr_intervals = np.diff(r_peaks)
        cv = np.std(rr_intervals) / np.mean(rr_intervals)
        components['regularity'] = max(0, 1 - cv)
    else:
        components['regularity'] = 0.0
    
    # SNR
    if len(signal) > 0:
        signal_power = np.var(signal)
        noise_est = np.var(np.diff(signal))
        snr = signal_power / max(noise_est, 1e-6)
        components['snr'] = min(1.0, snr / 100)
    else:
        components['snr'] = 0.0
    
    # Baseline stability
    if len(signal) >= 100:
        mid = len(signal) // 2
        drift = np.abs(np.mean(signal[:mid]) - np.mean(signal[mid:]))
        components['baseline_stability'] = max(0, 1 - drift / 100)
    else:
        components['baseline_stability'] = 0.0
    
    overall_sqi = np.mean(list(components.values()))
    
    return overall_sqi, components

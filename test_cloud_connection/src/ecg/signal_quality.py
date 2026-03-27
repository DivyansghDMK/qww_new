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

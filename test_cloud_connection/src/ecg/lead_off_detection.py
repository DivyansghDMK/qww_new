"""Lead-Off Detection Module

This module provides lead-off detection functionality for ECG signals.
Lead-off detection identifies when ECG electrodes are disconnected or have poor contact.

Calibrated for 12-bit ADC (0-4096) at 500 Hz sampling rate.
"""

import numpy as np
from typing import Dict

# Calibrated thresholds for 12-bit ADC (0-4096) at 500 Hz
LEAD_OFF_THRESHOLDS = {
    'amplitude_min': 20,      # < 0.1 mV p-p → flatline/disconnected
    'amplitude_max': 3000,    # > 14.6 mV p-p → saturation
    'variance_min':  4,       # < variance → flatline (disconnected)
    'variance_max':  250000,  # > variance → full noise (disconnected) - UPDATED from 10,000
    'saturation_lo': 10,      # ADC < 10 → rail low
    'saturation_hi': 4085,    # ADC > 4085 → rail high
}


def detect_lead_off(signal: np.ndarray, sampling_rate: float = 500, window_size: float = 1.0) -> bool:
    """
    Detect if ECG lead is disconnected (lead-off condition)
    
    Algorithm:
    1. Check signal amplitude (lead-off → very low or very high)
    2. Check signal variance (lead-off → near zero or very high noise)
    3. Check for saturation (ADC at min/max values)
    
    Lead-off indicators:
    - Signal amplitude < 0.1 mV (too low)
    - Signal amplitude > 14.6 mV (too high, saturation)
    - Signal variance < 4 (flatline)
    - Signal variance > 250,000 (excessive noise) - CRITICAL: Updated from 10,000
    - ADC values stuck at 0 or 4095 (saturation)
    
    Args:
        signal: ECG signal (ADC values 0-4096)
        sampling_rate: Sampling rate in Hz (default: 500)
        window_size: Window size in seconds (default: 1.0)
    
    Returns:
        bool: True if lead is disconnected, False if connected
    """
    window_samples = int(window_size * sampling_rate)
    
    if len(signal) < window_samples:
        return False
    
    # Get recent window
    window = signal[-window_samples:]
    
    # Check 1: Signal amplitude (peak-to-peak)
    amplitude = np.ptp(window)
    if amplitude < LEAD_OFF_THRESHOLDS['amplitude_min']:
        return True  # Lead off - too low (flatline)
    if amplitude > LEAD_OFF_THRESHOLDS['amplitude_max']:
        return True  # Lead off - saturation
    
    # Check 2: Signal variance
    variance = np.var(window)
    if variance < LEAD_OFF_THRESHOLDS['variance_min']:
        return True  # Lead off - flatline (near zero variance)
    if variance > LEAD_OFF_THRESHOLDS['variance_max']:
        return True  # Lead off - excessive noise (CRITICAL FIX: 250,000 threshold)
    
    # Check 3: Saturation (ADC at min/max values)
    min_val = np.min(window)
    max_val = np.max(window)
    if min_val <= LEAD_OFF_THRESHOLDS['saturation_lo'] or max_val >= LEAD_OFF_THRESHOLDS['saturation_hi']:
        return True  # Lead off - ADC saturation
    
    return False  # Lead connected


def check_all_leads_quality(lead_data: Dict[str, np.ndarray], sampling_rate: float = 500) -> Dict[str, str]:
    """
    Check lead quality for all ECG leads
    
    Args:
        lead_data: Dictionary mapping lead names to signal arrays
        sampling_rate: Sampling rate in Hz (default: 500)
    
    Returns:
        Dictionary mapping lead names to quality status ('OK' or 'OFF')
    
    Example:
        lead_data = {
            'I': np.array([...]),
            'II': np.array([...]),
            'III': np.array([...]),
            ...
        }
        quality = check_all_leads_quality(lead_data)
        # Returns: {'I': 'OK', 'II': 'OFF', 'III': 'OK', ...}
    """
    lead_quality = {}
    
    for lead_name, signal in lead_data.items():
        try:
            is_off = detect_lead_off(signal, sampling_rate)
            lead_quality[lead_name] = 'OFF' if is_off else 'OK'
        except Exception as e:
            print(f" Error checking lead {lead_name}: {e}")
            lead_quality[lead_name] = 'UNKNOWN'
    
    return lead_quality


def get_lead_quality_summary(lead_quality: Dict[str, str]) -> str:
    """
    Get a summary of lead quality status
    
    Args:
        lead_quality: Dictionary mapping lead names to quality status
    
    Returns:
        Human-readable summary string
    
    Example:
        quality = {'I': 'OK', 'II': 'OFF', 'III': 'OK'}
        summary = get_lead_quality_summary(quality)
        # Returns: "1 lead disconnected (II)"
    """
    off_leads = [name for name, status in lead_quality.items() if status == 'OFF']
    
    if len(off_leads) == 0:
        return "All leads connected"
    elif len(off_leads) == 1:
        return f"1 lead disconnected ({off_leads[0]})"
    else:
        return f"{len(off_leads)} leads disconnected ({', '.join(off_leads)})"

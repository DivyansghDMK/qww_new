"""ECG interval calculations with smoothing and improved detection

This module provides interval calculations (PR, QRS, QT, P duration) with:
- Ectopic beat rejection
- EMA + median smoothing for stability
- Adaptive search windows
- Improved onset/offset detection
"""

import numpy as np
from typing import Optional, Tuple
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
from ..clinical_measurements import measure_rv5_sv1_from_median_beat, build_median_beat


# Global smoothing buffers for interval stabilization (WARNING FIX #8)
_pr_smoothing_buffers = {}
_qrs_smoothing_buffers = {}
_qt_smoothing_buffers = {}
_p_dur_smoothing_buffers = {}


def apply_interval_smoothing(value: int, buffer_key: str, buffer_dict: dict, buffer_size: int = 15) -> int:
    """
    Apply EMA + median smoothing to interval measurements (WARNING FIX #8)
    
    This prevents flickering of interval values while maintaining responsiveness
    to real changes. Uses the same pattern as heart rate smoothing.
    
    Args:
        value: Current interval value in milliseconds
        buffer_key: Unique identifier for this buffer (e.g., instance_id)
        buffer_dict: Dictionary holding all buffers for this interval type
        buffer_size: Size of median buffer (default: 15)
    
    Returns:
        Smoothed interval value in milliseconds
    """
    if buffer_key not in buffer_dict:
        buffer_dict[buffer_key] = {
            'buffer': deque(maxlen=buffer_size),
            'ema': float(value),
            'last_stable': value
        }
    
    state = buffer_dict[buffer_key]
    state['buffer'].append(value)
    
    # Calculate median from buffer
    if len(state['buffer']) >= 5:
        median_val = int(round(np.median(list(state['buffer']))))
    else:
        median_val = value
    
    # Adaptive EMA: fast response to large changes, slow for small changes
    current_display = int(round(state['ema']))
    alpha = 0.5 if abs(median_val - current_display) >= 2 else 0.10
    state['ema'] = (1 - alpha) * state['ema'] + alpha * median_val
    
    smoothed = int(round(state['ema']))
    
    # Only update if changed (prevents unnecessary UI updates)
    if abs(smoothed - state['last_stable']) >= 1:
        state['last_stable'] = smoothed
        return smoothed
    else:
        return state['last_stable']


def calculate_qtcf_interval(qt_ms: float, rr_ms: float) -> int:
    """Calculate QTcF using Fridericia formula: QTcF = QT / RR^(1/3)
    
    Args:
        qt_ms: QT interval in milliseconds
        rr_ms: RR interval in milliseconds
    
    Returns:
        QTcF in milliseconds (rounded to integer)
    """
    try:
        if not qt_ms or qt_ms <= 0 or not rr_ms or rr_ms <= 0:
            return 0
        
        # Convert to seconds
        qt_sec = qt_ms / 1000.0
        rr_sec = rr_ms / 1000.0
        
        # Fridericia formula: QTcF = QT / RR^(1/3)
        qtcf_sec = qt_sec / (rr_sec ** (1.0 / 3.0))
        
        # Convert back to milliseconds
        qtcf_ms = int(round(qtcf_sec * 1000.0))
        
        return qtcf_ms
    except Exception as e:
        print(f" Error calculating QTcF: {e}")
        return 0


def calculate_qtc_bazett(qt_ms: float, rr_ms: float) -> int:
    """Calculate QTc using Bazett formula: QTc = QT / sqrt(RR)
    
    Args:
        qt_ms: QT interval in milliseconds
        rr_ms: RR interval in milliseconds
    
    Returns:
        QTc in milliseconds (rounded to integer)
    """
    try:
        if not qt_ms or qt_ms <= 0 or not rr_ms or rr_ms <= 0:
            return 0
        
        # Convert to seconds
        qt_sec = qt_ms / 1000.0
        rr_sec = rr_ms / 1000.0
        
        # Bazett formula: QTc = QT / sqrt(RR)
        qtc_sec = qt_sec / np.sqrt(rr_sec)
        
        # Convert back to milliseconds
        qtc_ms = int(round(qtc_sec * 1000.0))
        
        return qtc_ms
    except Exception as e:
        print(f" Error calculating QTc (Bazett): {e}")
        return 0


def calculate_qtc_auto(qt_ms: float, rr_ms: float, heart_rate: int) -> int:
    """
    Calculate QTc using appropriate formula based on heart rate (INFO FIX #13)
    
    - Bazett: Best for HR 60-100 BPM
    - Fridericia: Best for HR <55 or >100 BPM
    
    Args:
        qt_ms: QT interval in milliseconds
        rr_ms: RR interval in milliseconds
        heart_rate: Heart rate in BPM
    
    Returns:
        QTc in milliseconds using appropriate formula
    """
    if heart_rate < 55 or heart_rate > 100:
        # Use Fridericia for bradycardia or tachycardia
        return calculate_qtcf_interval(qt_ms, rr_ms)
    else:
        # Use Bazett for normal heart rates
        return calculate_qtc_bazett(qt_ms, rr_ms)


def calculate_rv5_sv1_from_median(data: list, r_peaks: np.ndarray, fs: float) -> Tuple[Optional[float], Optional[float]]:
    """Calculate RV5 and SV1 from median beats (GE/Philips standard).
    
    Args:
        data: List of ECG data arrays (12 leads)
        r_peaks: R-peak indices from Lead II
        fs: Sampling rate in Hz
    
    Returns:
        Tuple of (rv5_mv, sv1_mv) in millivolts, or (None, None) if calculation fails
    """
    try:
        if len(data) < 11:
            return None, None
        
        # CRITICAL: Correct lead indices for 12-lead ECG
        # LEADS_MAP: ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        # Index 6 = V1, Index 10 = V5
        lead_v5_raw = np.asarray(data[10], dtype=float) if len(data) > 10 else None
        lead_v1_raw = np.asarray(data[6], dtype=float) if len(data) > 6 else None
        lead_ii = np.asarray(data[1], dtype=float)
        
        if lead_v5_raw is None or lead_v1_raw is None:
            return None, None
        
        if len(r_peaks) < 8:
            return None, None
        
        # Call measurement function using RAW data and shared R-peaks
        # ADC factors for V5/V1 (Marquette standards)
        rv5_mv, sv1_mv = measure_rv5_sv1_from_median_beat(
            lead_v5_raw, lead_v1_raw, 
            r_peaks, r_peaks,  # Use shared R-peaks for alignment
            fs, 
            v5_adc_per_mv=2048.0, 
            v1_adc_per_mv=1441.0
        )
        
        return rv5_mv, sv1_mv
    except Exception as e:
        print(f" Error calculating RV5/SV1 from median: {e}")
        return None, None

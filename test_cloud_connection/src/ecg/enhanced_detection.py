"""Enhanced ECG interval detection with improved algorithms

This module provides improved P, Q, R, S, T detection with:
- PR interval: P-onset to R-peak (WARNING FIX #5)
- QRS duration: Wider Q search window (WARNING FIX #6)
- QT interval: Adaptive T-wave search (WARNING FIX #7)
- P detection: Lower amplitude threshold (INFO FIX #11)
"""

import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple, Optional


def detect_p_onset(filtered_signal: np.ndarray, p_peak_idx: int, sampling_rate: float = 500) -> int:
    """
    Detect P-wave onset (start of P-wave) - WARNING FIX #5
    
    Args:
        filtered_signal: Filtered ECG signal
        p_peak_idx: P-peak index
        sampling_rate: Sampling rate in Hz
    
    Returns:
        P-onset index
    """
    # Search 80ms before P-peak
    search_start = max(0, p_peak_idx - int(0.08 * sampling_rate))
    segment = filtered_signal[search_start:p_peak_idx]
    
    if len(segment) == 0:
        return p_peak_idx
    
    # Find baseline
    baseline = np.median(segment)
    
    # Find onset: first point where signal exceeds 10% of P amplitude
    p_amp = filtered_signal[p_peak_idx] - baseline
    thresh = 0.10 * abs(p_amp)
    
    for i in range(len(segment)):
        if abs(segment[i] - baseline) > thresh:
            return search_start + i
    
    return p_peak_idx  # Fallback


def detect_q_peak_wide_window(filtered_signal: np.ndarray, r_peak_idx: int, sampling_rate: float = 500) -> int:
    """
    Detect Q-wave with wider search window - WARNING FIX #6
    
    Args:
        filtered_signal: Filtered ECG signal
        r_peak_idx: R-peak index
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Q-peak index
    """
    # UPDATED: 60 ms before R (was 40 ms) - WARNING FIX #6
    q_start = max(0, r_peak_idx - int(0.06 * sampling_rate))
    
    if q_start >= r_peak_idx:
        return r_peak_idx
    
    q_segment = filtered_signal[q_start:r_peak_idx]
    if len(q_segment) == 0:
        return r_peak_idx
    
    q_idx = q_start + np.argmin(q_segment)
    return q_idx


def detect_s_peak(filtered_signal: np.ndarray, r_peak_idx: int, sampling_rate: float = 500) -> int:
    """
    Detect S-wave (minimum after R-peak)
    
    Args:
        filtered_signal: Filtered ECG signal
        r_peak_idx: R-peak index
        sampling_rate: Sampling rate in Hz
    
    Returns:
        S-peak index
    """
    # Search 80ms after R-peak
    s_end = min(len(filtered_signal), r_peak_idx + int(0.08 * sampling_rate))
    
    if s_end <= r_peak_idx:
        return r_peak_idx
    
    s_segment = filtered_signal[r_peak_idx:s_end]
    if len(s_segment) == 0:
        return r_peak_idx
    
    s_idx = r_peak_idx + np.argmin(s_segment)
    return s_idx


def detect_t_peak_adaptive(filtered_signal: np.ndarray, r_peak_idx: int, rr_interval_ms: Optional[float], 
                          sampling_rate: float = 500) -> Optional[int]:
    """
    Detect T-wave peak with adaptive search window - WARNING FIX #7
    
    Args:
        filtered_signal: Filtered ECG signal
        r_peak_idx: R-peak index
        rr_interval_ms: RR interval in milliseconds (None for fixed window)
        sampling_rate: Sampling rate in Hz
    
    Returns:
        T-peak index or None if not found
    """
    # ADAPTIVE window based on RR interval - WARNING FIX #7
    if rr_interval_ms is not None and rr_interval_ms > 0:
        # T-wave search: 0.2*RR to 0.7*RR after R-peak
        t_start = r_peak_idx + int(0.2 * rr_interval_ms * sampling_rate / 1000)
        t_end = r_peak_idx + int(0.7 * rr_interval_ms * sampling_rate / 1000)
    else:
        # Fallback to fixed window
        t_start = r_peak_idx + int(0.2 * sampling_rate)  # 200ms
        t_end = r_peak_idx + int(0.4 * sampling_rate)    # 400ms
    
    # Clamp to signal bounds
    t_end = min(t_end, len(filtered_signal))
    
    if t_end <= t_start:
        return None
    
    t_segment = filtered_signal[t_start:t_end]
    if len(t_segment) == 0:
        return None
    
    # Find T-peak (maximum absolute value in segment)
    t_idx = t_start + np.argmax(np.abs(t_segment))
    return t_idx


def detect_p_peaks_enhanced(filtered_signal: np.ndarray, r_peaks: np.ndarray, sampling_rate: float = 500) -> List[int]:
    """
    Detect P-waves with lower amplitude threshold - INFO FIX #11
    
    Args:
        filtered_signal: Filtered ECG signal
        r_peaks: R-peak indices
        sampling_rate: Sampling rate in Hz
    
    Returns:
        List of P-peak indices
    """
    p_peaks = []
    
    for r_idx in r_peaks:
        # Search 80-200ms before R-peak
        p_start = max(0, r_idx - int(0.2 * sampling_rate))
        p_end = max(0, r_idx - int(0.08 * sampling_rate))
        
        if p_end <= p_start:
            continue
        
        p_segment = filtered_signal[p_start:p_end]
        if len(p_segment) == 0:
            continue
        
        # Find P-peak candidate
        p_candidate_idx = p_start + np.argmax(np.abs(p_segment))
        p_amp = abs(filtered_signal[p_candidate_idx])
        r_amp = abs(filtered_signal[r_idx])
        
        # INFO FIX #11: Lower threshold from 20% to 10% of R-peak
        # Also add 0.05mV absolute floor (≈10 ADC counts)
        min_amp = max(0.10 * r_amp, 10)  # 10% of R or 10 ADC counts
        
        if p_amp > min_amp:
            p_peaks.append(p_candidate_idx)
    
    return p_peaks


def calculate_pr_interval_improved(p_peaks: List[int], r_peaks: List[int], 
                                   filtered_signal: np.ndarray, sampling_rate: float = 500) -> List[float]:
    """
    Calculate PR interval using P-onset (not P-peak) - WARNING FIX #5
    
    Args:
        p_peaks: P-peak indices
        r_peaks: R-peak indices
        filtered_signal: Filtered ECG signal
        sampling_rate: Sampling rate in Hz
    
    Returns:
        List of PR intervals in milliseconds
    """
    pr_intervals = []
    
    for p_peak, r_peak in zip(p_peaks, r_peaks):
        if r_peak > p_peak:
            # Use P-onset instead of P-peak - WARNING FIX #5
            p_onset = detect_p_onset(filtered_signal, p_peak, sampling_rate)
            pr_ms = (r_peak - p_onset) / sampling_rate * 1000
            
            # UPDATED range: 80-200 ms (was 120-200 ms) - WARNING FIX #5
            # This allows detection of WPW syndrome (PR <120ms)
            if 80 <= pr_ms <= 200:
                pr_intervals.append(pr_ms)
    
    return pr_intervals


def calculate_qrs_duration_improved(q_peaks: List[int], s_peaks: List[int], sampling_rate: float = 500) -> List[float]:
    """
    Calculate QRS duration with updated range - WARNING FIX #6
    
    Args:
        q_peaks: Q-peak indices
        s_peaks: S-peak indices
        sampling_rate: Sampling rate in Hz
    
    Returns:
        List of QRS durations in milliseconds
    """
    qrs_durations = []
    
    for q_idx, s_idx in zip(q_peaks, s_peaks):
        if s_idx > q_idx:
            qrs_ms = (s_idx - q_idx) / sampling_rate * 1000
            
            # UPDATED range: 40-120 ms (was 60-120 ms) - WARNING FIX #6
            if 40 <= qrs_ms <= 120:
                qrs_durations.append(qrs_ms)
    
    return qrs_durations

"""ECG axis calculations from median beats"""
import numpy as np
from typing import Optional, List
from ..clinical_measurements import calculate_axis_from_median_beat, build_median_beat, get_tp_baseline


def calculate_qrs_axis_from_median(data: List[np.ndarray], leads: List[str], 
                                   r_peaks: np.ndarray, fs: float) -> Optional[int]:
    """Calculate QRS axis from median beat vectors (GE/Philips standard).
    
    Args:
        data: List of ECG data arrays for all leads
        leads: List of lead names
        r_peaks: R-peak indices
        fs: Sampling rate in Hz
    
    Returns:
        QRS axis in degrees, or None if calculation fails
    """
    try:
        if len(data) < 6:
            return None
        
        # Need Lead I (index 0), Lead II (index 1), and Lead aVF (index 5)
        lead_i_data = data[0] if len(data) > 0 else None
        lead_ii_data = data[1] if len(data) > 1 else None
        lead_avf_data = data[5] if len(data) > 5 else None
        
        if lead_i_data is None or lead_ii_data is None or lead_avf_data is None:
            return None
        
        if len(r_peaks) < 4:  # Absolute minimum reduced to 4
            return None
        
        # Calculate HR to determine min_beats (Fixed Bug P-2: Lower requirement for high BPM)
        rr_intervals = np.diff(r_peaks) / fs
        mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0.8
        hr_est = 60 / mean_rr if mean_rr > 0 else 75
        
        # Use 4 beats for high HR (>150), otherwise 8 (standard)
        min_beats_req = 4 if hr_est > 150 else 8
        
        # Build median beats for Lead I, II, and aVF
        time_axis_i, median_beat_i = build_median_beat(lead_i_data, r_peaks, fs, min_beats=min_beats_req)
        time_axis_ii, median_beat_ii = build_median_beat(lead_ii_data, r_peaks, fs, min_beats=min_beats_req)
        time_axis_avf, median_beat_avf = build_median_beat(lead_avf_data, r_peaks, fs, min_beats=min_beats_req)
        
        if median_beat_i is None or median_beat_ii is None or median_beat_avf is None:
            return None
        
        # Find R-peak index in median beat (should be at center)
        r_idx = len(median_beat_i) // 2
        
        # Get TP baseline for Lead I and aVF
        # We set these to None to force calculate_axis_from_median_beat to compute
        # a much more stable baseline from the median beat itself (which is already averaged),
        # preventing wild axis jumps due to single-beat baseline drift.
        r_mid = r_peaks[len(r_peaks) // 2]
        prev_r_idx = r_peaks[len(r_peaks) // 2 - 1] if len(r_peaks) > 1 else None
        tp_baseline_i = None
        tp_baseline_avf = None
        
        # Calculate QRS axis using standardized function
        qrs_axis = calculate_axis_from_median_beat(
            lead_i_data, lead_ii_data, lead_avf_data,
            median_beat_i, median_beat_ii, median_beat_avf,
            r_idx, fs, tp_baseline_i=tp_baseline_i, tp_baseline_avf=tp_baseline_avf,
            time_axis=time_axis_i, wave_type='QRS'
        )
        
        return qrs_axis
    except Exception as e:
        # OPTIMIZED: Reduced error print frequency for better performance
        if not hasattr(calculate_qrs_axis_from_median, '_error_count'):
            calculate_qrs_axis_from_median._error_count = 0
        calculate_qrs_axis_from_median._error_count += 1
        if calculate_qrs_axis_from_median._error_count % 100 == 1:  # Print every 100th error
            print(f" Error calculating QRS axis: {e}")
        return None


def calculate_p_axis_from_median(data: List[np.ndarray], leads: List[str],
                                 r_peaks: np.ndarray, fs: float,
                                 pr_ms: Optional[float] = None) -> Optional[int]:
    """Calculate P-wave axis from median beat vectors (GE/Philips standard).
    
    Args:
        data: List of ECG data arrays for all leads
        leads: List of lead names
        r_peaks: R-peak indices
        fs: Sampling rate in Hz
        pr_ms: PR interval in ms (optional, for P-wave window estimation)
    
    Returns:
        P-wave axis in degrees, or None if calculation fails
    """
    try:
        if len(data) < 6:
            return None
        
        lead_i_data = data[0] if len(data) > 0 else None
        lead_ii_data = data[1] if len(data) > 1 else None
        lead_avf_data = data[5] if len(data) > 5 else None
        
        if lead_i_data is None or lead_ii_data is None or lead_avf_data is None:
            return None
        
        if len(r_peaks) < 4: # Absolute minimum reduced to 4
            return None
        
        # Calculate HR to determine min_beats (Fixed Bug P-2)
        rr_intervals = np.diff(r_peaks) / fs
        mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0.8
        hr_est = 60 / mean_rr if mean_rr > 0 else 75
        min_beats_req = 4 if hr_est > 150 else 8
        
        # Build median beats
        time_axis_i, median_beat_i = build_median_beat(lead_i_data, r_peaks, fs, min_beats=min_beats_req)
        time_axis_ii, median_beat_ii = build_median_beat(lead_ii_data, r_peaks, fs, min_beats=min_beats_req)
        time_axis_avf, median_beat_avf = build_median_beat(lead_avf_data, r_peaks, fs, min_beats=min_beats_req)
        
        if median_beat_i is None or median_beat_ii is None or median_beat_avf is None:
            return None
        
        r_idx = len(median_beat_i) // 2
        r_mid = r_peaks[len(r_peaks) // 2]
        prev_r_idx = r_peaks[len(r_peaks) // 2 - 1] if len(r_peaks) > 1 else None
        
        # P-axis uses PRE-P baseline [-300ms, -200ms] before R
        tp_baseline_i = get_tp_baseline(lead_i_data, r_mid, fs, prev_r_peak_idx=prev_r_idx, use_measurement_channel=True)
        tp_baseline_avf = get_tp_baseline(lead_avf_data, r_mid, fs, prev_r_peak_idx=prev_r_idx, use_measurement_channel=True)
        
        # Calculate P axis
        p_axis = calculate_axis_from_median_beat(
            lead_i_data, lead_ii_data, lead_avf_data,
            median_beat_i, median_beat_ii, median_beat_avf,
            r_idx, fs, tp_baseline_i=tp_baseline_i, tp_baseline_avf=tp_baseline_avf,
            time_axis=time_axis_i, wave_type='P', pr_ms=pr_ms
        )
        
        return p_axis
    except Exception as e:
        # OPTIMIZED: Reduced error print frequency for better performance
        if not hasattr(calculate_p_axis_from_median, '_error_count'):
            calculate_p_axis_from_median._error_count = 0
        calculate_p_axis_from_median._error_count += 1
        if calculate_p_axis_from_median._error_count % 100 == 1:  # Print every 100th error
            print(f" Error calculating P axis: {e}")
        return None


def calculate_t_axis_from_median(data: List[np.ndarray], leads: List[str],
                                 r_peaks: np.ndarray, fs: float) -> Optional[int]:
    """Calculate T-wave axis from median beat vectors (GE/Philips standard).
    
    Args:
        data: List of ECG data arrays for all leads
        leads: List of lead names
        r_peaks: R-peak indices
        fs: Sampling rate in Hz
    
    Returns:
        T-wave axis in degrees, or None if calculation fails
    """
    try:
        if len(data) < 6:
            return None
        
        lead_i_data = data[0] if len(data) > 0 else None
        lead_ii_data = data[1] if len(data) > 1 else None
        lead_avf_data = data[5] if len(data) > 5 else None
        
        if lead_i_data is None or lead_ii_data is None or lead_avf_data is None:
            return None
        
        if len(r_peaks) < 4:
            return None
        
        # Calculate HR to determine min_beats (Fixed Bug P-2)
        rr_intervals = np.diff(r_peaks) / fs
        mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0.8
        hr_est = 60 / mean_rr if mean_rr > 0 else 75
        min_beats_req = 4 if hr_est > 150 else 8
        
        # Build median beats
        time_axis_i, median_beat_i = build_median_beat(lead_i_data, r_peaks, fs, min_beats=min_beats_req)
        time_axis_ii, median_beat_ii = build_median_beat(lead_ii_data, r_peaks, fs, min_beats=min_beats_req)
        time_axis_avf, median_beat_avf = build_median_beat(lead_avf_data, r_peaks, fs, min_beats=min_beats_req)
        
        if median_beat_i is None or median_beat_ii is None or median_beat_avf is None:
            return None
        
        r_idx = len(median_beat_i) // 2
        r_mid = r_peaks[len(r_peaks) // 2]
        prev_r_idx = r_peaks[len(r_peaks) // 2 - 1] if len(r_peaks) > 1 else None
        
        # T-axis uses post-T TP baseline [700ms, 800ms] after R
        # We pass None to let calculate_axis_from_median_beat compute this cleanly from the baseline of the median beat
        tp_baseline_i = None
        tp_baseline_avf = None
        
        # Calculate T axis
        t_axis = calculate_axis_from_median_beat(
            lead_i_data, lead_ii_data, lead_avf_data,
            median_beat_i, median_beat_ii, median_beat_avf,
            r_idx, fs, tp_baseline_i=tp_baseline_i, tp_baseline_avf=tp_baseline_avf,
            time_axis=time_axis_i, wave_type='T'
        )
        
        return t_axis
    except Exception as e:
        # OPTIMIZED: Reduced error print frequency for better performance
        if not hasattr(calculate_t_axis_from_median, '_error_count'):
            calculate_t_axis_from_median._error_count = 0
        calculate_t_axis_from_median._error_count += 1
        if calculate_t_axis_from_median._error_count % 100 == 1:  # Print every 100th error
            print(f" Error calculating T axis: {e}")
        return None

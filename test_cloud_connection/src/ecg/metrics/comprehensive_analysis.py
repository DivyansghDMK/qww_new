import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import Dict, Any, List, Optional, Tuple
from ..qrs_detection import qrs_duration_from_raw_signal

# -----------------------------------------------------------------------------
# Helper Classes & Functions (Translated from User Kotlin Code)
# -----------------------------------------------------------------------------

class AdaptiveWindows:
    def __init__(self, minPtoQRS, maxPtoQRS, pSearchWindow, qrsOnsetSearch, qrsOffsetFromR, tSearchStart, tSearchEnd):
        self.minPtoQRS = minPtoQRS
        self.maxPtoQRS = maxPtoQRS
        self.pSearchWindow = pSearchWindow
        self.qrsOnsetSearch = qrsOnsetSearch
        self.qrsOffsetFromR = qrsOffsetFromR
        self.tSearchStart = tSearchStart
        self.tSearchEnd = tSearchEnd

def calculate_adaptive_windows(heart_rate: int, rr_interval_sec: float, fs: float) -> AdaptiveWindows:
    rr_samples = int(rr_interval_sec * fs)
    
    if heart_rate < 40:
        return AdaptiveWindows(
            75, 150, 200, 50, 80, 150, min(300, int(rr_samples * 0.65))
        )
    elif heart_rate < 60:
        return AdaptiveWindows(
            60, 120, 150, 40, 60, 120, min(250, int(rr_samples * 0.60))
        )
    elif 60 <= heart_rate <= 100:
        return AdaptiveWindows(
            45, 85, 100, 30, 45, 100, min(175, int(rr_samples * 0.55))
        )
    elif 100 < heart_rate <= 150:
        return AdaptiveWindows(
            30, 75, 95, 25, 35, 50, min(155, int(rr_samples * 0.55))
        )
    elif 150 < heart_rate <= 250:
        return AdaptiveWindows(
            15, 35, 45, 20, 25, 40, min(200, int(rr_samples * 0.80))
        )
    else:
        return AdaptiveWindows(10, 20, 25, 15, 20, 25, min(110, int(rr_samples * 0.75)))

def calculate_expected_qt(rr_interval_sec: float, heart_rate: int, fs: float) -> int:
    # Fallback / Estimation for QT if T-wave not found
    # Using a generic expectation: QTc ~ 400ms -> QT = 0.4 * sqrt(RR)
    # User didn't provide this function, implementing reasonable default
    qt_sec = 0.42 * np.sqrt(rr_interval_sec)
    return int(qt_sec * fs)

def calculate_baseline_pre_qrs(data: np.ndarray, qrs_start: int) -> float:
    # Not provided in snippet, implementing logical baseline check
    # Typically 10-30ms before QRS onset
    start = max(0, qrs_start - 30)
    end = max(0, qrs_start - 10)
    if end > start:
        return float(np.mean(data[start:end]))
    return float(data[qrs_start]) if qrs_start < len(data) else 0.0

def detect_qrs_start_adaptive(data: np.ndarray, r_peak: int, windows: AdaptiveWindows) -> int:
    search_start = max(0, r_peak - windows.qrsOnsetSearch)
    if search_start >= r_peak - 7:
        return max(0, r_peak - (windows.qrsOnsetSearch // 2))
    
    baseline_start = max(0, r_peak - (windows.qrsOnsetSearch + 40))
    baseline_end = max(0, r_peak - (windows.qrsOnsetSearch + 10))
    
    if baseline_end > baseline_start:
        baseline = float(np.mean(data[baseline_start:baseline_end]))
    else:
        baseline = float(data[max(0, r_peak - 50)])
        
    r_amplitude = data[r_peak] - baseline
    threshold = r_amplitude * 0.07
    
    for i in range(search_start, r_peak - 7):
        if abs(data[i] - baseline) > threshold:
            return i
            
    return r_peak - (windows.qrsOnsetSearch // 2)

def detect_qrs_end_adaptive(data: np.ndarray, r_peak: int, windows: AdaptiveWindows) -> int:
    search_start = r_peak + (windows.qrsOffsetFromR // 2)
    search_end = min(len(data) - 1, r_peak + windows.qrsOffsetFromR)

    if search_start >= search_end:
        return r_peak + (windows.qrsOffsetFromR // 2)

    if search_end + 20 < len(data):
        st_baseline = float(np.mean(data[search_end:search_end + 20]))
    else:
        st_baseline = float(data[min(len(data) - 1, search_end + 5)])

    # Scale-adaptive threshold: 15% of R-peak amplitude
    # (replaces hardcoded 0.15 which only works for mV-scale signals)
    r_peak_amp = abs(float(data[r_peak]) - st_baseline) if r_peak < len(data) else 1.0
    if r_peak_amp < 1e-9:
        r_peak_amp = 1.0
    amp_threshold = 0.15 * r_peak_amp

    j_point = search_start
    min_slope = float('inf')

    for i in range(search_start, search_end - 2):
        slope = abs(data[i + 1] - data[i])
        if slope < min_slope and abs(data[i] - st_baseline) < amp_threshold:
            min_slope = slope
            j_point = i

    return j_point

def detect_tend_by_tangent(data: np.ndarray, t_peak: int, baseline: float, search_end: int, 
                           heart_rate: int, qrs_start: int, expected_qt: int, err_thr: float) -> int:
    sw = 50
    if heart_rate > 280: sw = 90
    elif heart_rate > 250: sw = 80
    elif heart_rate > 150: sw = 70
    
    max_slope = 0.0
    slope_idx = t_peak
    
    limit = min(t_peak + sw, search_end - 5)
    for i in range(t_peak, limit):
        s = abs(data[i] - data[i + 5]) / 5.0
        if s > max_slope:
            max_slope = s
            slope_idx = i
            
    min_slope_thr = 0.004
    if heart_rate > 280: min_slope_thr = 0.002
    elif heart_rate > 250: min_slope_thr = 0.0025
    elif heart_rate > 180: min_slope_thr = 0.003
    elif heart_rate > 150: min_slope_thr = 0.0035
    
    if max_slope < min_slope_thr:
        return -1
        
    flat_mul = 0.12
    if heart_rate > 280: flat_mul = 0.18
    elif heart_rate > 250: flat_mul = 0.16
    elif heart_rate > 180: flat_mul = 0.15
    elif heart_rate > 150: flat_mul = 0.13
    
    flat_thr = max_slope * flat_mul
    
    stab_mul = 1.6
    if heart_rate > 280: stab_mul = 2.0
    elif heart_rate > 250: stab_mul = 1.9
    elif heart_rate > 180: stab_mul = 1.8
    elif heart_rate > 150: stab_mul = 1.7
    
    for i in range(slope_idx, search_end - 5):
        if abs(data[i + 1] - data[i]) < flat_thr:
            # Check next 3 samples stability
            is_stable = True
            for it in range(1, 4):
                idx1 = min(i + it + 1, len(data) - 1)
                idx2 = min(i + it, len(data) - 1)
                if abs(data[idx1] - data[idx2]) >= flat_thr * stab_mul:
                    is_stable = False
                    break
            
            if is_stable and abs(i - qrs_start - expected_qt) < expected_qt * err_thr:
                return i
                
    return -1

def detect_t_wave_end_adaptive(data: np.ndarray, r_peak: int, qrs_start: int, next_r_peak: int, 
                               windows: AdaptiveWindows, rr_interval_sec: float, heart_rate: int, fs: float) -> int:
    search_start = r_peak + windows.tSearchStart
    search_end = min(len(data) - 1, r_peak + windows.tSearchEnd)
    margin = 5 if heart_rate > 250 else 10
    
    safe_end = search_end
    if next_r_peak > 0:
        safe_end = min(search_end, next_r_peak - margin)
        
    expected_qt = calculate_expected_qt(rr_interval_sec, heart_rate, fs)
    
    if search_start >= safe_end:
        return min(len(data)-1, qrs_start + expected_qt)
        
    t_peak_search_window = 50
    if heart_rate > 250: t_peak_search_window = 80
    elif heart_rate > 150: t_peak_search_window = 70
    
    # Slice for T-peak search
    t_slice_end = min(safe_end, search_start + t_peak_search_window)
    t_slice = data[search_start:t_slice_end]
    
    if len(t_slice) == 0:
        return min(len(data)-1, qrs_start + expected_qt)
        
    t_peak_relative = np.argmax(t_slice) # Kotlin code: indices.maxByOrNull { tSlice[it] } (assumes positive T??)
    # The Kotlin code maxByOrNull implies finding the MAX value. 
    # If T is inverted, this might find just noise or the least negative point? 
    # For now assuming standard positive T-wave as per user snippet.
    t_peak = search_start + t_peak_relative
    
    baseline = calculate_baseline_pre_qrs(data, qrs_start)
    t_peak_value = data[t_peak]
    
    # Parameters based on HR
    if heart_rate > 280:
        min_slope_thr, stability_window, min_descent = 0.100, 2, 0.010
    elif heart_rate > 250:
        min_slope_thr, stability_window, min_descent = 0.080, 2, 0.012
    elif heart_rate > 200:
        min_slope_thr, stability_window, min_descent = 0.050, 3, 0.015
    elif heart_rate > 180:
        min_slope_thr, stability_window, min_descent = 0.040, 3, 0.018
    elif heart_rate > 150:
        min_slope_thr, stability_window, min_descent = 0.035, 3, 0.020
    elif heart_rate > 120:
        min_slope_thr, stability_window, min_descent = 0.003, 4, 0.030
    elif heart_rate > 100:
        min_slope_thr, stability_window, min_descent = 0.0025, 4, 0.035
    else:
        min_slope_thr, stability_window, min_descent = 0.002, 5, 0.050
        
    if heart_rate > 280: stability_thr = 0.120
    elif heart_rate > 250: stability_thr = 0.100
    elif heart_rate > 200: stability_thr = 0.080
    elif heart_rate > 180: stability_thr = 0.070
    elif heart_rate > 150: stability_thr = 0.060
    elif heart_rate > 120: stability_thr = 0.022
    elif heart_rate > 100: stability_thr = 0.020
    else: stability_thr = 0.015
    
    if heart_rate > 280: err_thr = 0.70
    elif heart_rate > 250: err_thr = 0.65
    elif heart_rate > 200: err_thr = 0.60
    elif heart_rate > 180: err_thr = 0.55
    elif heart_rate > 150: err_thr = 0.50
    elif heart_rate > 120: err_thr = 0.40
    elif heart_rate > 100: err_thr = 0.38
    else: err_thr = 0.35
    
    start_offset = 8
    if heart_rate > 280: start_offset = 3
    elif heart_rate > 250: start_offset = 4
    elif heart_rate > 180: start_offset = 4
    elif heart_rate > 150: start_offset = 6
    
    # First Pass: Slope & Stability
    for i in range(t_peak + start_offset, safe_end - stability_window):
        descent = abs(t_peak_value - data[i])
        if descent < min_descent:
            continue
            
        max_local_slope = 0.0
        for k in range(3):
             slope_val = abs(data[i + k + 1] - data[i + k])
             if slope_val > max_local_slope:
                 max_local_slope = slope_val
        
        if max_local_slope > min_slope_thr:
            continue
            
        # Range max-min
        win_vals = data[i : i + stability_window + 1]
        volt_range = np.max(win_vals) - np.min(win_vals)
        
        if volt_range > stability_thr:
            continue
            
        qt_err = abs(i - qrs_start - expected_qt)
        if qt_err < expected_qt * err_thr:
            return i
            
    # Tangent fallback
    tangent_end = detect_tend_by_tangent(
        data, t_peak, baseline, safe_end, heart_rate, qrs_start, expected_qt, err_thr
    )
    if tangent_end > 0:
        return tangent_end
        
    return min(len(data) - 1, qrs_start + expected_qt)


# -----------------------------------------------------------------------------
# Main Analysis Function
# -----------------------------------------------------------------------------

def bandpass(x, fs):
    """
    Apply 0.5-40Hz bandpass filter.
    """
    if not np.isfinite(fs) or fs <= 0:
        return x
    nyquist = fs / 2.0
    low = max(0.5 / nyquist, 0.001)
    high = min(40.0 / nyquist, 0.99)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high >= 1 or low >= high:
        return x
    b, a = butter(2, [low, high], 'band')
    return filtfilt(b, a, x)

def calculate_comprehensive_metrics(lead_data: np.ndarray, fs: float = 500.0) -> Dict[str, Any]:
    """
    Calculate comprehensive ECG metrics using User's Adaptive Logic.
    """
    results = {
        "heart_rate": None,
        "rr_interval": None,
        "pr_interval": None,
        "qrs_duration": None,
        "qt_interval": None,
        "qtc_interval": None
    }
    
    sig = np.array(lead_data, float)
    if len(sig) < 2000:
        return results

    # Remove DC
    sig -= np.mean(sig)
    
    # Filter for analysis (User logic likely expects clean signal)
    filt = bandpass(sig, fs)
    
    # Detect R-peaks
    try:
        from ..pan_tompkins import pan_tompkins
        r_peaks = pan_tompkins(filt, fs=fs)
    except Exception:
        r_peaks = np.array([], dtype=int)

    # Fallback R-peaks
    if len(r_peaks) < 3: # Need at least 3 peaks to have a full 'middle' interval
        energy = np.diff(filt) ** 2
        peaks, _ = find_peaks(energy, distance=int(0.3 * fs), height=np.mean(energy) * 5)
        if len(peaks) < 3:
            return results
        r_peaks = peaks
        
    # Select the second to last R-peak to ensure we have a next R-peak for T-wave search
    # [ ... R_prev, R_curr, R_next ... ]
    # We analyze the cycle starting at R_curr (actually QT is R_curr to T_end)
    # Wait, detectTWaveEndAdaptive uses rPeak and nextRPeak.
    
    r_curr_idx = int(r_peaks[-2])
    r_next_idx = int(r_peaks[-1])
    r_prev_idx = int(r_peaks[-3])
    
    # RR Interval (Current)
    rr_samples = r_next_idx - r_curr_idx
    rr_sec = rr_samples / fs
    rr_ms = rr_sec * 1000.0
    
    # Heart Rate
    if rr_sec > 0:
        hr = int(round(60.0 / rr_sec))
    else:
        hr = 72 # fallback
        
    results["heart_rate"] = hr
    results["rr_interval"] = rr_ms
    
    # Adaptive Windows
    windows = calculate_adaptive_windows(hr, rr_sec, fs)
    
    # QRS Detection (Paper-based: Curtin et al. 2018, Stage 6-10)
    qrs_dur_ms = qrs_duration_from_raw_signal(filt, r_curr_idx, fs, adc_per_mv=1200.0)
    results["qrs_duration"] = qrs_dur_ms

    # QRS start/end still needed for QT and PR calculations below
    qrs_start = detect_qrs_start_adaptive(filt, r_curr_idx, windows)
    qrs_end = detect_qrs_end_adaptive(filt, r_curr_idx, windows)
    
    # QT Detection (Adaptive)
    t_end = detect_t_wave_end_adaptive(filt, r_curr_idx, qrs_start, r_next_idx, windows, rr_sec, hr, fs)
    
    qt_ms = (t_end - qrs_start) / fs * 1000.0
    results["qt_interval"] = qt_ms
    
    # QTc (Bazett only as requested)
    if rr_sec > 0:
        # QTc = QT / sqrt(RR_sec)
        # Standard unit: QT in sec? No, usually QT in Same Unit, but sqrt(RR) in sec.
        # Bazett: QTc = QT(s) / sqrt(RR(s)). Result in sec. Then * 1000 for ms.
        # OR: QTc(ms) = QT(ms) / sqrt(RR(s)). This is the same.
        qtc_ms = qt_ms / np.sqrt(rr_sec)
        results["qtc_interval"] = qtc_ms
    
    # PR Interval (Using Adaptive P search window)
    # Treating AdaptiveWindows values as SAMPLES
    
    p_search_end = qrs_start - windows.minPtoQRS 
    p_search_start = max(0, qrs_start - windows.maxPtoQRS)
    
    # Simple P-detection
    if p_search_end > p_search_start:
        p_seg = filt[p_search_start:p_search_end]
        if len(p_seg) > 0:
            p_peak_rel = np.argmax(np.abs(p_seg))
            p_peak = p_search_start + p_peak_rel
            
            # P-onset: backtrack from P-peak
            # Simple threshold
            p_amp = abs(filt[p_peak])
            p_onset = p_peak
            for i in range(p_peak, p_search_start, -1):
                if abs(filt[i]) < p_amp * 0.1: # 10% threshold
                    p_onset = i
                    break
            
            pr_ms = (qrs_start - p_onset) / fs * 1000.0
            results["pr_interval"] = pr_ms

    return results

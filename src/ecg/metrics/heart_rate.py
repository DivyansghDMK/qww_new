"""Heart rate calculation from ECG signals"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import platform
import time
from collections import deque


# Global smoothing buffers for BPM stabilization
_bpm_smoothing_buffers = {}  # Key: instance_id, Value: deque buffer
_bpm_ema_values = {}  # Key: instance_id, Value: EMA value
_last_stable_bpm = {}  # Key: instance_id, Value: Last stable BPM value
_bpm_last_success_ts = {}  # Key: instance_id, Value: last success timestamp


def calculate_heart_rate_from_signal(lead_data, sampling_rate=None, sampler=None, instance_id=None):
    """Calculate heart rate from Lead II data using R-R intervals
    
    Args:
        lead_data: Raw ECG signal data (numpy array or list)
        sampling_rate: Sampling rate in Hz (optional, defaults to 500 Hz)
        sampler: SamplingRateCalculator instance (optional, used if sampling_rate not provided)
    
    Returns:
        int: Heart rate in BPM (10-300 range), or 0 if calculation fails
    """
    try:
        buffer_key = instance_id if instance_id is not None else 'global'
        def _fallback_value():
            last = _last_stable_bpm.get(buffer_key, None)
            last_success = _bpm_last_success_ts.get(buffer_key, 0.0)
            now = time.time()
            if last is not None and (now - last_success) <= 0.5:
                return last
            return 0
        # Early exit: if no real signal, report 0 instead of fallback
        try:
            arr = np.asarray(lead_data, dtype=float)
            if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
                return 0
        except Exception:
            return 0

        # Validate input data
        if not isinstance(lead_data, (list, np.ndarray)) or len(lead_data) < 200:
            print(" Insufficient data for heart rate calculation")
            return _fallback_value()

        # Convert to numpy array for processing
        try:
            lead_data = np.asarray(lead_data, dtype=float)
        except Exception as e:
            print(f" Error converting lead data to array: {e}")
            return _fallback_value()

        # Check for invalid values
        if np.any(np.isnan(lead_data)) or np.any(np.isinf(lead_data)):
            print(" Invalid values (NaN/Inf) in lead data")
            return _fallback_value()

        # Get sampling rate
        is_windows = platform.system() == 'Windows'
        platform_tag = "[Windows]" if is_windows else "[macOS/Linux]"
        
        fs = 500.0  # Standard fallback for all platforms
        if sampling_rate is not None and sampling_rate > 10:
            fs = float(sampling_rate)
        elif sampler is not None and hasattr(sampler, 'sampling_rate') and sampler.sampling_rate > 10:
            detected_rate = sampler.sampling_rate
            if detected_rate > 10 and np.isfinite(detected_rate):
                fs = float(detected_rate)
        
        # Validation
        if fs <= 0 or not np.isfinite(fs):
            fs = 500.0  # Fallback

        # Apply display filter for R-peak detection (0.5-40 Hz)
        try:
            from ..signal_paths import display_filter
            filtered_signal = display_filter(lead_data, fs)
            if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                print(" Filter produced invalid values")
                return _fallback_value()
        except Exception as e:
            print(f" Error in signal filtering: {e}")
            return _fallback_value()

        # Find R-peaks using scipy with robust parameters
        try:
            signal_mean = np.mean(filtered_signal)
            signal_std = np.std(filtered_signal)
            if signal_std == 0:
                print(" No signal variation detected")
                return _fallback_value()
            
            # SMART ADAPTIVE PEAK DETECTION (10-300 BPM with BPM-based selection)
            height_threshold = signal_mean + 0.5 * signal_std
            prominence_threshold = signal_std * 0.4
            
            # Run 3 detection strategies
            detection_results = []
            
            # Strategy 1: Conservative (best for 10-120 BPM)
            peaks_conservative, _ = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=int(0.4 * fs),  # 400ms
                prominence=prominence_threshold
            )
            if len(peaks_conservative) >= 2:
                rr_cons = np.diff(peaks_conservative) * (1000 / fs)
                valid_cons = rr_cons[(rr_cons >= 200) & (rr_cons <= 6000)]
                if len(valid_cons) > 0:
                    bpm_cons = 60000 / np.median(valid_cons)
                    std_cons = np.std(valid_cons)
                    detection_results.append(('conservative', peaks_conservative, bpm_cons, std_cons))
            
            # Strategy 2: Normal (best for 100-180 BPM)
            peaks_normal, _ = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=int(0.3 * fs),  # 240ms
                prominence=prominence_threshold
            )
            if len(peaks_normal) >= 2:
                rr_norm = np.diff(peaks_normal) * (1000 / fs)
                valid_norm = rr_norm[(rr_norm >= 200) & (rr_norm <= 6000)]
                if len(valid_norm) > 0:
                    bpm_norm = 60000 / np.median(valid_norm)
                    std_norm = np.std(valid_norm)
                    detection_results.append(('normal', peaks_normal, bpm_norm, std_norm))
            
            # Strategy 3: Tight (best for 160-300 BPM)
            peaks_tight, _ = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=int(0.2 * fs),  # 160ms
                prominence=prominence_threshold
            )
            if len(peaks_tight) >= 2:
                rr_tight = np.diff(peaks_tight) * (1000 / fs)
                valid_tight = rr_tight[(rr_tight >= 200) & (rr_tight <= 6000)]
                if len(valid_tight) > 0:
                    bpm_tight = 60000 / np.median(valid_tight)
                    std_tight = np.std(valid_tight)
                    detection_results.append(('tight', peaks_tight, bpm_tight, std_tight))
            
            # Select based on BPM consistency (lowest std deviation = most stable)
            if detection_results:
                detection_results.sort(key=lambda x: x[3])  # Sort by std
                best_method, peaks, best_bpm, best_std = detection_results[0]
            else:
                # Fallback - use conservative distance
                peaks, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.4 * fs),
                    prominence=prominence_threshold
                )
        except Exception as e:
            print(f" Error in peak detection: {e}")
            return _fallback_value()

        if len(peaks) < 2:
            print(f" Insufficient peaks detected: {len(peaks)}")
            return _fallback_value()

        # Calculate R-R intervals in milliseconds
        try:
            rr_intervals_ms = np.diff(peaks) * (1000 / fs)
            if len(rr_intervals_ms) == 0:
                print(" No R-R intervals calculated")
                return _fallback_value()
        except Exception as e:
            print(f" Error calculating R-R intervals: {e}")
            return _fallback_value()

        # Filter physiologically reasonable intervals (200-6000 ms)
        try:
            valid_intervals = rr_intervals_ms[(rr_intervals_ms >= 200) & (rr_intervals_ms <= 6000)]
            if len(valid_intervals) == 0:
                print(" No valid R-R intervals found")
                return _fallback_value()
        except Exception as e:
            print(f" Error filtering intervals: {e}")
            return _fallback_value()

        # Calculate heart rate from median R-R interval
        try:
            median_rr = np.median(valid_intervals)
            if median_rr <= 0:
                print(" Invalid median R-R interval")
                return _fallback_value()
            heart_rate = 60000 / median_rr
            # Extended: stable 10–300 BPM range
            heart_rate = max(10, min(300, heart_rate))
            
            # Extra guard: avoid falsely reporting very high BPM when real rate is very low
            try:
                window_sec = len(lead_data) / float(fs)
            except Exception:
                window_sec = 0
            if heart_rate > 150 and window_sec >= 5.0:
                expected_peaks = (heart_rate * window_sec) / 60.0
                if expected_peaks > len(peaks) * 3:
                    print(f" Suspicious high BPM ({heart_rate:.1f}) with too few peaks. Clamping to 10 bpm.")
                    heart_rate = 10.0
            
            if np.isnan(heart_rate) or np.isinf(heart_rate):
                print(" Invalid heart rate calculated")
                return _fallback_value()
            
            # ENHANCED BPM SMOOTHING: Prevent flickering between 99-101 BPM
            hr_int = int(round(heart_rate))
            
            # Use instance_id for per-instance smoothing (or default to 'global')
            
            # Initialize smoothing buffer (larger buffer for better stability)
            if buffer_key not in _bpm_smoothing_buffers:
                _bpm_smoothing_buffers[buffer_key] = deque(maxlen=15)  # Increased from 10 to 15
            
            buffer = _bpm_smoothing_buffers[buffer_key]
            buffer.append(hr_int)
            
            # Initialize EMA if needed
            if buffer_key not in _bpm_ema_values:
                _bpm_ema_values[buffer_key] = float(hr_int)
            
            # Calculate median of buffer (rejects outliers)
            if len(buffer) >= 5:
                median_hr = int(round(np.median(list(buffer))))
            else:
                median_hr = hr_int
            
            try:
                current_display = int(round(_bpm_ema_values[buffer_key]))
            except Exception:
                current_display = int(round(hr_int))
            alpha = 0.5 if abs(median_hr - current_display) >= 1 else 0.10
            _bpm_ema_values[buffer_key] = (1 - alpha) * _bpm_ema_values[buffer_key] + alpha * median_hr
            
            smoothed_hr = int(round(_bpm_ema_values[buffer_key]))
            
            if buffer_key not in _last_stable_bpm:
                _last_stable_bpm[buffer_key] = smoothed_hr
            
            last_stable = _last_stable_bpm[buffer_key]
            
            if abs(smoothed_hr - last_stable) >= 1:
                _last_stable_bpm[buffer_key] = smoothed_hr
                _bpm_last_success_ts[buffer_key] = time.time()
                return smoothed_hr
            else:
                _bpm_last_success_ts[buffer_key] = time.time()
                return last_stable
            
        except Exception as e:
            print(f" Error calculating final BPM: {e}")
            return _fallback_value()
    except Exception as e:
        print(f" Critical error in calculate_heart_rate_from_signal: {e}")
        return 0

"""Heart rate calculation from ECG signals"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import platform
import time
from collections import deque


# Global smoothing buffers for BPM stabilization
_bpm_smoothing_buffers = {}  # Key: instance_id, Value: deque buffer
_bpm_ema_values = {}         # Key: instance_id, Value: EMA value
_last_stable_bpm = {}        # Key: instance_id, Value: Last stable BPM value
_bpm_last_success_ts = {}    # Key: instance_id, Value: last success timestamp

# FIX #5: Startup beat counter - do not emit HR until enough stable beats seen
_bpm_beat_count = {}         # Key: instance_id, Value: cumulative beat count

# PR Stabilization (FIX #6): 10-second stability requirement
_bpm_displayed_value = {}    # Key: instance_id, Value: Currently displayed stable BPM
_bpm_pending_value = {}      # Key: instance_id, Value: New value being monitored for stability
_bpm_stability_start_ts = {} # Key: instance_id, Value: Timestamp when pending value first appeared


def cleanup_instance(instance_id: str):
    """
    Remove all smoothing state for a given instance_id.

    Call this when a monitoring session ends to prevent memory leaks
    in long-running processes that create many instance_ids over time.

    Args:
        instance_id: The instance key used in calculate_heart_rate_from_signal()
    """
    for d in (_bpm_smoothing_buffers, _bpm_ema_values,
              _last_stable_bpm, _bpm_last_success_ts, _bpm_beat_count,
              _bpm_displayed_value, _bpm_pending_value, _bpm_stability_start_ts):
        d.pop(instance_id, None)


# FIX #5: Startup parameters
_STARTUP_LOCKOUT_BEATS = 5    # Ignore first N beat detections
_STARTUP_RR_MAX_MS     = 2000 # RR > 2000ms (< 30 BPM) at startup = noise floor
_STARTUP_ECTOPIC_TOL   = 0.10 # Tighter ectopic rejection during warmup (10%)
_NORMAL_ECTOPIC_TOL    = 0.20 # Normal tolerance after warmup (20%)


def calculate_heart_rate_from_signal(lead_data, sampling_rate=None, sampler=None, instance_id=None):
    """Calculate heart rate from Lead II data using R-R intervals.

    Args:
        lead_data:      Raw ECG signal data (numpy array or list).
        sampling_rate:  Sampling rate in Hz (optional, defaults to 500 Hz).
        sampler:        SamplingRateCalculator instance (optional).
        instance_id:    Key for per-instance smoothing buffers.

    Returns:
        int: Heart rate in BPM (10-300 range), or 0 if calculation fails.
    """
    try:
        buffer_key = instance_id if instance_id is not None else 'global'

        def _fallback_value():
            last = _last_stable_bpm.get(buffer_key, None)
            last_success = _bpm_last_success_ts.get(buffer_key, 0.0)
            if last is not None and (time.time() - last_success) <= 0.5:
                return last
            return 0

        # -- Early exit: flat / silent signal -----------------------------
        try:
            arr = np.asarray(lead_data, dtype=float)
            if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
                return 0
        except Exception:
            return 0

        # -- Validate input ------------------------------------------------
        if not isinstance(lead_data, (list, np.ndarray)) or len(lead_data) < 200:
            print(" Insufficient data for heart rate calculation")
            return _fallback_value()

        try:
            lead_data = np.asarray(lead_data, dtype=float)
        except Exception as e:
            print(f" Error converting lead data to array: {e}")
            return _fallback_value()

        if np.any(np.isnan(lead_data)) or np.any(np.isinf(lead_data)):
            print(" Invalid values (NaN/Inf) in lead data")
            return _fallback_value()

        # -- Sampling rate -------------------------------------------------
        fs = 500.0
        if sampling_rate is not None and sampling_rate > 10:
            fs = float(sampling_rate)
        elif (sampler is not None and hasattr(sampler, 'sampling_rate')
              and sampler.sampling_rate > 10):
            detected = sampler.sampling_rate
            if np.isfinite(detected):
                fs = float(detected)
        if fs <= 0 or not np.isfinite(fs):
            fs = 500.0

        # -- Filter --------------------------------------------------------
        try:
            from ..signal_paths import display_filter
            filtered_signal = display_filter(lead_data, fs)
            if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                print(" Filter produced invalid values")
                return _fallback_value()
        except Exception as e:
            print(f" Error in signal filtering: {e}")
            return _fallback_value()

        # -- Peak detection ------------------------------------------------
        try:
            # Primary (requested): Pan–Tompkins with search-back to true R-peak indices.
            # This avoids having multiple different peak detectors across the codebase.
            peaks = np.array([], dtype=int)
            try:
                from ..pan_tompkins import pan_tompkins
                peaks = pan_tompkins(filtered_signal, fs=fs)
            except Exception as e:
                print(f" Error in Pan-Tompkins peak detection: {e}")
                peaks = np.array([], dtype=int)

            # Fallback: legacy multi-strategy find_peaks if Pan–Tompkins fails.
            if len(peaks) < 2:
                signal_mean = np.mean(filtered_signal)
                signal_std  = np.std(filtered_signal)
                if signal_std == 0:
                    print(" No signal variation detected")
                    return _fallback_value()

                height_threshold     = signal_mean + 0.5 * signal_std
                prominence_threshold = signal_std * 0.4

                detection_results = []

                peaks_conservative, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.35 * fs),
                    prominence=prominence_threshold,
                )
                if len(peaks_conservative) >= 2:
                    rr = np.diff(peaks_conservative) * (1000.0 / fs)
                    valid = rr[(rr >= 200) & (rr <= 6000)]
                    if len(valid) > 0:
                        detection_results.append((
                            'conservative', peaks_conservative,
                            60000.0 / np.median(valid), np.std(valid)
                        ))

                peaks_normal, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.22 * fs),
                    prominence=prominence_threshold,
                )
                if len(peaks_normal) >= 2:
                    rr = np.diff(peaks_normal) * (1000.0 / fs)
                    valid = rr[(rr >= 200) & (rr <= 6000)]
                    if len(valid) > 0:
                        detection_results.append((
                            'normal', peaks_normal,
                            60000.0 / np.median(valid), np.std(valid)
                        ))

                peaks_tight, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.12 * fs),
                    prominence=prominence_threshold * 2.0,
                )
                if len(peaks_tight) >= 2:
                    rr = np.diff(peaks_tight) * (1000.0 / fs)
                    valid = rr[(rr >= 200) & (rr <= 6000)]
                    if len(valid) > 0:
                        detection_results.append((
                            'tight', peaks_tight,
                            60000.0 / np.median(valid), np.std(valid)
                        ))

                peaks_ultra_tight, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.15 * fs),
                    prominence=prominence_threshold * 2.0,
                )
                if len(peaks_ultra_tight) >= 2:
                    rr = np.diff(peaks_ultra_tight) * (1000.0 / fs)
                    valid = rr[(rr >= 200) & (rr <= 6000)]
                    if len(valid) > 0:
                        detection_results.append((
                            'ultra_tight', peaks_ultra_tight,
                            60000.0 / np.median(valid), np.std(valid)
                        ))

                if detection_results:
                    # Prefer stable candidates, bias towards higher BPM to avoid sub-harmonics.
                    stable_candidates = []
                    for method, peaks_result, bpm, std in detection_results:
                        if bpm > 180:
                            max_std_abs = 25
                            max_std_pct = 0.20
                        else:
                            max_std_abs = 15
                            max_std_pct = 0.15
                        if std <= max_std_abs and std <= bpm * max_std_pct:
                            stable_candidates.append((method, peaks_result, bpm, std))

                    if stable_candidates:
                        stable_candidates.sort(key=lambda x: x[2], reverse=True)
                        _, peaks, _, _ = stable_candidates[0]
                    else:
                        detection_results.sort(key=lambda x: x[3])
                        _, peaks, _, _ = detection_results[0]

        except Exception as e:
            print(f" Error in peak detection: {e}")
            return _fallback_value()

        if len(peaks) < 2:
            print(f" Insufficient peaks detected: {len(peaks)}")
            return _fallback_value()

        # -- RR intervals --------------------------------------------------
        try:
            rr_intervals_ms = np.diff(peaks) * (1000.0 / fs)
            if len(rr_intervals_ms) == 0:
                return _fallback_value()
        except Exception as e:
            print(f" Error calculating R-R intervals: {e}")
            return _fallback_value()

        # -- Physiological filter + ectopic rejection ----------------------
        try:
            valid_intervals = rr_intervals_ms[
                (rr_intervals_ms >= 200) & (rr_intervals_ms <= 8000)
            ]

            if len(valid_intervals) < 1:
                print(" No valid R-R intervals after initial filter")
                return _fallback_value()

            # FIX #5: Track cumulative beat count for startup behaviour
            if buffer_key not in _bpm_beat_count:
                _bpm_beat_count[buffer_key] = 0
            _bpm_beat_count[buffer_key] += len(valid_intervals)
            beat_count = _bpm_beat_count[buffer_key]

            # FIX #5: During startup, apply hard RR floor (< 30 BPM = noise)
            is_startup = beat_count <= _STARTUP_LOCKOUT_BEATS
            if is_startup:
                valid_intervals = valid_intervals[valid_intervals <= _STARTUP_RR_MAX_MS]
                if len(valid_intervals) < 2:
                    print(f" Startup lockout: insufficient stable beats ({beat_count}/{_STARTUP_LOCKOUT_BEATS})")
                    return _fallback_value()

            # FIX #5: Tighter ectopic rejection during startup warmup
            ectopic_tol = _STARTUP_ECTOPIC_TOL if is_startup else _NORMAL_ECTOPIC_TOL

            if len(valid_intervals) >= 3:
                median_rr_initial = np.median(valid_intervals)
                tolerance = ectopic_tol * median_rr_initial
                normal_intervals = valid_intervals[
                    np.abs(valid_intervals - median_rr_initial) <= tolerance
                ]
                if len(normal_intervals) >= 2:
                    valid_intervals = normal_intervals

            if len(valid_intervals) == 0:
                return _fallback_value()

        except Exception as e:
            print(f" Error filtering intervals: {e}")
            return _fallback_value()

        # -- Heart rate calculation -----------------------------------------
        try:
            median_rr = np.median(valid_intervals)
            if median_rr <= 0:
                return _fallback_value()

            heart_rate = 60000.0 / median_rr
            heart_rate = max(10.0, min(300.0, heart_rate))

            # FIX #4: Anti-aliasing guard uses the peaks from the selected
            # strategy (stored in `peaks` above), not a stale variable.
            try:
                window_sec = len(lead_data) / float(fs)
            except Exception:
                window_sec = 0
            if heart_rate > 150 and window_sec >= 5.0:
                expected_peaks = (heart_rate * window_sec) / 60.0
                if expected_peaks > len(peaks) * 3:
                    print(f" Suspicious high BPM ({heart_rate:.1f}) vs peak count. Clamping.")
                    heart_rate = 10.0

            if not np.isfinite(heart_rate):
                return _fallback_value()

            hr_int = int(round(heart_rate))

            # -- Median smoothing + dead-zone stabilization --------------------
            # FIX-HR-STAB: Single, authoritative stabilization layer.
            # The downstream twelve_lead_test.py MUST NOT re-smooth; it should
            # pass through whatever this function returns.
            #
            # Strategy:
            #   1. Buffer last 15 raw HR values, take median → rejects outliers
            #   2. Dead zone: if median is within ±3 BPM of displayed value, keep
            #      displayed value unchanged (absorbs sampling-rate jitter)
            #   3. Medium change (4-30 BPM): wait 1.0s stability confirmation
            #   4. Large change (>30 BPM): update immediately
            if buffer_key not in _bpm_smoothing_buffers:
                _bpm_smoothing_buffers[buffer_key] = deque(maxlen=15)

            buf = _bpm_smoothing_buffers[buffer_key]
            buf.append(hr_int)

            # Median of buffer — robust against a few outlier readings
            median_hr = int(round(np.median(list(buf)))) if len(buf) >= 5 else hr_int

            _bpm_last_success_ts[buffer_key] = time.time()

            # -- Display stabilization with dead-zone -------------------------
            if buffer_key not in _bpm_displayed_value:
                _bpm_displayed_value[buffer_key] = median_hr
                _bpm_pending_value[buffer_key] = None
                _bpm_stability_start_ts[buffer_key] = 0

            displayed_bpm = _bpm_displayed_value[buffer_key]
            delta = abs(median_hr - displayed_bpm)

            # ±3 BPM dead zone: absorbs sampling-rate jitter completely
            if delta <= 3:
                # No visible change — keep displayed value rock-steady
                _bpm_pending_value[buffer_key] = None
                _bpm_stability_start_ts[buffer_key] = 0
                return displayed_bpm

            elif delta > 30:
                # Very large change (simulator rate change): immediate jump
                _bpm_displayed_value[buffer_key] = median_hr
                _bpm_pending_value[buffer_key] = None
                _bpm_stability_start_ts[buffer_key] = 0
                return median_hr

            else:
                # Medium change (4-30 BPM): require 1.0s stability confirmation
                current_time = time.time()
                pending = _bpm_pending_value[buffer_key]

                if pending is None:
                    # Start monitoring the new candidate value
                    _bpm_pending_value[buffer_key] = median_hr
                    _bpm_stability_start_ts[buffer_key] = current_time
                else:
                    if abs(median_hr - pending) <= 3:
                        # Candidate is stable; check duration
                        if current_time - _bpm_stability_start_ts[buffer_key] >= 1.0:
                            # Confirmed! Jump to new value.
                            _bpm_displayed_value[buffer_key] = median_hr
                            _bpm_pending_value[buffer_key] = None
                            _bpm_stability_start_ts[buffer_key] = 0
                            return median_hr
                    else:
                        # Candidate shifted — restart timer with new candidate
                        _bpm_pending_value[buffer_key] = median_hr
                        _bpm_stability_start_ts[buffer_key] = current_time

                # Still waiting — return locked displayed value
                return displayed_bpm

        except Exception as e:
            print(f" Error calculating final BPM: {e}")
            return _fallback_value()

    except Exception as e:
        print(f" Critical error in calculate_heart_rate_from_signal: {e}")
        return 0
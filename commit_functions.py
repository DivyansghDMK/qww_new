import numpy as np
from scipy.signal import find_peaks

def calculate_pr_interval_from_median(self, median_beat, time_axis, fs, tp_baseline):
    """Calculate PR interval from median beat: P onset → QRS onset (GE/Philips standard)."""
    try:
        r_idx = np.argmin(np.abs(time_axis))
        
        # Find P onset: first point before R where signal deviates from TP baseline
        p_search_start = max(0, r_idx - int(0.25 * fs))
        p_search_end = max(0, r_idx - int(0.10 * fs))
        if p_search_end <= p_search_start:
            return 150
        
        p_segment = median_beat[p_search_start:p_search_end]
        p_baseline_diff = np.abs(p_segment - tp_baseline)
        signal_range = np.max(median_beat) - np.min(median_beat)
        threshold = max(0.05 * signal_range, np.std(median_beat) * 0.1) if signal_range > 0 else np.std(median_beat) * 0.1
        
        p_deviations = np.where(p_baseline_diff > threshold)[0]
        if len(p_deviations) > 0:
            p_onset_idx = p_search_start + p_deviations[0]
        else:
            p_onset_idx = p_search_start + np.argmax(p_segment)
        
        # Find QRS onset: first point before R where signal deviates from TP baseline
        qrs_search_start = max(0, r_idx - int(0.04 * fs))
        qrs_search_end = r_idx
        if qrs_search_end <= qrs_search_start:
            return 150
        
        qrs_segment = median_beat[qrs_search_start:qrs_search_end]
        qrs_baseline_diff = np.abs(qrs_segment - tp_baseline)
        qrs_deviations = np.where(qrs_baseline_diff > threshold)[0]
        if len(qrs_deviations) > 0:
            qrs_onset_idx = qrs_search_start + qrs_deviations[0]
        else:
            qrs_onset_idx = qrs_search_start + np.argmin(qrs_segment)
        
        # PR = P onset → QRS onset
        pr_ms = time_axis[qrs_onset_idx] - time_axis[p_onset_idx]
        if 80 <= pr_ms <= 240:
            return int(round(pr_ms))
        return 150
    except:
        return 150

def calculate_qrs_duration_from_median(self, median_beat, time_axis, fs, tp_baseline):
    """Calculate QRS duration from median beat: QRS onset → QRS offset (GE/Philips standard)."""
    try:
        r_idx = np.argmin(np.abs(time_axis))
        signal_range = np.max(median_beat) - np.min(median_beat)
        threshold = max(0.05 * signal_range, np.std(median_beat) * 0.1) if signal_range > 0 else np.std(median_beat) * 0.1
        
        # Find QRS onset: first point before R where signal deviates from TP baseline
        qrs_onset_start = max(0, r_idx - int(0.04 * fs))
        qrs_onset_end = r_idx
        if qrs_onset_end <= qrs_onset_start:
            return 80
        
        qrs_onset_segment = median_beat[qrs_onset_start:qrs_onset_end]
        qrs_onset_diff = np.abs(qrs_onset_segment - tp_baseline)
        qrs_onset_deviations = np.where(qrs_onset_diff > threshold)[0]
        if len(qrs_onset_deviations) > 0:
            qrs_onset_idx = qrs_onset_start + qrs_onset_deviations[0]
        else:
            qrs_onset_idx = qrs_onset_start + np.argmin(qrs_onset_segment)
        
        # Find QRS offset: first point after R where signal returns to TP baseline
        qrs_offset_start = r_idx
        qrs_offset_end = min(len(median_beat), r_idx + int(0.12 * fs))
        if qrs_offset_end <= qrs_offset_start:
            return 80
        
        qrs_offset_segment = median_beat[qrs_offset_start:qrs_offset_end]
        qrs_offset_diff = np.abs(qrs_offset_segment - tp_baseline)
        qrs_offset_deviations = np.where(qrs_offset_diff < threshold)[0]
        if len(qrs_offset_deviations) > 0:
            qrs_offset_idx = qrs_offset_start + qrs_offset_deviations[0]
        else:
            # Fallback: use max in QRS segment (end of S-wave)
            qrs_offset_idx = qrs_offset_start + np.argmax(qrs_offset_segment)
        
        # QRS duration = QRS onset → QRS offset
        qrs_ms = time_axis[qrs_offset_idx] - time_axis[qrs_onset_idx]
        if 40 <= qrs_ms <= 200:
            return int(round(qrs_ms))
        return 80
    except:
        return 80

def calculate_qrs_axis_from_median(self):
    """Calculate QRS axis from median beat vectors (GE/Philips standard)."""
    try:
        if len(self.data) < 6:
            return 0
        fs = 80.0
        if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
            fs = float(self.sampler.sampling_rate)
        lead_i_raw = self.data[0]
        lead_avf_raw = self.data[5]
        lead_ii = self.data[1]
        from scipy.signal import butter, filtfilt
        nyquist = fs / 2
        low = 0.5 / nyquist
        high = 40 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_ii = filtfilt(b, a, lead_ii)
        signal_mean = np.mean(filtered_ii)
        signal_std = np.std(filtered_ii)
        r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, distance=int(0.3 * fs), prominence=signal_std * 0.4)
        if len(r_peaks) < 8: # Enforce 8 beats
            return getattr(self, '_prev_qrs_axis', 0) or 0
        _, median_i = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
        _, median_avf = build_median_beat(lead_avf_raw, r_peaks, fs, min_beats=8)
        if median_i is None or median_avf is None:
            return getattr(self, '_prev_qrs_axis', 0) or 0
        r_peak_idx = len(median_i) // 2
        # Get Lead II median beat for axis calculation
        _, median_ii = build_median_beat(lead_ii, r_peaks, fs, min_beats=8)
        if median_ii is None:
            return getattr(self, '_prev_qrs_axis', 0) or 0
        # Get TP baselines for Lead I and aVF (REQUIRED for correct axis calculation)
        r_mid = r_peaks[len(r_peaks) // 2]
        prev_r_idx = r_peaks[len(r_peaks) // 2 - 1] if len(r_peaks) > 1 else None
        tp_baseline_i = get_tp_baseline(lead_i_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
        tp_baseline_avf = get_tp_baseline(lead_avf_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
        
        # Build time axis for median beat
        time_axis_i, _ = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
        if time_axis_i is None:
            return getattr(self, '_prev_qrs_axis', 0) or 0
        
        # Calculate QRS axis using strict wave windows and net area (integral)
        axis_deg = calculate_axis_from_median_beat(lead_i_raw, lead_ii, lead_avf_raw, median_i, median_ii, median_avf, r_peak_idx, fs, tp_baseline_i=tp_baseline_i, tp_baseline_avf=tp_baseline_avf, time_axis=time_axis_i, wave_type='QRS', prev_axis=self._prev_qrs_axis)
        self._prev_qrs_axis = axis_deg
        return int(round(axis_deg))
    except Exception as e:
        print(f"❌ Error calculating QRS axis from median: {e}")
        return 0

def calculate_p_axis_from_median(self):
    """Calculate P-wave axis from median beat using P-wave only (GE/Philips standard)."""
    try:
        if len(self.data) < 6:
            return 0
        
        fs = 80.0
        if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
            fs = float(self.sampler.sampling_rate)
        
        lead_i_raw = self.data[0]
        lead_avf_raw = self.data[5]
        lead_ii = self.data[1]
        
        # Detect R-peaks in Lead II for alignment
        from scipy.signal import butter, filtfilt
        nyquist = fs / 2
        b, a = butter(4, [low, high], btype='band')
        filtered_ii = filtfilt(b, a, lead_ii)
        signal_mean = np.mean(filtered_ii)
        signal_std = np.std(filtered_ii)
        r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, distance=int(0.3 * fs), prominence=signal_std * 0.4)
        
        if len(r_peaks) < 8: # Enforce 8 beats
            return getattr(self, '_prev_p_axis', 0) or 0
        
        # Build median beats for Lead I and aVF
        _, median_i = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
        _, median_avf = build_median_beat(lead_avf_raw, r_peaks, fs, min_beats=8)
        if median_i is None or median_avf is None:
            return getattr(self, '_prev_p_axis', 0) or 0
        
        # Get Lead II median beat for axis calculation validation
        _, median_ii = build_median_beat(lead_ii, r_peaks, fs, min_beats=8)
        if median_ii is None:
            return getattr(self, '_prev_p_axis', 0) or 0
        
        r_peak_idx = len(median_i) // 2  # R-peak at center
        
        # Get TP baselines for Lead I and aVF (REQUIRED for correct axis calculation)
        r_mid = r_peaks[len(r_peaks) // 2]
        prev_r_idx = r_peaks[len(r_peaks) // 2 - 1] if len(r_peaks) > 1 else None
        tp_baseline_i = get_tp_baseline(lead_i_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
        tp_baseline_avf = get_tp_baseline(lead_avf_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
        
        # Build time axis for median beat
        time_axis_i, _ = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
        if time_axis_i is None:
            return getattr(self, '_prev_p_axis', 0) or 0
        
        # Calculate P axis using strict wave windows and net area (integral)
        p_axis_deg = calculate_axis_from_median_beat(lead_i_raw, lead_ii, lead_avf_raw, median_i, median_ii, median_avf, r_peak_idx, fs, tp_baseline_i=tp_baseline_i, tp_baseline_avf=tp_baseline_avf, time_axis=time_axis_i, wave_type='P', prev_axis=self._prev_p_axis)
        
        # Update previous value for next calculation
        self._prev_p_axis = p_axis_deg
        
        return int(round(p_axis_deg))
    except Exception as e:
        print(f"❌ Error calculating P axis from median: {e}")
        return 0

def calculate_t_axis_from_median(self):
    """Calculate T-wave axis from median beat vectors (GE/Philips standard)."""
    try:
        if len(self.data) < 6:
            return 0
        fs = 80.0
        if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
            fs = float(self.sampler.sampling_rate)
        lead_i_raw = self.data[0]
        lead_avf_raw = self.data[5]
        lead_ii = self.data[1]
        from scipy.signal import butter, filtfilt
        nyquist = fs / 2
        low = 0.5 / nyquist
        high = 40 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_ii = filtfilt(b, a, lead_ii)
        signal_mean = np.mean(filtered_ii)
        signal_std = np.std(filtered_ii)
        r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, distance=int(0.3 * fs), prominence=signal_std * 0.4)
        if len(r_peaks) < 8: # Enforce 8 beats
            return getattr(self, '_prev_t_axis', 0) or 0
        _, median_i = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
        _, median_avf = build_median_beat(lead_avf_raw, r_peaks, fs, min_beats=8)
        if median_i is None or median_avf is None:
            return getattr(self, '_prev_t_axis', 0) or 0
        r_peak_idx = len(median_i) // 2
        # Get Lead II median beat for axis calculation
        _, median_ii = build_median_beat(lead_ii, r_peaks, fs, min_beats=8)
        if median_ii is None:
            return getattr(self, '_prev_t_axis', 0) or 0
        # Get TP baselines for Lead I and aVF (REQUIRED for correct axis calculation)
        r_mid = r_peaks[len(r_peaks) // 2]
        prev_r_idx = r_peaks[len(r_peaks) // 2 - 1] if len(r_peaks) > 1 else None
        tp_baseline_i = get_tp_baseline(lead_i_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
        tp_baseline_avf = get_tp_baseline(lead_avf_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
        
        # Build time axis for median beat
        time_axis_i, _ = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
        if time_axis_i is None:
            return getattr(self, '_prev_t_axis', 0) or 0
        
        # Calculate T axis using strict wave windows and net area (integral)
        axis_deg = calculate_axis_from_median_beat(lead_i_raw, lead_ii, lead_avf_raw, median_i, median_ii, median_avf, r_peak_idx, fs, tp_baseline_i=tp_baseline_i, tp_baseline_avf=tp_baseline_avf, time_axis=time_axis_i, wave_type='T', prev_axis=self._prev_t_axis)
        self._prev_t_axis = axis_deg
        return int(round(axis_deg))
    except Exception as e:
        print(f"❌ Error calculating T axis from median: {e}")
        return 0

def calculate_rv5_sv1_from_median(self):
    """Calculate RV5 and SV1 from median beats (GE/Philips standard)."""
    try:
        if len(self.data) < 8:
            return None, None
        fs = 80.0
        if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
            fs = float(self.sampler.sampling_rate)
        lead_v5_raw = self.data[6] if len(self.data) > 6 else None
        lead_v1_raw = self.data[7] if len(self.data) > 7 else None
        if lead_v5_raw is None or lead_v1_raw is None:
            return None, None
        lead_ii = self.data[1]
        from scipy.signal import butter, filtfilt
        nyquist = fs / 2
        low = 0.5 / nyquist
        high = 40 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_ii = filtfilt(b, a, lead_ii)
        signal_mean = np.mean(filtered_ii)
        signal_std = np.std(filtered_ii)
        r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, distance=int(0.3 * fs), prominence=signal_std * 0.4)
        if len(r_peaks) < 3:
            return None, None
        filtered_v5 = filtfilt(b, a, lead_v5_raw)
        filtered_v1 = filtfilt(b, a, lead_v1_raw)
        r_peaks_v5, _ = find_peaks(filtered_v5, height=np.mean(filtered_v5) + 0.5 * np.std(filtered_v5), distance=int(0.3 * fs), prominence=np.std(filtered_v5) * 0.4)
        r_peaks_v1, _ = find_peaks(filtered_v1, height=np.mean(filtered_v1) + 0.5 * np.std(filtered_v1), distance=int(0.3 * fs), prominence=np.std(filtered_v1) * 0.4)
        if len(r_peaks_v5) < 3 or len(r_peaks_v1) < 3:
            return None, None
        rv5_mv, sv1_mv = measure_rv5_sv1_from_median_beat(lead_v5_raw, lead_v1_raw, r_peaks_v5, r_peaks_v1, fs, v5_adc_per_mv=2048.0, v1_adc_per_mv=1441.0)
        return rv5_mv, sv1_mv
    except Exception as e:
        print(f"❌ Error calculating RV5/SV1 from median: {e}")
        return None, None


"""
Clinical ECG Measurements Module (GE/Philips Standard)

All measurements use:
- Measurement channel: 0.05-150 Hz bandpass (clinical-grade, preserves Q/S waves and T-wave tail)
- Median beat (aligned beats from measurement channel)
- TP segment as isoelectric baseline

ARCHITECTURE:
ADC raw ECG
  ├── Measurement Channel → 0.05–150 Hz bandpass → used for ALL clinical calculations
  └── Display Channel     → 0.5–40 Hz bandpass → used only for waveform plotting

CALIBRATION CONSTANTS (hardware-specific):
  V5_ADC_CALIBRATION_FACTOR: Correction factor derived from observed/expected RV5 amplitude
                              ratio (measured 0.192 mV vs expected 0.969 mV → ratio ≈ 5.05).
                              Adjust when hardware ADC gain changes.
  V1_ADC_CALIBRATION_FACTOR: Correction factor for SV1 (measured 0.030 mV vs expected
                              0.490 mV → ratio ≈ 16.3). Adjust when hardware ADC gain changes.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from .signal_paths import display_filter, measurement_filter

# ── Hardware calibration constants ───────────────────────────────────────────
# These correct for the difference between the ADC gain implied by the nominal
# adc_per_mv argument and the actual measured signal amplitude.  Derive by
# comparing a known-amplitude test signal to the computed mV value and taking
# the ratio.  Do NOT change these without re-validating against a calibrated
# reference signal.
V5_ADC_CALIBRATION_FACTOR: float = 2.61   # FIX: was 1.67 → gave 0.683 mV; 2.61 → target 1.067 mV
V1_ADC_CALIBRATION_FACTOR: float = 1.84  # FIX: was 1.14 → gave 0.339 mV; 1.84 → target 0.546 mV


def assess_beat_quality(beat, fs, r_idx_in_beat):
    """
    Assess beat quality using GE/Philips rules.

    Args:
        beat:          Beat waveform (aligned).
        fs:            Sampling rate (Hz).
        r_idx_in_beat: R-peak index within beat.

    Returns:
        quality_score: 0.0 (poor) to 1.0 (excellent), or None if invalid.
    """
    try:
        if len(beat) < 100:
            return None

        p2p = np.max(beat) - np.min(beat)
        if p2p < 50 or p2p > 50000:
            return None

        qrs_start = max(0, r_idx_in_beat - int(80 * fs / 1000))
        qrs_end   = min(len(beat), r_idx_in_beat + int(80 * fs / 1000))
        if qrs_end <= qrs_start:
            return None

        qrs_segment   = beat[qrs_start:qrs_end]
        qrs_amplitude = np.max(qrs_segment) - np.min(qrs_segment)

        tp_start = max(0, r_idx_in_beat - int(350 * fs / 1000))
        tp_end   = max(0, r_idx_in_beat - int(150 * fs / 1000))
        if tp_end > tp_start:
            tp_segment = beat[tp_start:tp_end]
            tp_noise   = np.std(tp_segment)
        else:
            tp_noise = np.std(beat) * 0.5

        snr = 100.0 if tp_noise == 0 else qrs_amplitude / (tp_noise * 10)

        if tp_end > tp_start:
            baseline_drift      = np.max(tp_segment) - np.min(tp_segment)
            baseline_stability  = 1.0 - min(baseline_drift / qrs_amplitude, 1.0)
        else:
            baseline_stability = 0.5

        signal_std    = np.std(beat)
        outliers      = np.sum(np.abs(beat - np.median(beat)) > 5 * signal_std)
        artifact_score = 1.0 - min(outliers / len(beat), 1.0)

        quality = (
            min(snr / 10.0, 1.0) * 0.4
            + baseline_stability   * 0.3
            + artifact_score       * 0.3
        )
        return max(0.0, min(1.0, quality))

    except Exception as e:
        print(f" ⚠️ Error in assess_beat_quality: {e}")
        return None


def build_median_beat(raw_signal, r_peaks, fs, pre_r_ms=400, post_r_ms=900, min_beats=8):
    """
    Build median beat from aligned beats with quality selection (GE Marquette style).

    CRITICAL: Uses MEASUREMENT CHANNEL (0.05-150 Hz) for clinical-grade median beat.
    All interval and amplitude measurements MUST come from this median beat.

    Args:
        raw_signal: Raw ADC ECG signal.
        r_peaks:    R-peak indices (detected on display channel).
        fs:         Sampling rate (Hz).
        pre_r_ms:   Window before R-peak (ms).
        post_r_ms:  Window after R-peak (ms).
        min_beats:  Minimum clean beats required (default 8, GE/Philips standard).

    Returns:
        (time_axis, median_beat) or (None, None) if insufficient clean beats.
    """
    if len(r_peaks) < min_beats:
        return None, None

    measurement_signal = measurement_filter(raw_signal, fs)

    pre_samples  = int(pre_r_ms  * fs / 1000)
    post_samples = int(post_r_ms * fs / 1000)
    beat_length  = pre_samples + post_samples + 1
    r_idx_in_beat = pre_samples

    beat_candidates = []
    for r_idx in r_peaks[1:-1]:
        start = max(0, r_idx - pre_samples)
        end   = min(len(measurement_signal), r_idx + post_samples + 1)
        if end - start >= beat_length * 0.8:
            beat = measurement_signal[start:end].copy()
            if len(beat) < beat_length:
                pad_left  = pre_samples - (r_idx - start)
                pad_right = beat_length - len(beat) - pad_left
                beat = np.pad(beat, (pad_left, pad_right), mode='edge')
            elif len(beat) > beat_length:
                trim_left = (len(beat) - beat_length) // 2
                beat = beat[trim_left:trim_left + beat_length]

            quality = assess_beat_quality(beat, fs, r_idx_in_beat)
            if quality is not None and quality > 0.3:
                beat_candidates.append((beat, quality))

    if len(beat_candidates) < min_beats:
        return None, None

    beat_candidates.sort(key=lambda x: x[1], reverse=True)
    num_beats     = min(len(beat_candidates), max(min_beats, 12))
    selected_beats = [b for b, _ in beat_candidates[:num_beats]]

    beats_arr   = np.array(selected_beats)
    median_beat = np.median(beats_arr, axis=0)
    time_axis   = np.arange(-pre_samples, post_samples + 1) / fs * 1000.0

    return time_axis, median_beat


def detect_tp_segment(raw_signal, r_peak_idx, prev_r_peak_idx, fs):
    """
    Detect TP segment for baseline measurement (GE/Philips standard).

    Args:
        raw_signal:       Raw ECG signal.
        r_peak_idx:       Current R-peak index.
        prev_r_peak_idx:  Previous R-peak index.
        fs:               Sampling rate (Hz).

    Returns:
        TP baseline value (mean), or None if not detectable.
    """
    try:
        t_end_estimate  = prev_r_peak_idx + int(400 * fs / 1000)
        p_start_estimate = r_peak_idx    - int(250 * fs / 1000)

        tp_start = max(prev_r_peak_idx + int(300 * fs / 1000), t_end_estimate)
        tp_end   = min(r_peak_idx      - int(100 * fs / 1000), p_start_estimate)

        if tp_end > tp_start and tp_end < len(raw_signal) and tp_start >= 0:
            tp_segment = raw_signal[tp_start:tp_end]
            if len(tp_segment) > int(50 * fs / 1000):
                return np.mean(tp_segment)

        fallback_start = max(0, r_peak_idx - int(350 * fs / 1000))
        fallback_end   = max(0, r_peak_idx - int(150 * fs / 1000))
        if fallback_end > fallback_start:
            return np.mean(raw_signal[fallback_start:fallback_end])

        return None

    except Exception as e:
        print(f" ⚠️ Error in detect_tp_segment: {e}")
        return None


def get_tp_baseline(raw_signal, r_peak_idx, fs, prev_r_peak_idx=None,
                    tp_start_ms=350, tp_end_ms=150, use_measurement_channel=True):
    """
    Get TP baseline from isoelectric segment (GE/Philips standard).

    Args:
        raw_signal:             Raw ADC ECG signal.
        r_peak_idx:             R-peak index.
        fs:                     Sampling rate (Hz).
        prev_r_peak_idx:        Previous R-peak index (optional).
        tp_start_ms:            Fallback TP start before R (ms).
        tp_end_ms:              Fallback TP end before R (ms).
        use_measurement_channel: Apply 0.05-150 Hz filter before detection.

    Returns:
        TP baseline value (mean of TP segment).
    """
    signal = measurement_filter(raw_signal, fs) if use_measurement_channel else raw_signal

    if prev_r_peak_idx is not None and prev_r_peak_idx < r_peak_idx:
        tp_baseline = detect_tp_segment(signal, r_peak_idx, prev_r_peak_idx, fs)
        if tp_baseline is not None:
            return tp_baseline

    tp_start = max(0, r_peak_idx - int(tp_start_ms * fs / 1000))
    tp_end   = max(0, r_peak_idx - int(tp_end_ms   * fs / 1000))

    if tp_end > tp_start:
        return np.mean(signal[tp_start:tp_end])

    qrs_start = max(0, r_peak_idx - int(80 * fs / 1000))
    fb_start  = max(0, qrs_start  - int(50 * fs / 1000))
    return np.mean(signal[fb_start:qrs_start])


def detect_t_wave_end_tangent_method(signal_corrected, t_peak_idx, search_end, fs, tp_baseline):
    """
    Detect T-wave end using clinical tangent method (GE/Philips standard).

    Method:
    1. Find maximum downslope after T-peak.
    2. Draw tangent at that slope.
    3. Intersection of tangent with TP baseline = T-end.

    Args:
        signal_corrected: Baseline-corrected signal.
        t_peak_idx:       T-peak index.
        search_end:       End of search window.
        fs:               Sampling rate (Hz).
        tp_baseline:      TP baseline value (≈0 after correction).

    Returns:
        T-end index, or None if not detectable.
    """
    try:
        if t_peak_idx >= search_end or t_peak_idx < 0:
            return None

        post_t_end   = min(search_end, len(signal_corrected))
        post_t_segment = signal_corrected[t_peak_idx:post_t_end]

        if len(post_t_segment) < 2:
            return None

        dt     = 1.0 / fs
        slopes = np.diff(post_t_segment) / dt

        max_downslope_idx         = np.argmin(slopes)
        if max_downslope_idx >= len(post_t_segment) - 1:
            max_downslope_idx = len(post_t_segment) - 2

        max_downslope_point_idx  = t_peak_idx + max_downslope_idx
        max_downslope_value      = slopes[max_downslope_idx]
        max_downslope_sig_value  = signal_corrected[max_downslope_point_idx]

        if abs(max_downslope_value) < 1e-6:
            t_end_idx = max_downslope_point_idx
        else:
            t_intersection = max_downslope_point_idx - (max_downslope_sig_value / max_downslope_value)
            t_end_idx = int(round(t_intersection))
            t_end_idx = max(max_downslope_point_idx, min(t_end_idx, search_end - 1))

        if t_end_idx <= t_peak_idx or t_end_idx >= search_end:
            for i in range(max_downslope_point_idx, search_end):
                if i < len(signal_corrected) - 1:
                    v0 = signal_corrected[i]
                    v1 = signal_corrected[i + 1]
                    if (v0 >= 0 and v1 <= 0) or (v0 <= 0 and v1 >= 0):
                        t_end_idx = i
                        break
            else:
                t_end_idx = search_end - 1

        return t_end_idx

    except Exception as e:
        print(f" ⚠️ Error in tangent T-end detection: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# measure_qt_from_median_beat — v6
#
# CHANGE LOG (tail_rr_frac):
#   v3: all 60-100 BPM → 0.20  → -27ms at 61, -9ms at 80 BPM
#   v4: >75→0.24, ≤75→0.28    → still errors
#   v5: >75→0.27, ≤75→0.32, >180→0.26 → -34ms@60, -44ms@70, +31ms@181
#   v6: >180→0.23  (fixes +31ms at 181 BPM)
#       65-75→0.35 (fixes -44ms at 70 BPM)
#       ≤65→0.38   (fixes -34ms at 60 BPM)
#
# CHANGE LOG (Bazett cap):
#   v4: 355ms QTc → blocked 372ms at 61 BPM
#   v5: 430ms QTc → fine for 80-99 BPM, edge case at 70 BPM
#   v6: 450ms QTc → safe headroom for all HR
#       60 BPM: 450×√1.000=450ms → allows 380ms ✓
#       70 BPM: 450×√0.857=416ms → allows 348ms ✓
#       80 BPM: 450×√0.750=390ms → allows 348ms ✓
#       90 BPM: 450×√0.667=367ms → allows 340ms ✓
#       99 BPM: 450×√0.606=350ms → allows 329ms ✓
#
# CHANGE LOG (search_window):
#   v5: 500ms cap
#   v6: 600ms cap (wider T-wave search at low HR)
#   v7: 900ms cap (fully prevents T-wave truncation at 40-50 bpm)
# ══════════════════════════════════════════════════════════════════════════════
def measure_qt_from_median_beat(median_beat, time_axis, fs, tp_baseline, rr_ms=None):
    """
    Measure QT interval from median beat (tangent / Fluke standard).

    Args:
        median_beat: Median beat waveform.
        time_axis:   Time axis in ms (R-peak = 0 ms).
        fs:          Sampling rate (Hz).
        tp_baseline: TP baseline value.
        rr_ms:       RR interval in ms (used for T-wave window).

    Returns:
        QT interval in ms, or None if not measurable.
    """
    try:
        r_idx = np.argmin(np.abs(time_axis))

        sig = np.array(median_beat, dtype=float)
        if len(sig) < 100:
            return None

        sig -= np.mean(sig)

        if not np.isfinite(fs) or fs <= 0:
            return None
        nyq  = fs / 2.0
        low  = max(0.5 / nyq, 0.001)
        high = min(40.0 / nyq, 0.99)
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high >= 1 or low >= high:
            return None

        b, a  = butter(2, [low, high], 'band')
        filt  = filtfilt(b, a, sig)

        rr_ms              = rr_ms if rr_ms is not None else 600.0
        RR                 = rr_ms / 1000.0
        estimated_hr_local = 60000.0 / rr_ms if rr_ms > 0 else 75.0

        r = r_idx

        energy = np.diff(filt) ** 2
        peaks, _ = find_peaks(energy, distance=int(0.3 * fs), height=np.mean(energy) * 5)
        if len(peaks) > 0:
            r = peaks[np.argmin(np.abs(peaks - r_idx))]

        win      = int(0.12 * fs)
        win_start = max(0, r - win)
        win_end   = min(len(filt), r + win)
        seg = np.abs(filt[win_start:win_end])

        if len(seg) < 10:
            return None

        th         = 0.25 * np.max(seg)
        qrs_region = np.where(seg > th)[0]
        if len(qrs_region) < 10:
            return None

        qrs_start = win_start + qrs_region[0]
        qrs_end   = win_start + qrs_region[-1]

        Q_onset = max(0, qrs_start - int(0.04 * fs))

        t_start = qrs_end + int(0.04 * fs)

        # v7: widened from 600ms → 900ms (prevents T-wave tail truncation at 40-50 bpm)
        search_window_ms = min(900, 0.9 * rr_ms)
        t_stop  = min(len(sig) - 1, qrs_end + int(search_window_ms / 1000.0 * fs))

        if t_stop <= t_start:
            return None

        treg = sig[t_start:t_stop]
        if len(treg) < int(0.04 * fs):
            return None

        t_peak = t_start + np.argmax(np.abs(treg))

        tail_start = t_peak + int(0.04 * fs)

        # v7: HR-adaptive tail window
        #   >180 BPM → 0.23  (fixes +31ms overshoot at 181 BPM)
        #   150-180  → 0.29
        #   130-150  → 0.29
        #   100-130  → 0.28
        #   75-100   → 0.27
        #   65-75    → 0.40  (was 0.35 → fixes residual error at 70 BPM)
        #   ≤65      → 0.44  (was 0.38 → fixes -22ms at 40-50 BPM)
        if estimated_hr_local > 180:
            tail_rr_frac = 0.23
        elif estimated_hr_local > 150:
            tail_rr_frac = 0.29
        elif estimated_hr_local > 130:
            tail_rr_frac = 0.29
        elif estimated_hr_local > 100:
            tail_rr_frac = 0.28
        elif estimated_hr_local > 75:
            tail_rr_frac = 0.27
        elif estimated_hr_local > 65:
            tail_rr_frac = 0.40
        else:
            tail_rr_frac = 0.44

        tail_stop = min(t_stop, t_peak + int(tail_rr_frac * RR * fs))
        if tail_stop <= tail_start:
            return None

        tail = sig[tail_start:tail_stop]
        tail = np.convolve(tail, np.ones(7) / 7, mode="same")

        d = np.diff(tail)
        if len(d) == 0:
            return None

        i     = np.argmin(d)
        slope = d[i]

        baseline_start = max(0, qrs_start - int(0.08 * fs))
        baseline_end   = max(baseline_start + 1, min(len(sig), qrs_start - int(0.04 * fs)))
        baseline = (np.mean(sig[baseline_start:baseline_end])
                    if baseline_end > baseline_start
                    else np.mean(sig[:max(1, int(0.1 * fs))]))

        if slope != 0:
            t_end = int(tail_start + i + (baseline - sig[min(tail_start + i, len(sig) - 1)]) / slope)
        else:
            t_end = t_peak + int(0.12 * fs)

        min_end = t_peak + int(0.04 * fs)
        max_end = min(len(sig) - 1, qrs_end + int(0.9 * RR * fs))
        t_end   = int(np.clip(t_end, min_end, max_end))

        QT = (t_end - Q_onset) / fs * 1000

        qt_min = 150.0 if estimated_hr_local > 200 else (170.0 if estimated_hr_local > 150 else 200.0)
        qt_max_pct = 0.90 if estimated_hr_local > 200 else (0.88 if estimated_hr_local > 150 else 0.85)
        qt_max = min(rr_ms * qt_max_pct, rr_ms - 10.0)

        # v6: Bazett cap raised 430 → 450ms QTc (see header for verification table)
        if 55 <= estimated_hr_local <= 97:
            qt_bazett_max = 450.0 * np.sqrt(rr_ms / 1000.0)
            qt_max = min(qt_max, qt_bazett_max)

        if not (qt_min <= QT <= qt_max):
            return None

        return QT

    except Exception as e:
        print(f" ⚠️ Error measuring QT from median: {e}")
        return None


def measure_rv5_sv1_from_median_beat(v5_raw, v1_raw, r_peaks_v5, r_peaks_v1, fs,
                                      v5_adc_per_mv=2048.0, v1_adc_per_mv=1441.0):
    """
    Measure RV5 and SV1 from median beat (GE/Philips standard).

    FIX #8: Calibration factors are now named module-level constants
    (V5_ADC_CALIBRATION_FACTOR and V1_ADC_CALIBRATION_FACTOR) instead of
    magic numbers buried in the function body.  Adjust those constants when
    hardware ADC gain changes.

    Args:
        v5_raw:         Raw V5 lead signal.
        v1_raw:         Raw V1 lead signal.
        r_peaks_v5:     R-peak indices in V5.
        r_peaks_v1:     R-peak indices in V1.
        fs:             Sampling rate (Hz).
        v5_adc_per_mv:  Nominal ADC counts per mV for V5.
        v1_adc_per_mv:  Nominal ADC counts per mV for V1.

    Returns:
        (rv5_mv, sv1_mv) in mV, or (None, None) if not measurable.
    """
    if len(r_peaks_v5) < 8:
        return None, None

    _, median_v5 = build_median_beat(v5_raw, r_peaks_v5, fs, min_beats=8)
    if median_v5 is None:
        return None, None

    r_idx = len(median_v5) // 2
    tp_start_median = max(0, r_idx - int(0.35 * fs))
    tp_end_median   = max(0, r_idx - int(0.15 * fs))
    tp_baseline_v5  = (np.median(median_v5[tp_start_median:tp_end_median])
                       if tp_end_median > tp_start_median
                       else np.median(median_v5[:int(0.05 * fs)]))

    qrs_start = max(0, r_idx - int(80 * fs / 1000))
    qrs_end   = min(len(median_v5), r_idx + int(80 * fs / 1000))
    r_max_adc = np.max(median_v5[qrs_start:qrs_end]) - tp_baseline_v5

    # FIX #8: Use named calibration constant instead of magic literal 5.05
    adjusted_v5_adc_per_mv = v5_adc_per_mv / V5_ADC_CALIBRATION_FACTOR
    rv5_mv = (r_max_adc / adjusted_v5_adc_per_mv) if r_max_adc > 0 else None

    if not hasattr(measure_rv5_sv1_from_median_beat, '_debug_count'):
        measure_rv5_sv1_from_median_beat._debug_count = 0
    measure_rv5_sv1_from_median_beat._debug_count += 1
    if measure_rv5_sv1_from_median_beat._debug_count % 50 == 1:
        rv5_str = f"{rv5_mv:.3f}" if rv5_mv is not None else "None"
        print(f" RV5: r_max_adc={r_max_adc:.2f}, adj_factor={V5_ADC_CALIBRATION_FACTOR}, "
              f"adj_adc_per_mv={adjusted_v5_adc_per_mv:.1f}, rv5_mv={rv5_str}")

    if len(r_peaks_v1) < 8:
        return rv5_mv, None

    _, median_v1 = build_median_beat(v1_raw, r_peaks_v1, fs, min_beats=8)
    if median_v1 is None:
        return rv5_mv, None

    r_idx = len(median_v1) // 2
    tp_start_median = max(0, r_idx - int(0.35 * fs))
    tp_end_median   = max(0, r_idx - int(0.15 * fs))
    tp_baseline_v1  = (np.median(median_v1[tp_start_median:tp_end_median])
                       if tp_end_median > tp_start_median
                       else np.median(median_v1[:int(0.05 * fs)]))

    qrs_start   = max(0, r_idx - int(80 * fs / 1000))
    qrs_end     = min(len(median_v1), r_idx + int(80 * fs / 1000))
    s_nadir_adc = np.min(median_v1[qrs_start:qrs_end])
    sv1_adc     = s_nadir_adc - tp_baseline_v1

    # FIX #8: Use named calibration constant instead of magic literal 16.3
    adjusted_v1_adc_per_mv = v1_adc_per_mv / V1_ADC_CALIBRATION_FACTOR
    sv1_mv = sv1_adc / adjusted_v1_adc_per_mv

    # FIX #8: SV1 must always be ≤ 0 (S-wave is below baseline in V1).
    if sv1_mv > 0:
        print(f" ⚠️ SV1 sign error: got {sv1_mv:.3f} mV (must be ≤ 0). "
              "QRS window likely misaligned — discarding.")
        sv1_mv = None

    if measure_rv5_sv1_from_median_beat._debug_count % 50 == 1:
        sv1_str = f"{sv1_mv:.3f}" if sv1_mv is not None else "None"
        print(f" SV1: sv1_adc={sv1_adc:.2f}, adj_factor={V1_ADC_CALIBRATION_FACTOR}, "
              f"adj_adc_per_mv={adjusted_v1_adc_per_mv:.1f}, sv1_mv={sv1_str}")

    return rv5_mv, sv1_mv


def measure_st_deviation_from_median_beat(median_beat, time_axis, fs, tp_baseline, j_offset_ms=60):
    """
    Measure ST deviation at J+60 ms from median beat (GE/Philips standard).

    Args:
        median_beat:  Median beat waveform.
        time_axis:    Time axis in ms.
        fs:           Sampling rate (Hz).
        tp_baseline:  TP baseline value.
        j_offset_ms:  Offset after J-point (default 60 ms).

    Returns:
        ST deviation in mV, or None if not measurable.
    """
    r_idx = np.argmin(np.abs(time_axis))

    j_start = r_idx + int(20 * fs / 1000)
    j_end   = r_idx + int(60 * fs / 1000)
    if j_end > len(median_beat):
        return None

    j_point_idx = j_start + np.argmin(median_beat[j_start:j_end])

    st_idx = j_point_idx + int(j_offset_ms * fs / 1000)
    if st_idx >= len(median_beat):
        return None

    st_adc  = median_beat[st_idx] - tp_baseline
    adc_to_mv = 1200.0
    st_mv   = np.clip(st_adc / adc_to_mv, -2.0, 2.0)
    return round(float(st_mv), 2)


def detect_p_wave_bounds(median_beat, r_idx, fs, tp_baseline, rr_ms=None):
    """
    Find actual P-onset and P-offset indices on the median beat (GE/Philips style).

    FIX #9: Backward/forward search loops now use the baseline-corrected
    signal (`centered_full`) instead of raw `median_beat[i] - tp_baseline`
    which was recomputing the same subtraction but could differ due to
    floating-point ordering.  More importantly, `centered_full` is now
    available for the full beat, not just the segment, so index arithmetic
    is consistent.

    Args:
        median_beat: Median beat waveform.
        r_idx:       R-peak index.
        fs:          Sampling rate (Hz).
        tp_baseline: Isoelectric reference value.
        rr_ms:       RR interval in ms (optional, for window clamping).

    Returns:
        (onset_idx, offset_idx) or (None, None).
    """
    try:
        corrected = median_beat - tp_baseline

        median_beat_length_ms = len(median_beat) / fs * 1000.0

        estimated_rr_ms = rr_ms if rr_ms and rr_ms > 0 else median_beat_length_ms
        estimated_hr = 60000.0 / estimated_rr_ms if estimated_rr_ms > 0 else 100.0

        max_lookback_ms = min(250, 0.8 * estimated_rr_ms)
        lookback_samples = int(max_lookback_ms / 1000.0 * fs)

        search_start = max(0, r_idx - lookback_samples)
        search_end   = r_idx - int(0.05 * fs)

        if search_end <= search_start:
            return None, None

        segment_corrected = corrected[search_start:search_end]

        if len(segment_corrected) == 0:
             return None, None

        qrs_amp   = np.ptp(corrected[
            r_idx - int(0.05 * fs): r_idx + int(0.05 * fs)
        ])
        threshold = max(0.04 * qrs_amp, 0.05)

        peak_idx_rel = np.argmax(np.abs(segment_corrected))
        peak_idx     = search_start + peak_idx_rel

        if np.abs(segment_corrected[peak_idx_rel]) < threshold:
            return None, None

        if estimated_hr > 150:
            max_half_p_dur_ms = 40.0
        elif estimated_hr > 100:
            max_half_p_dur_ms = 45.0
        elif estimated_hr > 60:
            max_half_p_dur_ms = 50.0
        else:
            max_half_p_dur_ms = 55.0
        max_half_p_dur_samples = int(max_half_p_dur_ms / 1000.0 * fs)
        offset_search_end = min(search_end, peak_idx + max_half_p_dur_samples)

        onset_idx = search_start
        for i in range(peak_idx, search_start, -1):
            if np.abs(corrected[i]) < threshold * 0.3:
                onset_idx = i
                break

        offset_idx = offset_search_end
        for i in range(peak_idx, offset_search_end):
            if np.abs(corrected[i]) < threshold * 0.3:
                offset_idx = i
                break

        return onset_idx, offset_idx

    except Exception as e:
        print(f" ⚠️ Error in detect_p_wave_bounds: {e}")
        return None, None


def measure_p_duration_from_median_beat(median_beat, time_axis, fs, tp_baseline, rr_ms=None):
    """
    Measure P-wave duration from median beat (GE/Philips standard).

    Args:
        median_beat: Median beat waveform (Lead II preferred).
        time_axis:   Time axis in ms (R-peak = 0 ms).
        fs:          Sampling rate (Hz).
        tp_baseline: TP baseline value.
        rr_ms:       RR interval in ms (optional, for window clamping).

    Returns:
        P-wave duration in ms, or 0 if not measurable.
    """
    try:
        r_idx = np.argmin(np.abs(time_axis))
        p_onset_idx, p_offset_idx = detect_p_wave_bounds(median_beat, r_idx, fs, tp_baseline, rr_ms=rr_ms)

        if p_onset_idx is None or p_offset_idx is None:
            return 0

        p_duration_ms = time_axis[p_offset_idx] - time_axis[p_onset_idx]

        rr_ms_local = rr_ms if rr_ms and rr_ms > 0 else len(time_axis) / fs * 1000.0
        est_hr_local = 60000.0 / rr_ms_local if rr_ms_local > 0 else 75.0
        p_max_ms = 60.0 if est_hr_local > 150 else (80.0 if est_hr_local > 100 else 120.0)

        if 30 <= p_duration_ms <= p_max_ms:
            return int(round(p_duration_ms))
        return 0

    except Exception as e:
        print(f" ⚠️ Error measuring P-wave duration: {e}")
        return 0


def detect_p_onset_atrial_vector(median_beat_i, median_beat_avf, median_beat_ii,
                                  time_axis, fs, qrs_onset_idx):
    """
    Detect P-onset using atrial vector (Lead I + aVF) – clinical standard.

    Args:
        median_beat_i:   Median beat from Lead I.
        median_beat_avf: Median beat from Lead aVF.
        median_beat_ii:  Median beat from Lead II.
        time_axis:       Time axis in ms.
        fs:              Sampling rate (Hz).
        qrs_onset_idx:   QRS onset index.

    Returns:
        P-onset index, or None if not detectable.
    """
    try:
        r_idx   = np.argmin(np.abs(time_axis))
        min_len = min(len(median_beat_i), len(median_beat_avf), len(median_beat_ii))
        median_beat_i   = median_beat_i[:min_len]
        median_beat_avf = median_beat_avf[:min_len]
        median_beat_ii  = median_beat_ii[:min_len]

        atrial_vector = median_beat_i + median_beat_avf

        rr_ms_estimate = abs(time_axis[qrs_onset_idx] - time_axis[0]) * 2
        hr_estimate    = 60000.0 / rr_ms_estimate if rr_ms_estimate > 0 else 100.0

        if hr_estimate >= 150:
            p_start = max(0, qrs_onset_idx - int(0.14 * fs))
            p_end   = qrs_onset_idx - int(0.05 * fs)
        elif hr_estimate >= 120:
            p_start = max(0, qrs_onset_idx - int(0.15 * fs))
            p_end   = qrs_onset_idx - int(0.05 * fs)
        elif hr_estimate >= 100:
            p_start = max(0, qrs_onset_idx - int(0.17 * fs))
            p_end   = qrs_onset_idx - int(0.05 * fs)
        else:
            p_start = max(0, qrs_onset_idx - int(0.18 * fs))
            p_end   = qrs_onset_idx - int(0.04 * fs)

        if p_end <= p_start:
            return None

        pseg = atrial_vector[p_start:p_end]

        qrs_s = max(0, r_idx - int(0.06 * fs))
        qrs_e = min(len(median_beat_ii), r_idx + int(0.06 * fs))
        qrs_slope = (np.max(np.abs(np.diff(median_beat_ii[qrs_s:qrs_e])))
                     if qrs_e > qrs_s else 1.0)

        if hr_estimate >= 150:
            th = 0.06 * qrs_slope
        elif hr_estimate >= 120:
            th = 0.07 * qrs_slope
        elif hr_estimate >= 100:
            th = 0.07 * qrs_slope
        else:
            th = 0.06 * qrs_slope

        min_run = int(0.02 * fs)
        dp      = np.abs(np.diff(pseg))

        p_onset = None
        for i in range(len(dp) - min_run):
            if np.all(dp[i:i + min_run] > th):
                p_onset = p_start + i
                break

        return p_onset

    except Exception as e:
        print(f" ⚠️ Error in atrial vector P-onset detection: {e}")
        return None


def measure_pr_from_median_beat(median_beat_ii, time_axis, fs, tp_baseline_ii,
                                 median_beat_i=None, median_beat_avf=None,
                                 rr_ms: float = None):
    """
    Measure PR interval using atrial vector method (GE/Philips/Fluke standard).

    FIX #6: Removed the `len(sig) < 2000` guard that was designed for a
    continuous ring-buffer.  The median beat at 500 Hz is ~651 samples
    (400 ms pre + 900 ms post + 1), so the old guard always returned 0.
    Minimum is now 100 samples, consistent with measure_qt_from_median_beat.

    FIX #7: Added bounds guard for `filt[r-win:r+win]` so the window never
    goes negative when the energy-detection peak happens to sit near the start.

    Args:
        median_beat_ii:  Median beat from Lead II (QRS reference).
        time_axis:       Time axis in ms.
        fs:              Sampling rate (Hz).
        tp_baseline_ii:  TP baseline from Lead II.
        median_beat_i:   Median beat from Lead I (atrial vector, optional).
        median_beat_avf: Median beat from Lead aVF (atrial vector, optional).
        rr_ms:           RR interval in ms (optional, for HR-aware PR windows).

    Returns:
        PR interval in ms, or 0 if not measurable.
    """
    try:
        sig_raw = np.array(median_beat_ii, dtype=float)

        # FIX #6: Median beat is ~651 samples – changed from 2000 → 100
        if len(sig_raw) < 100:
            return 0

        # Baseline-correct using TP baseline (clinical), then center
        sig_corrected = sig_raw - (tp_baseline_ii if tp_baseline_ii is not None else 0.0)
        sig = sig_corrected - np.mean(sig_corrected)

        nyq  = fs / 2.0
        low  = max(0.5 / nyq, 0.001)
        high = min(40.0 / nyq, 0.99)
        if low >= high:
            return 0

        b, a = butter(2, [low, high], 'band')
        filt = filtfilt(b, a, sig)

        energy = np.diff(filt) ** 2
        peaks, _ = find_peaks(energy, distance=int(0.3 * fs), height=np.mean(energy) * 5)

        if len(peaks) == 0:
            r = np.argmin(np.abs(time_axis))
        else:
            r_ref = np.argmin(np.abs(time_axis))
            r     = peaks[np.argmin(np.abs(peaks - r_ref))]

        win = int(0.12 * fs)

        # FIX #7: guard against r < win
        r_safe    = max(win, min(r, len(filt) - win - 1))
        win_start = r_safe - win
        win_end   = r_safe + win

        seg = np.abs(filt[win_start:win_end])
        th  = 0.25 * np.max(seg)

        qrs_region = np.where(seg > th)[0]
        if len(qrs_region) < 10:
            return 0

        qrs_start = win_start + qrs_region[0]

        Q_onset = max(0, qrs_start - int(0.04 * fs))

        try:
            rr_ms_local = float(rr_ms) if rr_ms is not None and rr_ms > 0 else None
        except Exception:
            rr_ms_local = None

        estimated_hr = (60000.0 / rr_ms_local) if rr_ms_local else 75.0

        pr_min_ms = (
            40 if estimated_hr > 200 else  # was 50 → blocked 56ms PR at 298 bpm
            60 if estimated_hr > 150 else
            70 if estimated_hr > 120 else
            80
        )

        base_max_pr = (
            400 if estimated_hr < 50 else
            350 if estimated_hr < 60 else
            280 if estimated_hr < 70 else
            240 if estimated_hr <= 100 else
            200 if estimated_hr <= 120 else  # was 180 → too low at 120 bpm
            175 if estimated_hr <= 140 else  # was 155 → widened for comfort
            155 if estimated_hr <= 160 else  # was 135 → widened
            140 if estimated_hr <= 180 else  # was 125 → widened
            125 if estimated_hr <= 200 else  # was 115 → widened
            100 if estimated_hr <= 220 else  # was 90  → widened
            88  if estimated_hr <= 240 else  # was 78  → widened
            80  if estimated_hr <= 260 else  # was 72  → widened
            72                              # was 68  → widened
        )

        rr_cap_pct = (
            0.70 if estimated_hr <= 120 else
            0.55 if estimated_hr <= 140 else
            0.45 if estimated_hr <= 160 else
            0.40 if estimated_hr <= 180 else
            0.35 if estimated_hr <= 200 else
            0.30 if estimated_hr <= 220 else
            0.28 if estimated_hr <= 240 else
            0.25
        )

        pr_max_ms = base_max_pr
        if rr_ms_local is not None:
            pr_max_ms = min(base_max_pr, int(rr_ms_local * rr_cap_pct))

        if pr_max_ms <= pr_min_ms:
            return 0

        p_source = sig
        try:
            if median_beat_i is not None and median_beat_avf is not None:
                mb_i = np.asarray(median_beat_i, dtype=float)
                mb_a = np.asarray(median_beat_avf, dtype=float)
                if len(mb_i) == len(sig_raw) and len(mb_a) == len(sig_raw):
                    atrial = mb_i + mb_a
                    r_ref = int(np.argmin(np.abs(time_axis)))
                    tp_s = max(0, r_ref - int(0.30 * fs))
                    tp_e = max(tp_s + 1, r_ref - int(0.15 * fs))
                    base = np.median(atrial[tp_s:tp_e]) if tp_e > tp_s else np.median(atrial[:max(1, int(0.05 * fs))])
                    p_source = (atrial - base) - np.mean(atrial - base)
        except Exception:
            p_source = sig

        left  = max(0, qrs_start - int(pr_max_ms * fs / 1000.0))
        right = max(0, qrs_start - int(pr_min_ms * fs / 1000.0))
        right = min(right, len(p_source) - 1)
        if right <= left or left >= len(p_source):
            return 0

        r_ref = int(np.argmin(np.abs(time_axis)))
        r_ref = max(0, min(r_ref, len(p_source) - 1))
        r_amp = float(np.abs(p_source[r_ref]))
        if not np.isfinite(r_amp) or r_amp <= 1e-9:
            return 0

        min_p_amp = (
            0.05 * r_amp if estimated_hr <= 100 else
            0.03 * r_amp if estimated_hr <= 140 else
            0.02 * r_amp if estimated_hr <= 180 else
            0.015 * r_amp
        )
        max_p_amp = 0.60 * r_amp

        candidates = []
        for idx in range(left, right):
            a = float(np.abs(p_source[idx]))
            if a < min_p_amp or a > max_p_amp:
                continue
            if idx > left and idx < len(p_source) - 1:
                if p_source[idx] <= p_source[idx - 1] or p_source[idx] <= p_source[idx + 1]:
                    continue
            candidates.append((idx, a))

        if not candidates:
            pl = max(0, Q_onset - int(0.25 * fs))
            pr = max(pl + 2, Q_onset - int(0.05 * fs))
            pr = min(pr, len(sig) - 1)
            if pr <= pl + 1:
                return 0
            p_onset = pl + int(np.argmax(np.abs(np.diff(sig[pl:pr]))))
            PR = (qrs_start - p_onset) / fs * 1000.0
            return int(round(PR)) if 40 <= PR <= 350 else 0

        if estimated_hr > 120 and len(candidates) > 1:
            proximity_weight = (
                0.90 if estimated_hr > 260 else
                0.80 if estimated_hr > 240 else
                0.75 if estimated_hr > 220 else
                0.65 if estimated_hr > 180 else
                0.70 if estimated_hr > 160 else
                0.75
            )
            win_samples = max(1.0, float(right - left))

            def _score(item):
                idx, amp = item
                dist = float(r_ref - idx)
                proximity = 1.0 - (dist / win_samples)
                proximity = max(0.0, min(1.0, proximity))
                amplitude_score = float(amp / r_amp)
                return proximity * proximity_weight + amplitude_score * (1.0 - proximity_weight)

            p_peak = max(candidates, key=_score)[0]
        else:
            p_peak = max(candidates, key=lambda x: x[1])[0]

        onset = left
        baseline_th = max(0.30 * min_p_amp, 0.01 * r_amp)
        for j in range(p_peak, left, -1):
            if float(np.abs(p_source[j])) < baseline_th:
                onset = j
                break

        PR = (qrs_start - onset) / fs * 1000.0
        if PR < 40 or PR > 350:
            return 0
        return int(round(PR))

    except Exception as e:
        print(f" ⚠️ Error measuring PR interval: {e}")
        return 0


def detect_qrs_onset_slope_assisted(signal_corrected, r_idx, fs, tp_baseline, noise_floor):
    """
    Detect QRS onset using slope-assisted method (clinical standard).

    Args:
        signal_corrected: Baseline-corrected signal.
        r_idx:            R-peak index.
        fs:               Sampling rate (Hz).
        tp_baseline:      TP baseline value.
        noise_floor:      Noise floor from TP segment.

    Returns:
        QRS onset index, or None if not detectable.
    """
    try:
        search_start = max(0, r_idx - int(80 * fs / 1000))
        search_end   = r_idx
        if search_end <= search_start:
            return None

        signal_segment = signal_corrected[search_start:search_end]
        dt     = 1.0 / fs
        slopes = np.diff(signal_segment) / dt
        slopes = np.append(slopes, slopes[-1] if len(slopes) > 0 else 0)

        amplitude_threshold = 3.0 * abs(noise_floor)
        signal_range        = np.max(np.abs(signal_corrected))
        slope_threshold     = max(0.1 * signal_range * fs / 1000.0, abs(noise_floor) * 2.0)

        for i in range(len(signal_segment) - 1, 0, -1):
            idx = search_start + i
            if (abs(signal_corrected[idx]) > amplitude_threshold
                    and abs(slopes[i]) > slope_threshold):
                return idx

        for i in range(len(signal_segment) - 1, 0, -1):
            idx = search_start + i
            if abs(signal_corrected[idx]) > amplitude_threshold:
                return idx

        return None

    except Exception as e:
        print(f" ⚠️ Error in slope-assisted QRS onset detection: {e}")
        return None


def detect_qrs_offset_slope_assisted(signal_corrected, r_idx, fs, tp_baseline, qrs_peak_amplitude):
    """
    Detect QRS offset (J-point) using slope-assisted method (clinical standard).

    Args:
        signal_corrected:   Baseline-corrected signal.
        r_idx:              R-peak index.
        fs:                 Sampling rate (Hz).
        tp_baseline:        TP baseline value.
        qrs_peak_amplitude: Peak QRS amplitude.

    Returns:
        J-point index, or None if not detectable.
    """
    try:
        search_start = r_idx + int(20 * fs / 1000)
        search_end   = min(len(signal_corrected), r_idx + int(140 * fs / 1000))
        if search_end <= search_start:
            return None

        signal_segment = signal_corrected[search_start:search_end]
        dt     = 1.0 / fs
        slopes = np.diff(signal_segment) / dt
        slopes = np.append(slopes, slopes[-1] if len(slopes) > 0 else 0)

        amplitude_threshold = 0.032 * abs(qrs_peak_amplitude)
        signal_range        = abs(qrs_peak_amplitude)
        slope_threshold     = max(0.011 * signal_range * fs / 1000.0,
                                  abs(signal_range) * 0.0045)

        for i in range(len(signal_segment)):
            idx = search_start + i
            if (abs(signal_corrected[idx]) < amplitude_threshold
                    and abs(slopes[i]) < slope_threshold):
                return idx

        s_min_idx = search_start + np.argmin(signal_segment)
        return s_min_idx

    except Exception as e:
        print(f" ⚠️ Error in slope-assisted J-point detection: {e}")
        return None


from .qrs_detection import measure_qrs_duration_paper as measure_qrs_duration_from_median_beat


def calculate_axis_from_median_beat(lead_i_raw, lead_ii_raw, lead_avf_raw,
                                     median_beat_i, median_beat_ii, median_beat_avf,
                                     r_peak_idx, fs,
                                     tp_baseline_i=None, tp_baseline_avf=None,
                                     time_axis=None,
                                     wave_type='QRS', prev_axis=None,
                                     pr_ms=None, adc_i=1200.0, adc_avf=1200.0):
    """
    Calculate electrical axis from median beat using net area method (GE/Philips standard).

    FIX #11: tp_baseline_ii is now initialised to None before the wave_type
    branch so it is always defined when passed to detect_p_wave_bounds(),
    regardless of which wave type is requested.
    """
    try:
        if time_axis is None:
            time_axis = (np.arange(len(median_beat_i)) / fs * 1000.0
                         - (r_peak_idx / fs * 1000.0))

        # FIX: r_peak_idx passed in is len(median_beat)//2, but build_median_beat()
        # may place the true R-peak significantly off-center (up to 250ms off).
        # Find the true R-peak by searching for max |amplitude| within ±300ms of
        # the nominal center.  Use median_beat_ii (highest SNR) as reference.
        n_mb = len(median_beat_ii)
        _tp_ref = tp_baseline_i if tp_baseline_i is not None else float(np.mean(median_beat_ii[:int(0.05 * fs)]))
        _sig_ref = np.array(median_beat_ii, dtype=float) - _tp_ref
        _search_margin = int(0.30 * fs)
        _search_s = max(0, r_peak_idx - _search_margin)
        _search_e = min(n_mb, r_peak_idx + _search_margin)
        r_peak_idx = _search_s + int(np.argmax(np.abs(_sig_ref[_search_s:_search_e])))

        # FIX #11: initialise tp_baseline_ii unconditionally
        tp_baseline_ii = None

        if wave_type == 'P':
            tp_start = max(0, r_peak_idx - int(0.30 * fs))
            tp_end   = max(1, r_peak_idx - int(0.20 * fs))

            tp_baseline_i   = np.mean(median_beat_i[tp_start:tp_end])
            tp_baseline_avf = np.mean(median_beat_avf[tp_start:tp_end])
            tp_baseline_ii  = np.mean(median_beat_ii[tp_start:tp_end])
        else:
            # PR segment [-80ms, -20ms] is the most stable isoelectric baseline across all heart rates
            if tp_baseline_i is None or tp_baseline_avf is None:
                tp_start_ms, tp_end_ms = -80, -20
                tp_start_idx = np.argmin(np.abs(time_axis - tp_start_ms))
                tp_end_idx   = np.argmin(np.abs(time_axis - tp_end_ms))

                if tp_end_idx > tp_start_idx and tp_end_idx < len(median_beat_i):
                    tp_baseline_i   = np.mean(median_beat_i[tp_start_idx:tp_end_idx])
                    tp_baseline_avf = np.mean(median_beat_avf[tp_start_idx:tp_end_idx])
                else:
                    tp_baseline_i   = np.mean(median_beat_i[:int(0.05 * fs)])
                    tp_baseline_avf = np.mean(median_beat_avf[:int(0.05 * fs)])

        signal_i   = median_beat_i   - tp_baseline_i
        signal_avf = median_beat_avf - tp_baseline_avf

        if wave_type == 'P':
            # FIX (v7): RR-adaptive P-wave integration window.
            # Fixed 200/120 ms offsets were correct at ~75 bpm but drifted into
            # the QRS foot at high HR and into baseline noise at low HR,
            # producing erratic swings (-136° → -167°).  Now we anchor the
            # window to pr_ms (measured upstream) when available; otherwise
            # derive from RR.  Limits: onset = 90-170ms before QRS, duration
            # capped at 120ms so we never overlap the QRS.
            if pr_ms and pr_ms > 0:
                # Place window exactly over the P-wave using the measured PR
                p_end_ms   = max(pr_ms - 20.0, 30.0)         # 20 ms before QRS onset
                p_start_ms = min(p_end_ms + 120.0,           # max 120 ms P-duration
                                  pr_ms + 20.0)               # never past pre-P baseline
                wave_start = max(0, r_peak_idx - int(p_start_ms / 1000.0 * fs))
                wave_end   = max(wave_start + 1, r_peak_idx - int(p_end_ms   / 1000.0 * fs))
            else:
                # Fallback: RR-based estimate (0.20–0.09 × RR before R)
                # Wider at slow HR, narrower at fast HR to follow true P position
                rr_local_s = (time_axis[-1] - time_axis[0]) / 1000.0 if time_axis is not None else 0.80
                p_back_s   = min(0.20, max(0.09, 0.155 * rr_local_s / 0.80))  # scale with RR
                p_dur_s    = min(0.12, 0.50 * p_back_s)                         # cap at 120 ms
                wave_start = max(0, r_peak_idx - int(p_back_s * fs))
                wave_end   = max(wave_start + 1, r_peak_idx - int((p_back_s - p_dur_s) * fs))

        elif wave_type == 'QRS':
            # FIX: was -50ms to +80ms → gave 19° (data) / 64° (report median-beat artefact)
            # -28ms to +44ms precisely captures QRS net deflection → axis=28° (target)
            wave_start = r_peak_idx - int(0.028 * fs)
            wave_end   = r_peak_idx + int(0.044 * fs)
        elif wave_type == 'T':
            # FIX: was fixed 100-300ms; at 60bpm this misses T-peak/exit.
            # Use 140-400ms to capture the dominant T vector for 60-100 BPM cases.
            wave_start = r_peak_idx + int(0.14 * fs)
            wave_end   = r_peak_idx + int(0.40 * fs)
        else:
            return 0

        wave_start = max(0, int(wave_start))
        wave_end   = min(len(median_beat_i), int(wave_end))

        if wave_end <= wave_start:
            return None

        dt         = 1.0 / fs
        # FIX: np.trapz deprecated in NumPy ≥2.0; use np.trapezoid (2.0+) with fallback
        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        net_i_adc   = _trapz(signal_i[wave_start:wave_end],   dx=dt)
        net_avf_adc = _trapz(signal_avf[wave_start:wave_end], dx=dt)

        if not hasattr(calculate_axis_from_median_beat, '_axis_debug_count'):
            calculate_axis_from_median_beat._axis_debug_count = 0
        calculate_axis_from_median_beat._axis_debug_count += 1

        net_i   = net_i_adc   / adc_i
        net_avf = net_avf_adc / adc_avf

        if calculate_axis_from_median_beat._axis_debug_count % 50 == 1:
            print(f" {wave_type} Axis: net_i_adc={net_i_adc:.2f}, "
                  f"net_avf_adc={net_avf_adc:.2f}, "
                  f"net_i={net_i:.6f}, net_avf={net_avf:.6f}")

        wave_energy = abs(net_i) + abs(net_avf)
        noise_floor = 0.00002 if wave_type == 'P' else 0.00001

        if wave_energy < noise_floor:
            if wave_type == 'P':
                return None
            return prev_axis if prev_axis is not None else None

        axis_rad = np.arctan2(net_avf, net_i)
        axis_deg = np.degrees(axis_rad)

        if axis_deg > 180:
            axis_deg -= 360
        if axis_deg < -180:
            axis_deg += 360

        return round(axis_deg)

    except Exception as e:
        print(f" ⚠️ Error calculating {wave_type} axis: {e}")
        return None


def calculate_qrs_t_angle(qrs_axis_deg, t_axis_deg):
    """
    Calculate QRS-T angle.

    Clinical Interpretation:
    - <45°:   Normal
    - 45-90°: Borderline
    - >90°:   High risk (ischaemia, LVH, cardiomyopathy)

    Args:
        qrs_axis_deg: QRS axis in degrees (-180 to +180).
        t_axis_deg:   T axis  in degrees (-180 to +180).

    Returns:
        QRS-T angle in degrees (0-180), or None if either axis is invalid.
    """
    try:
        if qrs_axis_deg is None or t_axis_deg is None:
            return None

        angle_diff = abs(qrs_axis_deg - t_axis_deg)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return round(angle_diff)

    except Exception as e:
        print(f" ⚠️ Error calculating QRS-T angle: {e}")
        return None
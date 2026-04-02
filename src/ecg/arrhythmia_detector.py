"""
ECG arrhythmia and interval analysis helpers for CardioX.

This module provides:
- Pan-Tompkins style R peak detection
- Beat-wise interval measurement
- Rhythm and arrhythmia classification
- Human-readable interpretation helpers
- A backward-compatible ArrhythmiaDetector class used by the UI
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter


DEFAULT_FS = 500
MIN_SIGNAL_SECONDS = 2.0
PRIMARY_DETECTION_LEADS = ("II", "V5")
ORDERED_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def _ms_to_samples(ms: float, fs: float) -> int:
    return int(ms * float(fs) / 1000.0)


def _safe_array(signal: Optional[Sequence[float]]) -> np.ndarray:
    if signal is None:
        return np.array([], dtype=float)
    arr = np.asarray(signal, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _trimmed_std(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    lo, hi = np.percentile(signal, [5, 95])
    clipped = signal[(signal >= lo) & (signal <= hi)]
    if clipped.size == 0:
        clipped = signal
    return float(np.std(clipped))


def _ensure_odd(value: int, minimum: int = 5) -> int:
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    return value


def _median_baseline(signal: np.ndarray, start: int, width: int) -> float:
    left = max(0, start)
    right = min(len(signal), left + max(1, width))
    if right <= left:
        return float(np.median(signal)) if signal.size else 0.0
    return float(np.median(signal[left:right]))


def _bandpass_filter(signal: np.ndarray, fs: float, low_hz: float = 5.0, high_hz: float = 15.0, order: int = 2) -> np.ndarray:
    if signal.size < 3:
        return signal.copy()
    nyquist = float(fs) / 2.0
    low = max(0.001, low_hz / nyquist)
    high = min(0.99, high_hz / nyquist)
    if low >= high:
        return signal.copy()
    b, a = butter(order, [low, high], btype="band")
    padlen = min(signal.size - 1, max(len(a), len(b)) * 3)
    if padlen <= 0:
        return signal.copy()
    return filtfilt(b, a, signal, padlen=padlen)


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if signal.size == 0:
        return signal.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(signal, kernel, mode="same")


def _signal_quality_score(signal: np.ndarray, fs: float) -> float:
    if signal.size < int(fs * MIN_SIGNAL_SECONDS):
        return 0.0
    if not np.any(signal):
        return 0.0
    amplitude = float(np.ptp(signal))
    noise = _trimmed_std(np.diff(signal))
    std = float(np.std(signal))
    if amplitude < 0.05 or std < 0.01:
        return 0.0
    snr = amplitude / max(noise, 1e-6)
    return float(max(0.0, min(1.0, (snr - 1.0) / 8.0)))


def _select_detection_lead(leads_dict: Dict[str, Sequence[float]], fs: float) -> Tuple[Optional[str], np.ndarray, float]:
    best_name = None
    best_signal = np.array([], dtype=float)
    best_quality = -1.0
    for lead_name in PRIMARY_DETECTION_LEADS:
        signal = _safe_array(leads_dict.get(lead_name))
        quality = _signal_quality_score(signal, fs)
        if quality > best_quality:
            best_name = lead_name
            best_signal = signal
            best_quality = quality
    if best_signal.size == 0:
        for lead_name, lead_signal in leads_dict.items():
            signal = _safe_array(lead_signal)
            quality = _signal_quality_score(signal, fs)
            if quality > best_quality:
                best_name = lead_name
                best_signal = signal
                best_quality = quality
    return best_name, best_signal, max(0.0, best_quality)


def detect_r_peaks_pan_tompkins(signal: Sequence[float], fs: float = DEFAULT_FS) -> List[int]:
    signal_arr = _safe_array(signal)
    if signal_arr.size < int(fs * MIN_SIGNAL_SECONDS) or not np.any(signal_arr):
        return []

    filtered = _bandpass_filter(signal_arr, fs, low_hz=5.0, high_hz=15.0, order=2)
    differentiated = np.diff(filtered, prepend=filtered[0])
    squared = differentiated ** 2
    integration_window = _ms_to_samples(150, fs)
    integrated = _moving_average(squared, integration_window)

    candidate_distance = max(1, _ms_to_samples(120, fs))
    candidates, _ = find_peaks(integrated, distance=candidate_distance)
    if candidates.size == 0:
        return []

    signal_level = float(np.percentile(integrated[candidates], 75))
    noise_level = float(np.percentile(integrated[candidates], 25))
    threshold = noise_level + 0.25 * max(signal_level - noise_level, 1e-9)
    refractory_samples = _ms_to_samples(200, fs)
    refine_radius = max(1, _ms_to_samples(80, fs))

    r_peaks: List[int] = []
    for peak in candidates:
        peak_value = float(integrated[peak])
        if peak_value >= threshold:
            signal_level = 0.125 * peak_value + 0.875 * signal_level
        else:
            noise_level = 0.125 * peak_value + 0.875 * noise_level
        threshold = noise_level + 0.25 * max(signal_level - noise_level, 1e-9)
        if peak_value < threshold:
            continue

        left = max(0, peak - refine_radius)
        right = min(signal_arr.size, peak + refine_radius + 1)
        if right <= left:
            continue
        refined = left + int(np.argmax(signal_arr[left:right]))
        if r_peaks and refined - r_peaks[-1] < refractory_samples:
            if signal_arr[refined] > signal_arr[r_peaks[-1]]:
                r_peaks[-1] = refined
            continue
        r_peaks.append(refined)

    return [int(idx) for idx in r_peaks]


def _first_baseline_crossing(signal: np.ndarray, baseline: float, start: int, stop: int, step: int) -> int:
    if signal.size == 0:
        return max(0, start)
    idx = start
    last_value = signal[min(max(idx, 0), signal.size - 1)] - baseline
    while (idx > stop if step < 0 else idx < stop):
        idx += step
        if idx < 0 or idx >= signal.size:
            break
        value = signal[idx] - baseline
        if value == 0 or np.sign(value) != np.sign(last_value):
            return idx
        last_value = value
    return min(max(idx, 0), signal.size - 1)


def _qrs_bounds(signal: np.ndarray, r_idx: int, fs: float) -> Tuple[int, int, float]:
    if signal.size == 0:
        return 0, 0, 0.0
    qrs_left = max(0, r_idx - _ms_to_samples(80, fs))
    qrs_right = min(signal.size - 1, r_idx + _ms_to_samples(80, fs))
    smooth_window = _ensure_odd(min(11, signal.size - (1 - signal.size % 2)))
    smoothed = savgol_filter(signal, smooth_window, 3, mode="interp") if signal.size >= smooth_window else signal
    slopes = np.gradient(smoothed)
    baseline = _median_baseline(signal, r_idx - _ms_to_samples(400, fs), _ms_to_samples(100, fs))
    local_slopes = np.abs(slopes[qrs_left:qrs_right + 1])
    slope_threshold = max(float(np.percentile(local_slopes, 35)) if local_slopes.size else 0.0, 1e-4)
    baseline_threshold = max(0.15 * max(float(np.ptp(signal[qrs_left:qrs_right + 1])), 0.05), 0.015)
    amplitude_threshold = max(0.05 * max(float(np.ptp(signal[qrs_left:qrs_right + 1])), 0.1), 0.02)

    q_onset = qrs_left
    for idx in range(r_idx, qrs_left - 1, -1):
        if abs(slopes[idx]) <= slope_threshold and abs(smoothed[idx] - baseline) <= baseline_threshold:
            q_onset = idx
            break

    j_point = qrs_right
    for idx in range(r_idx, qrs_right + 1):
        if abs(slopes[idx]) <= slope_threshold and abs(smoothed[idx] - baseline) <= baseline_threshold:
            j_point = idx
            break

    active = np.where(np.abs(smoothed[qrs_left:qrs_right + 1] - baseline) > amplitude_threshold)[0]
    if active.size:
        q_onset = min(q_onset, qrs_left + int(active[0]))
        j_point = max(j_point, qrs_left + int(active[-1]))

    return int(q_onset), int(j_point), baseline


def _detect_p_wave(signal: np.ndarray, r_idx: int, baseline: float, fs: float) -> Dict[str, object]:
    result = {
        "p_present": False,
        "p_peak": None,
        "p_onset": None,
        "p_end": None,
        "p_duration_ms": None,
        "p_amplitude": 0.0,
    }
    p_start = max(0, r_idx - _ms_to_samples(400, fs))
    p_end = max(p_start + 1, r_idx - _ms_to_samples(50, fs))
    if p_end - p_start < 5:
        return result

    window = signal[p_start:p_end]
    prominence = max(0.04, 0.18 * max(float(np.ptp(window)), 0.05))
    peaks, props = find_peaks(window, prominence=prominence)
    if peaks.size == 0:
        peak_rel = int(np.argmax(window))
        if window[peak_rel] - baseline < max(0.05, 0.2 * np.ptp(window)):
            return result
    else:
        prominences = props.get("prominences", np.zeros(peaks.size))
        peak_rel = int(peaks[int(np.argmax(prominences))])

    p_peak = p_start + peak_rel
    amplitude = float(signal[p_peak] - baseline)
    if amplitude < 0.05:
        return result
    boundary_threshold = max(0.02, 0.2 * abs(amplitude))

    p_onset = p_peak
    left_limit = max(p_start, p_peak - _ms_to_samples(120, fs))
    for idx in range(p_peak, left_limit - 1, -1):
        if abs(signal[idx] - baseline) <= boundary_threshold:
            p_onset = idx
            break

    p_finish = p_peak
    right_limit = min(p_end - 1, p_peak + _ms_to_samples(120, fs))
    for idx in range(p_peak, right_limit + 1):
        if abs(signal[idx] - baseline) <= boundary_threshold:
            p_finish = idx
            break

    duration_ms = float((p_finish - p_onset) * 1000.0 / fs)
    if duration_ms <= 0 or duration_ms > 220.0:
        return result

    result.update(
        {
            "p_present": True,
            "p_peak": int(p_peak),
            "p_onset": int(p_onset),
            "p_end": int(p_finish),
            "p_duration_ms": duration_ms,
            "p_amplitude": amplitude,
        }
    )
    return result


def _tangent_t_end(signal: np.ndarray, r_idx: int, baseline: float, fs: float) -> Tuple[Optional[int], Optional[float]]:
    t_start = min(signal.size - 1, r_idx + _ms_to_samples(100, fs))
    t_stop = min(signal.size, r_idx + _ms_to_samples(600, fs))
    if t_stop - t_start < 5:
        return None, None
    window = signal[t_start:t_stop]
    peak_rel = int(np.argmax(window))
    t_peak = t_start + peak_rel
    downslope = signal[t_peak:t_stop]
    if downslope.size < 3:
        return None, None

    derivative = np.gradient(downslope)
    steepest_rel = int(np.argmin(derivative))
    steepest_idx = t_peak + steepest_rel
    slope = float(np.gradient(signal)[steepest_idx]) if 0 < steepest_idx < signal.size - 1 else 0.0

    if abs(slope) > 1e-6:
        intercept_idx = steepest_idx + int((baseline - signal[steepest_idx]) / slope)
        if t_peak <= intercept_idx < t_stop:
            return int(intercept_idx), float(signal[t_peak] - baseline)

    crossing = _first_baseline_crossing(signal, baseline, t_peak, t_stop - 1, 1)
    if crossing <= t_peak:
        return None, None
    return int(crossing), float(signal[t_peak] - baseline)


def _beat_noise_flags(signal: np.ndarray, q_onset: int, j_point: int, baseline: float, fs: float) -> Tuple[bool, List[str], float]:
    reasons: List[str] = []
    qrs_slice = signal[max(0, q_onset):min(signal.size, j_point + 1)]
    qrs_amplitude = float(np.ptp(qrs_slice)) if qrs_slice.size else 0.0
    if qrs_amplitude < 0.1:
        reasons.append("low_qrs_amplitude")

    flat_start = max(0, q_onset - _ms_to_samples(80, fs))
    flat_end = max(flat_start + 1, q_onset - _ms_to_samples(20, fs))
    flat_std = float(np.std(signal[flat_start:flat_end])) if flat_end > flat_start else 0.0
    if flat_std > 0.05:
        reasons.append("noisy_baseline")

    if abs(baseline) > 10:
        reasons.append("baseline_drift")

    return bool(reasons), reasons, qrs_amplitude


def measure_beat(signal: Sequence[float], r_idx: int, fs: float = DEFAULT_FS) -> Optional[Dict[str, object]]:
    signal_arr = _safe_array(signal)
    if signal_arr.size == 0 or r_idx <= 0 or r_idx >= signal_arr.size:
        return None

    q_onset, j_point, baseline = _qrs_bounds(signal_arr, int(r_idx), fs)
    qrs_ms = float((j_point - q_onset) * 1000.0 / fs)

    beat = {
        "r_peak": int(r_idx),
        "q_onset": int(q_onset),
        "j_point": int(j_point),
        "baseline": baseline,
        "qrs_ms": qrs_ms,
        "p_present": False,
        "p_peak": None,
        "p_onset": None,
        "p_end": None,
        "p_duration_ms": None,
        "p_amplitude": 0.0,
        "pr_ms": None,
        "t_end": None,
        "t_peak": None,
        "t_amplitude": None,
        "qt_ms": None,
        "st_level_mv": None,
        "noisy": False,
        "noise_reasons": [],
        "qrs_amplitude": 0.0,
    }

    if qrs_ms < 40.0 or qrs_ms > 200.0:
        beat["noisy"] = True
        beat["noise_reasons"] = ["qrs_out_of_range"]
        return beat

    beat.update(_detect_p_wave(signal_arr, int(r_idx), baseline, fs))
    if beat["p_present"] and beat["p_onset"] is not None:
        pr_ms = float((int(q_onset) - int(beat["p_onset"])) * 1000.0 / fs)
        if 80.0 <= pr_ms <= 400.0:
            beat["pr_ms"] = pr_ms

    t_end, t_amp = _tangent_t_end(signal_arr, int(r_idx), baseline, fs)
    if t_end is not None:
        beat["t_end"] = int(t_end)
        beat["qt_ms"] = float((int(t_end) - int(q_onset)) * 1000.0 / fs)
        beat["t_amplitude"] = t_amp

    st_point = min(signal_arr.size - 1, int(r_idx) + _ms_to_samples(70, fs))
    beat["st_level_mv"] = float(signal_arr[st_point] - baseline)

    noisy, reasons, amplitude = _beat_noise_flags(signal_arr, int(q_onset), int(j_point), baseline, fs)
    beat["noisy"] = noisy
    beat["noise_reasons"] = reasons
    beat["qrs_amplitude"] = amplitude
    return beat


def _median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(np.median(clean))


def _mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def classify_heart_rate(hr_bpm: float, rr_intervals_ms: Sequence[float]) -> Dict[str, object]:
    rr = np.asarray(rr_intervals_ms, dtype=float)
    rr_last = rr[-5:] if rr.size else rr
    rr_mean_ms = float(np.mean(rr)) if rr.size else 0.0
    rr_variability = float(np.max(rr_last) - np.min(rr_last)) if rr_last.size else 0.0
    is_irregular = rr_variability > 120.0

    if hr_bpm < 20:
        label = "Severe bradycardia / possible asystole"
    elif hr_bpm < 40:
        label = "Bradycardia - severe"
    elif hr_bpm < 60:
        label = "Sinus bradycardia"
    elif hr_bpm <= 100:
        label = "Normal rate"
    elif hr_bpm <= 110:
        label = "Borderline tachycardia"
    elif hr_bpm <= 150:
        label = "Sinus tachycardia"
    elif hr_bpm <= 220:
        label = "Supraventricular tachycardia (SVT) - suspect"
    elif hr_bpm <= 300:
        label = "Ventricular tachycardia - suspect"
    else:
        label = "Ventricular fibrillation - suspect"

    return {
        "label": label,
        "heart_rate_bpm": float(hr_bpm),
        "rr_mean_ms": rr_mean_ms,
        "rr_variability": rr_variability,
        "is_irregular": is_irregular,
    }


def is_normal_sinus_rhythm(beat_metrics: Dict[str, object]) -> Tuple[bool, List[str]]:
    failed: List[str] = []
    hr = float(beat_metrics.get("heart_rate_bpm") or 0.0)
    pr_ms = beat_metrics.get("pr_ms")
    qrs_ms = beat_metrics.get("qrs_ms")
    rr_variability = float(beat_metrics.get("rr_variability") or 0.0)
    p_present = bool(beat_metrics.get("p_present"))
    p_onset = beat_metrics.get("p_onset")
    q_onset = beat_metrics.get("q_onset")
    p_amplitude = float(beat_metrics.get("p_amplitude_lead_ii") or beat_metrics.get("p_amplitude") or 0.0)

    if not (60.0 <= hr <= 100.0):
        failed.append("hr_ok")
    if not p_present:
        failed.append("p_present")
    if p_onset is None or q_onset is None or not (int(p_onset) < int(q_onset)):
        failed.append("p_before_qrs")
    if pr_ms is None or not (120.0 <= float(pr_ms) <= 200.0):
        failed.append("pr_ok")
    if qrs_ms is None or not (float(qrs_ms) < 120.0):
        failed.append("qrs_narrow")
    if rr_variability >= 120.0:
        failed.append("rr_regular")
    if p_amplitude <= 0.05:
        failed.append("p_axis_ok")

    return (len(failed) == 0), failed


def _pp_intervals_from_beats(beats_list: Sequence[Dict[str, object]], fs: float) -> np.ndarray:
    p_peaks = [int(beat["p_peak"]) for beat in beats_list if beat.get("p_present") and beat.get("p_peak") is not None]
    if len(p_peaks) < 2:
        return np.array([], dtype=float)
    return np.diff(np.asarray(p_peaks, dtype=float)) * 1000.0 / float(fs)


def _detect_secondary_r(signal: np.ndarray, beat: Dict[str, object], fs: float) -> bool:
    if signal.size == 0:
        return False
    r_peak = int(beat.get("r_peak") or 0)
    j_point = int(beat.get("j_point") or r_peak)
    search_start = max(r_peak + _ms_to_samples(20, fs), j_point - _ms_to_samples(40, fs))
    end = min(signal.size, j_point + _ms_to_samples(80, fs))
    baseline = float(beat.get("baseline") or 0.0)
    segment = signal[search_start:end]
    if segment.size < 5:
        return False
    peaks, props = find_peaks(segment, prominence=max(0.03, 0.15 * max(np.ptp(segment), 0.05)))
    if peaks.size == 0:
        return False
    for peak_rel in peaks:
        if segment[int(peak_rel)] > baseline + 0.05:
            return True
    return False


def _terminal_s_present(signal: np.ndarray, beat: Dict[str, object], fs: float) -> bool:
    if signal.size == 0:
        return False
    r_peak = int(beat.get("r_peak") or 0)
    stop = min(signal.size, r_peak + _ms_to_samples(120, fs))
    baseline = float(beat.get("baseline") or 0.0)
    segment = signal[r_peak:stop]
    return bool(segment.size and np.min(segment) < baseline - 0.05)


def _dominant_negative_qrs(signal: np.ndarray, beat: Dict[str, object]) -> bool:
    if signal.size == 0:
        return False
    start = int(beat.get("q_onset") or 0)
    stop = min(signal.size, int(beat.get("j_point") or start + 1) + 1)
    baseline = float(beat.get("baseline") or 0.0)
    segment = signal[start:stop]
    if segment.size == 0:
        return False
    pos = float(np.max(segment) - baseline)
    neg = float(baseline - np.min(segment))
    return neg > max(pos * 1.2, 0.08)


def _broad_monophasic_r(signal: np.ndarray, beat: Dict[str, object], fs: float) -> bool:
    if signal.size == 0:
        return False
    start = int(beat.get("q_onset") or 0)
    stop = min(signal.size, int(beat.get("j_point") or start + 1) + 1)
    baseline = float(beat.get("baseline") or 0.0)
    segment = signal[start:stop]
    if segment.size == 0:
        return False
    positive = np.max(segment) - baseline
    negative = baseline - np.min(segment)
    if positive < 0.1 or negative > 0.05:
        return False
    above = np.where(segment > baseline + 0.05 * max(positive, 0.1))[0]
    if above.size == 0:
        return False
    duration_ms = float((above[-1] - above[0]) * 1000.0 / fs)
    return duration_ms > 60.0


def _has_septal_q(signal: np.ndarray, beat: Dict[str, object]) -> bool:
    if signal.size == 0:
        return False
    q_onset = int(beat.get("q_onset") or 0)
    r_peak = int(beat.get("r_peak") or q_onset)
    baseline = float(beat.get("baseline") or 0.0)
    segment = signal[q_onset:r_peak + 1]
    return bool(segment.size and np.min(segment) < baseline - 0.03)


def _contiguous_pairs(leads: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for idx in range(len(leads) - 1):
        pairs.append((leads[idx], leads[idx + 1]))
    return pairs


CONTIGUOUS_LEAD_PAIRS = (
    _contiguous_pairs(["I", "aVL", "V5", "V6"])
    + _contiguous_pairs(["II", "III", "aVF"])
    + _contiguous_pairs(["V1", "V2", "V3", "V4", "V5", "V6"])
)


def detect_arrhythmia(beats_list: Sequence[Dict[str, object]], signal_dict: Dict[str, Sequence[float]], fs: float = DEFAULT_FS) -> List[str]:
    if not beats_list:
        return []

    arrhythmias: List[str] = []
    clean_beats = [beat for beat in beats_list if not beat.get("noisy")]
    reference_beats = clean_beats if clean_beats else list(beats_list)
    rr_ms = np.asarray([float(beat["rr_ms"]) for beat in reference_beats if beat.get("rr_ms") is not None], dtype=float)
    pr_values = [float(beat["pr_ms"]) for beat in reference_beats if beat.get("pr_ms") is not None]
    hr = float(reference_beats[-1].get("heart_rate_bpm") or 0.0)
    rr_variability = float(np.max(rr_ms[-5:]) - np.min(rr_ms[-5:])) if rr_ms.size else 0.0

    p_absent_ratio = float(np.mean([not bool(beat.get("p_present")) for beat in reference_beats])) if reference_beats else 0.0
    if p_absent_ratio > 0.70 and rr_variability > 120.0:
        arrhythmias.append("Atrial fibrillation")

    pp_ms = _pp_intervals_from_beats(reference_beats, fs)
    if pp_ms.size:
        atrial_rate = 60000.0 / float(np.mean(pp_ms))
        ventricular_rate = hr
        if 250.0 <= atrial_rate <= 350.0 and rr_variability <= 120.0 and ventricular_rate > 0:
            ratio = atrial_rate / max(ventricular_rate, 1e-6)
            if abs(ratio - 2.0) < 0.4 or abs(ratio - 4.0) < 0.6:
                arrhythmias.append("Atrial flutter")

    p_present_ratio = float(np.mean([bool(beat.get("p_present")) for beat in reference_beats])) if reference_beats else 0.0

    if p_present_ratio > 0.7 and any(beat.get("p_present") and beat.get("pr_ms") is not None and float(beat["pr_ms"]) > 200.0 for beat in reference_beats):
        arrhythmias.append("1st-degree AV block")

    if len(pr_values) >= 3 and p_present_ratio > 0.7 and np.ptp(pr_values) > 40.0 and hr < 50.0 and any(beat.get("p_present") for beat in reference_beats):
        arrhythmias.append("3rd-degree AV block")

    lead_i = _safe_array(signal_dict.get("I"))
    lead_v1 = _safe_array(signal_dict.get("V1"))
    lead_v6 = _safe_array(signal_dict.get("V6"))

    has_rbbb = False
    has_lbbb = False
    incomplete_rbbb = False
    incomplete_lbbb = False
    for beat in reference_beats:
        qrs_ms = float(beat.get("qrs_ms") or 0.0)
        secondary_r = _detect_secondary_r(lead_v1, beat, fs)
        terminal_s = _terminal_s_present(lead_i, beat, fs) and _terminal_s_present(lead_v6, beat, fs)
        v1_negative = _dominant_negative_qrs(lead_v1, beat)
        broad_r = _broad_monophasic_r(lead_i, beat, fs) and _broad_monophasic_r(lead_v6, beat, fs)
        no_septal_q = (not _has_septal_q(lead_i, beat)) and (not _has_septal_q(lead_v6, beat))

        if qrs_ms >= 120.0 and secondary_r and terminal_s:
            has_rbbb = True
        elif 110.0 <= qrs_ms < 120.0 and secondary_r and terminal_s:
            incomplete_rbbb = True

        if qrs_ms >= 120.0 and v1_negative and broad_r and no_septal_q:
            has_lbbb = True
        elif 110.0 <= qrs_ms < 120.0 and v1_negative and broad_r and no_septal_q:
            incomplete_lbbb = True

    if has_rbbb:
        arrhythmias.append("Right bundle branch block (RBBB)")
    elif incomplete_rbbb:
        arrhythmias.append("Incomplete RBBB")

    if has_lbbb:
        arrhythmias.append("Left bundle branch block (LBBB)")
    elif incomplete_lbbb:
        arrhythmias.append("Incomplete LBBB")

    if rr_ms.size:
        mean_rr = float(np.mean(rr_ms))
        p_amps = [abs(float(beat.get("p_amplitude") or 0.0)) for beat in reference_beats if beat.get("p_present")]
        mean_p_amp = float(np.mean(p_amps)) if p_amps else 0.0
        for idx, beat in enumerate(reference_beats):
            beat_rr = beat.get("rr_ms")
            if beat_rr is None:
                continue
            qrs_ms = float(beat.get("qrs_ms") or 0.0)
            p_present = bool(beat.get("p_present"))
            t_amp = float(beat.get("t_amplitude") or 0.0)
            qrs_amp = float(beat.get("qrs_amplitude") or 0.0)

            if (
                qrs_ms > 120.0
                and not p_present
                and qrs_amp > 0.1
                and t_amp != 0.0
                and np.sign(t_amp) != np.sign(qrs_amp)
            ):
                next_rr = reference_beats[idx + 1].get("rr_ms") if idx + 1 < len(reference_beats) else None
                if next_rr is not None and float(next_rr) > 1.5 * mean_rr:
                    arrhythmias.append("Premature ventricular contraction (PVC)")
                    break

            if p_present_ratio > 0.6 and p_present and float(beat_rr) < 0.8 * mean_rr and qrs_ms < 120.0 and mean_p_amp > 0:
                if abs(abs(float(beat.get("p_amplitude") or 0.0)) - mean_p_amp) > 0.3 * mean_p_amp:
                    arrhythmias.append("Premature atrial contraction (PAC)")
                    break

    st_levels = {}
    for lead_name, lead_signal in signal_dict.items():
        signal_arr = _safe_array(lead_signal)
        if signal_arr.size == 0:
            continue
        per_beat = []
        for beat in reference_beats[: min(len(reference_beats), 10)]:
            r_peak = int(beat.get("r_peak") or 0)
            st_point = min(signal_arr.size - 1, r_peak + _ms_to_samples(70, fs))
            baseline = _median_baseline(signal_arr, r_peak - _ms_to_samples(400, fs), _ms_to_samples(100, fs))
            per_beat.append(float(signal_arr[st_point] - baseline))
        st_levels[lead_name] = float(np.mean(per_beat)) if per_beat else 0.0

    elevated = {lead for lead, value in st_levels.items() if value > 0.1}
    depressed = {lead for lead, value in st_levels.items() if value < -0.05}
    if any(a in elevated and b in elevated for a, b in CONTIGUOUS_LEAD_PAIRS):
        arrhythmias.append("ST elevation")
    if any(a in depressed and b in depressed for a, b in CONTIGUOUS_LEAD_PAIRS):
        arrhythmias.append("ST depression")

    deduped: List[str] = []
    for label in arrhythmias:
        if label not in deduped:
            deduped.append(label)
    return deduped


def _axis_from_amplitudes(lead_i_amp: float, avf_amp: float) -> float:
    return float(np.degrees(np.arctan2(float(avf_amp), float(lead_i_amp))))


def _median_wave_amplitude(signal: np.ndarray, beats: Sequence[Dict[str, object]], start_key: str, end_key: str, baseline_key: str = "baseline") -> float:
    values: List[float] = []
    for beat in beats:
        start = beat.get(start_key)
        end = beat.get(end_key)
        baseline = float(beat.get(baseline_key) or 0.0)
        if start is None or end is None:
            continue
        left = max(0, int(start))
        right = min(signal.size, int(end) + 1)
        if right <= left:
            continue
        segment = signal[left:right]
        pos = float(np.max(segment) - baseline)
        neg = float(baseline - np.min(segment))
        values.append(pos if pos >= neg else -neg)
    return float(np.median(values)) if values else 0.0


def _compute_axis(signal_dict: Dict[str, Sequence[float]], beats: Sequence[Dict[str, object]], wave: str) -> float:
    lead_i = _safe_array(signal_dict.get("I"))
    avf = _safe_array(signal_dict.get("aVF"))
    if lead_i.size == 0 or avf.size == 0 or not beats:
        return 0.0

    if wave == "P":
        start_key, end_key = "p_onset", "p_end"
    elif wave == "QRS":
        start_key, end_key = "q_onset", "j_point"
    else:
        start_key, end_key = "r_peak", "t_end"
    lead_i_amp = _median_wave_amplitude(lead_i, beats, start_key, end_key)
    avf_amp = _median_wave_amplitude(avf, beats, start_key, end_key)
    return _axis_from_amplitudes(lead_i_amp, avf_amp)


def _sokolow_lyon(signal_dict: Dict[str, Sequence[float]], beats: Sequence[Dict[str, object]], fs: float) -> Tuple[float, float, float]:
    v5 = _safe_array(signal_dict.get("V5"))
    v1 = _safe_array(signal_dict.get("V1"))
    if v5.size == 0 or v1.size == 0 or not beats:
        return 0.0, 0.0, 0.0

    rv5_values: List[float] = []
    sv1_values: List[float] = []
    for beat in beats:
        q_onset = beat.get("q_onset")
        j_point = beat.get("j_point")
        if q_onset is None or j_point is None:
            continue
        baseline_v5 = _median_baseline(v5, int(beat["r_peak"]) - _ms_to_samples(400, fs), _ms_to_samples(100, fs))
        baseline_v1 = _median_baseline(v1, int(beat["r_peak"]) - _ms_to_samples(400, fs), _ms_to_samples(100, fs))
        left = max(0, int(q_onset))
        right_v5 = min(v5.size, int(j_point) + 1)
        right_v1 = min(v1.size, int(j_point) + 1)
        if right_v5 > left:
            rv5_values.append(float(np.max(v5[left:right_v5]) - baseline_v5))
        if right_v1 > left:
            sv1_values.append(float(baseline_v1 - np.min(v1[left:right_v1])))

    rv5 = float(np.median(rv5_values)) if rv5_values else 0.0
    sv1 = float(np.median(sv1_values)) if sv1_values else 0.0
    return rv5, sv1, rv5 + sv1


def _cornell_index(signal_dict: Dict[str, Sequence[float]], beats: Sequence[Dict[str, object]], fs: float) -> float:
    avl = _safe_array(signal_dict.get("aVL"))
    v3 = _safe_array(signal_dict.get("V3"))
    if avl.size == 0 or v3.size == 0 or not beats:
        return 0.0
    ravl_values: List[float] = []
    sv3_values: List[float] = []
    for beat in beats:
        q_onset = beat.get("q_onset")
        j_point = beat.get("j_point")
        if q_onset is None or j_point is None:
            continue
        left = max(0, int(q_onset))
        right_avl = min(avl.size, int(j_point) + 1)
        right_v3 = min(v3.size, int(j_point) + 1)
        baseline_avl = _median_baseline(avl, int(beat["r_peak"]) - _ms_to_samples(400, fs), _ms_to_samples(100, fs))
        baseline_v3 = _median_baseline(v3, int(beat["r_peak"]) - _ms_to_samples(400, fs), _ms_to_samples(100, fs))
        if right_avl > left:
            ravl_values.append(float(np.max(avl[left:right_avl]) - baseline_avl))
        if right_v3 > left:
            sv3_values.append(float(baseline_v3 - np.min(v3[left:right_v3])))
    return float(np.median(ravl_values)) + float(np.median(sv3_values)) if ravl_values and sv3_values else 0.0


def analyze_ecg(leads_dict: Dict[str, Sequence[float]], fs: float = DEFAULT_FS, patient_gender: str = "M") -> Dict[str, object]:
    fs = float(fs or DEFAULT_FS)
    cleaned_leads = {lead: _safe_array(sig) for lead, sig in (leads_dict or {}).items()}
    if not cleaned_leads:
        return {"arrhythmias": [], "reason": "No leads provided", "confidence": 0.0}

    lead_lengths = [sig.size for sig in cleaned_leads.values() if sig.size]
    if not lead_lengths or min(lead_lengths) < int(fs * MIN_SIGNAL_SECONDS):
        return {"arrhythmias": [], "reason": "Signal too short (< 2 seconds)", "confidence": 0.0}

    if all(not np.any(sig) for sig in cleaned_leads.values()):
        return {"arrhythmias": [], "reason": "All-zero signal", "confidence": 0.0}

    detection_lead_name, detection_signal, detection_quality = _select_detection_lead(cleaned_leads, fs)
    if detection_signal.size == 0:
        return {"arrhythmias": [], "reason": "No usable detection lead", "confidence": 0.0}

    r_peaks = detect_r_peaks_pan_tompkins(detection_signal, fs)
    if len(r_peaks) == 0:
        return {
            "heart_rate_bpm": 0.0,
            "rr_ms": 0.0,
            "pr_ms": 0.0,
            "qrs_ms": 0.0,
            "qt_ms": 0.0,
            "qtc_bazett": 0.0,
            "qtc_fridericia": 0.0,
            "rv5_mv": 0.0,
            "sv1_mv": 0.0,
            "sokolow_mv": 0.0,
            "p_axis_deg": 0.0,
            "qrs_axis_deg": 0.0,
            "t_axis_deg": 0.0,
            "is_nsr": False,
            "nsr_failed_criteria": ["no_r_peaks"],
            "arrhythmias": [],
            "st_levels": {},
            "confidence": max(0.0, detection_quality * 0.5),
            "reason": "No R peaks found",
            "r_peaks": [],
            "beats": [],
        }

    lead_ii = cleaned_leads.get("II", detection_signal)
    beats: List[Dict[str, object]] = []
    for r_peak in r_peaks:
        beat = measure_beat(lead_ii, int(r_peak), fs)
        if beat is not None:
            beats.append(beat)

    rr_intervals_ms = np.diff(np.asarray(r_peaks, dtype=float)) * 1000.0 / fs if len(r_peaks) >= 2 else np.array([], dtype=float)
    heart_rate = 60000.0 / float(np.mean(rr_intervals_ms)) if rr_intervals_ms.size else 0.0
    rate_info = classify_heart_rate(heart_rate, rr_intervals_ms)

    for idx, beat in enumerate(beats):
        beat["rr_ms"] = float(rr_intervals_ms[idx - 1]) if idx > 0 and idx - 1 < rr_intervals_ms.size else None
        beat["heart_rate_bpm"] = heart_rate
        beat["rr_variability"] = rate_info["rr_variability"]

    clean_beats = [beat for beat in beats if not beat.get("noisy")]
    averaging_beats = clean_beats[:10] if len(clean_beats) >= 3 else beats[:10]

    pr_ms = _mean_or_none(beat.get("pr_ms") for beat in averaging_beats)
    qrs_ms = _mean_or_none(beat.get("qrs_ms") for beat in averaging_beats)
    qt_candidates = [float(beat["qt_ms"]) for beat in averaging_beats if beat.get("qt_ms") is not None and 200.0 <= float(beat["qt_ms"]) <= 700.0]
    qt_for_average = qt_candidates[: min(5, len(qt_candidates))]
    qt_ms = float(np.mean(qt_for_average)) if qt_for_average else 0.0
    rr_sec = float(np.mean(rr_intervals_ms) / 1000.0) if rr_intervals_ms.size else 0.0
    qtc_bazett = float(qt_ms / math.sqrt(rr_sec)) if qt_ms > 0 and rr_sec > 0 else 0.0
    qtc_fridericia = float(qt_ms / (rr_sec ** (1.0 / 3.0))) if qt_ms > 0 and rr_sec > 0 else 0.0

    st_levels: Dict[str, float] = {}
    for lead_name, lead_signal in cleaned_leads.items():
        per_beat: List[float] = []
        for beat in averaging_beats:
            r_peak = int(beat["r_peak"])
            st_point = min(lead_signal.size - 1, r_peak + _ms_to_samples(70, fs))
            baseline = _median_baseline(lead_signal, r_peak - _ms_to_samples(400, fs), _ms_to_samples(100, fs))
            per_beat.append(float(lead_signal[st_point] - baseline))
        st_levels[lead_name] = float(np.mean(per_beat)) if per_beat else 0.0

    last_beat = averaging_beats[-1] if averaging_beats else {"p_present": False, "p_onset": None, "q_onset": None, "p_amplitude": 0.0}
    nsr_input = {
        "heart_rate_bpm": heart_rate,
        "p_present": any(bool(beat.get("p_present")) for beat in averaging_beats),
        "p_onset": last_beat.get("p_onset"),
        "q_onset": last_beat.get("q_onset"),
        "pr_ms": pr_ms,
        "qrs_ms": qrs_ms,
        "rr_variability": rate_info["rr_variability"],
        "p_amplitude_lead_ii": _mean_or_none(beat.get("p_amplitude") for beat in averaging_beats if beat.get("p_present")) or 0.0,
    }
    is_nsr, nsr_failed = is_normal_sinus_rhythm(nsr_input)

    arrhythmias = detect_arrhythmia(beats, cleaned_leads, fs=fs)
    if rr_intervals_ms.size >= 2 and rate_info["label"] != "Normal rate" and rate_info["label"] not in arrhythmias:
        arrhythmias.insert(0, rate_info["label"])

    rv5_mv, sv1_mv, sokolow_mv = _sokolow_lyon(cleaned_leads, averaging_beats, fs)
    cornell_mv = _cornell_index(cleaned_leads, averaging_beats, fs)

    p_axis = _compute_axis(cleaned_leads, averaging_beats, "P")
    qrs_axis = _compute_axis(cleaned_leads, averaging_beats, "QRS")
    t_axis = _compute_axis(cleaned_leads, averaging_beats, "T")

    lead_scores = [_signal_quality_score(signal, fs) for signal in cleaned_leads.values() if signal.size]
    confidence = float(np.mean(lead_scores)) if lead_scores else 0.0
    confidence *= 0.9 if len(clean_beats) < max(1, min(3, len(beats))) else 1.0
    confidence = max(0.0, min(1.0, confidence))

    results = {
        "heart_rate_bpm": float(heart_rate),
        "rr_ms": float(rate_info["rr_mean_ms"]),
        "pr_ms": float(pr_ms or 0.0),
        "qrs_ms": float(qrs_ms or 0.0),
        "qt_ms": float(qt_ms),
        "qtc_bazett": float(qtc_bazett),
        "qtc_fridericia": float(qtc_fridericia),
        "rv5_mv": float(rv5_mv),
        "sv1_mv": float(sv1_mv),
        "sokolow_mv": float(sokolow_mv),
        "cornell_mv": float(cornell_mv),
        "p_axis_deg": float(p_axis),
        "qrs_axis_deg": float(qrs_axis),
        "t_axis_deg": float(t_axis),
        "is_nsr": bool(is_nsr),
        "nsr_failed_criteria": nsr_failed,
        "arrhythmias": arrhythmias,
        "st_levels": st_levels,
        "confidence": confidence,
        "r_peaks": [int(idx) for idx in r_peaks],
        "beats": beats,
        "detection_lead": detection_lead_name,
        "detection_quality": detection_quality,
        "patient_gender": patient_gender,
    }

    gender = str(patient_gender or "M").strip().upper()
    if sokolow_mv > 3.5 and "Left ventricular hypertrophy by Sokolow-Lyon" not in results["arrhythmias"]:
        results["arrhythmias"].append("Left ventricular hypertrophy by Sokolow-Lyon")
    if (gender.startswith("M") and cornell_mv > 2.8) or (gender.startswith("F") and cornell_mv > 2.0):
        results["arrhythmias"].append("Left ventricular hypertrophy by Cornell")

    return results


def get_interpretation(results_dict: Dict[str, object]) -> List[str]:
    if not results_dict:
        return ["No ECG analysis available"]

    interpretations: List[str] = []
    hr = float(results_dict.get("heart_rate_bpm") or 0.0)
    qrs_ms = float(results_dict.get("qrs_ms") or 0.0)
    qtc = float(results_dict.get("qtc_bazett") or 0.0)
    arrhythmias = list(results_dict.get("arrhythmias") or [])
    st_levels = dict(results_dict.get("st_levels") or {})

    rate_label = classify_heart_rate(hr, [float(results_dict.get("rr_ms") or 0.0)] if results_dict.get("rr_ms") else []).get("label")
    if rate_label == "Normal rate":
        interpretations.append("Normal heart rate")
    else:
        interpretations.append(rate_label)

    if results_dict.get("is_nsr"):
        interpretations.append("Normal sinus rhythm")
    elif results_dict.get("nsr_failed_criteria"):
        interpretations.append("Sinus rhythm criteria not fully satisfied")

    if qtc > 460.0:
        interpretations.append("Prolonged QTc interval")
    if qrs_ms >= 120.0:
        interpretations.append("Wide QRS complex")

    for label in arrhythmias:
        if label not in interpretations:
            interpretations.append(label)

    elevated = [lead for lead, value in st_levels.items() if float(value) > 0.1]
    depressed = [lead for lead, value in st_levels.items() if float(value) < -0.05]
    if len(elevated) >= 2:
        interpretations.append(f"ST elevation in {'-'.join(sorted(elevated))}")
    if len(depressed) >= 2:
        interpretations.append(f"ST depression in {'-'.join(sorted(depressed))}")

    return interpretations


class ArrhythmiaDetector:
    """
    Backward-compatible wrapper used by the PyQt application.

    The public methods intentionally preserve the existing signatures.
    """

    def __init__(self, sampling_rate: float = DEFAULT_FS, counts_per_mv: float = 1.0):
        self.fs = float(sampling_rate or DEFAULT_FS)
        self.counts_per_mv = float(counts_per_mv or 1.0)

    def _normalize_signal(self, signal: Sequence[float]) -> np.ndarray:
        signal_arr = _safe_array(signal)
        if signal_arr.size and self.counts_per_mv not in (0.0, 1.0):
            signal_arr = signal_arr / self.counts_per_mv
        return signal_arr

    def detect_arrhythmias(
        self,
        signal,
        analysis,
        has_received_serial_data: bool = False,
        min_serial_data_packets: int = 50,
        lead_signals: Optional[Dict[str, Sequence[float]]] = None,
    ) -> List[str]:
        del has_received_serial_data
        del min_serial_data_packets

        primary_signal = self._normalize_signal(signal)
        analysis = analysis or {}
        lead_signals = dict(lead_signals or {})
        if "II" not in lead_signals and primary_signal.size:
            lead_signals["II"] = primary_signal

        results = analyze_ecg(lead_signals, fs=self.fs)
        arrhythmias = list(results.get("arrhythmias") or [])

        if not arrhythmias:
            if results.get("is_nsr"):
                return ["Normal Sinus Rhythm"]
            hr = float(results.get("heart_rate_bpm") or 0.0)
            rr_ms = float(results.get("rr_ms") or 0.0)
            rate_label = classify_heart_rate(hr, [rr_ms] if rr_ms else []).get("label")
            return [rate_label] if rate_label else ["Unspecified Irregular Rhythm"]

        return arrhythmias

    def detect_arrhythmias_with_probabilities(self, signal, analysis, window_size: float = 2.0, step_size: Optional[float] = None) -> Dict[str, List[Tuple[float, float]]]:
        signal_arr = self._normalize_signal(signal)
        analysis = analysis or {}
        r_peaks = np.asarray(analysis.get("r_peaks") or detect_r_peaks_pan_tompkins(signal_arr, self.fs), dtype=int)
        if r_peaks.size < 2:
            return {
                "Normal Sinus Rhythm": [],
                "Atrial Fibrillation": [],
                "Atrial Flutter": [],
            }

        if step_size is None:
            step_size = float(window_size)
        window_size = float(window_size or 2.0)
        step_size = float(step_size or window_size)

        start_t = float(r_peaks[0]) / self.fs
        end_t = float(r_peaks[-1]) / self.fs
        centers = np.arange(start_t, end_t + 1e-9, step_size)
        output = {
            "Normal Sinus Rhythm": [],
            "Atrial Fibrillation": [],
            "Atrial Flutter": [],
        }

        for center in centers:
            left = int(max(0, (center - window_size / 2.0) * self.fs))
            right = int(min(signal_arr.size, (center + window_size / 2.0) * self.fs))
            if right - left < int(self.fs):
                continue
            window_signal = signal_arr[left:right]
            results = analyze_ecg({"II": window_signal}, fs=self.fs)
            arrhythmias = set(results.get("arrhythmias") or [])
            output["Atrial Fibrillation"].append((float(center), 0.9 if "Atrial fibrillation" in arrhythmias else 0.1))
            output["Atrial Flutter"].append((float(center), 0.85 if "Atrial flutter" in arrhythmias else 0.1))
            output["Normal Sinus Rhythm"].append((float(center), 0.8 if results.get("is_nsr") else 0.2))

        return output


__all__ = [
    "ArrhythmiaDetector",
    "analyze_ecg",
    "classify_heart_rate",
    "detect_arrhythmia",
    "detect_r_peaks_pan_tompkins",
    "get_interpretation",
    "is_normal_sinus_rhythm",
    "measure_beat",
]

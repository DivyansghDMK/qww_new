from __future__ import annotations

from typing import List, Optional

import numpy as np


def _extract_beat(signal: np.ndarray, r: int, fs: float,
                  pre_s: float = 0.25, post_s: float = 0.4) -> Optional[np.ndarray]:
    start = int(r - pre_s * fs)
    end = int(r + post_s * fs)
    if start < 0 or end > len(signal) or end <= start:
        return None
    beat = np.asarray(signal[start:end], dtype=float)
    return beat if beat.size >= 3 else None


def create_template(signal: np.ndarray, peaks: np.ndarray, fs: float) -> Optional[np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    beats: List[np.ndarray] = []

    for r in peaks:
        beat = _extract_beat(signal, int(r), fs)
        if beat is not None:
            beats.append(beat)

    if not beats:
        return None

    min_len = min(len(beat) for beat in beats)
    if min_len < 3:
        return None

    aligned = np.asarray([beat[:min_len] for beat in beats], dtype=float)
    return np.median(aligned, axis=0)


def compute_similarity(beat: np.ndarray, template: np.ndarray) -> float:
    beat = np.asarray(beat, dtype=float)
    template = np.asarray(template, dtype=float)
    if beat.size < 3 or template.size < 3:
        return 0.0

    min_len = min(beat.size, template.size)
    beat = beat[:min_len]
    template = template[:min_len]

    beat_std = float(np.std(beat))
    template_std = float(np.std(template))
    if beat_std <= 1e-9 or template_std <= 1e-9:
        return 0.0

    beat_norm = (beat - float(np.mean(beat))) / beat_std
    template_norm = (template - float(np.mean(template))) / template_std
    corr = np.correlate(beat_norm, template_norm, mode="valid")

    # Convert the raw dot product into a normalized similarity score.
    similarity = float(np.max(corr)) / float(min_len)
    return max(-1.0, min(1.0, similarity))


def classify_with_template(signal: np.ndarray, peaks: np.ndarray,
                           template: Optional[np.ndarray], fs: float) -> List[dict]:
    signal = np.asarray(signal, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    results: List[dict] = []

    if template is None:
        return results

    for r in peaks:
        beat = _extract_beat(signal, int(r), fs)
        if beat is None:
            continue

        similarity = compute_similarity(beat, template)
        if similarity > 0.9:
            label = "NORMAL"
        elif similarity > 0.7:
            label = "VARIANT"
        else:
            label = "ABNORMAL"

        results.append({
            "r": int(r),
            "beat": beat,
            "similarity": similarity,
            "template_type": label,
        })

    return results


def _estimate_local_qrs_width(signal: np.ndarray, r: int, fs: float) -> float:
    left = max(0, r - int(0.25 * fs))
    right = min(len(signal) - 1, r + int(0.40 * fs))
    segment = np.asarray(signal[left:right + 1], dtype=float)
    if segment.size < 3:
        return 0.0

    baseline = float(np.median(segment))
    centered = segment - baseline
    peak = int(np.argmax(np.abs(centered)))
    amp = float(np.max(np.abs(centered)))
    amp_width_ms = 0.0
    if amp > 1e-9:
        threshold = 0.10 * amp
        l_idx = peak
        while l_idx > 0 and abs(centered[l_idx]) > threshold:
            l_idx -= 1
        r_idx = peak
        while r_idx < centered.size - 1 and abs(centered[r_idx]) > threshold:
            r_idx += 1
        amp_width_ms = float((r_idx - l_idx) * 1000.0 / fs)

    grad = np.gradient(segment)
    max_grad = float(np.max(np.abs(grad))) if grad.size else 0.0
    grad_width_ms = 0.0
    if max_grad > 0:
        threshold = 0.2 * max_grad
        active = np.flatnonzero(np.abs(grad) >= threshold)
        if active.size >= 2:
            grad_width_ms = float((int(active[-1]) - int(active[0])) * 1000.0 / fs)

    width_ms = max(amp_width_ms, grad_width_ms)
    return width_ms if 40.0 <= width_ms <= 300.0 else 0.0


def _estimate_local_pr(signal: np.ndarray, r: int, fs: float) -> float:
    start = max(0, r - int(0.30 * fs))
    end = max(start, r - int(0.04 * fs))
    segment = np.asarray(signal[start:end], dtype=float)
    if segment.size < 3:
        return 0.0

    p_idx = int(np.argmax(np.abs(segment))) + start
    pr_samples = r - p_idx
    return float(max(pr_samples, 0) * 1000.0 / fs)


def classify_beats(signal: np.ndarray, peaks: np.ndarray, fs: float) -> List[dict]:
    signal = np.asarray(signal, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    beats: List[dict] = []

    if signal.size == 0 or peaks.size < 3 or fs <= 0:
        return beats

    template = create_template(signal, peaks, fs)
    template_results = classify_with_template(signal, peaks, template, fs)
    template_map = {int(item["r"]): item for item in template_results}

    for i in range(1, len(peaks) - 1):
        r = int(peaks[i])
        rr_prev = float((peaks[i] - peaks[i - 1]) * 1000.0 / fs)
        rr_next = float((peaks[i + 1] - peaks[i]) * 1000.0 / fs)
        width = _estimate_local_qrs_width(signal, r, fs)
        pr = _estimate_local_pr(signal, r, fs)

        template_result = template_map.get(r, {})
        similarity = float(template_result.get("similarity", 0.0))
        morphology_type = str(template_result.get("template_type", "NORMAL"))

        beat_type = morphology_type
        if width > 120 and rr_prev < rr_next:
            beat_type = "PVC"
        elif rr_prev > 1200:
            beat_type = "ESCAPE"
        elif morphology_type == "ABNORMAL" and width > 110:
            beat_type = "ABNORMAL"
        elif morphology_type == "VARIANT":
            beat_type = "VARIANT"
        else:
            beat_type = "NORMAL"

        beats.append(
            {
                "r": r,
                "qrs": width,
                "pr": pr,
                "rr_prev": rr_prev,
                "rr_next": rr_next,
                "similarity": similarity,
                "template_type": morphology_type,
                "type": beat_type,
            }
        )

    return beats


__all__ = [
    "create_template",
    "compute_similarity",
    "classify_with_template",
    "classify_beats",
]

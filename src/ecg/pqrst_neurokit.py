"""
pqrst_neurokit.py

NeuroKit2-backed PQRST analyzer.

This module is intended to be a drop-in replacement for the PQRSTAnalyzer
currently embedded in `expanded_lead_view.py`.

It returns the keys expected by `ArrhythmiaDetector`:
  - r_peaks, p_peaks, q_peaks, s_peaks, t_peaks

And it also returns additional keys (used opportunistically by the detector):
  - t_ends:    T-wave end indices (for QTc measurement)
  - p_absent_flags: per-beat flags for P-wave absence (for AF/Flutter hints)
  - p_amps:    per-beat P-wave amplitudes (0.0 if absent)
  - qrs_widths: per-beat QRS width in milliseconds (0.0 if unavailable)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

import neurokit2 as nk


class PQRSTAnalyzerNK:
    """Clinical-grade PQRST detection using NeuroKit2."""

    def __init__(self, sampling_rate: float = 500.0):
        fs = float(sampling_rate)
        # Avoid NaN sampling_rate breaking NeuroKit2.
        self.fs = fs if np.isfinite(fs) else 500.0

    def _to_int_index(self, v: Any) -> int:
        """Convert NeuroKit2 indices to int, using -1 for missing."""
        if v is None:
            return -1
        try:
            fv = float(v)
            if np.isnan(fv) or fv < 0:
                return -1
            return int(round(fv))
        except Exception:
            return -1

    def _coerce_indices(self, values: Any) -> List[int]:
        if values is None:
            return []
        try:
            return [self._to_int_index(v) for v in list(values)]
        except Exception:
            return []

    def _clean(self, signal: np.ndarray) -> np.ndarray:
        """Full India-tuned filter chain for 500 Hz hardware.

        Order matters:
          1) High-pass 0.5 Hz  — remove baseline wander / sweat / respiration.
          2) 50 Hz notch       — India powerline.
          3) Low-pass 150 Hz   — remove muscle noise, keep ECG band.
        """
        sig = np.asarray(signal, dtype=float)
        if sig.size == 0:
            return sig

        fs = float(self.fs) if self.fs > 0 else 500.0

        # Step 1: High-pass 0.5 Hz
        try:
            b, a = butter(2, 0.5 / (fs / 2.0), btype="high")
            if sig.size > len(b) * 3:
                sig = filtfilt(b, a, sig)
        except Exception:
            pass

        # Step 2: 50 Hz notch (Q≈30 is a good compromise for India mains)
        try:
            b, a = iirnotch(50.0, Q=30.0, fs=fs)
            sig = filtfilt(b, a, sig)
        except Exception:
            pass

        # Step 3: Low-pass 150 Hz
        try:
            b, a = butter(4, 150.0 / (fs / 2.0), btype="low")
            if sig.size > len(b) * 3:
                sig = filtfilt(b, a, sig)
        except Exception:
            pass

        return sig

    def _detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Adaptive Pan-Tompkins via NeuroKit2 with India-specific constraints."""
        cleaned = self._clean(signal)
        if cleaned.size < max(3, int(0.5 * self.fs)):
            return np.array([], dtype=int)

        sr_int = int(round(self.fs))
        try:
            _, info = nk.ecg_peaks(
                cleaned,
                sampling_rate=sr_int,
                method="pantompkins1985",
                correct_artifacts=True,
            )
            peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        except Exception as e:
            print(f"NK2 R-peak error: {e} — using fallback detector")
            return self._fallback_r_peaks(cleaned)

        # Enforce physiologic minimum RR (>= 200 ms).
        if peaks.size > 1:
            rr = np.diff(peaks)
            min_rr = int(0.2 * self.fs)
            valid = np.concatenate([[True], rr >= min_rr])
            peaks = peaks[valid]

        if peaks.size < 2:
            # Fallback if NK2 produced too few R-peaks.
            return self._fallback_r_peaks(cleaned)

        return peaks

    def _fallback_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Fallback R-peak detector when NeuroKit2 fails.

        Adaptive Pan-Tompkins style:
        - Differentiate → square → moving window integration.
        - Adaptive threshold that relaxes if no peaks are found.
        """
        sig = np.asarray(signal, dtype=float)
        if sig.size < 5:
            return np.array([], dtype=int)

        diff = np.diff(sig)
        squared = diff ** 2

        win = max(3, int(0.15 * self.fs))
        mwa = np.convolve(squared, np.ones(win) / win, mode="same")

        # Start with a relatively strict threshold and relax if needed.
        mean_mwa = float(np.mean(mwa))
        std_mwa = float(np.std(mwa))
        min_distance = max(3, int(0.2 * self.fs))

        for factor in (0.5, 0.3, 0.1):
            thresh = mean_mwa + factor * std_mwa
            peaks, _ = find_peaks(mwa, height=thresh, distance=min_distance)
            if peaks.size >= 2:
                return np.asarray(peaks, dtype=int)

        return np.array([], dtype=int)

    def analyze_signal(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Analyze an ECG segment and return P/Q/R/S/T peaks.
        """
        result: Dict[str, Any] = {
            "r_peaks": [],
            "p_peaks": [],
            "q_peaks": [],
            "s_peaks": [],
            "t_peaks": [],
            # extra keys (optional consumption)
            "p_amps": [],
            "t_ends": [],
            "qrs_widths": [],
            "p_absent_flags": [],
        }

        try:
            sig = np.asarray(signal, dtype=float)
            if sig.size < max(10, int(2.0 * self.fs)):
                return result

            cleaned = self._clean(sig)

            r_peaks = self._detect_r_peaks(cleaned)
            if r_peaks.size < 2:
                return result

            # ECG delineation provides P/Q/S/T peaks and T offsets.
            sr_int = int(round(self.fs))
            _, waves = nk.ecg_delineate(
                cleaned,
                r_peaks,
                sampling_rate=sr_int,
                # DWT provides better P/T delineation (useful for QTc + AF P-wave absence)
                method="dwt",
            )

            if not isinstance(waves, dict):
                return result

            p_raw = self._coerce_indices(waves.get("ECG_P_Peaks", []))
            q_raw = self._coerce_indices(waves.get("ECG_Q_Peaks", []))
            s_raw = self._coerce_indices(waves.get("ECG_S_Peaks", []))
            t_raw = self._coerce_indices(waves.get("ECG_T_Peaks", []))
            t_end_raw = self._coerce_indices(waves.get("ECG_T_Offsets", []))

            # Filter for the indices used by ArrhythmiaDetector directly.
            result["r_peaks"] = np.asarray(r_peaks, dtype=int).tolist()
            result["p_peaks"] = [i for i in p_raw if i >= 0]
            result["q_peaks"] = [i for i in q_raw if i >= 0]
            result["s_peaks"] = [i for i in s_raw if i >= 0]
            result["t_peaks"] = [i for i in t_raw if i >= 0]
            result["t_ends"] = [i for i in t_end_raw if i >= 0]

            # Beat-level P-wave absent confidence aligned to R peaks.
            # NeuroKit2 returns only *detected* P peaks; to estimate "P mostly absent"
            # for AF/flutter/AV blocks we must align a per-beat absent flag to each R.
            global_noise = float(np.std(cleaned)) if cleaned.size else 0.0
            noise_floor = max(1e-9, global_noise * 0.15)

            # Candidates for P peaks that lie within the signal bounds.
            p_candidates = [pi for pi in p_raw if 0 <= pi < cleaned.size]

            p_amps_per_r: List[float] = []
            p_absent_flags_per_r: List[bool] = []

            # Search P within [80ms .. 250ms] before each R (wider than earlier to
            # be tolerant to monitor variability).
            pre_start = int(0.25 * self.fs)  # 250 ms
            pre_end = int(0.08 * self.fs)    # 80 ms

            for r_idx in np.asarray(r_peaks, dtype=int):
                win_start = max(0, r_idx - pre_start)
                win_end = max(0, r_idx - pre_end)
                if win_end <= win_start:
                    p_amps_per_r.append(0.0)
                    p_absent_flags_per_r.append(True)
                    continue

                cand = [pi for pi in p_candidates if win_start <= pi < win_end]
                if not cand:
                    p_amps_per_r.append(0.0)
                    p_absent_flags_per_r.append(True)
                    continue

                # Choose the P candidate with maximum amplitude above local baseline.
                best_amp = 0.0
                best_ok = False
                for p_idx in cand:
                    half_win = max(2, int(0.05 * self.fs))
                    s0 = max(0, p_idx - half_win)
                    s1 = min(cleaned.size, p_idx + half_win)
                    seg = cleaned[s0:s1] if s1 > s0 else np.array([])
                    if seg.size < 3:
                        continue

                    baseline = float(np.percentile(seg, 10))
                    amp = float(cleaned[p_idx] - baseline)
                    if abs(amp) > abs(best_amp):
                        best_amp = amp
                    if abs(amp) >= noise_floor:
                        best_ok = True

                if not best_ok or abs(best_amp) < noise_floor:
                    p_amps_per_r.append(0.0)
                    p_absent_flags_per_r.append(True)
                else:
                    p_amps_per_r.append(best_amp)
                    p_absent_flags_per_r.append(False)

            result["p_amps"] = p_amps_per_r
            result["p_absent_flags"] = p_absent_flags_per_r

            # Per-beat QRS width (ms) computed from Q and S peak indices.
            qrs_widths: List[float] = []
            for q_idx, s_idx in zip(q_raw, s_raw):
                if q_idx >= 0 and s_idx > q_idx and s_idx < cleaned.size:
                    qrs_widths.append(float((s_idx - q_idx) / self.fs * 1000.0))
                else:
                    qrs_widths.append(0.0)
            result["qrs_widths"] = qrs_widths

            # Beat-level labels to help downstream arrhythmia logic.
            result["beat_labels"] = self.classify_beats(cleaned, r_peaks, result)

        except Exception as e:
            # Live-mode robustness: return partial/empty results.
            print(f"[PQRSTAnalyzerNK] analyze_signal error: {e}")

        return result

    def classify_beats(self, signal: np.ndarray, r_peaks: np.ndarray, analysis: Dict[str, Any]):
        """Classify each beat as Normal / PVC / PAC / Noise / Unknown.

        India-specific priority: PVCs are very common in hypertensive patients,
        so we use relatively sensitive PVC rules (premature + wide + no P).
        """
        sig = np.asarray(signal, dtype=float)
        r_arr = np.asarray(r_peaks, dtype=int)
        if r_arr.size < 2:
            return []

        qrs_widths = analysis.get("qrs_widths", []) or []
        p_absent = analysis.get("p_absent_flags", []) or []

        rr_intervals = np.diff(r_arr) / self.fs * 1000.0  # ms
        mean_rr = float(np.mean(rr_intervals)) if rr_intervals.size else 800.0

        labels: List[str] = []
        for i, r_idx in enumerate(r_arr):
            qrs_w = float(qrs_widths[i]) if i < len(qrs_widths) else 0.0
            p_abs = bool(p_absent[i]) if i < len(p_absent) else False

            if i < rr_intervals.size:
                this_rr = float(rr_intervals[i])
            else:
                this_rr = mean_rr

            # Rule 1: PVC — premature + wide QRS + P absent
            if (
                this_rr < mean_rr * 0.82  # PVC_PREMATURE_RATIO
                and qrs_w > 120.0        # PVC_QRS_WIDTH_MS
                and p_abs
            ):
                labels.append("PVC")
            # Rule 2: PAC — premature + narrow QRS + P present
            elif (
                this_rr < mean_rr * 0.85
                and qrs_w < 120.0
                and not p_abs
            ):
                labels.append("PAC")
            # Rule 3: Noise — RR < 250 ms (physiologically implausible)
            elif this_rr < 250.0:
                labels.append("Noise")
            else:
                labels.append("Normal")

        return labels


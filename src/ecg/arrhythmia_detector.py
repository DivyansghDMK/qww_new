"""
arrhythmia_detector.py — Complete Merged + Optimized
ALL original detections preserved + ALL new arrhythmias from clinical sheet.
"""

import numpy as np
from scipy.signal import find_peaks
import traceback


class ArrhythmiaDetector:
    def __init__(self, sampling_rate=500, counts_per_mv: float = 500.0):
        self.fs = sampling_rate
        # Hardware scale: at wave_gain=10mm/mV and 1mV=1280 ADC counts.
        # We convert ADC amplitudes to mV inside this detector for Philips-style thresholds.
        self.counts_per_mv = float(counts_per_mv) if counts_per_mv else 0.0
        # India-specific tuning
        self.AFIB_RR_CV_THRESHOLD = 0.18
        self.AFIB_P_ABSENT_RATIO = 0.60
        self.AFIB_MIN_R_PEAKS = 8
        self.PVC_BIGEMINY_RATIO = 0.45
        # Debug helper (kept tiny/optional). We only print a few times.
        self._flutter_debug_used = 0

    # ── Signal quality (Phase 1) ─────────────────────────────────────────────

    def _compute_sqi(self, signal_arr, r_peaks):
        """Signal Quality Index — returns 0.0 (unusable) to 1.0 (clean)."""
        sig = np.asarray(signal_arr, dtype=float)
        if sig.size < self.fs * 2:
            return 0.0
        sig_ptp = float(np.ptp(sig))
        sig_std = float(np.std(sig))
        # Flat line check (lead-off or disconnected)
        if sig_ptp < 0.05 or sig_std < 0.01:
            return 0.0
        # Clipping check — if >20% of samples are at max/min → saturated
        max_v, min_v = float(np.max(sig)), float(np.min(sig))
        if max_v != min_v:
            clipped = np.mean((sig >= max_v * 0.98) | (sig <= min_v * 0.98))
            if clipped > 0.20:
                return 0.2
        # High-frequency noise proxy via excessive R detections.
        r_arr = np.asarray(r_peaks, dtype=int)
        if r_arr.size >= 2:
            rr_mean_sec = float(np.mean(np.diff(r_arr))) / float(self.fs)
            if rr_mean_sec > 0:
                expected_beats = sig.size / float(self.fs) / rr_mean_sec
                beat_density = r_arr.size / max(expected_beats, 1.0)
                # Too many "R peaks" = noise peaks being detected.
                if beat_density > 3.0:
                    return 0.3
        return 1.0

    def detect_arrhythmias(self, signal, analysis,
                           has_received_serial_data=False,
                           min_serial_data_packets=50,
                           lead_signals=None):
        analysis   = analysis or {}
        r_peaks    = np.array(analysis.get('r_peaks', []), dtype=int)
        p_peaks    = np.array(analysis.get('p_peaks', []), dtype=int)
        q_peaks    = np.array(analysis.get('q_peaks', []), dtype=int)
        s_peaks    = np.array(analysis.get('s_peaks', []), dtype=int)
        t_peaks    = np.array(analysis.get('t_peaks', []), dtype=int)
        # Optional NeuroKit2-enhanced keys.
        p_absent_flags = analysis.get('p_absent_flags', [])
        t_ends = analysis.get('t_ends', [])
        qrs_widths = analysis.get('qrs_widths', [])
        signal_arr = np.asarray(signal, dtype=float) if signal is not None else np.array([])
        # Convert ADC counts to mV for amplitude-based checks.
        if signal_arr.size > 0 and self.counts_per_mv and self.counts_per_mv > 0:
            signal_arr = signal_arr / self.counts_per_mv

        # Multi-lead signals (optional — gracefully absent)
        _leads = lead_signals or {}
        lead_I   = np.asarray(_leads.get('I',   []), dtype=float)
        lead_II  = np.asarray(_leads.get('II',  []), dtype=float)
        lead_aVF = np.asarray(_leads.get('aVF', []), dtype=float)
        lead_V1  = np.asarray(_leads.get('V1',  []), dtype=float)
        lead_V5  = np.asarray(_leads.get('V5',  []), dtype=float)
        _has_multilead = (lead_I.size > 0 and lead_aVF.size > 0)

        # Phase 1: signal quality gate.
        sqi = self._compute_sqi(signal_arr, r_peaks)
        if sqi < 0.3:
            return ["Signal quality too low — check lead connection"]
        if sqi < 0.6:
            # Partial quality: skip morphology‑heavy detectors (ST/T, flutter, etc.).
            _skip_morphology = True
        else:
            _skip_morphology = False

        # NOTE (product requirement):
        # The user wants an India-focused subset of detections only.
        # We intentionally DO NOT emit other labels here (they are effectively disabled).

        if len(r_peaks) < 3:
            return ["Insufficient data for arrhythmia detection."]

        rr_ms        = np.diff(r_peaks) / self.fs * 1000.0
        mean_rr      = float(np.mean(rr_ms)) if len(rr_ms) > 0 else 0.0
        hr           = 60000.0 / mean_rr if mean_rr > 0 else 0.0
        pr_interval  = self._estimate_pr_interval(p_peaks, q_peaks)
        qrs_duration = self._estimate_qrs_duration(q_peaks, s_peaks)

        # If NeuroKit2 provided per-beat QRS widths (ms), prefer the median.
        try:
            widths = [float(w) for w in (qrs_widths or []) if w is not None and float(w) > 0.0]
            if widths:
                qrs_duration = float(np.median(widths))
        except Exception:
            pass

        # AF detection: if NeuroKit2 indicates "P mostly absent", AF becomes much more likely.
        p_peaks_for_af = p_peaks
        p_absent_ratio = None
        try:
            if isinstance(p_absent_flags, (list, tuple, np.ndarray)) and len(p_absent_flags) > 0:
                absent_count = sum(1 for f in p_absent_flags if bool(f))
                p_absent_ratio = absent_count / max(1, len(p_absent_flags))
                if p_absent_ratio > 0.70:
                    p_peaks_for_af = np.array([], dtype=int)
        except Exception:
            pass

        # For other atrial rhythms, require meaningful P-wave presence.
        # These are "P-based" rhythms; if P is mostly absent, avoid labeling flutter/sinus-arrhythmia
        # and some atrial tachycardias.
        p_peaks_for_atrial = p_peaks
        if p_absent_ratio is not None and p_absent_ratio > 0.95:
            p_peaks_for_atrial = np.array([], dtype=int)

        # For junctional/nodal rhythms, the key is *P absence*.
        # We use a stricter threshold here so nodal can still trigger when P is truly absent.
        p_peaks_for_junctional = p_peaks
        if p_absent_ratio is not None and p_absent_ratio > 0.55:
            p_peaks_for_junctional = np.array([], dtype=int)

        # Also evaluate AF independently, then decide which to display.
        af_candidate = False
        af_rvr_candidate = False
        flutter_candidate = False
        try:
            af_candidate = bool(self._is_atrial_fibrillation_india(
                signal_arr, r_peaks, p_peaks_for_af, rr_ms, qrs_duration, p_absent_ratio=p_absent_ratio
            ))
        except Exception:
            af_candidate = False
        try:
            af_rvr_candidate = bool(self._is_afib_rvr(r_peaks, p_peaks_for_af, rr_ms, qrs_duration, hr))
        except Exception:
            af_rvr_candidate = False
        if not _skip_morphology:
            try:
                flutter_candidate = bool(self._is_atrial_flutter(
                    hr, qrs_duration, rr_ms, p_peaks_for_atrial, r_peaks, p_absent_ratio=p_absent_ratio
                ))
            except Exception:
                flutter_candidate = False

        arrhythmias = []

        def _try(label, fn, *args, **kwargs):
            try:
                if fn is None:
                    return
                result = fn(*args, **kwargs)
                if result:
                    arrhythmias.append(result if isinstance(result, str) else label)
            except Exception as e:
                print(f"Error detecting {label}: {e}")

        def _m(name):
            fn = getattr(self, name, None)
            return fn if callable(fn) else None

        # Priority 1: life-threatening first (return immediately)
        try:
            if self._is_asystole(signal_arr, r_peaks, hr):
                return ["Asystole (Cardiac Arrest) — EMERGENCY"]
        except Exception:
            pass
        try:
            if self._is_ventricular_fibrillation(signal_arr, r_peaks, rr_ms):
                return ["Ventricular Fibrillation — EMERGENCY"]
        except Exception:
            pass

        # VT (subset)
        _try(
            "Possible Ventricular Tachycardia",
            _m("_is_ventricular_tachycardia"),
            rr_ms,
            qrs_duration,
        )

        # Priority 2: AFib / Flutter
        if af_candidate:
            arrhythmias.append("Atrial Fibrillation Detected")
        elif af_rvr_candidate:
            arrhythmias.append("Atrial Fibrillation 2 (with RVR)")
        elif flutter_candidate:
            arrhythmias.append("Possible Atrial Flutter")

        # PVCs & ectopics / patterns
        # If we already have an atrial candidate (AFib/Flutter), avoid letting PVC/ectopic
        # logic overwrite it in live noisy conditions.
        if not (af_candidate or af_rvr_candidate or flutter_candidate):
            _try("Ventricular Ectopics Detected",
                 _m("_is_ventricular_ectopics"), signal_arr, r_peaks, qrs_duration, p_peaks, rr_ms)
        for label in self._classify_pvcs(signal_arr, r_peaks, rr_ms, qrs_duration, p_peaks, q_peaks, s_peaks):
            arrhythmias.append(label)

        # Bigeminy / Trigeminy / Run
        try:
            if self._is_bigeminy(rr_ms, qrs_duration, signal_arr, r_peaks):
                arrhythmias.append("Bigeminy")
        except Exception as e:
            print(f"Error in bigeminy detection: {e}")
        _try("Trigeminy", _m("_is_trigeminy"), rr_ms, qrs_duration, signal_arr, r_peaks)
        _try("Run of PVCs (>=3 consecutive)",
             _m("_is_run_of_pvcs"), signal_arr, r_peaks, rr_ms, qrs_duration)

        # AV Blocks (Phase 2: Wenckebach-aware)
        try:
            pr_sequence = self._compute_pr_sequence(p_peaks, r_peaks)
            if self._is_wenckebach(pr_sequence, r_peaks, rr_ms):
                arrhythmias.append("Second-Degree AV Block (Type I — Wenckebach)")
            else:
                av = self._is_av_block(pr_interval, p_peaks, r_peaks, rr_ms, hr)
                if av:
                    arrhythmias.append(av)
        except Exception as e:
            print(f"Error in AV block: {e}")
        _try("High AV-Block",
             _m("_is_high_av_block"), pr_interval, p_peaks, r_peaks, rr_ms, hr)

        # Bundle branch blocks (RBBB/LBBB only)
        _try("Left Bundle Branch Block (LBBB)",
             _m("_is_left_bundle_branch_block"), qrs_duration, pr_interval, rr_ms, signal_arr, q_peaks, r_peaks)
        _try("Right Bundle Branch Block (RBBB)",
             _m("_is_right_bundle_branch_block"), qrs_duration, pr_interval, rr_ms, signal_arr, r_peaks)

        # Morphology detections — only on reasonably clean signal
        if not _skip_morphology:
            _try("ST Change Detected",
                 _m("_is_st_change"), signal_arr, q_peaks, s_peaks, p_peaks)
            _try("T-Wave Inversion Detected",
                 _m("_is_t_wave_inversion"), signal_arr, t_peaks, r_peaks)
            _try("QTc Prolongation",
                 _m("_is_qtc_prolonged"), q_peaks, t_peaks, rr_ms)

        # Multi-lead detections (only when lead data provided and morphology allowed)
        if _has_multilead and not _skip_morphology:
            _try("Left Anterior Fascicular Block (LAFB)",
                 _m("_is_lafb"), qrs_duration, lead_I, lead_aVF, r_peaks, s_peaks)
            _try("Left Posterior Fascicular Block (LPFB)",
                 _m("_is_lpfb"), qrs_duration, lead_I, lead_aVF, r_peaks)
            _try("WPW Syndrome",
                 _m("_is_wpw_multilead"), pr_interval, qrs_duration, lead_I, lead_V1, r_peaks)
            _try("Right Ventricular Hypertrophy",
                 _m("_is_rvh"), lead_V1, lead_V5, r_peaks, qrs_duration)

        # SVT (subset)
        _try("Supraventricular Tachycardia (SVT)",
             _m("_is_supraventricular_tachycardia"), hr, qrs_duration, rr_ms, p_peaks, r_peaks)
        _try("Paroxysmal Atrial Tachycardia (PAT)",
             _m("_is_pat"), hr, qrs_duration, rr_ms, p_peaks, r_peaks)

        # Bradycardia (subset)
        if not arrhythmias:
            try:
                if self._is_bradycardia(rr_ms):
                    arrhythmias.append("Sinus Bradycardia")
            except Exception as e:
                print(f"Error in bradycardia detection: {e}")

        if not arrhythmias and self._is_normal_sinus_rhythm(rr_ms):
            return ["Normal Sinus Rhythm"]

        # India-focused priority order:
        # AFib > PVCs/Bigeminy > AV Blocks > RBBB/LBBB > Bradycardia > SVT/VT
        if not arrhythmias:
            return ["Unspecified Irregular Rhythm"]

        def _rank(label: str) -> int:
            s = str(label or "")
            if "Atrial Fibrillation" in s:
                return 0
            if "Atrial Flutter" in s:
                return 1
            if ("PVC" in s) or ("Ventricular Ectopics" in s) or ("Bigeminy" in s) or ("Trigeminy" in s) or ("Run of PVCs" in s):
                return 1
            if "AV Block" in s:
                return 2
            if "WPW" in s:
                return 2
            if ("Bundle Branch Block" in s) or ("(LBBB)" in s) or ("(RBBB)" in s) or ("LBBB" in s) or ("RBBB" in s):
                return 3
            if "LAFB" in s or "Left Anterior Fascicular" in s:
                return 3
            if "LPFB" in s or "Left Posterior Fascicular" in s:
                return 3
            if "Sinus Bradycardia" in s:
                return 4
            if ("SVT" in s) or ("Tachycardia" in s):
                return 5
            if "Ventricular Tachycardia" in s:
                return 5
            if "Right Ventricular Hypertrophy" in s:
                return 13
            if "QTc Prolongation" in s:
                return 14
            if "ST Change" in s:
                return 15
            if "T-Wave Inversion" in s:
                return 15
            return 99

        arrhythmias_sorted = sorted(arrhythmias, key=_rank)

        # Phase 5: return up to top‑3 labels across different tiers.
        def _pick_top_n(sorted_labels, n=3):
            tiers_seen = set()
            picked = []
            for lab in sorted_labels:
                tier = _rank(lab) // 3
                if tier not in tiers_seen:
                    picked.append(lab)
                    tiers_seen.add(tier)
                if len(picked) >= n:
                    break
            return picked or sorted_labels[:1]

        return _pick_top_n(arrhythmias_sorted, n=3)

    def _is_atrial_fibrillation_india(self, signal, r_peaks, p_peaks, rr_intervals, qrs_duration, p_absent_ratio=None):
        """AFib detection tuned for Indian cohorts and live monitor windows."""
        if len(r_peaks) < self.AFIB_MIN_R_PEAKS:
            return False

        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 2:
            rr = np.diff(r_peaks) / self.fs * 1000.0
        if len(rr) < 2:
            return False

        mean_rr = float(np.mean(rr))
        if mean_rr <= 0:
            return False
        rr_cv = float(np.std(rr) / mean_rr)
        if rr_cv < self.AFIB_RR_CV_THRESHOLD:
            # Exception: if NeuroKit2 strongly reports P-wave absence, AFib can still
            # trigger even when RR irregularity looks modest on short windows.
            if not (p_absent_ratio is not None and p_absent_ratio >= (self.AFIB_P_ABSENT_RATIO + 0.05)):
                return False

        # Use NK2 per-beat P-absence when available; but tolerate occasional P detection
        # failures (common on live monitors).
        if p_absent_ratio is not None:
            # If P-absence ratio is too low, still allow AFib when RR irregularity is strong.
            if p_absent_ratio < (self.AFIB_P_ABSENT_RATIO - 0.10):
                if rr_cv < 0.28:
                    return False
        else:
            # Fallback on sparse P detections (fewer P peaks favors AF).
            p_ratio = float(len(np.asarray(p_peaks, dtype=int))) / float(max(1, len(r_peaks)))
            if p_ratio > 0.45:
                return False

        # Typical AFib has narrow QRS unless aberrancy; keep relaxed upper bound.
        if qrs_duration is not None and qrs_duration > 130:
            return False
        return True

    # ── Adaptive windowing (Phase 3) ─────────────────────────────────────────

    def _adaptive_window(self, rr_ms, min_beats=8, fallback_sec=6.0):
        """Returns window_size in seconds that captures at least min_beats."""
        rr = np.asarray(rr_ms, dtype=float)
        if rr.size < 2:
            return float(fallback_sec)
        mean_rr_sec = float(np.mean(rr)) / 1000.0
        required = mean_rr_sec * float(min_beats)
        return float(max(required, fallback_sec))

    def detect_arrhythmias_with_probabilities(self, signal, analysis, window_size=2.0, step_size=None):
        """
        Build a simple AFib vs Atrial Flutter probability heatmap.

        Returns a dict where each key maps to a list of (window_center_time_sec, probability).
        ExpandedLeadView overlays this as a background heatmap and extracts high-probability events.
        """
        analysis = analysis or {}
        signal_arr = np.asarray(signal, dtype=float) if signal is not None else np.array([])

        r_peaks = np.asarray(analysis.get("r_peaks", []), dtype=int)
        p_peaks = np.asarray(analysis.get("p_peaks", []), dtype=int)
        p_absent_flags = analysis.get("p_absent_flags", [])
        qrs_widths = analysis.get("qrs_widths", [])

        # If we can't time windows, return an empty heatmap.
        if r_peaks.size < 3:
            return {
                "Normal Sinus Rhythm": [],
                "Atrial Fibrillation": [],
                "Atrial Flutter": [],
            }

        # Basic timing for windows.
        start_t = float(r_peaks[0]) / float(self.fs)
        end_t = float(r_peaks[-1]) / float(self.fs)
        if end_t <= start_t:
            return {
                "Normal Sinus Rhythm": [],
                "Atrial Fibrillation": [],
                "Atrial Flutter": [],
            }

        if step_size is None:
            # Use the current/soon‑to‑be adaptive window size as a natural step.
            step_size = float(window_size) if window_size else 2.0
        step_size = float(step_size) if step_size else float(window_size or 2.0)

        # Phase 3: adapt the analysis window length based on RR.
        if window_size is None or float(window_size) < 2.0:
            rr_quick = np.diff(r_peaks) / float(self.fs) * 1000.0
            window_size = self._adaptive_window(rr_quick)
        else:
            window_size = float(window_size)

        num_windows = int(np.floor((end_t - start_t) / step_size)) + 1
        if num_windows <= 0:
            num_windows = 1

        # P absent ratio: used as a differentiator between AFib vs flutter.
        p_absent_ratio = None
        try:
            if isinstance(p_absent_flags, (list, tuple, np.ndarray)) and len(p_absent_flags) > 0:
                absent_count = sum(1 for f in p_absent_flags if bool(f))
                p_absent_ratio = float(absent_count) / float(max(1, len(p_absent_flags)))
        except Exception:
            p_absent_ratio = None

        # Prefer a median QRS width if provided.
        qrs_duration = None
        try:
            widths = [float(w) for w in (qrs_widths or []) if w is not None and float(w) > 0.0]
            if widths:
                qrs_duration = float(np.median(widths))
        except Exception:
            qrs_duration = None

        # Precompute for speed.
        p_peaks_sec = p_peaks.astype(float) / float(self.fs)
        r_peaks_sec = r_peaks.astype(float) / float(self.fs)

        out = {
            "Normal Sinus Rhythm": [],
            "Atrial Fibrillation": [],
            "Atrial Flutter": [],
        }

        def _clamp01(x):
            try:
                return float(max(0.0, min(1.0, x)))
            except Exception:
                return 0.0

        for w in range(num_windows):
            center_t = start_t + w * step_size
            half = window_size / 2.0
            t0 = center_t - half
            t1 = center_t + half

            # Select peaks that fall inside this window.
            r_win_mask = (r_peaks_sec >= t0) & (r_peaks_sec <= t1)
            r_win = r_peaks[r_win_mask]
            p_win_mask = (p_peaks_sec >= t0) & (p_peaks_sec <= t1)
            p_win = p_peaks[p_win_mask]

            if r_win.size >= 3:
                rr_ms_win = np.diff(r_win) / float(self.fs) * 1000.0
                mean_rr_win = float(np.mean(rr_ms_win)) if rr_ms_win.size else 0.0
                rr_cv_win = float(np.std(rr_ms_win) / mean_rr_win) if mean_rr_win > 0 else 0.0

                hr_win = 60000.0 / mean_rr_win if mean_rr_win > 0 else 0.0
            else:
                rr_ms_win = np.array([])
                rr_cv_win = 0.0
                hr_win = 0.0

            # AFib probability: irregular rhythm + often higher P-absence.
            p_abs_score = 0.5
            if p_absent_ratio is not None:
                # High absent ratio -> higher AFib likelihood.
                p_abs_score = _clamp01((p_absent_ratio - 0.65) / 0.25)

            # Continuous heuristic.
            rr_score_af = _clamp01((rr_cv_win - 0.15) / 0.25)  # 0 when rr_cv <= 0.15
            prob_af = rr_score_af * (0.55 + 0.45 * p_abs_score)

            # Hard boost if criteria pass.
            try:
                if self._is_atrial_fibrillation(
                    signal_arr,
                    r_win,
                    p_win,
                    rr_ms_win,
                    qrs_duration,
                    p_absent_ratio=p_absent_ratio,
                ):
                    prob_af = max(prob_af, 0.85)
            except Exception:
                pass

            # Flutter probability: relatively regular + 2:1-ish behavior + not-extreme P absence.
            p_to_r = 0.0
            if r_win.size > 0:
                p_to_r = float(p_win.size) / float(max(1, r_win.size))

            p_abs_flutter_penalty = 1.0
            if p_absent_ratio is not None:
                # If P is *too* absent, flutter detection is unreliable -> penalize.
                p_abs_flutter_penalty = _clamp01(1.0 - (p_absent_ratio - 0.75) / 0.25)
            rr_score_fl = _clamp01((0.32 - rr_cv_win) / 0.18)
            p_to_r_score = _clamp01((p_to_r - 0.25) / 0.40)  # prefers >= ~0.5

            prob_fl = 0.15 + 0.85 * rr_score_fl * p_to_r_score * p_abs_flutter_penalty

            try:
                # Use the same detector criteria as a hard boost.
                # Note: these internal methods use HR + QRS too.
                if self._is_atrial_flutter(
                    hr_win if hr_win > 0 else None,
                    qrs_duration,
                    rr_ms_win if rr_ms_win.size else np.diff(r_win) / float(self.fs) * 1000.0,
                    p_win,
                    r_win,
                    p_absent_ratio=p_absent_ratio,
                ):
                    prob_fl = max(prob_fl, 0.85)
            except Exception:
                pass

            # Normal probability is what's left (only between AFib/Flutter).
            prob_norm = _clamp01(1.0 - max(prob_af, prob_fl))

            out["Normal Sinus Rhythm"].append((center_t, prob_norm))
            out["Atrial Fibrillation"].append((center_t, prob_af))
            out["Atrial Flutter"].append((center_t, prob_fl))

        return out

    # ── Utilities ──────────────────────────────────────────────────────────

    # Phase 2: per‑beat PR tracking

    def _compute_pr_sequence(self, p_peaks, r_peaks):
        """Returns list of (beat_index, pr_ms) — one per beat where P found."""
        p_arr = np.asarray(p_peaks, dtype=int)
        r_arr = np.asarray(r_peaks, dtype=int)
        sequence = []
        if p_arr.size == 0 or r_arr.size == 0:
            return sequence
        for i, r in enumerate(r_arr):
            # Find P peak immediately before this R within a 300 ms window.
            win = int(0.30 * self.fs)
            candidates = p_arr[(p_arr < r) & (p_arr > r - win)]
            if candidates.size:
                pr_ms = (r - candidates[-1]) / self.fs * 1000.0
                if 80.0 <= pr_ms <= 350.0:
                    sequence.append((i, pr_ms))
        return sequence

    def _estimate_pr_interval(self, p_peaks, q_peaks):
        """Legacy mean PR estimate (kept for existing logic)."""
        p_arr = np.asarray(p_peaks, dtype=int)
        q_arr = np.asarray(q_peaks, dtype=int)
        if len(p_arr) == 0 or len(q_arr) == 0:
            return None
        intervals = []
        for p in p_arr:
            q_after = q_arr[q_arr > p]
            if len(q_after):
                v = (q_after[0] - p) / self.fs * 1000.0
                if v < 400:
                    intervals.append(v)
        return float(np.mean(intervals)) if intervals else None

    def _estimate_qrs_duration(self, q_peaks, s_peaks):
        q_arr = np.asarray(q_peaks, dtype=int)
        s_arr = np.asarray(s_peaks, dtype=int)
        if len(q_arr) == 0 or len(s_arr) == 0:
            return None
        durations = []
        for q in q_arr:
            s_after = s_arr[s_arr > q]
            if len(s_after):
                v = (s_after[0] - q) / self.fs * 1000.0
                if v < 200:
                    durations.append(v)
        return float(np.mean(durations)) if durations else None

    def _rr_cv(self, rr_ms):
        if len(rr_ms) < 2:
            return 0.0
        m = float(np.mean(rr_ms))
        return float(np.std(rr_ms) / m) if m > 0 else 0.0

    def _is_wenckebach(self, pr_sequence, r_peaks, rr_ms):
        """Detects Second‑degree AV block type I (Wenckebach)."""
        if len(pr_sequence) < 4:
            return False
        pr_values = [pr for _, pr in pr_sequence]
        # Look for runs of ≥3 beats with progressively lengthening PR.
        for i in range(len(pr_values) - 2):
            window = pr_values[i:i + 3]
            if window[1] > window[0] + 10.0 and window[2] > window[1] + 5.0:
                # Check for a pause (dropped beat) after this run.
                beat_indices = [idx for idx, _ in pr_sequence]
                if i + 3 < len(beat_indices):
                    next_idx = beat_indices[i + 3]
                    if next_idx > beat_indices[i + 2] + 1:
                        return True
                # Or check for a clearly long RR right after the run.
                rr = np.asarray(rr_ms, dtype=float)
                if rr.size > i + 2:
                    if rr[i + 2] > float(np.mean(rr)) * 1.4:
                        return True
        return False

    # ── Multi-lead helpers (Phase 4) ────────────────────────────────────────

    def _is_lafb(self, qrs_duration, lead_I, lead_aVF, r_peaks, s_peaks):
        """
        Left anterior fascicular block:
        - Left axis deviation: tall R in I, deep S in aVF.
        - QRS < 120 ms (not a full bundle branch block).
        """
        if qrs_duration is not None and qrs_duration >= 120:
            return False
        if lead_I.size == 0 or lead_aVF.size == 0:
            return False
        r_arr = np.asarray(r_peaks, dtype=int)
        if r_arr.size < 3:
            return False

        r_amps_I   = [float(lead_I[r])   for r in r_arr if r < lead_I.size]
        r_amps_aVF = [float(lead_aVF[r]) for r in r_arr if r < lead_aVF.size]

        # S wave in aVF: trough after each R
        s_amps_aVF = []
        for r in r_arr[:6]:
            window_end = min(lead_aVF.size, r + int(0.08 * self.fs))
            if window_end > r:
                s_amps_aVF.append(float(np.min(lead_aVF[r:window_end])))

        if len(r_amps_I) < 3 or len(s_amps_aVF) < 3:
            return False

        mean_R_I   = float(np.mean(r_amps_I))
        mean_R_aVF = float(np.mean(r_amps_aVF)) if r_amps_aVF else 0.0
        mean_S_aVF = float(np.mean(s_amps_aVF))

        return (
            mean_R_I > 0.0 and
            mean_S_aVF < 0.0 and
            abs(mean_S_aVF) > abs(mean_R_aVF) * 1.2
        )

    def _is_lpfb(self, qrs_duration, lead_I, lead_aVF, r_peaks):
        """
        Left posterior fascicular block:
        - Right axis deviation: deep S in I, tall R in aVF.
        - QRS < 120 ms.
        """
        if qrs_duration is not None and qrs_duration >= 120:
            return False
        if lead_I.size == 0 or lead_aVF.size == 0:
            return False
        r_arr = np.asarray(r_peaks, dtype=int)
        if r_arr.size < 3:
            return False

        r_amps_aVF = [float(lead_aVF[r]) for r in r_arr if r < lead_aVF.size]

        # S wave in lead I: trough after each R
        s_amps_I = []
        for r in r_arr[:6]:
            window_end = min(lead_I.size, r + int(0.08 * self.fs))
            if window_end > r:
                s_amps_I.append(float(np.min(lead_I[r:window_end])))

        if len(r_amps_aVF) < 3 or len(s_amps_I) < 3:
            return False

        mean_R_aVF = float(np.mean(r_amps_aVF))
        mean_S_I   = float(np.mean(s_amps_I))

        return (
            mean_R_aVF > 0.0 and
            mean_S_I < 0.0 and
            abs(mean_S_I) > mean_R_aVF * 1.2
        )

    def _is_wpw_multilead(self, pr_interval, qrs_duration, lead_I, lead_V1, r_peaks):
        """
        WPW: short PR + wide QRS + delta wave slurring in V1 or lead I.
        Delta wave = slow initial upstroke before the main R deflection.
        """
        if pr_interval is None or qrs_duration is None:
            return False
        if not (pr_interval < 120 and qrs_duration > 110):
            return False

        delta_found = 0
        r_arr = np.asarray(r_peaks, dtype=int)
        if r_arr.size == 0:
            return False

        for lead in (lead_V1, lead_I):
            if lead.size == 0:
                continue
            for r in r_arr[:6]:
                onset = max(0, r - int(0.06 * self.fs))   # 60 ms before R
                seg = lead[onset:r]
                if seg.size < 6:
                    continue
                diffs = np.diff(seg.astype(float))
                if diffs.size < 4:
                    continue
                peak_slope = float(np.max(np.abs(diffs)))
                if peak_slope <= 0.0:
                    continue
                first_half_slope = float(np.mean(np.abs(diffs[: diffs.size // 2])))
                if 0.0 < first_half_slope < peak_slope * 0.45:
                    delta_found += 1
                    break

        return delta_found >= 1

    def _is_rvh(self, lead_V1, lead_V5, r_peaks, qrs_duration):
        """
        Right ventricular hypertrophy: dominant R in V1 (R > S).
        """
        if lead_V1.size == 0:
            return False
        r_arr = np.asarray(r_peaks, dtype=int)
        if r_arr.size < 3:
            return False

        r_amps_V1 = [float(lead_V1[r]) for r in r_arr if r < lead_V1.size]
        s_amps_V1 = []
        for r in r_arr[:6]:
            end = min(lead_V1.size, r + int(0.08 * self.fs))
            if end > r:
                s_amps_V1.append(float(np.min(lead_V1[r:end])))

        if len(r_amps_V1) < 3 or len(s_amps_V1) < 3:
            return False

        mean_R_V1 = float(np.mean(r_amps_V1))
        mean_S_V1 = float(np.mean(s_amps_V1))

        return mean_R_V1 > 0.0 and mean_R_V1 > abs(mean_S_V1)

    # ── Original preserved detections ──────────────────────────────────────

    def _is_normal_sinus_rhythm(self, rr_intervals):
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3: return False
        mean_rr = float(np.mean(rr))
        if mean_rr <= 0:
            return False
        rr_std = float(np.std(rr))
        rr_cv = rr_std / mean_rr
        return 60 <= 60000.0/mean_rr <= 100 and rr_std < 60 and rr_cv < 0.08

    def _is_asystole(self, signal, r_peaks, heart_rate, min_data_packets=50):
        sig = np.asarray(signal, dtype=float)
        if len(sig) == 0 or len(sig)/self.fs < 2.0: return False
        amp = float(np.ptp(sig)); max_abs = float(np.max(np.abs(sig)))
        # Thresholds were historically written in ADC counts; we now run on mV.
        # Convert the original constants using counts_per_mv.
        cpmv = self.counts_per_mv if self.counts_per_mv else 1.0
        amp_threshold = 10.0 / cpmv
        flat_amp = (50.0 / cpmv) if max_abs > amp_threshold else (0.05 / cpmv)
        flat_std = (20.0 / cpmv) if max_abs > amp_threshold else (0.02 / cpmv)
        if len(r_peaks) == 0:
            return amp < flat_amp or (amp < flat_amp*5 and float(np.std(sig)) < flat_std)
        if len(r_peaks) <= 2:
            return amp < flat_amp*5 and float(np.std(sig)) < flat_std*6
        if heart_rate is not None and heart_rate < 20:
            return amp < flat_amp*10 and float(np.std(sig)) < flat_std*5
        dur = len(sig)/self.fs
        if dur > 3 and (len(r_peaks)/dur)*60 < 20 and amp < flat_amp*12:
            return True
        return False

    def _is_atrial_fibrillation(self, signal, r_peaks, p_peaks, rr_intervals, qrs_duration, p_absent_ratio=None):
        if len(r_peaks) < 8: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 2: rr = np.diff(r_peaks)/self.fs*1000.0
        if len(rr) < 2: return False
        mean_rr = float(np.mean(rr))
        if mean_rr <= 0: return False
        rr_cv = float(np.std(rr)/mean_rr)
        p_arr = np.asarray(p_peaks, dtype=int)
        p_ratio = float(len(p_arr)) / max(len(r_peaks), 1)
        # Make AF harder to trigger in sinus arrhythmia:
        # - Require stronger RR irregularity
        # - If P is "mostly absent", allow AF; otherwise avoid false positives
        if qrs_duration is not None and qrs_duration > 120:
            return False

        # If P absent ratio is known, use it.
        if p_absent_ratio is not None:
            if p_absent_ratio < 0.65:
                return False

        # Base irregularity requirement, relaxed when P is very likely absent.
        # (Some AF episodes appear "more regular" over short windows, especially with pauses
        # or rate-limited conduction; when P is strongly absent we allow a lower rr_cv.)
        min_rr_cv = 0.22
        if p_absent_ratio is not None:
            if p_absent_ratio >= 0.92:
                min_rr_cv = 0.15
            elif p_absent_ratio >= 0.85:
                min_rr_cv = 0.18
        else:
            # If we cannot compute p_absent_ratio, still allow AF when P detection is sparse.
            # This avoids flutter stealing AF when rr_cv is borderline.
            if p_ratio <= 0.35:
                min_rr_cv = 0.18

        if rr_cv < min_rr_cv:
            return False

        # If we have no P peaks, still require high RR irregularity.
        if len(p_arr) == 0:
            return rr_cv >= max(min_rr_cv, 0.20)

        # If P exists but is sparse, require high RR irregularity.
        if p_ratio < 0.55:
            return rr_cv >= max(min_rr_cv, 0.22)

        # Otherwise: only call AF if P timing is highly inconsistent.
        if len(p_arr) >= 3:
            p_iv = np.diff(p_arr) / self.fs * 1000.0
            if len(p_iv) > 1 and float(np.mean(p_iv)) > 0:
                p_cv = float(np.std(p_iv) / np.mean(p_iv))
                return p_cv > 0.15

        return False

    def _is_afib_rvr(self, r_peaks, p_peaks, rr_ms, qrs_duration, hr):
        if len(rr_ms) < 4: return False
        return self._rr_cv(rr_ms) > 0.10 and hr > 110 and len(p_peaks) < len(r_peaks)*0.5

    def _is_ventricular_tachycardia(self, rr_intervals, qrs_duration):
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 5 or qrs_duration is None or qrs_duration <= 120: return False
        mean_rr = float(np.mean(rr))
        if mean_rr <= 0: return False
        return 60000.0/mean_rr > 120 and float(np.std(rr)) < 80

    def _is_ventricular_fibrillation(self, signal, r_peaks, rr_intervals):
        if signal is None or len(signal) < int(self.fs * 3): return False
        sig = np.asarray(signal, dtype=float)
        sig_d = sig - float(np.mean(sig))
        ptp = float(np.ptp(sig_d))
        if ptp <= 0:
            return False
        sig_norm = sig_d / ptp
        rr  = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3 and len(r_peaks) >= 3: rr = np.diff(r_peaks)/self.fs*1000.0
        if len(r_peaks) >= 5:
            return False
        if len(rr) >= 3:
            mean_rr = float(np.mean(rr))
            if mean_rr > 0 and float(np.std(rr)/mean_rr) > 0.3:
                if float(np.std(sig_norm)) > 0.15: return True
        dur = len(sig)/self.fs
        if dur >= 2.0 and len(r_peaks) >= 3:
            crr = np.diff(r_peaks)/self.fs*1000.0
            if len(crr) >= 2:
                m = float(np.mean(crr))
                if m > 0 and float(np.std(crr)/m) > 0.25 and float(np.std(sig_norm)) > 0.13: return True
        if dur >= 3.0 and len(r_peaks) < 5:
            if float(np.std(sig_norm)) > 0.14 and float(np.mean(np.abs(sig_norm))) > 0.05: return True
        if len(sig) >= 1000:
            ma = float(np.mean(np.abs(sig_norm)))
            if ma > 0 and float(np.std(sig_norm))/ma > 1.0:
                crr2 = np.diff(r_peaks)/self.fs*1000.0 if len(r_peaks) >= 3 else np.array([])
                if len(r_peaks) < 8 or (len(crr2)>=3 and float(np.mean(crr2))>0 and float(np.std(crr2)/np.mean(crr2))>0.2): return True
        return False

    def _is_bradycardia(self, rr_intervals):
        rr = np.asarray(rr_intervals, dtype=float)
        return len(rr) >= 3 and 60000.0/float(np.mean(rr)) < 60

    def _is_tachycardia(self, rr_intervals):
        rr = np.asarray(rr_intervals, dtype=float)
        return len(rr) >= 3 and 60000.0/float(np.mean(rr)) >= 100

    def _is_ventricular_ectopics(self, signal, r_peaks, qrs_duration, p_peaks, rr_intervals):
        if len(r_peaks) < 5: return False
        rr_ms = np.asarray(rr_intervals, dtype=float)
        if len(rr_ms) == 0: rr_ms = np.diff(r_peaks)/self.fs*1000.0
        if len(rr_ms) < 2: return False
        mean_rr = float(np.mean(rr_ms))
        if mean_rr <= 0: return False
        if qrs_duration is not None and qrs_duration > 120:
            prem = int(np.sum(rr_ms < 0.85*mean_rr))
            comp = sum(1 for i in range(len(rr_ms)-1) if rr_ms[i]<0.85*mean_rr and rr_ms[i+1]>1.15*mean_rr)
            if prem >= 1 and comp >= 1: return True
            if prem >= 2: return True
        rr_sec = np.diff(r_peaks)/self.fs
        if len(rr_sec) < 2: return False
        mean_s = float(np.mean(rr_sec))
        p_arr  = np.asarray(p_peaks, dtype=int)
        for i in range(len(rr_sec)):
            if rr_sec[i] < 0.8*mean_s and i+1 < len(rr_sec) and rr_sec[i+1] > 1.2*mean_s:
                pr = r_peaks[i+1] if i+1 < len(r_peaks) else None
                if pr is not None:
                    if not any(120 <= (pr-p)/self.fs*1000 <= 200 for p in p_arr): return True
        return False

    def _is_bigeminy(self, rr_intervals, qrs_duration, signal, r_peaks):
        try:
            rr = np.asarray(rr_intervals, dtype=float)
            if len(rr) < 4 or len(r_peaks) < 5: return False
            if float(np.max(rr)) < 10: rr = rr*1000.0
            mean_rr = float(np.mean(rr))
            if mean_rr <= 0: return False
            sh, lg = 0.75*mean_rr, 1.03*mean_rr
            alt = sum(1 for i in range(len(rr)-1)
                      if (bool(rr[i]<sh) and bool(rr[i+1]>lg)) or (bool(rr[i]>lg) and bool(rr[i+1]<sh)))
            min_alt = max(2, int(len(rr)*0.25))
            if alt < min_alt: return False
            short_ivs = [float(v) for v in rr if float(v) < sh]
            consistent = True
            if len(short_ivs) >= 2:
                cm = float(np.mean(short_ivs))
                consistent = (float(np.std(short_ivs)/cm) <= 0.25) if cm > 0 else False
            has_wide = qrs_duration is not None and qrs_duration > 120
            if alt >= min_alt:
                if has_wide: return True
                if alt >= max(2, int(len(rr)*0.3)):
                    if consistent: return True
                    if alt >= max(2, int(len(rr)*0.5)): return True
                    if alt >= 3: return True
            return False
        except Exception as e:
            print(f"Error in bigeminy: {e}")
            return False

    def _is_asynchronous_75_bpm(self, heart_rate, rr_intervals, p_peaks, r_peaks):
        try:
            if heart_rate is None: return False
            rr = np.asarray(rr_intervals, dtype=float)
            if len(rr) < 3: return False
            if float(np.max(rr)) < 10: rr = rr*1000.0
            mean_rr = float(np.mean(rr)); std_rr = float(np.std(rr))
            if mean_rr <= 0: return False
            cv = std_rr/mean_rr
            if 70 <= heart_rate <= 80:
                if cv < 0.005 or cv > 0.25 or std_rr < 5 or std_rr > 300: return False
                p_c = len(p_peaks) if p_peaks is not None else 0
                r_c = len(r_peaks) if r_peaks is not None else 0
                if r_c > 0 and p_c < r_c*0.05: return False
                return True
            if not (60 <= heart_rate <= 90) or not (0.03 <= cv <= 0.15 and 30 <= std_rr <= 250): return False
            p_c = len(p_peaks) if p_peaks is not None else 0
            r_c = len(r_peaks) if r_peaks is not None else 0
            if r_c > 0 and p_c < r_c*0.2: return False
            for i in range(len(rr)-1):
                if abs(float(rr[i+1])-float(rr[i])) > 200: return False
            return True
        except Exception: return False

    def _is_left_bundle_branch_block(self, qrs_duration, pr_interval, rr_intervals, signal, q_peaks, r_peaks):
        if qrs_duration is None or qrs_duration < 130: return False
        if pr_interval is not None and pr_interval > 220: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3: return False
        rr_ms = rr
        mean_rr = float(np.mean(rr_ms))
        if mean_rr <= 0 or float(np.std(rr_ms)/mean_rr) > 0.15: return False
        q_arr = np.asarray(q_peaks, dtype=int); r_arr = np.asarray(r_peaks, dtype=int)
        if len(r_arr) == 0 or len(q_arr) > len(r_arr)*0.6: return False
        sig = np.asarray(signal, dtype=float); notched = total = 0
        for r in r_arr[:min(6,len(r_arr))]:
            st = max(0,r-int(0.02*self.fs)); en = min(len(sig),r+int(0.08*self.fs))
            if en-st < 5: continue
            seg = sig[st:en]-sig[st:en].min()
            try: pks,_ = find_peaks(seg, distance=max(2,int(0.01*self.fs)))
            except: continue
            if len(pks) >= 2: notched += 1
            total += 1
        return total > 0 and notched/total >= 0.3

    def _is_right_bundle_branch_block(self, qrs_duration, pr_interval, rr_intervals, signal, r_peaks):
        if qrs_duration is None or qrs_duration < 120: return False
        if pr_interval is not None and pr_interval > 220: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3: return False
        rr_ms = rr
        mean_rr = float(np.mean(rr_ms))
        if mean_rr <= 0 or float(np.std(rr_ms)/mean_rr) > 0.18: return False
        r_arr = np.asarray(r_peaks, dtype=int)
        if len(r_arr) < 3: return False
        sig = np.asarray(signal, dtype=float); ds = checked = 0
        for r in r_arr[:min(6,len(r_arr))]:
            st = max(0,r-int(0.015*self.fs)); en = min(len(sig),r+int(0.09*self.fs))
            if en-st < 6: continue
            seg = sig[st:en]-float(np.mean(sig[st:en]))
            fpv = float(np.max(seg))
            if fpv <= 0: continue
            try: pks,_ = find_peaks(seg, distance=max(2,int(0.008*self.fs)))
            except: continue
            for i in range(len(pks)-1):
                d = (pks[i+1]-pks[i])/self.fs*1000.0
                if 15 <= d <= 70 and seg[pks[i+1]]/fpv >= 0.3: ds += 1; break
            checked += 1
        return checked > 0 and ds/checked >= 0.3

    def _is_left_anterior_fascicular_block(self, qrs_duration, heart_rate, signal, r_peaks, s_peaks):
        if qrs_duration is None or qrs_duration > 130: return False
        if heart_rate is not None and not (45 <= heart_rate <= 120): return False
        r_arr = np.asarray(r_peaks, dtype=int); s_arr = np.asarray(s_peaks, dtype=int)
        n = min(len(r_arr),len(s_arr),6)
        if n < 3: return False
        sig = np.asarray(signal, dtype=float)
        r_a = [abs(float(sig[r_arr[i]])) for i in range(n) if r_arr[i]<len(sig)]
        s_a = [abs(float(sig[s_arr[i]])) for i in range(n) if s_arr[i]<len(sig)]
        if len(r_a) < 3 or len(s_a) < 3: return False
        avg_r,avg_s = float(np.mean(r_a)),float(np.mean(s_a))
        if avg_r<=0 or avg_s<=0 or avg_s/avg_r < 1.6: return False
        sl=ch=0
        for i in range(n):
            if r_arr[i]>=len(sig) or s_arr[i]>=len(sig): continue
            ch+=1
            seg = sig[min(r_arr[i],s_arr[i]):min(len(sig),s_arr[i]+int(0.04*self.fs))]
            if len(seg)<5: continue
            diff = np.diff(seg); thr = 0.2*float(np.max(np.abs(seg))) if float(np.max(np.abs(seg)))>0 else 0.05
            if thr>0 and float(np.mean(np.abs(diff)<thr))>0.6: sl+=1
        return ch>0 and sl/ch >= 0.4

    def _is_left_posterior_fascicular_block(self, qrs_duration, heart_rate, signal, r_peaks, s_peaks):
        if qrs_duration is None or qrs_duration > 130: return False
        if heart_rate is not None and not (45 <= heart_rate <= 120): return False
        r_arr = np.asarray(r_peaks, dtype=int); s_arr = np.asarray(s_peaks, dtype=int)
        n = min(len(r_arr),len(s_arr),6)
        if n < 3: return False
        sig = np.asarray(signal, dtype=float)
        r_a = [abs(float(sig[r_arr[i]])) for i in range(n) if r_arr[i]<len(sig)]
        s_a = [abs(float(sig[s_arr[i]])) for i in range(n) if s_arr[i]<len(sig)]
        if len(r_a)<3 or len(s_a)<3: return False
        avg_r,avg_s = float(np.mean(r_a)),float(np.mean(s_a))
        if avg_r<=0 or avg_s<=0 or avg_r/avg_s < 1.6: return False
        pt=insp=0
        for i in range(n):
            if s_arr[i]>=len(sig): continue
            insp+=1
            seg = sig[s_arr[i]:min(len(sig),s_arr[i]+int(0.05*self.fs))]
            if len(seg)<4: continue
            if float(np.mean(np.diff(seg)>0))>0.6: pt+=1
        return insp>0 and pt/insp >= 0.4

    def _is_junctional_rhythm(self, heart_rate, qrs_duration, pr_interval, rr_intervals, p_peaks, r_peaks, p_absent_ratio=None):
        if heart_rate is None or qrs_duration is None: return False
        if not (40 <= heart_rate <= 100) or qrs_duration > 120: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3 or float(np.std(rr)) >= 120: return False
        p_arr = np.asarray(p_peaks, dtype=int); r_arr = np.asarray(r_peaks, dtype=int)
        r_c = max(len(r_arr), len(rr)+1, 1)
        # Primary criterion: P mostly absent (supports nodal/junctional).
        if p_absent_ratio is not None and p_absent_ratio >= 0.55:
            return True
        # Fallback: if PR is short (or P count is low).
        return len(p_arr)/r_c < 0.4 or (pr_interval is not None and pr_interval <= 120)

    def _is_atrial_flutter(self, heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks, p_absent_ratio=None):
        if heart_rate is None or qrs_duration is None: return False
        debug = (80.0 <= float(heart_rate) <= 100.0 and self._flutter_debug_used < 3)
        # Keep HR flexible (flutter can be 2:1/3:1 conduction and appear slow).
        # We avoid mislabeling sinus using P-missingness / P-to-R ratio constraints below.
        if not (45 <= heart_rate <= 220) or qrs_duration > 120:
            return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3: return False
        mean_rr = float(np.mean(rr))
        if mean_rr <= 0: return False
        rr_cv = float(np.std(rr) / mean_rr)
        # Flutter is relatively regular in the conducted ventricular response.
        if rr_cv > 0.30:
            if debug:
                self._flutter_debug_used += 1
                print(f"[FlutterDebug] HR={heart_rate:.1f} rr_cv={rr_cv:.3f} -> fail rr_cv>0.30")
            return False
        p_arr = np.asarray(p_peaks, dtype=int)
        r_arr = np.asarray(r_peaks, dtype=int)
        if len(r_arr) == 0: return False
        if len(r_arr) < 6:  # need enough beats to see flutter pattern
            return False

        p_to_r = float(len(p_arr)) / float(max(len(r_arr), 1))
        min_p_needed = max(2, int(len(r_arr) * 0.10))

        # If we detected no P peaks, decide using P-absence ratio only.
        if len(p_arr) == 0:
            if p_absent_ratio is None:
                return False
            # P is mostly absent but not "AFib-like".
            return (0.35 <= p_absent_ratio <= 0.85)

        # Core differentiation vs AFib/sinus:
        # - If p_absent_ratio is very high -> AFib-like (avoid flutter).
        # - If p_absent_ratio is low (P usually present) -> require higher P-to-R evidence.
        if p_absent_ratio is not None:
            if p_absent_ratio > 0.85:
                return False
            if p_absent_ratio < 0.45 and p_to_r < 1.05:
                if debug:
                    self._flutter_debug_used += 1
                    print(f"[FlutterDebug] HR={heart_rate:.1f} p_abs_ratio={p_absent_ratio:.3f} p_to_r={p_to_r:.3f} -> fail (p_abs<0.45 and p_to_r<1.05)")
                return False
        else:
            # If NK didn't provide p_absent_flags, still require some P-to-R support.
            if p_to_r < 0.75:
                return False

        # With 2:1 flutter, P detection may be incomplete; be a bit tolerant.
        if len(p_arr) < min_p_needed:
            if debug:
                self._flutter_debug_used += 1
                print(f"[FlutterDebug] HR={heart_rate:.1f} len(P)={len(p_arr)} < min_p_needed={min_p_needed} -> fail")
            return False

        ok = p_to_r >= 0.65
        if debug and not ok:
            self._flutter_debug_used += 1
            print(f"[FlutterDebug] HR={heart_rate:.1f} p_to_r={p_to_r:.3f} -> fail p_to_r<0.65")
        return ok

    def _is_av_block(self, pr_interval, p_peaks, r_peaks, rr_intervals, heart_rate):
        p_arr = np.asarray(p_peaks, dtype=int); r_arr = np.asarray(r_peaks, dtype=int)
        if len(p_arr) < 2 or len(r_arr) < 2: return None
        if pr_interval is not None and pr_interval > 200: return "First-Degree AV Block"
        p_c,r_c = len(p_arr),len(r_arr)
        if p_c > r_c*1.2:
            dropped = (p_c-r_c)/max(p_c,1)
            if dropped > 0.5 and len(p_arr)>=3 and len(r_arr)>=3:
                p_iv = np.diff(p_arr)/self.fs*1000.0
                r_iv = np.asarray(rr_intervals,dtype=float)
                if len(r_iv)==0: r_iv = np.diff(r_arr)/self.fs*1000.0
                p_reg = bool(float(np.std(p_iv))<100) if len(p_iv)>0 else False
                r_reg = bool(float(np.std(r_iv))<100) if len(r_iv)>0 else False
                if p_reg and r_reg and heart_rate is not None and heart_rate < 60:
                    return "Third-Degree AV Block (Complete Heart Block)"
            if dropped > 0.2:
                if pr_interval is not None:
                    return ("Second-Degree AV Block (Type I - Wenckebach)"
                            if pr_interval > 180 else "Second-Degree AV Block (Type II)")
                return "Second-Degree AV Block"
        return None

    def _is_high_av_block(self, pr_interval, p_peaks, r_peaks, rr_intervals, heart_rate):
        p_arr = np.asarray(p_peaks, dtype=int); r_arr = np.asarray(r_peaks, dtype=int)
        if len(p_arr)<3 or len(r_arr)<2: return False
        p_c,r_c = len(p_arr),len(r_arr)
        if p_c <= r_c*1.1: return False
        dropped = (p_c-r_c)/max(p_c,1)
        if dropped > 0.5 and len(p_arr)>=3 and len(r_arr)>=3:
            p_iv = np.diff(p_arr)/self.fs*1000.0
            r_iv = np.asarray(rr_intervals,dtype=float)
            if len(r_iv)==0: r_iv = np.diff(r_arr)/self.fs*1000.0
            p_reg = bool(float(np.std(p_iv))<100) if len(p_iv)>0 else False
            r_reg = bool(float(np.std(r_iv))<100) if len(r_iv)>0 else False
            if p_reg and r_reg:
                if heart_rate is not None and heart_rate < 60: return True
                if len(p_iv)>0 and len(r_iv)>0:
                    pr = 60000.0/float(np.mean(p_iv)) if float(np.mean(p_iv))>0 else 0
                    rr2 = 60000.0/float(np.mean(r_iv)) if float(np.mean(r_iv))>0 else 0
                    if abs(pr-rr2)>20: return True
        if dropped>0.25 and pr_interval is not None and pr_interval<=250: return True
        if dropped>0.3: return True
        return False

    def _is_wpw_syndrome(self, pr_interval, qrs_duration, signal, p_peaks, q_peaks, r_peaks):
        # Deprecated single-lead WPW check; use _is_wpw_multilead instead.
        return False

    def _is_atrial_tachycardia(self, heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks):
        if heart_rate is None or qrs_duration is None: return False
        if heart_rate < 100 or qrs_duration > 120: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3: return False
        mean_rr = float(np.mean(rr))
        is_reg = float(np.std(rr))<120 or (mean_rr>0 and float(np.std(rr)/mean_rr)<0.1)
        if not is_reg: return False
        if heart_rate > 150: return True
        return len(np.asarray(p_peaks, dtype=int)) > 0

    def _is_supraventricular_tachycardia(self, heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks):
        if heart_rate is None or qrs_duration is None: return False
        if heart_rate < 150 or qrs_duration > 120: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3: return False
        mean_rr = float(np.mean(rr))
        is_reg = float(np.std(rr))<120 or (mean_rr>0 and float(np.std(rr)/mean_rr)<0.1)
        return is_reg

    # ── New detections ──────────────────────────────────────────────────────

    def _is_poly_vtach(self, signal, r_peaks, rr_ms, qrs_duration):
        if len(rr_ms)<4 or qrs_duration is None: return False
        hr = 60000.0/float(np.mean(rr_ms)); cv = self._rr_cv(rr_ms)
        if hr<100 or qrs_duration<120 or cv<0.12: return False
        sig = np.asarray(signal,dtype=float); r_arr = np.asarray(r_peaks,dtype=int)
        win = int(self.fs*0.1)
        amps = [float(np.ptp(sig[max(0,p-win):min(len(sig),p+win)])) for p in r_arr if p-win>=0]
        if len(amps)<4: return False
        m = float(np.mean(amps))
        return float(np.std(amps)/m)>0.30 if m>0 else False

    def _is_torsade_de_pointes(self, signal, r_peaks, rr_ms, qrs_duration):
        if len(rr_ms)<6 or qrs_duration is None: return False
        hr = 60000.0/float(np.mean(rr_ms))
        if hr<150 or qrs_duration<120: return False
        sig = np.asarray(signal,dtype=float); r_arr = np.asarray(r_peaks,dtype=int)
        win = int(self.fs*0.12)
        amps = [float(np.ptp(sig[max(0,p-win):min(len(sig),p+win)])) for p in r_arr if p-win>=0]
        if len(amps)<6: return False
        amp_arr = np.array(amps); m = float(np.mean(amp_arr))
        amp_cv = float(np.std(amp_arr)/m) if m>0 else 0
        env_peaks,_ = find_peaks(amp_arr, distance=3)
        return amp_cv>0.35 and len(env_peaks)>=2

    def _is_pvc_r_on_t(self, signal, r_peaks, rr_ms, qrs_duration, side='LV'):
        if len(rr_ms)<4 or qrs_duration is None: return False
        mean_rr = float(np.mean(rr_ms)); qt_est = mean_rr*0.42
        return any(rr < qt_est*0.95 and rr < mean_rr*0.80 for rr in rr_ms)

    def _classify_pvcs(self, signal, r_peaks, rr_ms, qrs_duration, p_peaks, q_peaks, s_peaks):
        results = []
        if len(rr_ms)<4 or qrs_duration is None or qrs_duration<100: return results
        mean_rr = float(np.mean(rr_ms)); r_arr = np.asarray(r_peaks,dtype=int)
        sig = np.asarray(signal,dtype=float); win = int(self.fs*0.06)
        ectopic_idx = [i+1 for i,rr in enumerate(rr_ms) if rr < mean_rr*0.82]
        if not ectopic_idx: return results
        morphs=[]; early=0
        for idx in ectopic_idx:
            if idx>=len(r_arr): continue
            pos=r_arr[idx]; sl=sig[max(0,pos-win):min(len(sig),pos+win)]
            if len(sl)<4: continue
            morphs.append('LV' if float(np.max(sl))>abs(float(np.min(sl)))*1.2 else 'RV')
            if idx-1<len(rr_ms) and rr_ms[idx-1]<mean_rr*0.70: early+=1
        if not morphs: return results
        n_morphs=len(set(morphs)); freq=len(ectopic_idx)/max(len(rr_ms),1)
        lv=morphs.count('LV'); rv=morphs.count('RV')
        if n_morphs>=2:
            results.append("Frequent Multi-focal PVCs" if freq>0.2 else "Multi-focal PVCs")
        elif lv>=rv:
            results.append("PVC1 LV Early" if early>0 else "PVC1 Left Ventricle")
        else:
            results.append("PVC2 RV Early" if early>0 else "PVC2 Right Ventricle")
        return results

    def _is_trigeminy(self, rr_ms, qrs_duration, signal, r_peaks):
        if len(rr_ms)<6: return False
        thirds = list(range(2,len(rr_ms),3))
        if not thirds: return False
        other = [i for i in range(len(rr_ms)) if i not in thirds]
        if not other: return False
        mo = float(np.mean([rr_ms[i] for i in other]))
        if mo <= 0: return False
        return sum(1 for i in thirds if rr_ms[i]<mo*0.80)/len(thirds) > 0.60

    def _is_run_of_pvcs(self, signal, r_peaks, rr_ms, qrs_duration):
        if len(rr_ms)<3 or qrs_duration is None or qrs_duration<100: return False
        mean_rr = float(np.mean(rr_ms)); consec=mx=0
        for rr in rr_ms:
            if rr<mean_rr*0.82: consec+=1; mx=max(mx,consec)
            else: consec=0
        return mx>=3

    def _is_pat(self, heart_rate, qrs_duration, rr_ms, p_peaks, r_peaks):
        if heart_rate is None or len(rr_ms)<4: return False
        # PAT is a narrow, often regular SV tachy. In practice your P/QRS
        # delineation can be imperfect, so relax slightly.
        return (130 <= heart_rate <= 250 and self._rr_cv(rr_ms) < 0.15
                and (qrs_duration is None or qrs_duration < 130))

    def _is_pac(self, signal, r_peaks, rr_ms, qrs_duration, p_peaks):
        if len(rr_ms)<4: return False
        if not (qrs_duration is None or qrs_duration<120): return False
        mean_rr = float(np.mean(rr_ms)); n_prem = sum(1 for rr in rr_ms if rr<mean_rr*0.85)
        p_arr = np.asarray(p_peaks,dtype=int)
        return n_prem>=1 and n_prem<len(rr_ms)*0.30 and len(p_arr)>=len(r_peaks)*0.5

    def _is_pnc(self, signal, r_peaks, rr_ms, qrs_duration, p_peaks, pr_ms):
        if len(rr_ms)<4: return False
        narrow = qrs_duration is None or qrs_duration<120
        short_pr = pr_ms is not None and pr_ms<120
        p_arr = np.asarray(p_peaks,dtype=int)
        absent_p = len(p_arr)<len(r_peaks)*0.6
        mean_rr = float(np.mean(rr_ms))
        n_prem = sum(1 for rr in rr_ms if rr<mean_rr*0.88)
        return narrow and (short_pr or absent_p) and n_prem>=1

    def _is_sinus_arrhythmia(self, rr_ms, hr, p_peaks, r_peaks):
        if len(rr_ms)<4 or hr is None: return False
        cv = self._rr_cv(rr_ms); p_arr = np.asarray(p_peaks,dtype=int)
        return 50<=hr<=110 and 0.10<=cv<=0.25 and len(p_arr)>=len(r_peaks)*0.7

    def _is_missed_beat(self, r_peaks, rr_ms, hr):
        if len(rr_ms)<4: return None
        mean_rr = float(np.mean(rr_ms))
        # Detect a "pause" as one interval substantially longer than the mean.
        # NeuroKit2/our R detector may slightly shift rr, so we use a tolerant multiplier.
        pause_mult = 1.6
        for rr in rr_ms:
            if rr > mean_rr * pause_mult:
                # HR is derived from mean_rr, so these ranges correspond to expected
                # nominal rates for the pause pattern you want to see.
                if 60 <= hr <= 105:
                    return "Missed Beat at ~80 BPM (SA Block / Sinus Pause)"
                if 95 <= hr <= 150:
                    return "Missed Beat at ~120 BPM (SA Block / Sinus Pause)"
                return "Sinus Pause / Missed Beat"
        return None

    def _compute_hrv(self, rr_ms):
        rr = np.asarray(rr_ms, dtype=float)
        if rr.size < 3:
            return {"sdnn_ms": 0.0, "rmssd_ms": 0.0, "pnn50_pct": 0.0}
        diffs = np.diff(rr)
        sdnn = float(np.std(rr))
        rmssd = float(np.sqrt(np.mean(diffs ** 2))) if diffs.size else 0.0
        pnn50 = float(np.mean(np.abs(diffs) > 50.0) * 100.0) if diffs.size else 0.0
        return {"sdnn_ms": sdnn, "rmssd_ms": rmssd, "pnn50_pct": pnn50}

    def _is_qtc_prolonged(self, q_peaks, t_peaks, rr_ms):
        q_arr = np.asarray(q_peaks, dtype=int)
        t_arr = np.asarray(t_peaks, dtype=int)
        rr = np.asarray(rr_ms, dtype=float)
        if q_arr.size == 0 or t_arr.size == 0 or rr.size == 0:
            return False
        qt_vals = []
        for q in q_arr:
            t_after = t_arr[t_arr > q]
            if t_after.size:
                qt_ms = (t_after[0] - q) / self.fs * 1000.0
                if 200 <= qt_ms <= 700:
                    qt_vals.append(qt_ms)
        if not qt_vals:
            return False
        qt = float(np.mean(qt_vals))
        rr_mean = float(np.mean(rr))
        if rr_mean <= 0:
            return False
        qtc = qt / np.sqrt(rr_mean / 1000.0)
        return qtc > 460.0

    def _is_st_change(self, signal, q_peaks, s_peaks, p_peaks):
        sig = np.asarray(signal, dtype=float)
        s_arr = np.asarray(s_peaks, dtype=int)
        p_arr = np.asarray(p_peaks, dtype=int)
        if sig.size == 0 or s_arr.size == 0 or p_arr.size == 0:
            return False
        offsets = []
        j_shift = int(0.08 * self.fs)
        for s in s_arr[:min(10, len(s_arr))]:
            j_idx = s + j_shift
            if j_idx >= len(sig):
                continue
            base_idx = p_arr[p_arr < s]
            if base_idx.size == 0:
                continue
            b = base_idx[-1]
            b0, b1 = max(0, b - int(0.04 * self.fs)), min(len(sig), b)
            if b1 - b0 < 3:
                continue
            baseline = float(np.mean(sig[b0:b1]))
            offsets.append(float(sig[j_idx] - baseline))
        if not offsets:
            return False
        st_mean = float(np.mean(offsets))
        # Convert ST thresholds from ADC counts to mV.
        cpmv = self.counts_per_mv if self.counts_per_mv else 1.0
        return st_mean < (-80.0 / cpmv) or st_mean > (120.0 / cpmv)

    def _is_t_wave_inversion(self, signal, t_peaks, r_peaks):
        sig = np.asarray(signal, dtype=float)
        t_arr = np.asarray(t_peaks, dtype=int)
        r_arr = np.asarray(r_peaks, dtype=int)
        if sig.size == 0 or t_arr.size == 0 or r_arr.size == 0:
            return False
        inv = 0
        total = 0
        for t in t_arr[:min(12, len(t_arr))]:
            r_before = r_arr[r_arr < t]
            if r_before.size == 0 or t >= len(sig):
                continue
            r_idx = r_before[-1]
            r_amp = float(sig[r_idx])
            t_amp = float(sig[t])
            if abs(r_amp) < 1e-6:
                continue
            total += 1
            if np.sign(t_amp) != np.sign(r_amp):
                inv += 1
        return total >= 3 and inv / total >= 0.5

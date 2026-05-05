"""
ecg/holter/analysis_worker.py
==============================
Background thread that pulls 30-second data chunks from HolterStreamWriter's
queue and runs the full metric pipeline on them.

Uses existing codebase APIs:
  - ecg_calculations.calculate_all_ecg_metrics()
  - arrhythmia_detector.ArrhythmiaDetector.detect_arrhythmias()
  - signal_quality.calculate_signal_quality_index()

Output: one JSON line per chunk appended to metrics.jsonl
"""

import sys
import os
import json
import time
import queue
import threading
import traceback
import numpy as np
from typing import Optional, Callable

# Add project root to sys.path so imports work
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class HolterAnalysisWorker(threading.Thread):
    """
    Runs as a daemon thread. Pulls 30-sec chunks from the queue,
    runs analysis, writes metrics to JSONL.
    """

    def __init__(self,
                 analysis_queue: queue.Queue,
                 on_chunk_done: Optional[Callable] = None,
                 on_arrhythmia: Optional[Callable] = None,
                 fs: int = 500):
        super().__init__(daemon=True, name="HolterAnalysisWorker")
        self.analysis_queue = analysis_queue
        self.on_chunk_done = on_chunk_done    # callback(chunk_result_dict)
        self.on_arrhythmia = on_arrhythmia    # callback(arrhythmia_list, timestamp)
        self.fs = fs
        self._stop_event = threading.Event()

        # Import existing ECG analysis modules
        self._ecg_calc = None
        self._arrhythmia_detector = None
        self._load_modules()

    def _load_modules(self):
        try:
            from ecg.ecg_calculations import calculate_all_ecg_metrics
            self._calc_metrics = calculate_all_ecg_metrics
        except Exception as e:
            print(f"[HolterWorker] WARNING: could not import calculate_all_ecg_metrics: {e}")
            self._calc_metrics = None

        try:
            from ecg.arrhythmia_detector import ArrhythmiaDetector
            self._arrhythmia_detector = ArrhythmiaDetector(sampling_rate=self.fs)
        except Exception as e:
            print(f"[HolterWorker] WARNING: could not import ArrhythmiaDetector: {e}")

        try:
            from ecg.signal_quality import calculate_signal_quality_index
            self._calc_sqi = calculate_signal_quality_index
        except Exception as e:
            self._calc_sqi = None

        try:
            from ecg.holter.analysis_pipeline import (
                ArrhythmiaDetector as RuleArrhythmiaDetector,
                HolterConfig,
                MultiLeadSelector,
                QRSValidator,
                SignalPreprocessor,
                SignalQuality,
                TemplateCluster,
                clean_rr,
            )
            self._pipeline_config = HolterConfig()
            self._preprocessor = SignalPreprocessor(fs=self.fs, config=self._pipeline_config)
            self._signal_quality = SignalQuality(fs=self.fs)
            self._lead_selector = MultiLeadSelector(fs=self.fs, config=self._pipeline_config)
            self._qrs_validator = QRSValidator()
            self._template_cluster = TemplateCluster(similarity_threshold=0.90)
            self._clean_rr = clean_rr
            self._rule_arrhythmia_detector = RuleArrhythmiaDetector(config=self._pipeline_config)
        except Exception as e:
            print(f"[HolterWorker] WARNING: could not initialize Holter pipeline helpers: {e}")
            self._pipeline_config = None
            self._preprocessor = None
            self._signal_quality = None
            self._lead_selector = None
            self._qrs_validator = None
            self._template_cluster = None
            self._clean_rr = None
            self._rule_arrhythmia_detector = None

        try:
            from scipy.signal import find_peaks
            self._find_peaks = find_peaks
        except Exception:
            self._find_peaks = None

    # ── Main thread loop ───────────────────────────────────────────────────────

    def run(self):
        print("[HolterWorker] Analysis thread started")
        while not self._stop_event.is_set():
            try:
                item = self.analysis_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            if item is None:    # sentinel → stop
                break

            try:
                self._process_chunk(item)
            except Exception as e:
                print(f"[HolterWorker] Chunk analysis error: {e}")
                traceback.print_exc()

            self.analysis_queue.task_done()

        print("[HolterWorker] Analysis thread stopped")

    def stop(self):
        self._stop_event.set()

    # ── Chunk processing ───────────────────────────────────────────────────────

    def _process_chunk(self, item: dict):
        """
        Analyze one 30-second chunk.
        item keys: data (12xN ndarray), start_sec, fs, partial, jsonl_path
        """
        t0 = time.time()
        data: np.ndarray = item["data"]
        start_sec: float = item["start_sec"]
        fs: int = item.get("fs", self.fs)
        jsonl_path: str = item["jsonl_path"]
        partial: bool = item.get("partial", False)

        lead_ii = data[1] if data.shape[0] > 1 else data[0]
        n_samples = lead_ii.shape[0]
        if n_samples < fs * 5:
            return

        result = {
            "t": round(start_sec, 2),
            "duration": round(n_samples / fs, 2),
            "partial": partial,
        }

        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        lead_map = {lead_names[idx]: data[idx] for idx in range(min(len(lead_names), data.shape[0]))}

        best_lead_name = "II" if data.shape[0] > 1 else (lead_names[0] if data.shape[0] > 0 else "")
        best_signal = np.asarray(lead_ii, dtype=float)
        lead_scores = {}

        if self._lead_selector is not None:
            try:
                selected_name, selected_signal, lead_scores = self._lead_selector.select_best_lead(lead_map)
                if selected_signal is not None and selected_signal.size > 0:
                    best_lead_name = selected_name or best_lead_name
                    best_signal = np.asarray(selected_signal, dtype=float)
            except Exception as e:
                print(f"[HolterWorker] Lead selection error: {e}")
        elif self._preprocessor is not None:
            try:
                best_signal = self._preprocessor.process(best_signal)
            except Exception:
                pass

        result["best_lead"] = best_lead_name
        if lead_scores:
            result["lead_quality_scores"] = {k: round(float(v), 3) for k, v in lead_scores.items()}

        sqi_score = 1.0
        try:
            if self._signal_quality is not None:
                sqi_score = float(self._signal_quality.compute_sqi(best_signal))
            elif self._calc_sqi is not None:
                sqi_score = float(self._calc_sqi(best_signal, np.array([], dtype=int), sampling_rate=float(fs)))
        except Exception:
            sqi_score = 0.0
        result["quality"] = round(float(sqi_score), 3)

        quality_threshold = 0.6
        if self._pipeline_config is not None:
            quality_threshold = float(getattr(self._pipeline_config, "sqi_min", quality_threshold))
        if sqi_score < quality_threshold:
            result.update({
                "rejected": True,
                "hr_mean": 0.0,
                "hr_min": 0.0,
                "hr_max": 0.0,
                "beat_count": 0,
                "n_beats": 0,
                "rr_ms": 0.0,
                "pr_ms": 0.0,
                "qrs_ms": 0.0,
                "qt_ms": 0.0,
                "qtc_ms": 0.0,
                "rr_intervals_list": [],
                "rr_std": 0.0,
                "rmssd": 0.0,
                "pnn50": 0.0,
                "longest_rr": 0.0,
                "pauses": 0,
                "arrhythmias": [],
                "beat_class_counts": {},
                "template_summary": [],
                "classified_events": [],
                "template_count": 0,
                "ve_beats": 0,
                "sve_beats": 0,
                "brady_beats": 0,
                "tachy_beats": 0,
                "st_mv": 0.0,
            })
            result["analysis_ms"] = round((time.time() - t0) * 1000, 1)
            try:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"[HolterWorker] JSONL write error: {e}")
            if self.on_chunk_done:
                try:
                    self.on_chunk_done(result)
                except Exception:
                    pass
            return

        try:
            if self._calc_metrics is not None:
                metrics = self._calc_metrics(best_signal, fs=float(fs), instance_id=f"holter_{start_sec:.0f}")
                result["hr_mean"] = round(float(metrics.get("heart_rate", 0)), 1)
                result["rr_ms"] = round(float(metrics.get("rr_interval", 0)), 1)
                result["pr_ms"] = round(float(metrics.get("pr_interval", 0)), 1)
                result["qrs_ms"] = round(float(metrics.get("qrs_duration", 0)), 1)
                result["qt_ms"] = round(float(metrics.get("qt_interval", 0)), 1)
                result["qtc_ms"] = round(float(metrics.get("qtc_interval", 0)), 1)
            else:
                result["hr_mean"] = 0.0
                result["rr_ms"] = result["pr_ms"] = result["qrs_ms"] = 0.0
                result["qt_ms"] = result["qtc_ms"] = 0.0
        except Exception as e:
            print(f"[HolterWorker] Metrics error at t={start_sec:.0f}s: {e}")
            result["hr_mean"] = 0.0
            result["rr_ms"] = result["pr_ms"] = result["qrs_ms"] = 0.0
            result["qt_ms"] = result["qtc_ms"] = 0.0

        try:
            r_peaks, rr_intervals, per_beat_hr = self._detect_r_peaks(best_signal, fs)
            rr_intervals = self._clean_rr(rr_intervals) if self._clean_rr is not None else rr_intervals
            result["hr_min"] = round(float(np.min(per_beat_hr)), 1) if len(per_beat_hr) else 0.0
            result["hr_max"] = round(float(np.max(per_beat_hr)), 1) if len(per_beat_hr) else 0.0
            result["beat_count"] = len(r_peaks)
            result["rr_intervals_list"] = [round(float(r), 1) for r in rr_intervals]

            if len(rr_intervals) >= 3:
                result["rr_std"] = round(float(np.std(rr_intervals)), 1)
                diff_rr = np.diff(rr_intervals)
                result["rmssd"] = round(float(np.sqrt(np.mean(diff_rr ** 2))), 1)
                nn50 = int(np.sum(np.abs(diff_rr) > 50))
                result["pnn50"] = round(100.0 * nn50 / len(diff_rr), 2) if len(diff_rr) > 0 else 0.0
                result["longest_rr"] = round(float(np.max(rr_intervals)), 1)
                result["pauses"] = int(np.sum(rr_intervals > 2000))
            else:
                result["rr_std"] = result["rmssd"] = result["pnn50"] = 0.0
                result["longest_rr"] = 0.0
                result["pauses"] = 0
        except Exception as e:
            print(f"[HolterWorker] R-peak/HRV error: {e}")
            r_peaks, rr_intervals = np.array([]), np.array([])
            result.update({"hr_min": 0, "hr_max": 0, "beat_count": 0,
                           "rr_std": 0, "rmssd": 0, "pnn50": 0,
                           "longest_rr": 0, "pauses": 0})

        arrhythmias = []
        try:
            if self._arrhythmia_detector is not None and len(r_peaks) >= 3:
                analysis_dict = {"r_peaks": r_peaks, "p_peaks": [], "q_peaks": [],
                                 "s_peaks": [], "t_peaks": []}
                arrhythmias = self._arrhythmia_detector.detect_arrhythmias(
                    best_signal, analysis_dict,
                    has_received_serial_data=True,
                    min_serial_data_packets=50
                )
                arrhythmias = [a for a in arrhythmias if "Insufficient" not in a and "No " not in a]
            if not arrhythmias and self._rule_arrhythmia_detector is not None and len(r_peaks) >= 3:
                arrhythmias = self._rule_arrhythmia_detector.detect(
                    rr_intervals,
                    qrs_widths_ms=[result.get("qrs_ms", 0.0)] if result.get("qrs_ms", 0.0) else None,
                    beat_count=len(r_peaks),
                )
        except Exception as e:
            print(f"[HolterWorker] Arrhythmia error: {e}")

        result["arrhythmias"] = arrhythmias
        result["n_beats"] = len(r_peaks)
        result["brady_beats"] = int(np.sum(rr_intervals > 1500)) if len(rr_intervals) else 0
        result["tachy_beats"] = int(np.sum(rr_intervals < 400)) if len(rr_intervals) else 0
        try:
            beat_classes, template_summary, classified_events = self._classify_beats(
                lead_ii=best_signal,
                r_peaks=r_peaks,
                rr_intervals=rr_intervals,
                fs=fs,
                chunk_start_sec=float(start_sec),
            )
        except Exception as e:
            print(f"[HolterWorker] Beat classification error: {e}")
            beat_classes, template_summary, classified_events = ({}, [], [])
        result["beat_class_counts"] = beat_classes
        result["template_summary"] = template_summary
        result["classified_events"] = classified_events
        result["template_count"] = len(template_summary)
        result["ve_beats"] = int(beat_classes.get("VE", 0))
        result["sve_beats"] = int(beat_classes.get("SVE", 0))

        try:
            result["st_mv"] = round(self._estimate_st(best_signal, r_peaks, fs), 4)
        except Exception:
            result["st_mv"] = 0.0

        try:
            if self._calc_sqi is not None and len(r_peaks) >= 3:
                sqi = self._calc_sqi(best_signal, r_peaks, sampling_rate=float(fs))
                result["quality"] = round(float(sqi), 3)
        except Exception:
            pass

        result["analysis_ms"] = round((time.time() - t0) * 1000, 1)
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            print(f"[HolterWorker] JSONL write error: {e}")

        if self.on_chunk_done:
            try:
                self.on_chunk_done(result)
            except Exception:
                pass

        if arrhythmias and self.on_arrhythmia:
            try:
                self.on_arrhythmia(arrhythmias, start_sec)
            except Exception:
                pass

        if result.get("hr_mean", 0) > 0:
            print(f"[HolterWorker] t={start_sec:.0f}s | HR={result['hr_mean']:.0f} | Beats={result['beat_count']} | Arrhy={arrhythmias} | took {result['analysis_ms']:.0f}ms")

    def _detect_r_peaks(self, lead_ii: np.ndarray, fs: int):
        """Simple Pan-Tompkins-inspired R-peak detection for HRV/classification."""
        from scipy.signal import butter, filtfilt, find_peaks
        b, a = butter(2, [5 / (fs / 2), 20 / (fs / 2)], btype='band')
        filtered = filtfilt(b, a, lead_ii)
        squared = filtered ** 2
        window = int(0.15 * fs)
        kernel = np.ones(window) / window
        mwa = np.convolve(squared, kernel, mode='same')
        threshold = np.mean(mwa) * 0.5
        min_dist = int(0.3 * fs)
        r_peaks, _ = find_peaks(mwa, height=threshold, distance=min_dist)

        rr_intervals = np.diff(r_peaks) / fs * 1000
        per_beat_hr = np.array([])
        if len(rr_intervals) > 0:
            per_beat_hr = 60000.0 / rr_intervals
        return r_peaks, rr_intervals, per_beat_hr


    def _estimate_st(self, lead_ii: np.ndarray, r_peaks: np.ndarray, fs: int) -> float:
        """Estimate average ST deviation (mV) from J+80ms point."""
        if len(r_peaks) < 3:
            return 0.0
        # ADC to mV: assume ADC center=2048, gain=1000 (1 mV = 1000 ADC units approx)
        adc_scale = 1.0 / 200.0   # rough conversion
        j_offset = int(0.08 * fs)  # J-point ~80ms after R
        st_offset = j_offset + int(0.02 * fs)  # ST measurement at J+20ms
        baseline_offset = -int(0.15 * fs)       # PR segment baseline

        st_vals = []
        for r in r_peaks[1:-1]:
            st_idx = r + st_offset
            bl_idx = r + baseline_offset
            if 0 <= bl_idx < len(lead_ii) and 0 <= st_idx < len(lead_ii):
                bl = lead_ii[bl_idx]
                st = lead_ii[st_idx]
                st_vals.append((st - bl) * adc_scale)

        return float(np.median(st_vals)) if st_vals else 0.0

    def _classify_beats(
        self,
        lead_ii: np.ndarray,
        r_peaks: np.ndarray,
        rr_intervals: np.ndarray,
        fs: int,
        chunk_start_sec: float,
    ):
        """
        Classify beats and build lightweight template clusters for review UI.
        Returns class_counts, template_summary, classified_events.
        """
        if len(r_peaks) == 0:
            return {}, [], []

        pre = int(0.16 * fs)
        post = int(0.24 * fs)
        rr_ms = rr_intervals if len(rr_intervals) else np.array([], dtype=float)
        rr_median = float(np.median(rr_ms)) if len(rr_ms) else 800.0

        class_counts = {'N': 0, 'VE': 0, 'SVE': 0, 'Brady': 0, 'Tachy': 0, 'Pause': 0}
        beat_records = []
        classified_events = []

        for idx, r in enumerate(r_peaks):
            i0 = int(r) - pre
            i1 = int(r) + post
            if i0 < 0 or i1 >= len(lead_ii):
                continue

            segment = np.asarray(lead_ii[i0:i1], dtype=np.float32)
            if segment.size < 8:
                continue

            rr_prev = float(rr_ms[idx - 1]) if idx - 1 >= 0 and idx - 1 < len(rr_ms) else rr_median
            qrs_width_ms = self._estimate_qrs_width_ms(segment, fs)
            qrs_amp = float(np.ptp(segment)) if segment.size else 0.0
            if self._qrs_validator is not None and not self._qrs_validator.is_valid({
                'width_ms': qrs_width_ms,
                'amplitude': qrs_amp,
                'qrs_ms': qrs_width_ms,
                'qrs_amplitude': qrs_amp,
            }):
                continue

            label = self._label_beat(rr_prev, rr_median, qrs_width_ms)
            class_counts[label] = class_counts.get(label, 0) + 1

            t_sec = float(chunk_start_sec + (float(r) / float(fs)))
            beat_records.append({
                'segment': segment,
                'label': label,
                'timestamp': round(t_sec, 3),
                'rr_ms': round(rr_prev, 1),
                'qrs_ms': round(qrs_width_ms, 1),
            })
            if label != 'N':
                classified_events.append({
                    'timestamp': round(t_sec, 3),
                    'label': self._event_label_from_class(label),
                    'template_label': label,
                    'rr_ms': round(rr_prev, 1),
                    'qrs_ms': round(qrs_width_ms, 1),
                })

        class_counts = {k: int(v) for k, v in class_counts.items() if v > 0}
        template_rows = []
        if beat_records and self._template_cluster is not None:
            try:
                clusters = self._template_cluster.cluster(beat_records)
            except Exception as e:
                print(f"[HolterWorker] Template clustering error: {e}")
                clusters = []
            for i, cluster in enumerate(sorted(clusters, key=lambda item: item.get('count', 0), reverse=True), start=1):
                beats = list(cluster.get('beats', []) or [])
                if not beats:
                    continue
                template_rows.append({
                    'template_id': f'T{i}',
                    'template_key': f'C{i:02d}',
                    'label': str(beats[0].get('label', 'N')),
                    'count': int(cluster.get('count', len(beats))),
                    'avg_rr_ms': round(float(np.mean([b.get('rr_ms', 0.0) for b in beats])) if beats else 0.0, 1),
                    'avg_qrs_ms': round(float(np.mean([b.get('qrs_ms', 0.0) for b in beats])) if beats else 0.0, 1),
                    'first_timestamp': round(float(min(b.get('timestamp', 0.0) for b in beats)), 3),
                })

        return class_counts, template_rows, classified_events

    @staticmethod
    def _estimate_qrs_width_ms(segment: np.ndarray, fs: int) -> float:
        centered = segment - float(np.median(segment))
        peak = float(np.max(np.abs(centered))) if centered.size else 0.0
        if peak <= 1e-6:
            return 0.0
        active = np.where(np.abs(centered) >= 0.35 * peak)[0]
        if active.size == 0:
            return 0.0
        width_samples = int(active[-1] - active[0] + 1)
        return float(width_samples) * 1000.0 / float(fs)

    @staticmethod
    def _label_beat(rr_prev_ms: float, rr_median_ms: float, qrs_width_ms: float) -> str:
        if rr_prev_ms > 2000:
            return 'Pause'
        hr_prev = 60000.0 / max(rr_prev_ms, 1.0)
        if hr_prev < 50:
            return 'Brady'
        if hr_prev > 130:
            return 'Tachy'
        if qrs_width_ms >= 130 and rr_prev_ms <= rr_median_ms * 0.95:
            return 'VE'
        if rr_prev_ms < rr_median_ms * 0.8 and qrs_width_ms < 130:
            return 'SVE'
        return 'N'

    @staticmethod
    def _event_label_from_class(label: str) -> str:
        mapping = {
            'VE': 'PVC Candidate',
            'SVE': 'PAC Candidate',
            'Brady': 'Brady Episode',
            'Tachy': 'Tachy Episode',
            'Pause': 'Pause Episode',
        }
        return mapping.get(label, label)

    @staticmethod
    def _template_key(label: str, segment: np.ndarray, qrs_width_ms: float) -> str:
        centered = segment - float(np.median(segment))
        r_sign = '+' if float(np.max(centered)) >= abs(float(np.min(centered))) else '-'
        width_bucket = int(round(qrs_width_ms / 20.0) * 20)
        return f"{label}-{r_sign}-QRS{width_bucket:03d}"

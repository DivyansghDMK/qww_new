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
        item keys: data (12×N ndarray), start_sec, fs, partial, jsonl_path
        """
        t0 = time.time()
        data: np.ndarray = item['data']          # shape (12, N)
        start_sec: float = item['start_sec']
        fs: int = item.get('fs', self.fs)
        jsonl_path: str = item['jsonl_path']
        partial: bool = item.get('partial', False)

        # Lead II is index 1 — primary analysis lead
        lead_ii = data[1] if data.shape[0] > 1 else data[0]
        n_samples = lead_ii.shape[0]

        if n_samples < fs * 5:   # skip tiny partial chunks < 5 seconds
            return

        result = {
            't': round(start_sec, 2),
            'duration': round(n_samples / fs, 2),
            'partial': partial,
        }

        # ── 1. Core ECG metrics ────────────────────────────────────────────────
        try:
            if self._calc_metrics is not None:
                metrics = self._calc_metrics(lead_ii, fs=float(fs),
                                             instance_id=f'holter_{start_sec:.0f}')
                result['hr_mean'] = round(float(metrics.get('heart_rate', 0)), 1)
                result['rr_ms']   = round(float(metrics.get('rr_interval', 0)), 1)
                result['pr_ms']   = round(float(metrics.get('pr_interval', 0)), 1)
                result['qrs_ms']  = round(float(metrics.get('qrs_duration', 0)), 1)
                result['qt_ms']   = round(float(metrics.get('qt_interval', 0)), 1)
                result['qtc_ms']  = round(float(metrics.get('qtc_interval', 0)), 1)
            else:
                result['hr_mean'] = 0.0
                result['rr_ms'] = result['pr_ms'] = result['qrs_ms'] = 0.0
                result['qt_ms'] = result['qtc_ms'] = 0.0
        except Exception as e:
            print(f"[HolterWorker] Metrics error at t={start_sec:.0f}s: {e}")
            result['hr_mean'] = 0.0
            result['rr_ms'] = result['pr_ms'] = result['qrs_ms'] = 0.0
            result['qt_ms'] = result['qtc_ms'] = 0.0

        # ── 2. Per-beat HR (min/max) ───────────────────────────────────────────
        try:
            r_peaks, rr_intervals, per_beat_hr = self._detect_r_peaks(lead_ii, fs)
            result['hr_min'] = round(float(np.min(per_beat_hr)), 1) if len(per_beat_hr) else 0.0
            result['hr_max'] = round(float(np.max(per_beat_hr)), 1) if len(per_beat_hr) else 0.0
            result['beat_count'] = len(r_peaks)
            result['rr_intervals_list'] = [round(float(r), 1) for r in rr_intervals]

            # HRV metrics
            if len(rr_intervals) >= 3:
                result['rr_std']  = round(float(np.std(rr_intervals)), 1)
                diff_rr = np.diff(rr_intervals)
                result['rmssd']   = round(float(np.sqrt(np.mean(diff_rr ** 2))), 1)
                nn50 = int(np.sum(np.abs(diff_rr) > 50))
                result['pnn50']   = round(100.0 * nn50 / len(diff_rr), 2) if len(diff_rr) > 0 else 0.0
                result['longest_rr'] = round(float(np.max(rr_intervals)), 1)
                # Detect pauses (RR > 2000 ms)
                result['pauses'] = int(np.sum(rr_intervals > 2000))
            else:
                result['rr_std'] = result['rmssd'] = result['pnn50'] = 0.0
                result['longest_rr'] = 0.0
                result['pauses'] = 0

        except Exception as e:
            print(f"[HolterWorker] R-peak/HRV error: {e}")
            r_peaks, rr_intervals = np.array([]), np.array([])
            result.update({'hr_min': 0, 'hr_max': 0, 'beat_count': 0,
                           'rr_std': 0, 'rmssd': 0, 'pnn50': 0,
                           'longest_rr': 0, 'pauses': 0})

        # ── 3. Arrhythmia detection ───────────────────────────────────────────
        arrhythmias = []
        try:
            if self._arrhythmia_detector is not None and len(r_peaks) >= 3:
                analysis_dict = {'r_peaks': r_peaks, 'p_peaks': [], 'q_peaks': [],
                                 's_peaks': [], 't_peaks': []}
                arrhythmias = self._arrhythmia_detector.detect_arrhythmias(
                    lead_ii, analysis_dict,
                    has_received_serial_data=True,
                    min_serial_data_packets=50
                )
                arrhythmias = [a for a in arrhythmias
                               if 'Insufficient' not in a and 'No ' not in a]
        except Exception as e:
            print(f"[HolterWorker] Arrhythmia error: {e}")

        result['arrhythmias'] = arrhythmias

        # ── 4. Beat classification counts ─────────────────────────────────────
        result['n_beats'] = len(r_peaks)
        brady = int(np.sum(rr_intervals > 1500)) if len(rr_intervals) else 0
        tachy = int(np.sum(rr_intervals < 400)) if len(rr_intervals) else 0
        result['brady_beats'] = brady
        result['tachy_beats'] = tachy

        # ── 5. ST deviation (simplified) ──────────────────────────────────────
        try:
            st_dev = self._estimate_st(lead_ii, r_peaks, fs)
            result['st_mv'] = round(st_dev, 4)
        except Exception:
            result['st_mv'] = 0.0

        # ── 6. Signal quality ─────────────────────────────────────────────────
        try:
            if self._calc_sqi is not None and len(r_peaks) >= 3:
                sqi = self._calc_sqi(lead_ii, r_peaks, sampling_rate=float(fs))
                result['quality'] = round(float(sqi), 3)
            else:
                result['quality'] = 1.0
        except Exception:
            result['quality'] = 1.0

        # ── 7. Write to JSONL ─────────────────────────────────────────────────
        result['analysis_ms'] = round((time.time() - t0) * 1000, 1)
        try:
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
        except Exception as e:
            print(f"[HolterWorker] JSONL write error: {e}")

        # ── 8. Callbacks ──────────────────────────────────────────────────────
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

        if result.get('hr_mean', 0) > 0:
            print(f"[HolterWorker] t={start_sec:.0f}s | "
                  f"HR={result['hr_mean']:.0f} | "
                  f"Beats={result['beat_count']} | "
                  f"Arrhy={arrhythmias} | "
                  f"took {result['analysis_ms']:.0f}ms")

    # ── Helper: R-peak detection ───────────────────────────────────────────────

    def _detect_r_peaks(self, lead_ii: np.ndarray, fs: int):
        """Simple Pan-Tompkins-inspired R-peak detection for HRV/classification."""
        from scipy.signal import butter, filtfilt, find_peaks
        # Bandpass 5–20 Hz
        b, a = butter(2, [5 / (fs / 2), 20 / (fs / 2)], btype='band')
        filtered = filtfilt(b, a, lead_ii)
        squared = filtered ** 2
        # Moving average
        window = int(0.15 * fs)
        kernel = np.ones(window) / window
        mwa = np.convolve(squared, kernel, mode='same')
        threshold = np.mean(mwa) * 0.5
        min_dist = int(0.3 * fs)
        r_peaks, _ = find_peaks(mwa, height=threshold, distance=min_dist)

        rr_intervals = np.diff(r_peaks) / fs * 1000  # ms
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

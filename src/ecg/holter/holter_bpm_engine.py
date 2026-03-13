"""
ecg/holter/holter_bpm_engine.py
================================
ALL Holter BPM logic in ONE file.

This file consolidates every piece of the Holter BPM pipeline:
  1.  HolterBPMCalculator  — calculates BPM from a data chunk (stateless)
  2.  HolterBPMStore       — thread-safe store for the current live BPM
  3.  HolterBPMWorker      — background thread: accumulates chunks, calculates, stores
  4.  HolterBPMDisplay     — Qt widget that SHOWS the BPM (green label + REC bar)
  5.  HolterBPMController  — wires everything together; call from any test window

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW BPM FLOWS (end-to-end):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Hardware Packet] ──push()──► [HolterBPMWorker queue]
                                        │
                         every CHUNK_SECONDS ───▼
                              [HolterBPMCalculator]
                              • bandpass Lead II (5–20 Hz)
                              • square + moving-average envelope
                              • find_peaks → R-peaks
                              • rr_intervals → median RR → BPM
                                        │
                               stores ─▼
                              [HolterBPMStore._live_bpm]   (thread-safe)
                                        │
                        Qt timer (3 s) ─▼
                              [HolterBPMDisplay.refresh()]
                              • reads _live_bpm from store
                              • calls label.setText(f"{bpm:.0f}")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USAGE in twelve_lead_test.py / hrv_test.py / hyperkalemia_test.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from ecg.holter.holter_bpm_engine import HolterBPMController

    # On test Start:
    self._bpm_ctrl = HolterBPMController(parent_widget=self, fs=500)
    self._bpm_ctrl.start(target_hours=0)  # target_hours=0 → compact mode
    layout.addWidget(self._bpm_ctrl.display_bar)

    # In packet loop (called 500× per second):
    self._bpm_ctrl.push(packet)  # dict: {"I": val, "II": val, ...}

    # On test Stop:
    self._bpm_ctrl.stop()
    self._bpm_ctrl.display_bar.hide()

    # Read latest BPM at any time (from main thread):
    bpm = self._bpm_ctrl.current_bpm()
"""

import os
import sys
import time
import json
import queue
import threading
import traceback
import numpy as np
from typing import Optional, List

# ── PyQt5 ──────────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QProgressBar, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

# ── Lead order (must match the hardware packet dict keys) ──────────────────────
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_II_IDX = 1          # Lead II is always index 1

# Sliding window configuration
WINDOW_SECONDS  = 10   # How many seconds of data to use per BPM calculation
CALC_INTERVAL   = 2.0  # Recalculate every N seconds (fast response to HR changes)
CHUNK_SECONDS   = WINDOW_SECONDS  # kept for backward-compat import
FS_DEFAULT      = 500

# Colours
COL_BG        = "#0D1117"
COL_RED       = "#B71C1C"
COL_ORANGE    = "#E65100"
COL_GREEN_ECG = "#00FF00"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  HolterBPMCalculator
#     Pure function: given a Lead-II numpy array + fs, returns BPM (float).
# ══════════════════════════════════════════════════════════════════════════════
class HolterBPMCalculator:
    """
    Stateless helper. Call calculate(lead_ii, fs) → BPM.

    Algorithm:
      1. Bandpass 5–20 Hz
      2. Square the signal
      3. Moving-average smoothing (150 ms window)
      4. find_peaks on the envelope → R-peaks
      5. RR intervals → filter physiological range (200–2000 ms = 30–300 BPM)
      6. median(RR) → BPM
    """

    def __init__(self, fs: int = FS_DEFAULT):
        self.fs = fs

    def calculate(self, lead_ii: np.ndarray, fs: int = None) -> float:
        """Returns heart rate in BPM, or 0.0 on failure."""
        fs = fs or self.fs
        if len(lead_ii) < fs * 3:   # need at least 3 seconds
            return 0.0

        try:
            from scipy.signal import butter, filtfilt, find_peaks

            # Step 1 – bandpass 5–20 Hz
            low  = max(0.01, min(5  / (fs / 2), 0.99))
            high = max(0.01, min(20 / (fs / 2), 0.99))
            if low >= high:
                high = min(low + 0.01, 0.99)
            b, a = butter(2, [low, high], btype='band')
            filtered = filtfilt(b, a, lead_ii.astype(float))

            # Step 2 – square
            squared = filtered ** 2

            # Step 3 – moving-average envelope (150 ms window)
            window   = max(1, int(0.15 * fs))
            kernel   = np.ones(window) / window
            envelope = np.convolve(squared, kernel, mode='same')

            # Step 4 – find R-peaks (min_dist = 120 ms)
            min_dist  = max(1, int(0.12 * fs))
            threshold = np.mean(envelope) * 0.4
            r_peaks, _ = find_peaks(envelope, height=threshold, distance=min_dist)

            if len(r_peaks) < 2:
                return 0.0

            # Step 5 – RR intervals, filter 200–2000 ms
            rr_ms = np.diff(r_peaks) / fs * 1000.0
            valid = rr_ms[(rr_ms >= 200) & (rr_ms <= 2000)]

            if len(valid) == 0:
                return 0.0

            # Step 6 – median BPM (last 12 intervals for recency)
            recent = valid[-12:] if len(valid) > 12 else valid
            bpm    = 60000.0 / np.median(recent)
            return float(np.clip(round(bpm, 1), 30, 300))

        except Exception as e:
            print(f"[HolterBPMCalculator] Error: {e}")
            return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  HolterBPMStore
#     Thread-safe container for the most recent BPM value + arrhythmia list.
# ══════════════════════════════════════════════════════════════════════════════
class HolterBPMStore:
    """Shared state between the worker thread and the Qt display timer."""

    def __init__(self):
        self._lock          = threading.Lock()
        self._bpm: float    = 0.0
        self._arrhythmias: List[str] = []
        self._last_update: float = 0.0

    # ── Writer (background thread) ────────────────────────────────────────
    def set_bpm(self, bpm: float, arrhythmias: List[str] = None):
        with self._lock:
            if bpm > 0:
                self._bpm = bpm
            if arrhythmias:
                for a in arrhythmias:
                    if a not in self._arrhythmias:
                        self._arrhythmias.insert(0, a)
                self._arrhythmias = self._arrhythmias[:10]
            self._last_update = time.time()

    # ── Reader (main thread) ──────────────────────────────────────────────
    def get_bpm(self) -> float:
        with self._lock:
            return self._bpm

    def get_arrhythmias(self) -> List[str]:
        with self._lock:
            return list(self._arrhythmias)

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                'bpm'         : self._bpm,
                'arrhythmias' : list(self._arrhythmias),
                'last_update' : self._last_update,
            }

    def reset(self):
        with self._lock:
            self._bpm = 0.0
            self._arrhythmias = []
            self._last_update = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3.  HolterBPMWorker
#     SLIDING-WINDOW design:
#     • keeps a ring-buffer of the last WINDOW_SECONDS of Lead II data
#     • every CALC_INTERVAL seconds, runs HolterBPMCalculator on those samples
#     • result is stored in HolterBPMStore → display reads it every 2 s
#     Benefit: latency to a real HR change ≈ CALC_INTERVAL (≈2 s) instead of
#              waiting for a full chunk to accumulate (was 10 s).
# ══════════════════════════════════════════════════════════════════════════════
class HolterBPMWorker(threading.Thread):
    """
    Sits between the serial reader and the UI.
    push() is called 500×/s — it ONLY writes into a ring buffer (O(1)).
    A separate ticker inside the thread tasks the calculator every CALC_INTERVAL.
    """

    def __init__(self,
                 store: HolterBPMStore,
                 fs: int = FS_DEFAULT,
                 chunk_seconds: int = WINDOW_SECONDS,   # kept for API compat
                 jsonl_path: str = "",
                 on_arrhythmia=None):
        super().__init__(daemon=True, name="HolterBPMWorker")
        self.store         = store
        self.fs            = fs
        self.window_size   = chunk_seconds * fs          # ring-buffer length
        self.jsonl_path    = jsonl_path
        self.on_arrhythmia = on_arrhythmia

        # ── Ring buffer for Lead II only (saves memory vs. storing all 12 leads) ──
        self._ring         = np.zeros(self.window_size, dtype=np.float32)
        self._ring_ptr     = 0       # next write position
        self._ring_count   = 0       # samples written so far (capped at window_size)
        self._ring_lock    = threading.Lock()

        self._stop_evt     = threading.Event()
        self._calculator   = HolterBPMCalculator(fs=fs)
        self._total_frames = 0

        # Optional arrhythmia detector
        self._arrhy_detector = None
        try:
            from ecg.arrhythmia_detector import ArrhythmiaDetector
            self._arrhy_detector = ArrhythmiaDetector(sampling_rate=fs)
        except Exception:
            pass

    # ── Called 500× per second from the Qt main thread ────────────────────
    def push(self, packet: dict):
        """
        Extracts Lead II and writes into the ring buffer.
        This is the hot path — must stay O(1) with no allocation.
        """
        val = float(packet.get("II", packet.get(LEAD_NAMES[LEAD_II_IDX], 2048)))
        with self._ring_lock:
            self._ring[self._ring_ptr] = val
            self._ring_ptr = (self._ring_ptr + 1) % self.window_size
            if self._ring_count < self.window_size:
                self._ring_count += 1
        self._total_frames += 1

    # ── Thread main loop — ticker every CALC_INTERVAL seconds ─────────────
    def run(self):
        print("[HolterBPMWorker] Started (sliding-window mode)")
        last_calc = 0.0
        min_samples = self.fs * 3   # require at least 3 s of data

        while not self._stop_evt.is_set():
            time.sleep(0.2)          # lightweight polling (200 ms)

            now = time.time()
            if now - last_calc < CALC_INTERVAL:
                continue
            last_calc = now

            # ── Snapshot the ring buffer ───────────────────────────────────
            with self._ring_lock:
                count = self._ring_count
                if count < min_samples:
                    continue        # not enough data yet
                # Build a chronological slice of the ring buffer
                if count >= self.window_size:
                    # Buffer is full — rearrange so oldest sample is first
                    ptr = self._ring_ptr
                    lead_ii = np.concatenate(
                        [self._ring[ptr:], self._ring[:ptr]]
                    ).copy()
                else:
                    # Buffer not yet full — just take first `count` samples
                    lead_ii = self._ring[:count].copy()

            self._process(lead_ii)

        print("[HolterBPMWorker] Stopped")

    def stop(self):
        self._stop_evt.set()

    # ── Core processing (runs in background thread) ───────────────────────
    def _process(self, lead_ii: np.ndarray):
        start_sec = self._total_frames / self.fs

        # ── Calculate BPM ─────────────────────────────────────────────────
        bpm = self._calculator.calculate(lead_ii, fs=self.fs)

        # ── Detect arrhythmias ────────────────────────────────────────────
        arrhythmias = []
        if self._arrhy_detector is not None and bpm > 0:
            try:
                from scipy.signal import butter, filtfilt, find_peaks
                low  = max(0.01, 5  / (self.fs / 2))
                high = min(0.99, 20 / (self.fs / 2))
                b, a = butter(2, [low, high], btype='band')
                filt = filtfilt(b, a, lead_ii.astype(float))
                sq   = filt ** 2
                win  = max(1, int(0.15 * self.fs))
                env  = np.convolve(sq, np.ones(win) / win, mode='same')
                thr  = np.mean(env) * 0.4
                r_peaks, _ = find_peaks(env, height=thr,
                                        distance=max(1, int(0.12 * self.fs)))
                if len(r_peaks) >= 3:
                    ad = {'r_peaks': r_peaks, 'p_peaks': [],
                          'q_peaks': [], 's_peaks': [], 't_peaks': []}
                    arrhythmias = self._arrhy_detector.detect_arrhythmias(
                        lead_ii, ad,
                        has_received_serial_data=True,
                        min_serial_data_packets=50
                    )
                    arrhythmias = [a for a in arrhythmias
                                   if 'Insufficient' not in a and 'No ' not in a]
            except Exception:
                pass

        # ── Push to store ─────────────────────────────────────────────────
        self.store.set_bpm(bpm, arrhythmias)

        # ── Optional arrhythmia callback ──────────────────────────────────
        if arrhythmias and self.on_arrhythmia:
            try:
                self.on_arrhythmia(arrhythmias, start_sec)
            except Exception:
                pass

        # ── Optional JSONL logging ─────────────────────────────────────────
        if self.jsonl_path:
            try:
                with open(self.jsonl_path, 'a') as f:
                    f.write(json.dumps({
                        't': round(start_sec, 2),
                        'hr_mean': bpm,
                        'arrhythmias': arrhythmias,
                    }) + '\n')
            except Exception:
                pass

        if bpm > 0:
            print(f"[HolterBPMWorker] t={start_sec:.0f}s | BPM={bpm:.0f} | Arrhy={arrhythmias}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HolterBPMDisplay
#     Qt widget: a compact status bar that shows  ● REC | HH:MM:SS | ♥ BPM
#     Reads from HolterBPMStore every 3 seconds via a QTimer.
# ══════════════════════════════════════════════════════════════════════════════
class HolterBPMDisplay(QFrame):
    """
    Drop-in status bar widget.

    Shows:
        ● REC  |  HH:MM:SS  |  ♥ BPM: 72  |  Events: …  [Stop]

    BPM is refreshed every 3 seconds from HolterBPMStore (no UI jitter).
    """

    stop_requested = pyqtSignal()

    def __init__(self,
                 store: HolterBPMStore,
                 parent=None,
                 target_hours: int = 0,
                 compact: bool = True):
        super().__init__(parent)
        self._store        = store
        self._start_time   = time.time()
        self._target_hours = target_hours
        self._blink_state  = True
        self._compact      = compact

        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COL_BG};
                border-bottom: 2px solid {COL_ORANGE};
                border-radius: 4px;
            }}
        """)

        self._build_ui()
        self._start_timers()

    # ── Build the bar ──────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)
        layout.setSpacing(12)

        # ● REC
        self._rec_lbl = QLabel("● REC")
        self._rec_lbl.setStyleSheet(
            f"color: {COL_RED}; font-size: 13px; font-weight: bold;")
        layout.addWidget(self._rec_lbl)

        layout.addWidget(self._sep())

        # Elapsed HH:MM:SS
        self._elapsed_lbl = QLabel("00:00:00")
        self._elapsed_lbl.setStyleSheet(
            "color: white; font-size: 14px; font-weight: bold;"
            " font-family: monospace;")
        layout.addWidget(self._elapsed_lbl)

        # Target (only if meaningful)
        if self._target_hours > 0:
            tgt = QLabel(f"/ {self._target_hours:02d}:00:00")
            tgt.setStyleSheet("color: #666; font-size: 11px;")
            layout.addWidget(tgt)

        layout.addWidget(self._sep())

        # ♥ BPM icon
        heart_lbl = QLabel("♥")
        heart_lbl.setStyleSheet(f"color: {COL_RED}; font-size: 16px;")
        layout.addWidget(heart_lbl)

        bpm_prefix = QLabel("BPM:")
        bpm_prefix.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(bpm_prefix)

        # ★ THE BPM VALUE — updated from store every 3 s
        self._bpm_lbl = QLabel("—")
        self._bpm_lbl.setStyleSheet(
            f"color: {COL_GREEN_ECG}; font-size: 18px; font-weight: bold;"
            " font-family: monospace;")
        self._bpm_lbl.setMinimumWidth(52)
        layout.addWidget(self._bpm_lbl)

        layout.addWidget(self._sep())

        # Arrhythmia ticker
        self._ev_lbl = QLabel("")
        self._ev_lbl.setStyleSheet("color: #FFA726; font-size: 10px;")
        self._ev_lbl.setMaximumWidth(260)
        layout.addWidget(self._ev_lbl, 1)

        # Progress bar (only for long recordings)
        if self._target_hours > 0:
            self._progress = QProgressBar()
            self._progress.setRange(0, self._target_hours * 3600)
            self._progress.setValue(0)
            self._progress.setFixedWidth(100)
            self._progress.setFixedHeight(10)
            self._progress.setStyleSheet(f"""
                QProgressBar {{ background: #333; border-radius: 5px; border: 1px solid #555; }}
                QProgressBar::chunk {{ background: {COL_ORANGE}; border-radius: 5px; }}
            """)
            self._progress.setTextVisible(False)
            layout.addWidget(self._progress)
        else:
            self._progress = None

        # Stop button
        stop_btn = QPushButton("■ Stop")
        stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COL_RED}; color: white; border: none;
                border-radius: 5px; padding: 4px 12px;
                font-size: 12px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #D32F2F; }}
        """)
        stop_btn.setFixedHeight(28)
        stop_btn.clicked.connect(self.stop_requested)
        layout.addWidget(stop_btn)

    def _sep(self) -> QLabel:
        s = QLabel("|")
        s.setStyleSheet("color: #333; font-size: 12px;")
        return s

    # ── Timers ─────────────────────────────────────────────────────────────
    def _start_timers(self):
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink)
        self._blink_timer.start(800)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)

        # ★ BPM refresh — reads from store every 2 s (Qt main thread only)
        self._bpm_timer = QTimer(self)
        self._bpm_timer.timeout.connect(self._refresh_bpm)
        self._bpm_timer.start(2000)

    # ── ★ BPM refresh ─────────────────────────────────────────────────────
    def _refresh_bpm(self):
        """Called every 3 s. Reads from HolterBPMStore — zero blocking."""
        snap = self._store.get_snapshot()

        bpm = snap['bpm']
        if bpm > 0:
            self._bpm_lbl.setText(f"{int(round(bpm))}")

        arrhythmias = snap['arrhythmias']
        if arrhythmias:
            self._ev_lbl.setText("  |  ".join(arrhythmias[:2]))
            self._ev_lbl.setStyleSheet(
                f"color: {COL_RED}; font-size: 10px; font-weight: bold;")

    def _blink(self):
        self._blink_state = not self._blink_state
        color = COL_RED if self._blink_state else "#444"
        self._rec_lbl.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold;")

    def _update_clock(self):
        elapsed = int(time.time() - self._start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        self._elapsed_lbl.setText(f"{h:02d}:{m:02d}:{s:02d}")
        if self._progress:
            self._progress.setValue(elapsed)

    # ── Cleanup ────────────────────────────────────────────────────────────
    def cleanup(self):
        for t in (self._blink_timer, self._clock_timer, self._bpm_timer):
            try:
                t.stop()
            except Exception:
                pass

    # ── Read latest BPM without touching Qt directly ────────────────────────
    def current_bpm(self) -> float:
        return self._store.get_bpm()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  HolterBPMController
#     ONE class to import everywhere. Wires store + worker + display.
# ══════════════════════════════════════════════════════════════════════════════
class HolterBPMController:
    """
    Single entry-point for all Holter BPM logic.

    Works for:
      • twelve_lead_test.py  (continuous, many packets/sec)
      • hrv_test.py          (5-minute short window)
      • hyperkalemia_test.py (30-second short window)

    Usage:
    ──────
        from ecg.holter.holter_bpm_engine import HolterBPMController

        # On Start:
        self._bpm_ctrl = HolterBPMController(parent_widget=self, fs=500)
        self._bpm_ctrl.start()
        layout.insertWidget(0, self._bpm_ctrl.display_bar)

        # In packet loop (per packet, called ~500/s):
        self._bpm_ctrl.push(packet)   # packet = {"I": v, "II": v, ...}

        # On Stop:
        self._bpm_ctrl.stop()
        self._bpm_ctrl.display_bar.hide()

        # Read BPM at any time (main thread):
        bpm = self._bpm_ctrl.current_bpm()
    """

    def __init__(self,
                 parent_widget=None,
                 fs: int = FS_DEFAULT,
                 chunk_seconds: int = CHUNK_SECONDS,
                 on_arrhythmia=None):
        self._parent        = parent_widget
        self._fs            = fs
        self._chunk_seconds = chunk_seconds
        self._on_arrhythmia = on_arrhythmia

        # Shared store always exists
        self.store: HolterBPMStore = HolterBPMStore()

        # Worker + display are created lazily in start()
        self._worker: Optional[HolterBPMWorker]  = None
        self.display_bar: Optional[HolterBPMDisplay] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def start(self,
              target_hours: int = 0,
              jsonl_path: str = ""):
        """Create and start worker thread + display bar."""
        self.store.reset()

        self._worker = HolterBPMWorker(
            store          = self.store,
            fs             = self._fs,
            chunk_seconds  = self._chunk_seconds,
            jsonl_path     = jsonl_path,
            on_arrhythmia  = self._on_arrhythmia,
        )
        self._worker.start()

        self.display_bar = HolterBPMDisplay(
            store        = self.store,
            parent       = self._parent,
            target_hours = target_hours,
            compact      = (target_hours == 0),
        )
        # Allow caller to wire stop_requested signal
        print("[HolterBPMController] Started")

    def stop(self):
        """Stop the worker thread and clean up the display."""
        if self._worker is not None:
            self._worker.stop()
            self._worker.join(timeout=5)
            self._worker = None

        if self.display_bar is not None:
            self.display_bar.cleanup()

        print("[HolterBPMController] Stopped")

    # ── Data input (called 500×/s from Qt main thread) ─────────────────────
    def push(self, packet: dict):
        """
        Forward a hardware packet to the worker's accumulation buffer.
        Must return in < 0.1 ms (never blocks).
        """
        if self._worker is not None and self._worker.is_alive():
            self._worker.push(packet)

    # ── Read current BPM ──────────────────────────────────────────────────
    def current_bpm(self) -> float:
        """Returns the most recently calculated BPM (thread-safe)."""
        return self.store.get_bpm()

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

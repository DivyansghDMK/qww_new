"""
ecg/holter/stream_writer.py
============================
HolterStreamWriter: central coordinator between the serial reader and
the Holter analysis pipeline.

Responsibilities:
  - Write every packet to disk via ECGHFileWriter (real-time, 500 Hz)
  - Maintain a 120-second circular RAM buffer for live display
  - Accumulate 30-second analysis chunks and enqueue to HolterAnalysisWorker
  - Track elapsed time, live BPM, detected arrhythmias

Integration (3 lines in twelve_lead_test.py):
    # In __init__:  self._holter = None
    # In packet loop:  if self._holter: self._holter.push(packet)
    # In Holter button:  self._holter = HolterStreamWriter(...)
"""

import os
import time
import queue
import threading
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict

from .file_format import ECGHFileWriter, LEAD_NAMES

# Analysis chunk: 30 seconds at 500 Hz
CHUNK_SECONDS = 30
FS_DEFAULT = 500
LEADS = 12
DISPLAY_BUFFER_SECONDS = 120   # how much raw data to keep in RAM for display


class HolterStreamWriter:
    """
    Sits between the serial reader and the rest of the Holter system.
    Must be fast — called 500× per second from the Qt main thread.
    """

    def __init__(self, output_dir: str, patient_info: dict,
                 fs: int = FS_DEFAULT,
                 on_chunk_ready=None,
                 on_arrhythmia=None):
        """
        Args:
            output_dir: Directory to save .ecgh + .jsonl files
            patient_info: dict with name, dob, gender, doctor
            fs: sampling rate (500 Hz)
            on_chunk_ready: callback(chunk_data) when 30s chunk is ready
            on_arrhythmia: callback(label, timestamp) for live arrhythmia ticker
        """
        self.fs = fs
        self.patient_info = patient_info
        self.output_dir = output_dir
        self.on_chunk_ready = on_chunk_ready
        self.on_arrhythmia = on_arrhythmia

        # State
        self._running = False
        self._start_time: Optional[float] = None
        self._total_frames = 0
        self._session_dir = ""
        self._ecgh_path = ""
        self._jsonl_path = ""

        # File writer (created on start)
        self._writer: Optional[ECGHFileWriter] = None

        # Display circular buffer: last 120s per lead
        self._display_buf_size = DISPLAY_BUFFER_SECONDS * fs
        self._display_buf = np.zeros((LEADS, self._display_buf_size), dtype=np.float32)
        self._display_ptr = 0    # next write position

        # 30-second analysis accumulator
        self._chunk_size = CHUNK_SECONDS * fs
        self._chunk_buf = np.zeros((LEADS, self._chunk_size), dtype=np.float32)
        self._chunk_ptr = 0

        # Analysis queue (consumed by HolterAnalysisWorker)
        self.analysis_queue: queue.Queue = queue.Queue(maxsize=10)

        # Live stats (updated by analysis worker via callbacks)
        self._live_bpm: float = 0.0
        self._live_arrhythmias: List[str] = []
        self._lock = threading.Lock()

        # Flush timer for disk writes
        self._flush_thread: Optional[threading.Thread] = None

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def start(self) -> str:
        """Creates session directory + .ecgh file. Returns session directory path."""
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        patient_name = self.patient_info.get('name', 'Unknown').replace(' ', '_')
        self._session_dir = os.path.join(self.output_dir, f"{ts}_{patient_name}")
        os.makedirs(self._session_dir, exist_ok=True)

        self._ecgh_path = os.path.join(self._session_dir, "recording.ecgh")
        self._jsonl_path = os.path.join(self._session_dir, "metrics.jsonl")

        self._writer = ECGHFileWriter(
            path=self._ecgh_path,
            patient_info=self.patient_info,
            fs=self.fs
        )

        self._start_time = time.time()
        self._running = True
        self._total_frames = 0
        self._chunk_ptr = 0
        self._display_ptr = 0

        # Reset buffers
        self._display_buf[:] = 0
        self._chunk_buf[:] = 0

        print(f"[Holter] Recording started → {self._session_dir}")
        return self._session_dir

    def stop(self) -> dict:
        """Flush, finalize file, return session summary."""
        if not self._running:
            return {}

        self._running = False

        # Flush partial chunk to analysis queue
        if self._chunk_ptr > 0:
            partial = self._chunk_buf[:, :self._chunk_ptr].copy()
            try:
                self.analysis_queue.put_nowait({
                    'data': partial,
                    'start_sec': self._total_frames / self.fs - self._chunk_ptr / self.fs,
                    'fs': self.fs,
                    'partial': True,
                    'jsonl_path': self._jsonl_path,
                })
            except queue.Full:
                pass

        # Finalize .ecgh file
        summary = {}
        if self._writer:
            summary = self._writer.finalize()
            summary['session_dir'] = self._session_dir
            summary['jsonl_path'] = self._jsonl_path

        # Signal analysis worker to stop
        try:
            self.analysis_queue.put_nowait(None)   # sentinel
        except queue.Full:
            pass

        print(f"[Holter] Recording stopped. Duration: {self.elapsed_seconds:.1f}s")
        return summary

    # ── Main push method (called 500× per second) ──────────────────────────────

    def push(self, packet: dict):
        """
        Fast path — must return in <0.1ms.
        Called from Qt main thread on every serial packet.
        """
        if not self._running or self._writer is None:
            return

        # 1. Write to disk
        self._writer.write_packet(packet)

        # 2. Update display circular buffer
        dp = self._display_ptr % self._display_buf_size
        for i, lead in enumerate(LEAD_NAMES):
            self._display_buf[i, dp] = float(packet.get(lead, 2048))
        self._display_ptr += 1

        # 3. Accumulate analysis chunk
        cp = self._chunk_ptr
        for i, lead in enumerate(LEAD_NAMES):
            self._chunk_buf[i, cp] = float(packet.get(lead, 2048))
        self._chunk_ptr += 1
        self._total_frames += 1

        # 4. Chunk full → enqueue for analysis
        if self._chunk_ptr >= self._chunk_size:
            chunk_data = self._chunk_buf.copy()
            chunk_start = (self._total_frames - self._chunk_size) / self.fs
            try:
                self.analysis_queue.put_nowait({
                    'data': chunk_data,
                    'start_sec': chunk_start,
                    'fs': self.fs,
                    'partial': False,
                    'jsonl_path': self._jsonl_path,
                })
            except queue.Full:
                pass   # analysis worker is slow, drop chunk (data still on disk)
            self._chunk_ptr = 0

    # ── Live display data ──────────────────────────────────────────────────────

    def get_display_data(self, lead_idx: int, n_samples: int) -> np.ndarray:
        """
        Returns last n_samples for the given lead from the circular display buffer.
        Safe to call from Qt main thread.
        """
        n = min(n_samples, self._display_buf_size)
        end = self._display_ptr % self._display_buf_size
        start = (end - n) % self._display_buf_size
        if start < end:
            return self._display_buf[lead_idx, start:end].copy()
        else:
            return np.concatenate([
                self._display_buf[lead_idx, start:],
                self._display_buf[lead_idx, :end]
            ])

    # ── Live stats (updated by analysis worker) ────────────────────────────────

    def update_live_stats(self, bpm: float, arrhythmias: List[str]):
        with self._lock:
            self._live_bpm = bpm
            for a in arrhythmias:
                if a not in self._live_arrhythmias:
                    self._live_arrhythmias.insert(0, a)
            self._live_arrhythmias = self._live_arrhythmias[:10]
            if self.on_arrhythmia:
                for a in arrhythmias:
                    self.on_arrhythmia(a, time.time())

    def get_live_stats(self) -> dict:
        with self._lock:
            return {
                'bpm': self._live_bpm,
                'arrhythmias': list(self._live_arrhythmias),
                'elapsed': self.elapsed_seconds,
                'frames': self._total_frames,
            }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def ecgh_path(self) -> str:
        return self._ecgh_path

    @property
    def jsonl_path(self) -> str:
        return self._jsonl_path

    @property
    def session_dir(self) -> str:
        return self._session_dir

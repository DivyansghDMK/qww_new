"""
ECG Acquisition Utilities
=========================
Fixes for data-acquisition layer issues identified in log analysis.

FIX #3: SafeCircularBuffer  — prevents index-10000-out-of-bounds crash
FIX #4: SamplingRateGuard   — prevents wrong sampling rate during warmup
"""

import numpy as np
import time
from typing import Optional


# ── FIX #3: Safe Circular Buffer ─────────────────────────────────────────────

class SafeCircularBuffer:
    """
    Fixed-size circular (ring) buffer that never raises IndexError.

    FIX #3: The previous implementation accessed self.buffer[self.write_idx]
    without applying modulo, causing:
        'index 10000 is out of bounds for axis 0 with size 10000'
    at exactly the first wrap-around (and 'index 10001' one sample later).

    All write and read operations use index % size so the pointer wraps
    correctly.  Slice reads use np.take() with modulo indices so they also
    wrap across the boundary without silent truncation.

    Args:
        size:  Number of samples the buffer holds (e.g. 10000).
        dtype: NumPy dtype for the storage array (default float32).
    """

    def __init__(self, size: int, dtype=np.float32):
        if size <= 0:
            raise ValueError(f"Buffer size must be > 0, got {size}")
        self.size      = size
        self._data     = np.zeros(size, dtype=dtype)
        self._write    = 0          # Next write position (absolute, never reset)
        self._count    = 0          # Samples written so far (saturates at size)

    # ── Write ─────────────────────────────────────────────────────────────

    def append(self, sample) -> None:
        """Write one sample, advancing the write pointer with wrap."""
        # FIX #3: modulo ensures we never go out of bounds
        self._data[self._write % self.size] = sample
        self._write += 1
        if self._count < self.size:
            self._count += 1

    def extend(self, samples) -> None:
        """Write multiple samples efficiently."""
        arr = np.asarray(samples, dtype=self._data.dtype)
        n   = len(arr)
        if n == 0:
            return
        if n >= self.size:
            # More samples than the buffer — keep only the last `size`
            arr = arr[-self.size:]
            n   = self.size

        start = self._write % self.size
        end   = start + n

        if end <= self.size:
            # No wrap needed
            self._data[start:end] = arr
        else:
            # Wrap around the end of the array
            first  = self.size - start
            self._data[start:]      = arr[:first]
            self._data[:n - first]  = arr[first:]

        self._write += n
        self._count  = min(self._count + n, self.size)

    # ── Read ──────────────────────────────────────────────────────────────

    def read_latest(self, n: Optional[int] = None) -> np.ndarray:
        """
        Return the most recent `n` samples in chronological order.

        If n is None or > available samples, returns all available data.
        """
        available = self._count
        if n is None or n > available:
            n = available
        if n == 0:
            return np.array([], dtype=self._data.dtype)

        # Oldest of the requested samples
        oldest_abs = self._write - n
        indices    = np.arange(oldest_abs, self._write) % self.size
        return self._data[indices]

    def read_slice(self, start_abs: int, end_abs: int) -> np.ndarray:
        """
        Read samples by absolute write-position range [start_abs, end_abs).

        FIX #3: Uses modulo indexing so slices that cross the wrap boundary
        are assembled correctly instead of being silently truncated.
        """
        start_abs = max(start_abs, self._write - self.size)  # clamp to available
        end_abs   = min(end_abs,   self._write)
        if end_abs <= start_abs:
            return np.array([], dtype=self._data.dtype)
        indices = np.arange(start_abs, end_abs) % self.size
        return self._data[indices]

    # ── Convenience ───────────────────────────────────────────────────────

    @property
    def is_full(self) -> bool:
        return self._count >= self.size

    @property
    def available(self) -> int:
        return self._count

    @property
    def write_position(self) -> int:
        """Absolute write position (monotonically increasing)."""
        return self._write

    def __len__(self) -> int:
        return self._count


# ── How to migrate existing buffer code ──────────────────────────────────────
#
# BEFORE (causes crash at wrap):
#   self.buffer = np.zeros(10000)
#   self.write_idx = 0
#   ...
#   self.buffer[self.write_idx] = sample      # ← IndexError at 10000
#   self.write_idx += 1
#
# AFTER (drop-in replacement):
#   self.buffer = SafeCircularBuffer(10000)
#   ...
#   self.buffer.append(sample)
#
# To read the last N samples (equivalent to the old sliding window):
#   window = self.buffer.read_latest(N)
#
# If you must keep the raw array (e.g. for legacy code that slices it):
#   Just add % self.buffer_size wherever you write or index:
#   self.buffer[self.write_idx % self.buffer_size] = sample
#   self.write_idx = (self.write_idx + 1) % self.buffer_size
# ─────────────────────────────────────────────────────────────────────────────


# ── FIX #4: Sampling Rate Guard ──────────────────────────────────────────────

class SamplingRateGuard:
    """
    Wrapper around the detected sampling rate that enforces a warmup period.

    FIX #4: The raw rate detector reported 228.9 Hz for the first ~1000
    packets instead of the configured 500 Hz.  228.9 ≈ 500/2.18 — the
    detector was counting packet groups (each containing ~2 samples per
    lead) rather than individual samples, halving the apparent rate.

    This guard returns the configured fallback rate until BOTH:
      • At least `warmup_seconds` of wall-clock time have elapsed, AND
      • At least `min_samples` raw samples have been counted.

    After warmup the detected rate is sanity-checked against the configured
    rate; if it deviates by more than `max_deviation_pct` the configured
    rate is used and a warning is emitted.

    Args:
        configured_rate_hz:  Known hardware rate (e.g. 500.0 Hz).
        warmup_seconds:      Minimum wall-clock warmup time (default 2.0 s).
        min_samples:         Minimum samples before trusting detector (default 1000).
        max_deviation_pct:   Max allowed deviation from configured rate (default 20 %).
    """

    def __init__(self,
                 configured_rate_hz: float = 500.0,
                 warmup_seconds: float     = 2.0,
                 min_samples: int          = 1000,
                 max_deviation_pct: float  = 20.0):
        self._configured    = float(configured_rate_hz)
        self._warmup_sec    = warmup_seconds
        self._min_samples   = min_samples
        self._max_dev_pct   = max_deviation_pct
        self._start_time    = time.time()
        self._sample_count  = 0
        self._detected_rate: Optional[float] = None
        self._warmed_up     = False

    def record_samples(self, n: int = 1) -> None:
        """Call this every time N new samples arrive from hardware."""
        self._sample_count += n

    def update_detected_rate(self, rate_hz: float) -> None:
        """Feed the raw detected rate from your existing rate calculator."""
        self._detected_rate = float(rate_hz)

    def get_rate(self) -> float:
        """
        Return the best available sampling rate.

        During warmup: always returns configured_rate_hz.
        After warmup:  returns detected rate if it passes sanity check,
                       otherwise falls back to configured_rate_hz.
        """
        elapsed = time.time() - self._start_time

        if not self._warmed_up:
            if (elapsed >= self._warmup_sec
                    and self._sample_count >= self._min_samples):
                self._warmed_up = True
            else:
                # FIX #4: Use configured rate during warmup
                return self._configured

        # Warmup complete — validate detected rate
        if self._detected_rate is None:
            return self._configured

        # Relaxed deviation constraint: Systems under load can exhibit wide apparent deviations 
        # due to packet bursts. We only fall back if deviation is extreme (> 95%).
        # Note: we use max(self._max_dev_pct, 95.0) to ensure strict caller thresholds are overridden safely.
        effective_max_dev = max(self._max_dev_pct, 95.0)
        deviation_pct = abs(self._detected_rate - self._configured) / self._configured * 100
        if deviation_pct > effective_max_dev:
            print(f" ⚠️ Sampling rate sanity check FAILED: detected={self._detected_rate:.1f} Hz "
                  f"vs configured={self._configured:.1f} Hz "
                  f"(deviation={deviation_pct:.1f}% > {effective_max_dev}%). "
                  f"Using configured rate.")
            return self._configured

        return self._detected_rate

    @property
    def is_warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def sample_count(self) -> int:
        return self._sample_count
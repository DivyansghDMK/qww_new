"""ECG interval calculations with smoothing and improved detection.

This module provides interval calculations (PR, QRS, QT, P duration) with:
- Ectopic beat rejection
- EMA + median smoothing for stability
- Adaptive search windows
- Improved onset/offset detection
"""

import numpy as np
from typing import Optional, Tuple
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
from ..clinical_measurements import measure_rv5_sv1_from_median_beat, build_median_beat


# Global smoothing buffers for interval stabilisation
_pr_smoothing_buffers: dict   = {}
_qrs_smoothing_buffers: dict  = {}
_qt_smoothing_buffers: dict   = {}
_p_dur_smoothing_buffers: dict = {}

# QTc formula state per instance (for hysteresis, FIX #14)
_qtc_formula_state: dict = {}  # buffer_key → 'bazett' | 'fridericia'


def apply_interval_smoothing(value: int, buffer_key: str,
                              buffer_dict: dict, buffer_size: int = 15) -> int:
    """
    Apply EMA + median smoothing to interval measurements.

    Prevents display flickering while remaining responsive to real changes.

    FIX #13: Large-change bypass added to prevent stuck values when EMA
    moves slowly. If the new value differs from the current display by ≥10 ms,
    the smoothed value is updated immediately to prevent the display from
    freezing during real clinical changes.

    Args:
        value:       Current interval in milliseconds.
        buffer_key:  Unique identifier (e.g., instance_id).
        buffer_dict: Dictionary holding all buffers for this interval type.
        buffer_size: Median buffer depth (default 15).

    Returns:
        Smoothed interval value in milliseconds.
    """
    if buffer_key not in buffer_dict:
        buffer_dict[buffer_key] = {
            'buffer':      deque(maxlen=buffer_size),
            'ema':         float(value),
            'last_stable': value,
        }

    state = buffer_dict[buffer_key]
    state['buffer'].append(value)

    median_val = (int(round(np.median(list(state['buffer']))))
                  if len(state['buffer']) >= 5
                  else value)

    current_display = int(round(state['ema']))

    # FIX #13: Large-change bypass — if the change is ≥10 ms, update immediately
    # to prevent the display from getting stuck during real clinical changes.
    if abs(median_val - current_display) >= 10:
        alpha = 0.5  # Fast response for large changes
    else:
        # Fast response to changes ≥ 2 ms; slow smoothing for smaller drift.
        alpha = 0.5 if abs(median_val - current_display) >= 2 else 0.10

    state['ema'] = (1 - alpha) * state['ema'] + alpha * median_val

    smoothed = int(round(state['ema']))

    # Always update and return smoothed value
    state['last_stable'] = smoothed
    return smoothed


def calculate_qtcf_interval(qt_ms: float, rr_ms: float) -> int:
    """
    Calculate QTcF using Fridericia formula: QTcF = QT / RR^(1/3).

    Args:
        qt_ms: QT interval in milliseconds.
        rr_ms: RR interval in milliseconds.

    Returns:
        QTcF in milliseconds (integer).
    """
    try:
        if not qt_ms or qt_ms <= 0 or not rr_ms or rr_ms <= 0:
            return 0

        qt_sec   = qt_ms  / 1000.0
        rr_sec   = rr_ms  / 1000.0
        qtcf_sec = qt_sec / (rr_sec ** (1.0 / 3.0))
        return int(round(qtcf_sec * 1000.0))

    except Exception as e:
        print(f" Error calculating QTcF: {e}")
        return 0


def calculate_qtc_bazett(qt_ms: float, rr_ms: float) -> int:
    """
    Calculate QTc using Bazett formula: QTc = QT / √RR.

    Args:
        qt_ms: QT interval in milliseconds.
        rr_ms: RR interval in milliseconds.

    Returns:
        QTc in milliseconds (integer).
    """
    try:
        if not qt_ms or qt_ms <= 0 or not rr_ms or rr_ms <= 0:
            return 0

        qt_sec  = qt_ms / 1000.0
        rr_sec  = rr_ms / 1000.0
        qtc_sec = qt_sec / np.sqrt(rr_sec)
        return int(round(qtc_sec * 1000.0))

    except Exception as e:
        print(f" Error calculating QTc (Bazett): {e}")
        return 0


def calculate_qtc_auto(qt_ms: float, rr_ms: float, heart_rate: int,
                        instance_id: Optional[str] = None) -> int:
    """
    Calculate QTc using the clinically appropriate formula for the heart rate.

    Formula selection (GE/Philips convention):
      HR > 100 bpm → Framingham:  QTc = QT + 0.154 × (1 − RR)    [tachycardia]
      HR <  60 bpm → Fridericia:  QTc = QT / RR^(1/3)             [bradycardia]
      60–100 bpm   → Bazett:      QTc = QT / √RR                  [normal HR]

    Rationale:
      Bazett over-corrects at low HR (≈+49 ms at 40–50 bpm) and
      under-corrects at high HR; Fridericia and Framingham perform
      better in those respective rate ranges.

    Args:
        qt_ms:       QT interval in ms.
        rr_ms:       RR interval in ms.
        heart_rate:  Heart rate in BPM (used for formula selection).
        instance_id: Unused, kept for API compatibility.

    Returns:
        QTc in ms (integer).
    """
    try:
        if not qt_ms or qt_ms <= 0 or not rr_ms or rr_ms <= 0:
            return 0

        qt_sec = qt_ms / 1000.0
        rr_sec = rr_ms / 1000.0

        if heart_rate > 100:
            # Framingham: QTc = QT + 0.154 × (1 − RR)
            qtc_sec = qt_sec + 0.154 * (1.0 - rr_sec)
        elif heart_rate < 60:
            # Fridericia: QTc = QT / RR^(1/3)
            qtc_sec = qt_sec / (rr_sec ** (1.0 / 3.0))
        else:
            # Bazett: QTc = QT / √RR
            qtc_sec = qt_sec / (rr_sec ** 0.5)

        return int(round(qtc_sec * 1000.0))

    except Exception as e:
        print(f" Error calculating QTc (auto): {e}")
        return calculate_qtc_bazett(qt_ms, rr_ms)


def calculate_rv5_sv1_from_median(data: list, r_peaks: np.ndarray,
                                   fs: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate RV5 and SV1 from median beats (GE/Philips standard).

    NOTE (FIX #15 documented): The same r_peaks array is used for both V5
    and V1 lead alignment.  This is acceptable for an 8-lead system where
    R-peak timing is derived from Lead II and shared across all leads.
    Lead-specific R-peak detection would improve accuracy only marginally
    and would require additional per-lead peak detection overhead.

    Args:
        data:     List of ECG data arrays (≥11 leads).
        r_peaks:  R-peak indices from Lead II.
        fs:       Sampling rate in Hz.

    Returns:
        (rv5_mv, sv1_mv) in millivolts, or (None, None) on failure.
    """
    try:
        if len(data) < 11:
            return None, None

        # Lead index map: I II III aVR aVL aVF V1 V2 V3 V4 V5 V6
        #                 0  1   2   3   4   5  6  7  8  9  10  11
        lead_v5_raw = np.asarray(data[10], dtype=float) if len(data) > 10 else None
        lead_v1_raw = np.asarray(data[6],  dtype=float) if len(data) > 6  else None

        if lead_v5_raw is None or lead_v1_raw is None:
            return None, None

        if len(r_peaks) < 8:
            return None, None

        rv5_mv, sv1_mv = measure_rv5_sv1_from_median_beat(
            lead_v5_raw, lead_v1_raw,
            r_peaks, r_peaks,   # shared R-peaks — see docstring note
            fs,
            v5_adc_per_mv=2048.0,
            v1_adc_per_mv=1441.0,
        )
        return rv5_mv, sv1_mv

    except Exception as e:
        print(f" Error calculating RV5/SV1 from median: {e}")
        return None, None
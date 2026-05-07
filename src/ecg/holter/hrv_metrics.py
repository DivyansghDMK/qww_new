"""HRV metric helpers used by Holter summary, replay, and reporting."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

try:
    from scipy.signal import welch
except Exception:  # pragma: no cover - scipy is already available in the app
    welch = None


def _safe_rr(rr_intervals_ms: Sequence[float]) -> np.ndarray:
    rr = np.asarray(rr_intervals_ms, dtype=float).reshape(-1)
    rr = rr[np.isfinite(rr)]
    rr = rr[rr > 0]
    return rr


def compute_time_domain_hrv(rr_intervals_ms: Sequence[float]) -> Dict[str, float]:
    rr = _safe_rr(rr_intervals_ms)
    if rr.size == 0:
        return {
            "sdnn": 0.0,
            "rmssd": 0.0,
            "pnn50": 0.0,
            "triangular_index": 0.0,
            "mean_nn": 0.0,
        }

    sdnn = float(np.std(rr)) if rr.size > 1 else 0.0
    if rr.size > 1:
        diff_rr = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if diff_rr.size else 0.0
        pnn50 = float(100.0 * np.sum(np.abs(diff_rr) > 50.0) / max(1, diff_rr.size))
    else:
        rmssd = 0.0
        pnn50 = 0.0

    mean_nn = float(np.mean(rr))
    bins = max(5, min(50, int(np.sqrt(rr.size) * 2)))
    hist, _ = np.histogram(rr, bins=bins)
    triangular_index = float(rr.size / max(1, np.max(hist))) if hist.size and np.max(hist) > 0 else 0.0

    return {
        "sdnn": round(sdnn, 1),
        "rmssd": round(rmssd, 1),
        "pnn50": round(pnn50, 2),
        "triangular_index": round(triangular_index, 2),
        "mean_nn": round(mean_nn, 1),
    }


def compute_frequency_domain_hrv(rr_intervals_ms: Sequence[float]) -> Dict[str, float]:
    rr = _safe_rr(rr_intervals_ms)
    if rr.size < 4 or welch is None:
        return {
            "vlf": 0.0,
            "lf": 0.0,
            "hf": 0.0,
            "lf_hf_ratio": 0.0,
            "total_power": 0.0,
        }

    # Convert RR intervals to a tachogram using 4 Hz interpolation.
    rr_sec = rr / 1000.0
    t = np.cumsum(rr_sec)
    t = np.insert(t, 0, 0.0)
    rr_series = np.insert(rr_sec, 0, rr_sec[0] if rr_sec.size else 0.0)
    fs_interp = 4.0
    t_uniform = np.arange(0.0, float(t[-1]), 1.0 / fs_interp)
    if t_uniform.size < 8:
        return {
            "vlf": 0.0,
            "lf": 0.0,
            "hf": 0.0,
            "lf_hf_ratio": 0.0,
            "total_power": 0.0,
        }
    interp = np.interp(t_uniform, t[:-1], rr_series[:-1])
    detrended = interp - np.mean(interp)
    nperseg = min(256, max(16, detrended.size))
    freqs, psd = welch(detrended, fs=fs_interp, nperseg=nperseg)
    if freqs.size == 0:
        return {
            "vlf": 0.0,
            "lf": 0.0,
            "hf": 0.0,
            "lf_hf_ratio": 0.0,
            "total_power": 0.0,
        }

    def band_power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]))

    vlf = band_power(0.0033, 0.04)
    lf = band_power(0.04, 0.15)
    hf = band_power(0.15, 0.40)
    total = vlf + lf + hf
    ratio = lf / hf if hf > 1e-9 else 0.0

    return {
        "vlf": round(vlf, 3),
        "lf": round(lf, 3),
        "hf": round(hf, 3),
        "lf_hf_ratio": round(ratio, 3),
        "total_power": round(total, 3),
    }


def compute_hrv_summary(rr_intervals_ms: Sequence[float]) -> Dict[str, float]:
    summary = compute_time_domain_hrv(rr_intervals_ms)
    summary.update(compute_frequency_domain_hrv(rr_intervals_ms))
    return summary

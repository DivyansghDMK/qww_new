"""
Shared helpers for deriving summary values used by Holter review panels.

These helpers keep UI views and replay/report summaries aligned on the same
logic for overall and sinus-specific heart-rate extremes.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _sec_to_hms(seconds: float) -> str:
    seconds = max(0.0, _safe_float(seconds, 0.0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _metric_hr(metric: Dict[str, object]) -> float:
    return _safe_float(metric.get("hr_mean", 0.0), 0.0)


def _has_major_arrhythmia(metric: Dict[str, object]) -> bool:
    arrhythmias = " ".join(str(a).lower() for a in (metric.get("arrhythmias") or []))
    for needle in (" af ", "flutter", "vt", "vf", "tachy", "brady", "pause"):
        if needle.strip() in arrhythmias:
            return True
    return False


def _is_sinus_like(metric: Dict[str, object]) -> bool:
    hr = _metric_hr(metric)
    if hr <= 0:
        return False

    beat_counts = metric.get("beat_class_counts") or {}
    try:
        total = int(sum(int(v or 0) for v in beat_counts.values()))
    except Exception:
        total = 0
    n_count = 0
    try:
        n_count = int(beat_counts.get("N", 0) or 0)
    except Exception:
        n_count = 0

    if _has_major_arrhythmia(metric):
        return False

    if total > 0:
        return n_count >= max(3, int(total * 0.60))

    # Fallback for older sessions where beat class counts may be sparse.
    return 40.0 <= hr <= 160.0


def _pick_extreme(metrics: List[Dict[str, object]], reverse: bool) -> Optional[Dict[str, object]]:
    candidates = [m for m in metrics if _metric_hr(m) > 0]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (_metric_hr(item), _safe_float(item.get("t", 0.0), 0.0)),
        reverse=reverse,
    )[0]


def derive_hr_focus_summary(metrics_list: Iterable[Dict[str, object]]) -> Dict[str, object]:
    metrics = list(metrics_list or [])
    result: Dict[str, object] = {
        "max_hr": 0.0,
        "max_hr_timestamp": 0.0,
        "max_hr_time": "",
        "min_hr": 0.0,
        "min_hr_timestamp": 0.0,
        "min_hr_time": "",
        "sinus_max_hr": 0.0,
        "sinus_max_hr_timestamp": 0.0,
        "sinus_max_hr_time": "",
        "sinus_min_hr": 0.0,
        "sinus_min_hr_timestamp": 0.0,
        "sinus_min_hr_time": "",
    }

    overall_max = _pick_extreme(metrics, reverse=True)
    overall_min = _pick_extreme(metrics, reverse=False)
    sinus_metrics = [m for m in metrics if _is_sinus_like(m)]
    if not sinus_metrics:
        sinus_metrics = metrics
    sinus_max = _pick_extreme(sinus_metrics, reverse=True)
    sinus_min = _pick_extreme(sinus_metrics, reverse=False)

    def _fill(prefix: str, metric: Optional[Dict[str, object]]):
        if not metric:
            return
        value = round(_metric_hr(metric), 1)
        timestamp = _safe_float(metric.get("t", 0.0), 0.0)
        result[f"{prefix}_hr"] = value
        result[f"{prefix}_hr_timestamp"] = timestamp
        result[f"{prefix}_hr_time"] = _sec_to_hms(timestamp)

    _fill("max", overall_max)
    _fill("min", overall_min)
    _fill("sinus_max", sinus_max)
    _fill("sinus_min", sinus_min)

    return result

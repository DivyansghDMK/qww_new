from __future__ import annotations

from typing import Dict, Optional

import numpy as np


BBB_MIN_QRS_MS = 110.0


def _polarity(signal: np.ndarray) -> tuple[float, float]:
    sig = np.asarray(signal, dtype=float)
    if sig.size == 0:
        return 0.0, 0.0
    baseline = float(np.median(sig))
    centered = sig - baseline
    return float(np.max(centered)), float(abs(np.min(centered)))


def detect_bundle_branch_fda(leads: Dict[str, np.ndarray], qrs_ms: float) -> Optional[str]:
    if not leads or qrs_ms < BBB_MIN_QRS_MS:
        return None

    v1 = leads.get("V1")
    if v1 is None:
        return None

    v1 = np.asarray(v1, dtype=float)
    if v1.size == 0:
        return None

    v1_pos, v1_neg = _polarity(v1)
    lateral_scores = []
    for lead_name in ("I", "aVL", "V5", "V6"):
        lead = leads.get(lead_name)
        if lead is None:
            continue
        pos, neg = _polarity(np.asarray(lead, dtype=float))
        lateral_scores.append(pos - neg)

    if not lateral_scores:
        return None

    lateral_positive = float(np.median(lateral_scores)) > 0
    if v1_neg > max(v1_pos * 1.05, 1e-9) and lateral_positive:
        return "LBBB"

    if v1_pos > max(v1_neg * 1.05, 1e-9) and not lateral_positive:
        return "RBBB"

    return None


def detect_bundle_branch(leads: Dict[str, np.ndarray], qrs_ms: float) -> Optional[str]:
    return detect_bundle_branch_fda(leads, qrs_ms)


__all__ = ["detect_bundle_branch", "detect_bundle_branch_fda"]

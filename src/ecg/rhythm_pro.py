from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _beat_prs(beats: Sequence[dict]) -> List[float]:
    return [
        float(b.get("pr", 0.0))
        for b in beats
        if float(b.get("pr", 0.0)) > 0.0
    ]


def _beat_rrs(beats: Sequence[dict]) -> np.ndarray:
    values = [float(b.get("rr_prev", 0.0)) for b in beats if float(b.get("rr_prev", 0.0)) > 0.0]
    return np.asarray(values, dtype=float)


def detect_pr_progression(beats: Sequence[dict]) -> bool:
    prs = _beat_prs(beats)
    if len(prs) < 4:
        return False
    return all(prs[i] < prs[i + 1] for i in range(len(prs) - 1))


def detect_dropped_beats(beats: Sequence[dict]) -> bool:
    rr_prev = _beat_rrs(beats)
    if rr_prev.size < 3:
        return False
    median_rr = float(np.median(rr_prev))
    if median_rr <= 0:
        return False
    return bool(np.any(rr_prev > 2.0 * median_rr))


def detect_av_dissociation(beats: Sequence[dict]) -> bool:
    prs = np.asarray(_beat_prs(beats), dtype=float)
    rr_prev = _beat_rrs(beats)
    if prs.size < 5 or rr_prev.size < 5:
        return False
    return float(np.std(prs)) > 50.0 and float(np.std(rr_prev)) < 120.0


def detect_rhythm_fda(
    hr: float,
    rr_ms: np.ndarray,
    beats: Sequence[dict],
    pr: float,
    qrs: float,
    qtc: float,
    qrs_count: int = 0,
    signal: Optional[np.ndarray] = None,
) -> Tuple[str, List[str], float]:
    rr_ms = np.asarray(rr_ms, dtype=float)
    findings: List[str] = []
    confidence = 0.90
    rr_var = float(np.std(rr_ms)) if rr_ms.size else 0.0

    if signal is not None and qrs_count == 0:
        sig = np.asarray(signal, dtype=float)
        if sig.size and float(np.var(sig)) > 0.02:
            return "Ventricular Fibrillation", [], 0.95

    if hr < 5 or qrs_count == 0:
        return "Asystole", [], 1.00

    if hr > 120 and qrs > 120:
        return "Ventricular Tachycardia", [], 0.92

    if rr_var > 120:
        return "Atrial Fibrillation", [], 0.90

    if pr > 200:
        findings.append("1st Degree AV Block")
        confidence = min(confidence, 0.88)

    if detect_pr_progression(beats):
        findings.append("2nd Degree AV Block (Mobitz I)")
        confidence = min(confidence, 0.86)

    if detect_dropped_beats(beats):
        findings.append("2nd Degree AV Block (Mobitz II)")
        confidence = min(confidence, 0.85)

    if detect_av_dissociation(beats):
        return "3rd Degree AV Block", [], 0.95

    pvc_count = sum(1 for beat in beats if beat.get("type") == "PVC")
    escape_count = sum(1 for beat in beats if beat.get("type") == "ESCAPE")
    variant_count = sum(1 for beat in beats if beat.get("type") == "VARIANT")
    abnormal_count = sum(1 for beat in beats if beat.get("type") == "ABNORMAL")
    if pvc_count:
        findings.append(f"PVCs ({pvc_count})")
        confidence = min(confidence, 0.87)
    if escape_count:
        findings.append(f"Escape Beats ({escape_count})")
        confidence = min(confidence, 0.87)
    if abnormal_count:
        findings.append(f"Abnormal QRS Morphology ({abnormal_count})")
        confidence = min(confidence, 0.86)
    elif variant_count:
        findings.append(f"Beat Morphology Variants ({variant_count})")
        confidence = min(confidence, 0.88)

    if hr < 60:
        diagnosis = "Sinus Bradycardia"
    elif hr > 100:
        diagnosis = "Sinus Tachycardia"
    else:
        diagnosis = "Normal Sinus Rhythm"

    if qtc > 480:
        findings.append("Prolonged QT")
        confidence = min(confidence, 0.88)

    return diagnosis, findings, confidence


def detect_rhythm_pro(
    hr: float,
    rr_ms: np.ndarray,
    pr: float,
    qrs: float,
    qtc: float,
    morphology: Optional[str],
    beats: Optional[Sequence[dict]] = None,
    qrs_count: int = 0,
    signal: Optional[np.ndarray] = None,
) -> Tuple[str, List[str]]:
    diagnosis, findings, _ = detect_rhythm_fda(
        hr=hr,
        rr_ms=rr_ms,
        beats=beats or [],
        pr=pr,
        qrs=qrs,
        qtc=qtc,
        qrs_count=qrs_count,
        signal=signal,
    )

    if morphology == "LBBB" and "Left Bundle Branch Block" not in findings:
        findings.append("Left Bundle Branch Block")
    elif morphology == "RBBB" and "Right Bundle Branch Block" not in findings:
        findings.append("Right Bundle Branch Block")

    return diagnosis, findings


def final_diagnosis(
    hr: float,
    rr: np.ndarray,
    pr: float,
    qrs: float,
    qtc: float,
    beats: Sequence[dict],
    leads: dict,
    qrs_count: int = 0,
    signal: Optional[np.ndarray] = None,
) -> dict:
    from ecg.morphology_pro import detect_bundle_branch_fda

    morphology = detect_bundle_branch_fda(leads, qrs)
    rhythm, findings, confidence = detect_rhythm_fda(
        hr=hr,
        rr_ms=rr,
        beats=beats,
        pr=pr,
        qrs=qrs,
        qtc=qtc,
        qrs_count=qrs_count,
        signal=signal,
    )

    if morphology == "LBBB" and "Left Bundle Branch Block" not in findings:
        findings.append("Left Bundle Branch Block")
    elif morphology == "RBBB" and "Right Bundle Branch Block" not in findings:
        findings.append("Right Bundle Branch Block")

    if qtc > 480 and "Prolonged QT" not in findings:
        findings.append("Prolonged QT")

    return {
        "Diagnosis": rhythm,
        "Findings": findings,
        "Confidence": round(float(confidence), 2),
        "Morphology": morphology,
    }


__all__ = [
    "detect_pr_progression",
    "detect_dropped_beats",
    "detect_av_dissociation",
    "detect_rhythm_fda",
    "detect_rhythm_pro",
    "final_diagnosis",
]

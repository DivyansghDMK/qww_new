"""Reference interval table (Fluke-equivalent) and helpers.

This module provides HR-dependent reference values for:
- RR, P, PR, QRS, QT, QTc (all in milliseconds, BPM as integer)

The table is taken from the user's calibration reference and is used for
lightweight calibration / verification of measured intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class ReferenceRow:
    bpm: int
    rr: int
    p: int
    pr: int
    qrs: int
    qt: int
    qtc: int


# Full reference table provided by the user (25–300 BPM)
REFERENCE_TABLE: List[ReferenceRow] = [
    ReferenceRow(25, 2400, 93, 170, 88, 387, 249),
    ReferenceRow(30, 2000, 93, 170, 88, 384, 271),
    ReferenceRow(35, 1714, 82, 150, 87, 340, 265),
    ReferenceRow(40, 1500, 93, 170, 86, 373, 304),
    ReferenceRow(45, 1333, 88, 160, 86, 360, 315),
    ReferenceRow(50, 1200, 92, 168, 85, 365, 333),
    ReferenceRow(55, 1091, 86, 155, 86, 350, 345),
    ReferenceRow(60, 1000, 92, 166, 91, 363, 363),
    ReferenceRow(65, 923, 81, 151, 87, 332, 348),
    ReferenceRow(70, 857, 81, 143, 87, 316, 341),
    ReferenceRow(75, 800, 76, 131, 86, 298, 333),
    ReferenceRow(80, 750, 92, 163, 85, 343, 396),
    ReferenceRow(85, 706, 87, 150, 87, 326, 390),
    ReferenceRow(90, 667, 92, 161, 86, 329, 403),
    ReferenceRow(95, 632, 77, 134, 86, 299, 378),
    ReferenceRow(100, 600, 92, 161, 86, 315, 407),
    ReferenceRow(105, 571, 87, 152, 87, 301, 400),
    ReferenceRow(110, 545, 79, 136, 87, 291, 394),
    ReferenceRow(120, 500, 91, 135, 86, 299, 423),
    ReferenceRow(130, 462, 83, 144, 86, 283, 416),
    ReferenceRow(140, 429, 77, 135, 86, 266, 406),
    ReferenceRow(150, 400, 72, 125, 87, 252, 398),
    ReferenceRow(160, 375, 73, 125, 85, 246, 402),
    ReferenceRow(170, 353, 69, 116, 85, 233, 393),
    ReferenceRow(180, 333, 65, 109, 84, 223, 386),
    ReferenceRow(190, 316, 61, 102, 86, 213, 379),
    ReferenceRow(200, 300, 54, 87, 85, 212, 387),
    ReferenceRow(210, 286, 51, 81, 86, 204, 382),
    ReferenceRow(220, 273, 47, 76, 86, 197, 377),
    ReferenceRow(230, 261, 44, 70, 85, 190, 372),
    ReferenceRow(240, 250, 44, 68, 85, 182, 364),
    ReferenceRow(250, 239, 41, 63, 85, 177, 362),
    ReferenceRow(260, 231, 39, 60, 86, 171, 357),
    ReferenceRow(270, 222, 38, 60, 86, 178, 384),
    ReferenceRow(280, 214, 41, 64, 86, 175, 380),
    ReferenceRow(290, 207, 42, 89, 86, 172, 378),
    ReferenceRow(300, 200, 47, 81, 88, 170, 380),
]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lookup_reference_intervals(bpm: float) -> Optional[dict]:
    """Return reference RR/P/PR/QRS/QT/QTc for a given BPM using linear interpolation.

    Args:
        bpm: Heart rate in beats per minute.

    Returns:
        dict with keys: 'RR', 'P', 'PR', 'QRS', 'QT', 'QTc'
        or None if bpm is not finite/positive.
    """
    try:
        if bpm is None:
            return None
        bpm = float(bpm)
        if not (bpm > 0):
            return None
    except Exception:
        return None

    rows = REFERENCE_TABLE
    if not rows:
        return None

    # Clamp to table range
    if bpm <= rows[0].bpm:
        r = rows[0]
        return {"RR": r.rr, "P": r.p, "PR": r.pr, "QRS": r.qrs, "QT": r.qt, "QTc": r.qtc}
    if bpm >= rows[-1].bpm:
        r = rows[-1]
        return {"RR": r.rr, "P": r.p, "PR": r.pr, "QRS": r.qrs, "QT": r.qt, "QTc": r.qtc}

    # Find bracketing rows
    lower = rows[0]
    upper = rows[-1]
    for i in range(len(rows) - 1):
        a = rows[i]
        b = rows[i + 1]
        if a.bpm <= bpm <= b.bpm:
            lower, upper = a, b
            break

    if upper.bpm == lower.bpm:
        t = 0.0
    else:
        t = (bpm - lower.bpm) / (upper.bpm - lower.bpm)

    def interp(attr: str) -> float:
        return _lerp(getattr(lower, attr), getattr(upper, attr), t)

    return {
        "RR": interp("rr"),
        "P": interp("p"),
        "PR": interp("pr"),
        "QRS": interp("qrs"),
        "QT": interp("qt"),
        "QTc": interp("qtc"),
    }



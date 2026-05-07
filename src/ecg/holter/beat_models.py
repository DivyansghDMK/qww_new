"""Core beat/event models for the Holter pipeline."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass(frozen=True)
class Beat:
    timestamp: float
    rr_ms: float = 0.0
    qrs_width_ms: float = 0.0
    morphology_type: str = "N"
    template_id: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BeatAnnotation:
    beat_id: str
    auto_label: str = ""
    clinician_label: str = ""
    confidence: float = 0.0
    edited_by: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Episode:
    start_sec: float
    end_sec: float
    label: str
    beat_count: int = 0
    max_hr: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


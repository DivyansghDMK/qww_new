from dataclasses import dataclass

from .theme import (
    ADC_TO_MV,
    MAGNIFIER_ZOOM_LEVELS,
    TOOL_ANNOTATE,
    TOOL_CALIPER,
    TOOL_CURSORS,
    TOOL_MAGNIFY,
    TOOL_RULER,
    TOOL_SELECT,
)


TOOL_LABELS = {
    TOOL_SELECT: "Select",
    TOOL_RULER: "Ruler",
    TOOL_CALIPER: "Caliper",
    TOOL_MAGNIFY: "Magnify",
    TOOL_ANNOTATE: "Annotate",
}

TOOL_TOOLTIPS = {
    TOOL_SELECT: "Select / Pan",
    TOOL_RULER: "Measurement ruler\nDrag to measure dt and dmV",
    TOOL_CALIPER: "Interval caliper\nClick x2 -> RR/PP bpm",
    TOOL_MAGNIFY: "Magnifier\nHover to inspect local waveform",
    TOOL_ANNOTATE: "Quick annotate\nClick start then end on lead",
}

TOOL_HINTS = {
    TOOL_SELECT: "",
    TOOL_RULER: "Ruler active. Drag on any lead to measure time and amplitude.",
    TOOL_CALIPER: "Caliper active. Click once for line 1, again for line 2.",
    TOOL_MAGNIFY: "Magnify active. Hover over waveform to inspect the local signal.",
    TOOL_ANNOTATE: "Annotate active. Click start point, then click end point.",
}

_ALIASES = {
    "normal": TOOL_SELECT,
    "measuring ruler": TOOL_RULER,
    "parallel ruler": TOOL_CALIPER,
    "magnifying glass": TOOL_MAGNIFY,
}


@dataclass
class ECGToolEngine:
    active_tool: str = TOOL_SELECT

    def set_tool(self, tool_id: str) -> str:
        self.active_tool = canonical_tool(tool_id)
        return self.active_tool


def canonical_tool(tool_id: str) -> str:
    if not tool_id:
        return TOOL_SELECT
    normalized = str(tool_id).strip().lower()
    return _ALIASES.get(normalized, normalized)


def button_label(tool_id: str) -> str:
    return TOOL_LABELS.get(canonical_tool(tool_id), str(tool_id))


def tooltip(tool_id: str) -> str:
    return TOOL_TOOLTIPS.get(canonical_tool(tool_id), "")


def hint(tool_id: str) -> str:
    return TOOL_HINTS.get(canonical_tool(tool_id), "")


def cursor(tool_id: str):
    return TOOL_CURSORS.get(canonical_tool(tool_id), TOOL_CURSORS[TOOL_SELECT])


def tool_specs(include_annotate: bool = True):
    ordered = [TOOL_SELECT, TOOL_RULER, TOOL_CALIPER, TOOL_MAGNIFY]
    if include_annotate:
        ordered.append(TOOL_ANNOTATE)
    return [(tool_id, button_label(tool_id), tooltip(tool_id)) for tool_id in ordered]


def ruler_label(dt_ms: float, dv_mv: float, bpm: float = None) -> str:
    label = f"Δt={dt_ms:.1f}ms  Δ={dv_mv:.2f}mV"
    if bpm is not None and bpm > 0:
        label += f"  {bpm:.0f} BPM"
    return label


def caliper_label(dt_ms: float) -> str:
    bpm = 60000.0 / dt_ms if dt_ms > 0 else 0.0
    return f"RR/PP = {dt_ms:.1f} ms  ->  {bpm:.0f} bpm"


def interval_ms_from_pixels(pixel_dx: float, width: int, sample_count: int, fs: float) -> float:
    if width <= 1 or sample_count <= 1 or fs <= 0:
        return 0.0
    return (abs(pixel_dx) / float(width)) * (sample_count / float(fs)) * 1000.0


def amplitude_mv_from_pixels(pixel_dy: float, height: int, adc_range: float = 4096.0, adc_to_mv: float = ADC_TO_MV) -> float:
    if height <= 0:
        return 0.0
    dy_adc = abs(pixel_dy) / float(height) * adc_range
    return dy_adc * adc_to_mv


def magnifier_zoom_levels():
    return list(MAGNIFIER_ZOOM_LEVELS)

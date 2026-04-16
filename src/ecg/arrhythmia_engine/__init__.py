"""
Advanced Arrhythmia Detection Engine

This package encapsulates advanced clinical-grade arrhythmia logic,
handling everything from basic AFib thresholds to complex morphology 
template matching and multi-beat sequence analysis.
"""

from .baseline_analyzer import BaselineAnalyzer
from .template_matcher import TemplateMatcher
from .tachycardia_analyzer import TachycardiaAnalyzer
from .block_analyzer import BlockAnalyzer
from .pattern_analyzer import PatternAnalyzer
from .pacemaker_detector import PacemakerDetector

__all__ = [
    "BaselineAnalyzer",
    "TemplateMatcher",
    "TachycardiaAnalyzer",
    "BlockAnalyzer",
    "PatternAnalyzer",
    "PacemakerDetector",
]

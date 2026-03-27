"""Signal processing utilities for ECG"""
from .signal_processing import (
    extract_low_frequency_baseline,
    detect_signal_source,
    apply_adaptive_gain,
    apply_realtime_smoothing
)

__all__ = [
    'extract_low_frequency_baseline',
    'detect_signal_source',
    'apply_adaptive_gain',
    'apply_realtime_smoothing'
]

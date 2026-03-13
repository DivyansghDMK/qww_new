"""Signal processing helper functions for ECG display and analysis"""
import numpy as np
from typing import List, Optional
from collections import deque


def extract_low_frequency_baseline(signal: np.ndarray, sampling_rate: float = 500.0) -> float:
    """
    Extract very-low-frequency baseline estimate (< 0.3 Hz) for display anchoring.
    
    This removes respiration artifacts (0.1-0.35 Hz) while preserving very-low-frequency drift.
    
    Args:
        signal: ECG signal array
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Baseline estimate value
    """
    try:
        if len(signal) < 100:
            return np.mean(signal) if len(signal) > 0 else 0.0
        
        # Use moving average with window ~3.3 seconds (removes respiration ~0.3 Hz)
        window_samples = int(3.3 * sampling_rate)
        window_samples = max(10, min(window_samples, len(signal) // 4))
        
        if window_samples >= len(signal):
            return np.mean(signal)
        
        # Simple moving average (low-pass filter)
        kernel = np.ones(window_samples) / window_samples
        baseline_estimate = np.convolve(signal, kernel, mode='same')
        
        # Return median of baseline estimate (more robust than mean)
        return float(np.median(baseline_estimate))
    except Exception as e:
        print(f" Error extracting low-frequency baseline: {e}")
        return float(np.mean(signal)) if len(signal) > 0 else 0.0


def detect_signal_source(data: np.ndarray) -> str:
    """Detect if signal is from hardware or human body based on amplitude characteristics
    
    Args:
        data: ECG signal array
    
    Returns:
        'hardware' or 'human' based on signal characteristics
    """
    try:
        if len(data) == 0:
            return "hardware"
        
        # Calculate signal statistics
        signal_std = np.std(data)
        signal_range = np.max(data) - np.min(data)
        signal_mean = np.abs(np.mean(data))
        
        # Hardware signals typically have:
        # - Higher amplitude (ADC counts: 0-4095 range)
        # - More consistent amplitude
        # - Less variation
        
        # Human body signals typically have:
        # - Lower amplitude (mV range: -5 to +5 mV typically)
        # - More variation
        # - More noise
        
        # Thresholds (calibrated for typical hardware vs human signals)
        if signal_range > 1000 or signal_mean > 500:
            return "hardware"
        elif signal_range < 100 and signal_std < 50:
            return "hardware"
        else:
            return "human"
    except Exception as e:
        print(f" Error detecting signal source: {e}")
        return "hardware"


def apply_adaptive_gain(data: np.ndarray, signal_source: str, gain_factor: float) -> np.ndarray:
    """Apply gain based on signal source with adaptive scaling
    
    Args:
        data: ECG signal array
        signal_source: 'hardware' or 'human'
        gain_factor: Display gain factor (from get_display_gain)
    
    Returns:
        Gain-adjusted signal array
    """
    try:
        if len(data) == 0:
            return data
        
        # Convert to numpy array
        data = np.asarray(data, dtype=float)
        
        # Apply source-specific scaling
        if signal_source == "hardware":
            # Hardware signals: apply gain directly
            scaled = data * gain_factor
        else:
            # Human body signals: may need different scaling
            # For now, apply same gain
            scaled = data * gain_factor
        
        return scaled
    except Exception as e:
        print(f" Error applying adaptive gain: {e}")
        return data


def apply_realtime_smoothing(new_value: float, lead_index: int, 
                            smoothing_buffers: Optional[dict] = None,
                            buffer_size: int = 5) -> float:
    """Apply real-time smoothing for individual data points - medical grade
    
    Uses exponential moving average (EMA) for smooth, responsive filtering.
    
    Args:
        new_value: New data point value
        lead_index: Lead index (0-11)
        smoothing_buffers: Dictionary of smoothing buffers (will create if None)
        buffer_size: Size of smoothing buffer
    
    Returns:
        Smoothed value
    """
    try:
        if smoothing_buffers is None:
            smoothing_buffers = {}
        
        # Initialize buffer for this lead if needed
        buffer_key = f'lead_{lead_index}'
        if buffer_key not in smoothing_buffers:
            smoothing_buffers[buffer_key] = deque(maxlen=buffer_size)
        
        buffer = smoothing_buffers[buffer_key]
        
        # Add new value
        buffer.append(new_value)
        
        # Use median for robustness (rejects outliers)
        if len(buffer) >= 3:
            smoothed = float(np.median(list(buffer)))
        else:
            # Not enough samples yet - use mean
            smoothed = float(np.mean(list(buffer))) if len(buffer) > 0 else new_value
        
        return smoothed
    except Exception as e:
        print(f" Error applying real-time smoothing: {e}")
        return new_value

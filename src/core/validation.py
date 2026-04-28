"""
Data validation utilities for ECG application
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .constants import (
    MIN_HEART_RATE, MAX_HEART_RATE, PR_INTERVAL_MIN, PR_INTERVAL_MAX,
    QRS_DURATION_MIN, QRS_DURATION_MAX, QT_INTERVAL_MIN, QT_INTERVAL_MAX,
    DEFAULT_SAMPLING_RATE, DEFAULT_BUFFER_SIZE
)
from .exceptions import ECGDataError, ECGSignalError


class ECGValidator:
    """Validation utilities for ECG data and parameters"""
    
    @staticmethod
    def validate_sampling_rate(sampling_rate: Union[int, float]) -> bool:
        """Validate ECG sampling rate"""
        if not isinstance(sampling_rate, (int, float)):
            raise ECGDataError(f"Sampling rate must be numeric, got {type(sampling_rate)}")
        
        if sampling_rate <= 0:
            raise ECGDataError(f"Sampling rate must be positive, got {sampling_rate}")
        
        if sampling_rate < 100 or sampling_rate > 2000:
            raise ECGDataError(f"Sampling rate {sampling_rate} Hz is outside valid range (100-2000 Hz)")
        
        return True
    
    @staticmethod
    def validate_ecg_signal(signal: np.ndarray, sampling_rate: float) -> bool:
        """Validate ECG signal data"""
        if not isinstance(signal, np.ndarray):
            raise ECGSignalError("ECG signal must be a numpy array")
        
        if signal.ndim != 1:
            raise ECGSignalError(f"ECG signal must be 1D array, got {signal.ndim}D")
        
        if len(signal) == 0:
            raise ECGSignalError("ECG signal cannot be empty")
        
        if not np.isfinite(signal).all():
            raise ECGSignalError("ECG signal contains non-finite values (NaN or Inf)")
        
        # Check for reasonable amplitude range
        signal_range = np.max(signal) - np.min(signal)
        if signal_range < 0.001:  # Less than 1mV range
            raise ECGSignalError("ECG signal amplitude range too small")
        
        if signal_range > 100:  # More than 100mV range
            raise ECGSignalError("ECG signal amplitude range too large")
        
        return True
    
    @staticmethod
    def validate_heart_rate(heart_rate: Union[int, float]) -> bool:
        """Validate heart rate value"""
        if not isinstance(heart_rate, (int, float)):
            raise ECGDataError(f"Heart rate must be numeric, got {type(heart_rate)}")
        
        if not np.isfinite(heart_rate):
            raise ECGDataError("Heart rate must be finite")
        
        if heart_rate < MIN_HEART_RATE or heart_rate > MAX_HEART_RATE:
            raise ECGDataError(f"Heart rate {heart_rate} BPM is outside valid range ({MIN_HEART_RATE}-{MAX_HEART_RATE} BPM)")
        
        return True
    
    @staticmethod
    def validate_pr_interval(pr_interval: Union[int, float]) -> bool:
        """Validate PR interval value"""
        if not isinstance(pr_interval, (int, float)):
            raise ECGDataError(f"PR interval must be numeric, got {type(pr_interval)}")
        
        if not np.isfinite(pr_interval):
            raise ECGDataError("PR interval must be finite")
        
        if pr_interval < PR_INTERVAL_MIN or pr_interval > PR_INTERVAL_MAX:
            raise ECGDataError(f"PR interval {pr_interval} ms is outside valid range ({PR_INTERVAL_MIN}-{PR_INTERVAL_MAX} ms)")
        
        return True
    
    @staticmethod
    def validate_qrs_duration(qrs_duration: Union[int, float]) -> bool:
        """Validate QRS duration value"""
        if not isinstance(qrs_duration, (int, float)):
            raise ECGDataError(f"QRS duration must be numeric, got {type(qrs_duration)}")
        
        if not np.isfinite(qrs_duration):
            raise ECGDataError("QRS duration must be finite")
        
        if qrs_duration < QRS_DURATION_MIN or qrs_duration > QRS_DURATION_MAX:
            raise ECGDataError(f"QRS duration {qrs_duration} ms is outside valid range ({QRS_DURATION_MIN}-{QRS_DURATION_MAX} ms)")
        
        return True
    
    @staticmethod
    def validate_qt_interval(qt_interval: Union[int, float]) -> bool:
        """Validate QT interval value"""
        if not isinstance(qt_interval, (int, float)):
            raise ECGDataError(f"QT interval must be numeric, got {type(qt_interval)}")
        
        if not np.isfinite(qt_interval):
            raise ECGDataError("QT interval must be finite")
        
        if qt_interval < QT_INTERVAL_MIN or qt_interval > QT_INTERVAL_MAX:
            raise ECGDataError(f"QT interval {qt_interval} ms is outside valid range ({QT_INTERVAL_MIN}-{QT_INTERVAL_MAX} ms)")
        
        return True
    
    @staticmethod
    def validate_qrs_axis(qrs_axis: Union[int, float]) -> bool:
        """Validate QRS axis value"""
        if not isinstance(qrs_axis, (int, float)):
            raise ECGDataError(f"QRS axis must be numeric, got {type(qrs_axis)}")
        
        if not np.isfinite(qrs_axis):
            raise ECGDataError("QRS axis must be finite")
        
        if qrs_axis < -180 or qrs_axis > 180:
            raise ECGDataError(f"QRS axis {qrs_axis}° is outside valid range (-180° to 180°)")
        
        return True
    
    @staticmethod
    def validate_buffer_size(buffer_size: int) -> bool:
        """Validate buffer size"""
        if not isinstance(buffer_size, int):
            raise ECGDataError(f"Buffer size must be integer, got {type(buffer_size)}")
        
        if buffer_size <= 0:
            raise ECGDataError(f"Buffer size must be positive, got {buffer_size}")
        
        if buffer_size > DEFAULT_BUFFER_SIZE * 2:
            raise ECGDataError(f"Buffer size {buffer_size} is too large (max: {DEFAULT_BUFFER_SIZE * 2})")
        
        return True
    
    @staticmethod
    def validate_lead_name(lead_name: str) -> bool:
        """Validate ECG lead name"""
        from .constants import ECG_LEADS
        
        if not isinstance(lead_name, str):
            raise ECGDataError(f"Lead name must be string, got {type(lead_name)}")
        
        if lead_name not in ECG_LEADS:
            raise ECGDataError(f"Invalid lead name '{lead_name}'. Valid leads: {ECG_LEADS}")
        
        return True
    
    @staticmethod
    def validate_metrics(metrics: dict) -> bool:
        """Validate ECG metrics dictionary"""
        if not isinstance(metrics, dict):
            raise ECGDataError("Metrics must be a dictionary")
        
        required_keys = ['heart_rate', 'pr_interval', 'qrs_duration', 'qt_interval', 'qrs_axis']
        
        for key in required_keys:
            if key not in metrics:
                raise ECGDataError(f"Missing required metric: {key}")
        
        # Validate each metric
        ECGValidator.validate_heart_rate(metrics['heart_rate'])
        ECGValidator.validate_pr_interval(metrics['pr_interval'])
        ECGValidator.validate_qrs_duration(metrics['qrs_duration'])
        ECGValidator.validate_qt_interval(metrics['qt_interval'])
        ECGValidator.validate_qrs_axis(metrics['qrs_axis'])
        
        return True


def validate_ecg_data(signal: np.ndarray, sampling_rate: float, metrics: Optional[dict] = None) -> bool:
    """Comprehensive ECG data validation"""
    validator = ECGValidator()
    
    # Validate signal
    validator.validate_sampling_rate(sampling_rate)
    validator.validate_ecg_signal(signal, sampling_rate)
    
    # Validate metrics if provided
    if metrics is not None:
        validator.validate_metrics(metrics)
    
    return True


def sanitize_ecg_signal(signal: np.ndarray) -> np.ndarray:
    """Sanitize ECG signal by removing outliers and ensuring finite values"""
    # Remove non-finite values
    signal = signal[np.isfinite(signal)]
    
    if len(signal) == 0:
        raise ECGSignalError("Signal contains no finite values")
    
    # Remove extreme outliers (beyond 5 standard deviations)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    
    if std_val > 0:
        outlier_mask = np.abs(signal - mean_val) <= 5 * std_val
        signal = signal[outlier_mask]
    
    return signal

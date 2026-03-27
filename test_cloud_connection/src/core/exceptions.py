"""
Custom exceptions for ECG application
"""


class ECGError(Exception):
    """Base exception for ECG-related errors"""
    pass


class ECGDeviceError(ECGError):
    """Exception raised for ECG device connection/communication errors"""
    pass


class ECGDataError(ECGError):
    """Exception raised for ECG data processing errors"""
    pass


class ECGConfigError(ECGError):
    """Exception raised for configuration-related errors"""
    pass


class ECGFileError(ECGError):
    """Exception raised for file operation errors"""
    pass


class ECGAnalysisError(ECGError):
    """Exception raised for ECG analysis errors"""
    pass


class ECGSignalError(ECGError):
    """Exception raised for signal processing errors"""
    pass


class ECGUIError(ECGError):
    """Exception raised for UI-related errors"""
    pass


class ECGAudioError(ECGError):
    """Exception raised for audio-related errors"""
    pass


class ECGAuthenticationError(ECGError):
    """Exception raised for authentication errors"""
    pass


class ECGNetworkError(ECGError):
    """Exception raised for network-related errors"""
    pass

from scipy.signal import butter, filtfilt
import numpy as np

def display_filter(raw, fs):
    # For plotting + R peak detection
    # CRITICAL FIX: Clamp upper frequency to 0.99 * Nyquist to prevent errors if fs is low (e.g. <= 80Hz)
    nyquist = fs / 2
    if nyquist <= 0:
        return raw
        
    low = 0.5 / nyquist
    high = 40 / nyquist
    
    # Ensure high < 1.0 (Scipy requirement: 0 < Wh < 1)
    if high >= 1.0:
        high = 0.99
        
    # Ensure low < high
    if low >= high:
        # Fallback for extremely low sampling rates or invalid inputs
        low = 0.01
        high = 0.99
        
    b, a = butter(4, [low, high], 'band')
    return filtfilt(b, a, raw)

def measurement_filter(raw, fs):
    # For PR, QRS, QT, QTc (clinical)
    nyquist = fs / 2
    if nyquist <= 0:
        return raw

    low = 0.05 / nyquist
    high = 150 / nyquist
    
    # Ensure high < 1.0
    if high >= 1.0:
        high = 0.99
        
    # Ensure low < high
    if low >= high:
        low = 0.001
        high = 0.99

    b, a = butter(4, [low, high], 'band')
    return filtfilt(b, a, raw)

import numpy as np
from scipy.signal import butter, lfilter

def pan_tompkins(ecg, fs=500):
    """
    Pan-Tompkins QRS detection algorithm implementation.
    Args:
        ecg: 1D numpy array of ECG signal
        fs: Sampling frequency (Hz)
    Returns:
        r_peaks: Indices of detected R peaks
    """
    # 1. Bandpass filter (5-15 Hz)
    def bandpass_filter(signal, lowcut, highcut, fs, order=1):
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.99)
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high >= 1 or low >= high:
            return signal
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, signal)
    filtered = bandpass_filter(ecg, 5, 15, fs)
    # 2. Differentiate
    diff = np.ediff1d(filtered)
    # 3. Square
    squared = diff ** 2
    # 4. Moving window integration (150 ms window)
    window_size = int(0.15 * fs)
    mwa = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    # 5. Find peaks (adaptive threshold)
    threshold = np.mean(mwa) + 0.5 * np.std(mwa)
    min_distance = int(0.2 * fs)  # 200 ms
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(mwa, height=threshold, distance=min_distance)
    return peaks

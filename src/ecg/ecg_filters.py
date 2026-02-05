"""
ECG Filter Module - Medical-Grade Filtering for ECG Signals

This module provides medical-grade filtering for ECG signals:
- Respiration-Preserving Baseline Correction (MONITOR-GRADE): Removes motion/electrode drift while preserving respiration
- Baseline Wander Removal (GOLD STANDARD): Median + Mean filter (commercial ECG monitor standard)
- AC Filter: Notch filter to remove 50Hz or 60Hz power line interference
- EMG Filter: High-pass filter to remove muscle artifacts (25Hz, 35Hz, 45Hz, 75Hz, 100Hz, 150Hz)
- DFT Filter: High-pass filter to remove baseline wander (0.05Hz, 0.5Hz)

Usage:
    from ecg.ecg_filters import apply_ecg_filters, ecg_with_respiratory_baseline
    
    # MONITOR-GRADE: Respiration-preserving baseline correction (RECOMMENDED)
    clean_ecg, respiration = ecg_with_respiratory_baseline(signal, sampling_rate=500)
    
    # GOLD STANDARD: Median + Mean filter for baseline removal
    clean_signal = apply_baseline_wander_median_mean(signal, sampling_rate=500)
    
    # Or use full filter chain
    filtered_signal = apply_ecg_filters(
        signal, 
        sampling_rate=500,
        ac_filter="50",  # "off", "50", or "60"
        emg_filter="150",  # "25", "35", "45", "75", "100", "150"
        dft_filter="0.5"  # "off", "0.05", or "0.5"
    )
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt, find_peaks
from scipy.ndimage import uniform_filter1d
from typing import Union, Optional, Tuple


def normalize_adc_signal(signal: np.ndarray, preserve_amplitude: bool = True) -> np.ndarray:
    """
    Normalize ADC signal at the very start of processing pipeline.
    Removes DC offset only (variance normalization disabled to prevent instability).
    
    Algorithm:
    1. Remove DC offset (mean subtraction)
    2. DO NOT normalize variance (preserves signal amplitude, prevents flickering)
    
    Args:
        signal: Raw ADC signal
        preserve_amplitude: If True, only remove DC offset (default: True)
    
    Returns:
        Normalized signal (zero mean, original amplitude preserved)
    """
    signal = np.asarray(signal, dtype=float)
    
    if len(signal) == 0:
        return signal
    
    # Remove DC offset only (do NOT normalize variance to prevent instability)
    normalized = signal - np.mean(signal)
    
    # Variance normalization disabled - causes waves to "come and go" when applied to sliding windows
    # The variance changes between updates, causing amplitude instability
    
    return normalized


def detect_qrs_regions(ecg: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """
    Detect QRS regions for gated sharpening.
    Returns boolean mask: True in QRS regions, False elsewhere.
    
    Algorithm:
    1. Find R-peaks using peak detection
    2. Create QRS windows (±80 ms around each R-peak)
    3. Return mask indicating QRS regions
    
    Args:
        ecg: ECG signal
        fs: Sampling rate (Hz)
    
    Returns:
        Boolean array: True where QRS regions are located
    """
    if len(ecg) < 10:
        return np.zeros(len(ecg), dtype=bool)
    
    try:
        # Find R-peaks (simple peak detection)
        # Use absolute value to handle inverted leads
        abs_ecg = np.abs(ecg)
        threshold = np.percentile(abs_ecg, 75)  # 75th percentile as threshold
        
        # Find peaks with minimum distance (corresponds to ~200 BPM max)
        min_distance = int(0.3 * fs)  # 300 ms minimum between peaks
        
        peaks, _ = find_peaks(abs_ecg, height=threshold, distance=min_distance)
        
        # Create QRS region mask (±80 ms around each R-peak)
        qrs_window_ms = 80.0  # ±80 ms QRS window
        qrs_window_samples = int(qrs_window_ms * fs / 1000.0)
        
        mask = np.zeros(len(ecg), dtype=bool)
        for peak_idx in peaks:
            start = max(0, peak_idx - qrs_window_samples)
            end = min(len(ecg), peak_idx + qrs_window_samples + 1)
            mask[start:end] = True
        
        return mask
    except Exception:
        # Fallback: return all False (no sharpening)
        return np.zeros(len(ecg), dtype=bool)


def sharpen_qrs_gated(ecg: np.ndarray, fs: float = 500.0, alpha: float = 0.3) -> np.ndarray:
    """
    Sharpen QRS complexes with gating - only applies to QRS regions, not P or T waves.
    Monitor-grade: Preserves ST segment and QT interval.
    
    Algorithm:
    1. Detect QRS regions
    2. Calculate first derivative (highlights QRS edges)
    3. Apply sharpening only in QRS regions
    4. Leave P and T waves unchanged
    
    Args:
        ecg: EMG-suppressed ECG signal
        fs: Sampling rate (Hz)
        alpha: Sharpening factor (0.2-0.4, default 0.3)
    
    Returns:
        Sharpened ECG with preserved P and T waves
    """
    if len(ecg) < 3:
        return ecg
    
    try:
        # Detect QRS regions
        qrs_mask = detect_qrs_regions(ecg, fs)
        
        if not np.any(qrs_mask):
            # No QRS detected, return original
            return ecg
        
        # Calculate first derivative (emphasizes QRS edges)
        dt = 1.0 / fs
        derivative = np.gradient(ecg, dt)
        
        # Normalize derivative to prevent amplitude distortion
        if np.std(derivative) > 1e-10:
            derivative = derivative / np.std(derivative) * np.std(ecg)
        
        # Apply sharpening only in QRS regions
        sharpened = ecg.copy()
        sharpened[qrs_mask] = ecg[qrs_mask] + alpha * derivative[qrs_mask]
        
        return sharpened
    except Exception:
        return ecg


def apply_ac_filter(signal: np.ndarray, sampling_rate: float, ac_filter: str) -> np.ndarray:
    """
    Apply AC (Notch) Filter to remove power line interference
    
    Args:
        signal: Input ECG signal
        sampling_rate: Sampling frequency in Hz
        ac_filter: "off", "50", or "60" (Hz)
    
    Returns:
        Filtered signal
    """
    if ac_filter == "off" or not ac_filter:
        return signal
    
    try:
        notch_freq = float(ac_filter)  # 50 or 60 Hz
        
        # Design notch filter (bandstop filter)
        nyquist = sampling_rate / 2.0
        quality_factor = 25.0  # Quality factor for notch filter (reduced from 30 to avoid ringing)
        
        # Normalize frequency
        w0 = notch_freq / nyquist
        
        # Ensure frequency is within valid range (0 < w0 < 1)
        if w0 <= 0 or w0 >= 1:
            print(f" AC filter frequency {notch_freq}Hz is invalid for sampling rate {sampling_rate}Hz")
            return signal
        
        # Design IIR notch filter
        b, a = iirnotch(w0, quality_factor)
        
        # Apply filter (zero-phase filtering)
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal
    
    except Exception as e:
        print(f" Error applying AC filter ({ac_filter}Hz): {e}")
        return signal


def apply_emg_filter(signal: np.ndarray, sampling_rate: float, emg_filter: str) -> np.ndarray:
    """
    Apply EMG Filter (Low-pass filter) to suppress muscle artifacts.
    CORRECTED: Uses 35-40 Hz low-pass instead of high-pass to preserve QRS while removing EMG noise.
    
    Args:
        signal: Input ECG signal
        sampling_rate: Sampling frequency in Hz
        emg_filter: Cutoff frequency - "25", "35", "40", "45", "75", "100", or "150" (Hz)
    
    Returns:
        Filtered signal
    """
    if not emg_filter or emg_filter == "off":
        return signal
    
    try:
        cutoff_freq = float(emg_filter)
        nyquist = sampling_rate / 2.0

        # Safety: if sampling_rate is unexpectedly low, fall back to 500 Hz
        if sampling_rate < 100:  # hardware should be 500 Hz; avoid unstable filters
            nyquist = 500.0 / 2.0
            sampling_rate = 500.0

        # Clamp cutoff so it is always below Nyquist for a valid low‑pass filter
        max_allowed = nyquist * 0.9
        if cutoff_freq >= max_allowed:
            cutoff_freq = max_allowed
        if cutoff_freq <= 0:
            return signal

        # Normalize cutoff frequency
        normalized_cutoff = cutoff_freq / nyquist
        
        # Design 4th order low-pass Butterworth filter (zero-phase)
        b, a = butter(4, normalized_cutoff, btype='low')
        
        # Apply filter (zero-phase filtering)
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal
    
    except Exception as e:
        print(f" Error applying EMG filter ({emg_filter}Hz): {e}")
        return signal


def apply_dft_filter(signal: np.ndarray, sampling_rate: float, dft_filter: str) -> np.ndarray:
    """
    Apply DFT Filter (High-pass filter) to remove baseline wander
    
    Args:
        signal: Input ECG signal
        sampling_rate: Sampling frequency in Hz
        dft_filter: Cutoff frequency - "off", "0.05", or "0.5" (Hz)
    
    Returns:
        Filtered signal
    """
    if dft_filter == "off" or not dft_filter:
        return signal
    
    try:
        cutoff_freq = float(dft_filter)  # Low cutoff frequency (0.05 or 0.5 Hz)
        
        # Design high-pass Butterworth filter for baseline wander removal
        nyquist = sampling_rate / 2.0
        
        # Normalize cutoff frequency
        normalized_cutoff = cutoff_freq / nyquist
        
        # Ensure cutoff is within valid range
        if normalized_cutoff <= 0 or normalized_cutoff >= 1:
            print(f" DFT filter cutoff {cutoff_freq}Hz is invalid for sampling rate {sampling_rate}Hz")
            return signal
        
        # Design 2nd order high-pass Butterworth filter (gentle for baseline)
        b, a = butter(2, normalized_cutoff, btype='high')
        
        # Apply filter (zero-phase filtering)
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal
    
    except Exception as e:
        print(f" Error applying DFT filter ({dft_filter}Hz): {e}")
        return signal


def process_ecg_monitor_grade(ecg: np.ndarray, fs: float = 500.0, apply_sharpening: bool = False) -> np.ndarray:
    """
    Complete monitor-grade ECG processing pipeline with all corrections.
    
    Pipeline order:
    1. ADC normalization (DC offset removal only - variance preserved to prevent instability)
    2. Powerline removal (50 Hz notch, Q=25)
    3. Baseline correction (respiration preserved, 120 ms median)
    4. EMG suppression (35-40 Hz low-pass)
    5. QRS sharpening (gated, only in QRS regions) - OPTIONAL to prevent instability
    
    Args:
        ecg: Raw ECG ADC signal
        fs: Sampling rate (Hz, default 500)
        apply_sharpening: If True, apply QRS sharpening (default: False to prevent instability)
    
    Returns:
        Processed ECG with sharp P-QRS-T waves, preserved ST/QT
    """
    ecg = np.asarray(ecg, dtype=float)
    
    if len(ecg) < 10:
        return ecg
    
    # Step 1: ADC normalization (remove DC offset only - preserve amplitude to prevent flickering)
    ecg = normalize_adc_signal(ecg, preserve_amplitude=True)
    
    # Step 2: Remove powerline interference (50 Hz, Q=25 to avoid ringing)
    ecg = notch_filter_butterworth(ecg, fs, freq=50.0, q=25.0)
    
    # Step 3: Correct baseline (preserve respiration, 120 ms median to avoid QRS erosion)
    ecg, _ = ecg_with_respiratory_baseline(ecg, fs)
    
    # Step 4: Suppress EMG noise (35-40 Hz low-pass, not high-pass)
    ecg = apply_emg_filter(ecg, fs, "35")
    
    # Step 5: Sharpen QRS (gated, only in QRS regions, preserves P/T waves)
    # DISABLED by default - can cause instability if QRS detection is inconsistent
    if apply_sharpening:
        ecg = sharpen_qrs_gated(ecg, fs, alpha=0.3)
    
    return ecg


def apply_ecg_filters(
    signal: Union[np.ndarray, list],
    sampling_rate: float = 500,
    ac_filter: Optional[str] = None,
    emg_filter: Optional[str] = None,
    dft_filter: Optional[str] = None
) -> np.ndarray:
    """
    Apply all ECG filters in the correct order:
    1. DFT Filter (baseline wander removal) - first
    2. EMG Filter (muscle artifact removal)
    3. AC Filter (power line interference removal) - last
    
    Args:
        signal: Input ECG signal (numpy array or list)
        sampling_rate: Sampling frequency in Hz (default: 500)
        ac_filter: AC filter setting - "off", "50", or "60"
        emg_filter: EMG filter setting - "25", "35", "45", "75", "100", "150"
        dft_filter: DFT filter setting - "off", "0.05", or "0.5"
    
    Returns:
        Filtered signal as numpy array
    """
    # Convert to numpy array if needed
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal, dtype=float)
    
    # Check minimum signal length
    if len(signal) < 10:
        return signal
    
    # Apply filters in correct order
    filtered = signal.copy()
    
    # 1. DFT Filter first (removes slow baseline wander)
    if dft_filter:
        filtered = apply_dft_filter(filtered, sampling_rate, dft_filter)
    
    # 2. EMG Filter second (removes muscle artifacts)
    emg_suppresses_ac = False
    if emg_filter and str(emg_filter).lower() != "off":
        try:
            if float(emg_filter) < 60:
                emg_suppresses_ac = True
        except ValueError:
            pass
        filtered = apply_emg_filter(filtered, sampling_rate, emg_filter)
    
    # 3. AC Filter last (removes power line interference)
    if ac_filter and not emg_suppresses_ac:
        filtered = apply_ac_filter(filtered, sampling_rate, ac_filter)
    
    return filtered


def apply_baseline_wander_median_mean(signal: np.ndarray, sampling_rate: float = 500) -> np.ndarray:
    """
    GOLD STANDARD: Median Filter + Mean Filter for baseline wander removal
    Used in many commercial ECG monitors.
    
    Why it's special:
    - Removes baseline without touching QRS or ST segments
    - No phase distortion
    - Excellent for real-time display
    
    Algorithm:
    1. Median filter (200-300 ms) → removes QRS influence
    2. Moving average (600-1000 ms) → smooths baseline
    3. Subtract baseline from original signal
    
    Typical parameters (500 Hz):
    - Median filter: 200-300 ms (100-150 samples)
    - Mean filter: 600-1000 ms (300-500 samples)
    
    Args:
        signal: Input ECG signal
        sampling_rate: Sampling frequency in Hz (default: 500)
    
    Returns:
        Clean ECG signal with baseline wander removed
    """
    if len(signal) < 50:  # Need minimum samples
        return signal - np.mean(signal)
    
    try:
        # Step 1: Median filter (120 ms window - reduced from 200 ms to avoid QRS erosion)
        median_window_ms = 120.0  # milliseconds
        median_window = int(median_window_ms * sampling_rate / 1000.0)
        median_window = max(3, min(median_window, len(signal) // 2))  # Ensure odd and reasonable
        
        # Make window odd (required for median filter)
        if median_window % 2 == 0:
            median_window += 1
        
        # Apply median filter to remove QRS influence
        b1 = medfilt(signal, kernel_size=median_window)
        
        # Step 2: Moving average (600-1000 ms window)
        # Use 800 ms as middle value: 0.8 * Fs
        mean_window_ms = 800.0  # milliseconds
        mean_window = int(mean_window_ms * sampling_rate / 1000.0)
        mean_window = max(10, min(mean_window, len(b1) // 2))  # Ensure reasonable
        
        # Apply moving average to smooth baseline
        baseline = uniform_filter1d(b1.astype(float), size=mean_window, mode='nearest')
        
        # Step 3: Subtract baseline from original signal
        clean_ecg = signal - baseline
        
        return clean_ecg
    
    except Exception as e:
        print(f" Error applying median+mean baseline filter: {e}")
        # Fallback: simple mean subtraction
        return signal - np.mean(signal)


def notch_filter_butterworth(ecg: np.ndarray, fs: float, freq: float = 50.0, q: float = 25.0) -> np.ndarray:
    """
    Notch filter using Butterworth design (for India → 50 Hz)
    
    Args:
        ecg: Input ECG signal
        fs: Sampling frequency in Hz
        freq: Notch frequency (default: 50.0 Hz for India)
        q: Quality factor (default: 30.0)
    
    Returns:
        Filtered signal with powerline noise removed
    """
    if len(ecg) < 10:
        return ecg
    
    try:
        w0 = freq / (fs / 2.0)
        if w0 <= 0 or w0 >= 1:
            return ecg
        
        b, a = butter(2, [w0 - w0/q, w0 + w0/q], btype='bandstop')
        return filtfilt(b, a, ecg)
    except Exception as e:
        print(f" Error applying notch filter: {e}")
        return ecg


def estimate_baseline_drift(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Estimate baseline drift (motion + electrode drift) using median + mean filter.
    This is the clinical gold standard method.
    
    Algorithm:
    1. Median filter (200 ms) → removes QRS influence
    2. Moving average (1.8 s) → smooths baseline drift
    
    Args:
        ecg: Input ECG signal
        fs: Sampling frequency in Hz
    
    Returns:
        Estimated baseline drift signal
    """
    if len(ecg) < 50:
        return np.zeros_like(ecg)
    
    try:
        # Median filter removes QRS influence (120 ms window - reduced from 200 ms to avoid QRS erosion)
        median_window = int(0.12 * fs) | 1  # Ensure odd
        if median_window < 3:
            median_window = 3
        if median_window > len(ecg) // 2:
            median_window = (len(ecg) // 2) | 1
        
        med = medfilt(ecg, kernel_size=median_window)
        
        # Moving average removes remaining slow drift (1.8 s window)
        mean_window = int(1.8 * fs)
        if mean_window < 10:
            mean_window = 10
        if mean_window > len(med):
            mean_window = len(med)
        
        # Use convolution for moving average
        kernel = np.ones(mean_window) / mean_window
        drift = np.convolve(med, kernel, mode='same')
        
        return drift
    except Exception as e:
        print(f" Error estimating baseline drift: {e}")
        return np.zeros_like(ecg)


def extract_respiration(drift_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Extract respiration component from baseline drift signal (EDR - ECG-Derived Respiration).
    CORRECTED: Extracts from drift signal, not raw ECG, for better accuracy.
    
    Respiration band: 0.1 - 0.35 Hz
    
    Args:
        drift_signal: Baseline drift signal (from estimate_baseline_drift)
        fs: Sampling frequency in Hz
    
    Returns:
        Respiration waveform (EDR) with safe amplitude scaling
    """
    if len(drift_signal) < 10:
        return np.zeros_like(drift_signal)
    
    try:
        # Low-pass filter at 0.35 Hz to extract respiration from drift signal
        nyquist = fs / 2.0
        cutoff = 0.35 / nyquist
        
        if cutoff <= 0 or cutoff >= 1:
            return np.zeros_like(drift_signal)
        
        b, a = butter(2, cutoff, btype='low')
        resp = filtfilt(b, a, drift_signal)
        
        # Remove DC offset
        resp = resp - np.mean(resp)
        
        # Safe amplitude scaling instead of hard clipping
        # Scale to ±0.6 mV range if amplitude exceeds threshold
        max_amplitude = np.max(np.abs(resp)) if len(resp) > 0 else 0.0
        if max_amplitude > 0.6:
            scale_factor = 0.6 / max_amplitude
            resp = resp * scale_factor
        
        return resp
    except Exception as e:
        print(f" Error extracting respiration: {e}")
        return np.zeros_like(drift_signal)


def ecg_with_respiratory_baseline(ecg: np.ndarray, fs: float = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    🫀 MONITOR-GRADE: ECG with Respiration-Controlled Baseline
    
    This is exactly how bedside ECG monitors behave:
    - ✅ Stable ECG baseline
    - ✅ Baseline moves only with respiration
    - ✅ Motion drift suppressed
    - ✅ Safe for ST segment
    - ✅ Real-time friendly
    
    Algorithm:
    1. Remove powerline noise (50 Hz notch for India)
    2. Estimate unwanted baseline drift (motion + electrode drift)
    3. Extract respiration component (0.1-0.35 Hz)
    4. Clamp respiration amplitude (prevents crazy baseline swing)
    5. Reconstruct ECG: clean_ecg = ecg_notched - drift + respiration
    
    Args:
        ecg: Input ECG signal
        fs: Sampling frequency in Hz (default: 500)
    
    Returns:
        tuple: (clean_ecg, respiration)
            - clean_ecg: Stable ECG with breathing baseline
            - respiration: Respiration waveform (EDR)
    
    Example:
        clean_ecg, respiration = ecg_with_respiratory_baseline(signal, fs=500)
        # clean_ecg: Flat baseline at rest, smooth sinusoidal motion while breathing
        # respiration: Can be used to calculate respiration rate
    """
    ecg = np.asarray(ecg, dtype=float)
    
    if len(ecg) < 50:
        # Too short for filtering, just center it
        centered = ecg - np.mean(ecg)
        return centered, np.zeros_like(centered)
    
    try:
        # 1. Remove powerline noise (50 Hz for India, Q=25 to avoid ringing)
        ecg_notched = notch_filter_butterworth(ecg, fs, freq=50.0, q=25.0)
        
        # 2. Estimate unwanted baseline drift (motion + electrode drift)
        drift = estimate_baseline_drift(ecg_notched, fs)
        
        # 3. Extract respiration component from drift signal (0.1-0.35 Hz)
        # CORRECTED: Extract from drift, not raw ECG, with safe amplitude scaling
        respiration = extract_respiration(drift, fs)
        
        # 5. Reconstruct ECG: remove drift, add back respiration
        clean_ecg = ecg_notched - drift + respiration
        
        return clean_ecg, respiration
    
    except Exception as e:
        print(f" Error in respiration-preserving baseline correction: {e}")
        # Fallback: simple mean subtraction
        centered = ecg - np.mean(ecg)
        return centered, np.zeros_like(centered)


def respiration_rate(resp: np.ndarray, fs: float) -> float:
    """
    Calculate respiration rate from respiration waveform (EDR).
    
    Args:
        resp: Respiration waveform (from extract_respiration or ecg_with_respiratory_baseline)
        fs: Sampling frequency in Hz
    
    Returns:
        Respiration rate in breaths per minute (BPM)
    """
    if len(resp) < 10:
        return 0.0
    
    try:
        # Count zero crossings (positive-going)
        zero_crossings = np.where(np.diff(np.sign(resp)) > 0)[0]
        breaths = len(zero_crossings)
        
        # Calculate duration in minutes
        duration_min = len(resp) / fs / 60.0
        
        if duration_min > 0:
            return breaths / duration_min
        else:
            return 0.0
    except Exception as e:
        print(f" Error calculating respiration rate: {e}")
        return 0.0


def apply_ecg_filters_from_settings(
    signal: Union[np.ndarray, list],
    sampling_rate: float = 500,
    settings_manager=None
) -> np.ndarray:
    """
    Apply ECG filters using settings from SettingsManager
    
    Args:
        signal: Input ECG signal
        sampling_rate: Sampling frequency in Hz
        settings_manager: SettingsManager instance (optional, will create if not provided)
    
    Returns:
        Filtered signal
    """
    # Import here to avoid circular imports
    if settings_manager is None:
        from utils.settings_manager import SettingsManager
        settings_manager = SettingsManager()
    
    # Get filter settings
    ac_filter = settings_manager.get_setting("filter_ac", "50")
    emg_filter = settings_manager.get_setting("filter_emg", "150")
    dft_filter = settings_manager.get_setting("filter_dft", "0.5")
    
    # Apply filters
    return apply_ecg_filters(
        signal=signal,
        sampling_rate=sampling_rate,
        ac_filter=ac_filter,
        emg_filter=emg_filter,
        dft_filter=dft_filter
    )


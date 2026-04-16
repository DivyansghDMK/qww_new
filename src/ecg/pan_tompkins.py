import numpy as np
from scipy.signal import butter, filtfilt

def pan_tompkins(ecg, fs=500):
    """
    Pan-Tompkins QRS detection algorithm implementation.
    Args:
        ecg: 1D numpy array of ECG signal
        fs: Sampling frequency (Hz)
    Returns:
        r_peaks: Indices of detected R peaks (aligned to the bandpassed ECG,
                 not the integrated envelope).
    """
    # 1. Bandpass filter (5-15 Hz) — classic QRS band.
    # Use filtfilt (zero-phase) so peak indices are not delayed by IIR phase shift.
    def bandpass_filter(signal, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.99)
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high >= 1 or low >= high:
            return signal
        b, a = butter(order, [low, high], btype='band')
        # If signal is too short for filtfilt padding, fall back to causal filtering
        try:
            if len(signal) < max(len(a), len(b)) * 3:
                from scipy.signal import lfilter
                return lfilter(b, a, signal)
            return filtfilt(b, a, signal)
        except Exception:
            return signal

    x = np.asarray(ecg, dtype=float)
    if x.size < 10 or not np.isfinite(fs) or fs <= 0:
        return np.array([], dtype=int)

    filtered = bandpass_filter(x, 5, 15, fs)
    # 2. Differentiate
    diff = np.ediff1d(filtered, to_end=0.0)
    # 3. Square
    squared = diff ** 2
    # 4. Moving window integration (150 ms window)
    window_size = max(1, int(round(0.15 * fs)))
    if window_size > squared.size:
        window_size = max(1, squared.size // 4)
    mwa = np.convolve(squared, np.ones(window_size) / float(window_size), mode='same')

    # 5. Find peaks (adaptive threshold with fallbacks)
    from scipy.signal import find_peaks

    # FIX-PT1: Adaptive refractory period — prevents T-wave double-detection.
    # FIX-PT2: Was 0.20*fs (200ms) = exactly RR at 300 BPM → find_peaks
    # missed every other peak → false 150 BPM.
    # Now 0.16*fs (160ms at 500Hz = 80 samples) → supports up to ~375 BPM.
    # Pass-2 below still adapts refractory for low HR to kill T-wave ghosts.
    min_distance = max(1, int(round(0.16 * fs)))
    mean_mwa = float(np.mean(mwa)) if mwa.size else 0.0
    std_mwa = float(np.std(mwa)) if mwa.size else 0.0

    peaks = np.array([], dtype=int)
    for k in (0.50, 0.30, 0.15, 0.10):
        thr = mean_mwa + k * std_mwa
        try:
            peaks, _ = find_peaks(mwa, height=thr, distance=min_distance)
        except Exception:
            peaks = np.array([], dtype=int)
        if peaks.size >= 2:
            break

    # FIX-PT1 continued: if HR looks < 100 BPM, rerun with tighter refractory
    # to kill T-wave false peaks that appear at ~0.33*RR after true R.
    if peaks.size >= 3:
        ipi = np.diff(peaks) / fs * 1000.0  # inter-peak intervals in ms
        median_ipi = float(np.median(ipi))
        if median_ipi > 600:  # HR < 100 BPM
            adaptive_refrac = max(int(round(0.20 * fs)),
                                  int(round(0.33 * median_ipi / 1000.0 * fs)))
            if adaptive_refrac > min_distance:
                min_distance = adaptive_refrac
                peaks = np.array([], dtype=int)
                for k in (0.50, 0.30, 0.15, 0.10):
                    thr = mean_mwa + k * std_mwa
                    try:
                        peaks, _ = find_peaks(mwa, height=thr, distance=min_distance)
                    except Exception:
                        peaks = np.array([], dtype=int)
                    if peaks.size >= 2:
                        break

    if peaks.size == 0:
        return np.array([], dtype=int)

    # 6. Search-back: map each envelope peak to the true R-peak location
    # by finding max(|bandpassed ECG|) within ±75 ms.
    search_half = max(1, int(round(0.075 * fs)))
    r_locs = []
    for p in peaks:
        left = max(0, int(p) - search_half)
        right = min(filtered.size, int(p) + search_half + 1)
        if right <= left:
            continue
        seg = filtered[left:right]
        # Use abs() to handle inverted leads (R can be negative)
        r = left + int(np.argmax(np.abs(seg)))
        r_locs.append(r)

    if not r_locs:
        return np.array([], dtype=int)

    r_locs = np.array(sorted(set(r_locs)), dtype=int)

    # 7. De-duplicate R-peaks within refractory period: keep the larger |amplitude|
    dedup = []
    for r in r_locs:
        if not dedup:
            dedup.append(int(r))
            continue
        if int(r) - dedup[-1] < min_distance:
            prev = dedup[-1]
            if abs(filtered[int(r)]) > abs(filtered[int(prev)]):
                dedup[-1] = int(r)
        else:
            dedup.append(int(r))

    return np.array(dedup, dtype=int)
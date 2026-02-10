import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import Dict, Any, Optional

def bandpass(x, fs):
    """
    Apply 0.5-40Hz bandpass filter as per user specification.
    """
    if not np.isfinite(fs) or fs <= 0:
        return x
    nyquist = fs / 2.0
    low = max(0.5 / nyquist, 0.001)
    high = min(40.0 / nyquist, 0.99)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high >= 1 or low >= high:
        return x
    b, a = butter(2, [low, high], 'band')
    return filtfilt(b, a, x)

def calculate_comprehensive_metrics(lead_data: np.ndarray, fs: float = 500.0) -> Dict[str, Any]:
    """
    Calculate comprehensive ECG metrics using the user-provided logic.
    
    Args:
        lead_data: Numpy array of Lead II data
        fs: Sampling rate in Hz
    
    Returns:
        Dictionary containing calculated metrics (HR, RR, PR, QRS, QT, QTc)
        Values are None if calculation fails.
    """
    results = {
        "heart_rate": None,
        "rr_interval": None,
        "pr_interval": None,
        "qrs_duration": None,
        "qt_interval": None,
        "qtc_interval": None
    }
    
    # Ensure numpy array
    sig = np.array(lead_data, float)
    
    # User's check: return if < 2000 samples
    # We might want to relax this for responsiveness, but let's stick to the logic for now
    # or maybe adapt it. 2000 samples @ 500Hz is 4 seconds.
    if len(sig) < 2000:
        return results

    sig -= np.mean(sig)
    filt = bandpass(sig, fs)

    energy = np.diff(filt)**2
    
    # User logic: peaks,_ = find_peaks(energy,distance=int(0.3*FS),height=np.mean(energy)*5)
    peaks, _ = find_peaks(energy, distance=int(0.3*fs), height=np.mean(energy)*5)
    
    if len(peaks) < 2:
        return results

    r = peaks[-1]
    last_r_idx = peaks[-2] 
    
    # Calculate RR and HR
    RR = (r - last_r_idx) / fs
    if RR <= 0:
        return results
        
    HR = round(60 / RR)
    RRms = RR * 1000
    
    results["heart_rate"] = HR
    results["rr_interval"] = RRms

    # ---------- QRS (energy envelope) ----------
    win = int(0.12 * fs)
    
    # Boundary checks
    if r - win < 0 or r + win >= len(filt):
        return results
        
    seg = np.abs(filt[r-win:r+win])
    if len(seg) == 0:
        return results
        
    th = 0.25 * np.max(seg)

    qrs_region = np.where(seg > th)[0]

    if len(qrs_region) < 10:
        return results

    qrs_start = r - win + qrs_region[0]
    qrs_end   = r - win + qrs_region[-1]

    # -------- True Q onset --------
    Q_onset = qrs_start - int(0.04*fs)
    Q_onset = max(Q_onset, 0)

    # -------- P --------
    pl = max(0, Q_onset - int(0.25*fs))
    pr = Q_onset - int(0.05*fs)
    
    if pr <= pl:
        return results
    
    # p_onset=pl+np.argmax(np.abs(np.diff(sig[pl:pr])))
    # Note: user used sig[pl:pr] (raw signal mean subtracted) for P-wave detection?
    # Wait, user code:
    # sig -= np.mean(sig)
    # filt = bandpass(sig)
    # ...
    # p_onset=pl+np.argmax(np.abs(np.diff(sig[pl:pr])))
    # Yes, it uses 'sig' (mean subtracted raw) not 'filt'.
    
    p_region = sig[pl:pr]
    if len(p_region) < 2:
        return results
        
    p_onset_idx = np.argmax(np.abs(np.diff(p_region)))
    p_onset = pl + p_onset_idx

    # -------- T (clinical tangent method) --------
    t_start = qrs_end + int(0.06 * fs)
    t_stop  = qrs_end + int(0.65 * RR * fs)

    t_stop = min(t_stop, len(sig)-1)
    if t_stop <= t_start:
        return results

    treg = sig[t_start:t_stop]
    if len(treg) < int(0.04 * fs):
        return results

    # T-peak
    t_peak_relative = np.argmax(np.abs(treg))
    t_peak = t_start + t_peak_relative

    # Use only the last half of T-wave for slope
    tail_start = t_peak + int(0.04 * fs)
    tail_stop  = min(t_stop, t_peak + int(0.25 * RR * fs))

    if tail_stop <= tail_start:
        return results

    tail = sig[tail_start:tail_stop]

    # Smooth tail
    tail = np.convolve(tail, np.ones(7)/7, mode="same")

    d = np.diff(tail)
    if len(d) == 0:
        return results

    i = np.argmin(d)
    slope = d[i]

    # baseline = np.mean(sig[qrs_start-80:qrs_start-40])
    base_start = max(0, qrs_start - 80)
    base_end = max(0, qrs_start - 40)
    
    if base_end > base_start:
        baseline = np.mean(sig[base_start:base_end])
    else:
        baseline = 0

    if slope != 0:
        # t_end = int(tail_start + i + (baseline - sig[tail_start + i]) / slope)
        t_end = int(tail_start + i + (baseline - sig[tail_start + i]) / slope)
    else:
        t_end = t_peak + int(0.12 * fs)

    # Safety clamp
    min_end = t_peak + int(0.04 * fs)
    max_end = qrs_end + int(0.7 * RR * fs)
    t_end = int(np.clip(t_end, min_end, max_end))

    # -------- INTERVALS --------
    PR  = (qrs_start - p_onset) / fs * 1000
    QRS = (qrs_end - qrs_start) / fs * 1000
    QT  = (t_end - Q_onset) / fs * 1000
    if RR > 0:
        QTc = (QT/1000) / np.sqrt(RR) * 1000
    else:
        QTc = 0

    results["pr_interval"] = PR
    results["qrs_duration"] = QRS
    results["qt_interval"] = QT
    results["qtc_interval"] = QTc
    
    return results

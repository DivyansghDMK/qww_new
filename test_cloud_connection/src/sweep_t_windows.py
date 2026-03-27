
import numpy as np
from ecg.utils.helpers import generate_realistic_ecg_waveform
from scipy.signal import find_peaks

fs = 500
np.random.seed(42)
ecg_i = generate_realistic_ecg_waveform(duration_seconds=10, sampling_rate=fs, heart_rate=60, lead_name='I') * 1200
ecg_avf = generate_realistic_ecg_waveform(duration_seconds=10, sampling_rate=fs, heart_rate=60, lead_name='aVF') * 1200
ecg_ii = generate_realistic_ecg_waveform(duration_seconds=10, sampling_rate=fs, heart_rate=60, lead_name='II') * 1200

r_peaks, _ = find_peaks(ecg_ii, height=300, distance=fs//2)
r_idx = r_peaks[2] # pick one beat

def get_axis(start_ms, end_ms):
    s = r_idx + int(start_ms * fs / 1000)
    e = r_idx + int(end_ms * fs / 1000)
    # simple baseline as mean before R
    base_i = np.mean(ecg_i[r_idx-100:r_idx-50])
    base_avf = np.mean(ecg_avf[r_idx-100:r_idx-50])
    net_i = np.sum(ecg_i[s:e] - base_i)
    net_avf = np.sum(ecg_avf[s:e] - base_avf)
    angle = np.degrees(np.arctan2(net_avf, net_i))
    return angle

print(f"100-300ms: {get_axis(100, 300):.1f}")
print(f"120-400ms: {get_axis(120, 400):.1f}")
print(f"140-450ms: {get_axis(140, 450):.1f}")
print(f"150-400ms: {get_axis(150, 400):.1f}")
print(f"100-400ms: {get_axis(100, 400):.1f}")
print(f"140-300ms: {get_axis(140, 300):.1f}")
print(f"140-350ms: {get_axis(140, 350):.1f}")
print(f"180-450ms: {get_axis(180, 450):.1f}") # capturing peak and tail

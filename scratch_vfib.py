import json
import numpy as np
with open('api_data/record_297.json') as f:
    data = json.load(f)['data']

sig = np.array([float(x) for x in data['lead1_reading'].split(',') if x.strip()])
# Center
sig = sig - np.mean(sig)

# Zero crossings
zero_crossings = np.where(np.diff(np.sign(sig)))[0]
zc_rate_hz = len(zero_crossings) / (len(sig) / 500.0)

# Amplitude
std = np.std(sig)
range_val = np.ptp(sig)

print(f"VFIB STATS (record 297):")
print(f"Zero Crossing Rate: {zc_rate_hz} Hz")
print(f"StdDev: {std}")
print(f"Peak-to-Peak: {range_val}")

with open('api_data/record_100.json') as f:
    data2 = json.load(f)['data']
sig2 = np.array([float(x) for x in data2['lead1_reading'].split(',') if x.strip()])
sig2 = sig2 - np.mean(sig2)
print(f"\nNORMAL STATS (record 100):")
print(f"Zero Crossing Rate: {len(np.where(np.diff(np.sign(sig2)))[0]) / (len(sig2) / 500.0)} Hz")
print(f"StdDev: {np.std(sig2)}")
print(f"Peak-to-Peak: {np.ptp(sig2)}")


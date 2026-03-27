
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

import ecg.clinical_measurements as cm
from ecg.metrics.intervals import calculate_qtc_auto

def validate():
    print("=== ECG MEASUREMENT LOGIC VALIDATION ===")
    fs = 500.0
    
    # 1. Test QTc Adaptive Logic
    print(f"\n[1] Testing QTc Formulas (RR threshold 600ms):")
    qt_ms = 400
    qtc_low_hr = calculate_qtc_auto(qt_ms, 1000.0, 60) # Bazett
    qtc_high_hr = calculate_qtc_auto(qt_ms, 500.0, 120) # Fridericia
    
    print(f"  - HR 60 (RR 1000ms): QT={qt_ms} -> QTc={qtc_low_hr} (Bazett expected 400)")
    print(f"  - HR 120 (RR 500ms): QT={qt_ms} -> QTc={qtc_high_hr} (Fridericia expected ~504)")

    # 2. Test Detection Functions Existence
    print(f"\n[2] Verifying core functions in clinical_measurements.py:")
    funcs = [
        'measure_pr_from_median_beat',
        'measure_qrs_duration_from_median_beat',
        'measure_qt_from_median_beat',
        'detect_qrs_onset_slope_assisted',
        'detect_qrs_offset_slope_assisted'
    ]
    for func in funcs:
        exists = hasattr(cm, func)
        print(f"  - {func}: {'FOUND' if exists else 'MISSING'}")

    # 3. Test PR detection on a simple ramp
    print(f"\n[3] Testing PR logic (QRS Onset - P Onset):")
    # P starts at 100, QRS starts at 200, R at 250
    signal = np.zeros(500)
    signal[100:150] = np.linspace(0, 100, 50) # P wave
    signal[150:200] = np.linspace(100, 0, 50)
    signal[200:250] = np.linspace(0, 1000, 50) # QRS onset to R
    
    time_axis = (np.arange(500) - 250) / fs * 1000.0
    tp_baseline = 0.0
    
    pr = cm.measure_pr_from_median_beat(signal, time_axis, fs, tp_baseline)
    print(f"  - Measured PR: {pr} ms (P-onset=100, QRS-onset=200, fs=500 -> 200ms expected)")

    print("\n=== VALIDATION COMPLETED ===")

if __name__ == "__main__":
    validate()

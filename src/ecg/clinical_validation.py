"""
Clinical Measurement Validation Module (GE/Philips Standard)

Internal assertions to ensure display filters never affect clinical measurements.
"""

import numpy as np


def validate_clinical_measurement_independence(raw_signal, filtered_signal, measurement_name):
    """
    Validate that clinical measurements are independent of display filters.
    
    Args:
        raw_signal: Raw ECG signal used for clinical measurements
        filtered_signal: Display-filtered signal (should NOT affect measurements)
        measurement_name: Name of measurement being validated
    
    Raises:
        AssertionError if raw and filtered signals are identical (indicating filter leakage)
    """
    # Ensure raw signal is actually raw (not filtered)
    assert raw_signal is not filtered_signal, \
        f"❌ VALIDATION FAILED: {measurement_name} - Raw signal is same object as filtered signal"
    
    # Check that signals differ (filtered should be different from raw)
    if len(raw_signal) == len(filtered_signal) and len(raw_signal) > 0:
        signal_diff = np.abs(raw_signal - filtered_signal)
        max_diff = np.max(signal_diff)
        assert max_diff > 1e-6, \
            f"❌ VALIDATION FAILED: {measurement_name} - Raw and filtered signals are identical (max diff={max_diff})"


def validate_rv5_sv1_signs(rv5_mv, sv1_mv):
    """
    Validate RV5/SV1 signs match GE/Philips standard.
    
    Args:
        rv5_mv: RV5 amplitude in mV
        sv1_mv: SV1 amplitude in mV
    
    Raises:
        AssertionError if signs are incorrect
    """
    if rv5_mv is not None:
        assert rv5_mv >= 0, \
            f"❌ VALIDATION FAILED: RV5 must be positive (got {rv5_mv:.3f} mV)"
    
    if sv1_mv is not None:
        assert sv1_mv <= 0, \
            f"❌ VALIDATION FAILED: SV1 must be negative (got {sv1_mv:.3f} mV)"


def validate_rv5_sv1_sum(rv5_mv, sv1_mv, rv5_sv1_sum):
    """
    Validate RV5+SV1 calculation: RV5 + abs(SV1).
    
    Args:
        rv5_mv: RV5 amplitude in mV
        sv1_mv: SV1 amplitude in mV (negative)
        rv5_sv1_sum: Calculated RV5+SV1 sum
    
    Raises:
        AssertionError if calculation is incorrect
    """
    if rv5_mv is not None and sv1_mv is not None:
        expected_sum = rv5_mv + abs(sv1_mv)
        if rv5_sv1_sum is not None:
            diff = abs(rv5_sv1_sum - expected_sum)
            assert diff < 0.001, \
                f"❌ VALIDATION FAILED: RV5+SV1 = {rv5_sv1_sum:.3f}, expected {expected_sum:.3f} (diff={diff:.3f})"


def validate_qtc_formulas(qt_ms, rr_ms, qtc_ms, qtcf_ms):
    """
    Validate QTc and QTcF formulas match GE/Philips standard.
    
    Args:
        qt_ms: QT interval in milliseconds
        rr_ms: RR interval in milliseconds
        qtc_ms: QTc (Bazett) in milliseconds
        qtcf_ms: QTcF (Fridericia) in milliseconds
    
    Raises:
        AssertionError if formulas are incorrect
    """
    if qt_ms > 0 and rr_ms > 0:
        # Convert to seconds
        qt_sec = qt_ms / 1000.0
        rr_sec = rr_ms / 1000.0
        
        # Bazett: QTc = QT / sqrt(RR)
        expected_qtc_sec = qt_sec / np.sqrt(rr_sec)
        expected_qtc_ms = expected_qtc_sec * 1000.0
        
        if qtc_ms is not None and qtc_ms > 0:
            diff_qtc = abs(qtc_ms - expected_qtc_ms)
            assert diff_qtc < 1.0, \
                f"❌ VALIDATION FAILED: QTc = {qtc_ms:.1f} ms, expected {expected_qtc_ms:.1f} ms (diff={diff_qtc:.1f})"
        
        # Fridericia: QTcF = QT / RR^(1/3)
        expected_qtcf_sec = qt_sec / (rr_sec ** (1.0 / 3.0))
        expected_qtcf_ms = expected_qtcf_sec * 1000.0
        
        if qtcf_ms is not None and qtcf_ms > 0:
            diff_qtcf = abs(qtcf_ms - expected_qtcf_ms)
            assert diff_qtcf < 1.0, \
                f"❌ VALIDATION FAILED: QTcF = {qtcf_ms:.1f} ms, expected {expected_qtcf_ms:.1f} ms (diff={diff_qtcf:.1f})"


def validate_qtcf_units(qtcf_value, qtcf_unit):
    """
    Validate QTcF is reported in milliseconds, never seconds.
    
    Args:
        qtcf_value: QTcF value
        qtcf_unit: Unit string (should be "ms" or "milliseconds")
    
    Raises:
        AssertionError if unit is incorrect
    """
    if qtcf_value is not None:
        assert qtcf_unit.lower() in ['ms', 'milliseconds', 'millisecond'], \
            f"❌ VALIDATION FAILED: QTcF unit must be ms, got '{qtcf_unit}'"
        
        # QTcF should be in reasonable range (200-600 ms)
        assert 200 <= qtcf_value <= 600, \
            f"❌ VALIDATION FAILED: QTcF = {qtcf_value:.1f} ms (outside normal range 200-600 ms)"


def validate_median_beat_beats(num_beats):
    """
    Validate median beat uses 8-12 beats (GE Marquette style).
    
    Args:
        num_beats: Number of beats used in median beat
    
    Raises:
        AssertionError if beat count is outside range
    """
    assert 8 <= num_beats <= 12, \
        f"❌ VALIDATION FAILED: Median beat must use 8-12 beats, got {num_beats}"


def validate_tp_baseline_usage(tp_baseline, signal_mean):
    """
    Validate TP baseline is used (not zero-line or signal mean).
    
    Args:
        tp_baseline: TP baseline value
        signal_mean: Mean of entire signal
    
    Raises:
        AssertionError if TP baseline equals signal mean (indicating wrong baseline)
    """
    if tp_baseline is not None and signal_mean is not None:
        diff = abs(tp_baseline - signal_mean)
        # TP baseline should differ from signal mean (unless signal is perfectly flat)
        signal_std = abs(signal_mean) * 0.1  # 10% threshold
        if signal_std > 1e-6:
            assert diff > signal_std, \
                f"❌ VALIDATION FAILED: TP baseline ({tp_baseline:.2f}) equals signal mean ({signal_mean:.2f})"


def validate_report_scaling(speed_mm_s, gain_mm_mv):
    """
    Validate report uses fixed diagnostic scaling (25 mm/s, 10 mm/mV).
    
    Args:
        speed_mm_s: Wave speed in mm/s
        gain_mm_mv: Wave gain in mm/mV
    
    Raises:
        AssertionError if scaling is not fixed diagnostic scale
    """
    assert abs(speed_mm_s - 25.0) < 0.1, \
        f"❌ VALIDATION FAILED: Report speed must be 25 mm/s, got {speed_mm_s:.1f} mm/s"
    
    assert abs(gain_mm_mv - 10.0) < 0.1, \
        f"❌ VALIDATION FAILED: Report gain must be 10 mm/mV, got {gain_mm_mv:.1f} mm/mV"


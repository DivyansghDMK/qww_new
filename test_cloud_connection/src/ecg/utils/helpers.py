"""ECG helper functions and utilities"""
import numpy as np
from typing import Tuple


def get_display_gain(wave_gain_mm: float) -> float:
    """
    ECG display gain calculation (hospital standard):
    10 mm/mV = 1.0x (clinical baseline)
    
    Args:
        wave_gain_mm: Wave gain setting in mm/mV (e.g., 2.5, 5, 10, 20)
    
    Returns:
        Display gain factor:
        - 2.5 mm/mV → 0.25x
        - 5 mm/mV → 0.5x
        - 10 mm/mV → 1.0x (baseline)
        - 20 mm/mV → 2.0x
    
    This matches GE / Philips monitor behavior.
    """
    try:
        return float(wave_gain_mm) / 10.0
    except (ValueError, TypeError):
        return 1.0  # Default to 10mm/mV baseline


def generate_realistic_ecg_waveform(duration_seconds=10, sampling_rate=500, heart_rate=72, lead_name="II"):
    """
    Generate realistic ECG waveform with proper PQRST complexes
    - duration_seconds: Length of waveform in seconds
    - sampling_rate: Samples per second (Hz)
    - heart_rate: Beats per minute
    - lead_name: Lead name for lead-specific characteristics
    """
    # Calculate parameters
    total_samples = int(duration_seconds * sampling_rate)
    rr_interval = 60.0 / heart_rate  # RR interval in seconds
    samples_per_beat = int(rr_interval * sampling_rate)
    
    # Create time array
    t = np.linspace(0, duration_seconds, total_samples)
    
    # Initialize waveform
    ecg = np.zeros(total_samples)
    
    # Lead-specific characteristics (amplitudes in mV)
    lead_characteristics = {
        "I": {"p_amp": 0.1, "qrs_amp": 0.8, "t_amp": 0.2, "baseline": 0.0},
        "II": {"p_amp": 0.15, "qrs_amp": 1.2, "t_amp": 0.3, "baseline": 0.0},
        "III": {"p_amp": 0.05, "qrs_amp": 0.6, "t_amp": 0.15, "baseline": 0.0},
        "aVR": {"p_amp": -0.1, "qrs_amp": -0.8, "t_amp": -0.2, "baseline": 0.0},
        "aVL": {"p_amp": 0.08, "qrs_amp": 0.7, "t_amp": 0.18, "baseline": 0.0},
        "aVF": {"p_amp": 0.12, "qrs_amp": 0.9, "t_amp": 0.25, "baseline": 0.0},
        "V1": {"p_amp": 0.05, "qrs_amp": 0.3, "t_amp": 0.1, "baseline": 0.0},
        "V2": {"p_amp": 0.08, "qrs_amp": 0.8, "t_amp": 0.2, "baseline": 0.0},
        "V3": {"p_amp": 0.1, "qrs_amp": 1.0, "t_amp": 0.25, "baseline": 0.0},
        "V4": {"p_amp": 0.12, "qrs_amp": 1.1, "t_amp": 0.3, "baseline": 0.0},
        "V5": {"p_amp": 0.1, "qrs_amp": 1.0, "t_amp": 0.25, "baseline": 0.0},
        "V6": {"p_amp": 0.08, "qrs_amp": 0.8, "t_amp": 0.2, "baseline": 0.0}
    }
    
    char = lead_characteristics.get(lead_name, lead_characteristics["II"])
    
    # Generate beats
    beat_start = 0
    while beat_start < total_samples:
        # P wave (atrial depolarization) - 80-120ms
        p_duration = 0.1  # 100ms
        p_samples = int(p_duration * sampling_rate)
        p_start = beat_start
        p_end = min(p_start + p_samples, total_samples)
        
        if p_start < total_samples:
            p_t = np.linspace(0, p_duration, p_end - p_start)
            p_wave = char["p_amp"] * np.sin(np.pi * p_t / p_duration) * np.exp(-2 * p_t / p_duration)
            ecg[p_start:p_end] += p_wave
        
        # PR interval (isoelectric line) - 120-200ms
        pr_duration = 0.16  # 160ms
        pr_samples = int(pr_duration * sampling_rate)
        pr_start = p_end
        pr_end = min(pr_start + pr_samples, total_samples)
        
        # QRS complex (ventricular depolarization) - 80-120ms
        qrs_duration = 0.08  # 80ms
        qrs_samples = int(qrs_duration * sampling_rate)
        qrs_start = pr_end
        qrs_end = min(qrs_start + qrs_samples, total_samples)
        
        if qrs_start < total_samples:
            qrs_t = np.linspace(0, qrs_duration, qrs_end - qrs_start)
            # Q wave (small negative deflection)
            q_wave = -char["qrs_amp"] * 0.1 * np.exp(-10 * qrs_t / qrs_duration)
            # R wave (large positive deflection)
            r_wave = char["qrs_amp"] * np.sin(np.pi * qrs_t / qrs_duration) * np.exp(-3 * qrs_t / qrs_duration)
            # S wave (negative deflection after R)
            s_wave = -char["qrs_amp"] * 0.3 * np.exp(-5 * qrs_t / qrs_duration)
            
            qrs_complex = q_wave + r_wave + s_wave
            ecg[qrs_start:qrs_end] += qrs_complex
        
        # ST segment (isoelectric) - 80-120ms
        st_duration = 0.08  # 80ms
        st_samples = int(st_duration * sampling_rate)
        st_start = qrs_end
        st_end = min(st_start + st_samples, total_samples)
        
        # T wave (ventricular repolarization) - 160-200ms
        t_duration = 0.18  # 180ms
        t_samples = int(t_duration * sampling_rate)
        t_start = st_end
        t_end = min(t_start + t_samples, total_samples)
        
        if t_start < total_samples:
            t_t = np.linspace(0, t_duration, t_end - t_start)
            t_wave = char["t_amp"] * np.sin(np.pi * t_t / t_duration) * np.exp(-3 * t_t / t_duration)
            ecg[t_start:t_end] += t_wave
        
        # Move to next beat
        beat_start += samples_per_beat
    
    # Add baseline
    ecg += char["baseline"]
    
    # Add noise (realistic ECG noise)
    noise = np.random.normal(0, 0.01, total_samples)
    ecg += noise
    
    return ecg


class SamplingRateCalculator:
    """
    Calculate sampling rate from sample count and time.
    
    FIX #4: Now uses SamplingRateGuard to prevent wrong sampling rate during
    warmup (was reporting 228.9 Hz instead of 500 Hz for first ~1000 packets).
    """
    def __init__(self, update_interval_sec=5, configured_rate_hz=500.0):
        from ..acquisition_utils import SamplingRateGuard
        import time
        
        self.sample_count = 0
        self.last_update_time = time.monotonic()
        self.update_interval = update_interval_sec
        self.sampling_rate = configured_rate_hz  # FIX #4: Start with configured rate
        
        # FIX #4: Sampling rate guard with warmup
        self._rate_guard = SamplingRateGuard(
            configured_rate_hz=configured_rate_hz,
            warmup_seconds=2.0,
            min_samples=1000,
            max_deviation_pct=40.0
        )

    def add_sample(self):
        import time
        self.sample_count += 1
        
        # FIX #4: Track samples for warmup guard
        self._rate_guard.record_samples(1)
        
        current_time = time.monotonic()
        elapsed = current_time - self.last_update_time
        
        if elapsed >= self.update_interval:
            # Calculate raw detected rate
            detected_rate = self.sample_count / elapsed
            
            # FIX #4: Feed to guard for validation
            self._rate_guard.update_detected_rate(detected_rate)
            
            # FIX #4: Get guarded rate (configured during warmup, validated after)
            self.sampling_rate = self._rate_guard.get_rate()
            
            self.sample_count = 0
            self.last_update_time = current_time
            
        return self.sampling_rate

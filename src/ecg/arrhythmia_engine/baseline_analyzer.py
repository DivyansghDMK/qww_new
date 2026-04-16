import numpy as np
from typing import Dict, List, Sequence, Optional
from scipy.signal import welch

class BaselineAnalyzer:
    """
    Analyzes the baseline of the ECG signal (specifically searching for 
    fibrillatory f-waves or flutter F-waves) to accurately differentiate 
    AFib and Atrial Flutter from normal baseline noise or frequent ectopics.
    """
    
    def __init__(self, fs: float):
        self.fs = fs

    def check_atrial_fibrillation_flutter(
        self, 
        signal_dict: Dict[str, np.ndarray], 
        beats: List[Dict[str, object]], 
        rr_intervals_ms: np.ndarray
    ) -> List[str]:
        """
        Differentiates between Afib, Flutter, and Normal based on RR variability
        and baseline characteristics. 
        """
        arrhythmias = []
        if len(beats) < 3 or rr_intervals_ms.size < 2:
            return arrhythmias

        rr_variability = float(np.max(rr_intervals_ms) - np.min(rr_intervals_ms))
        mean_rr = float(np.mean(rr_intervals_ms))
        
        p_absent_ratio = float(np.mean([not bool(beat.get("p_present")) for beat in beats]))
        
        # Pull lead V1 or II for baseline frequency analysis (if available)
        v1_sig = signal_dict.get("V1")
        primary_sig = signal_dict.get("II")
        analysis_sig = v1_sig if v1_sig is not None and v1_sig.size > 0 else primary_sig

        # Fallback to pure rule-based if signal is missing or too short
        if analysis_sig is None or analysis_sig.size < self.fs * 2:
            if p_absent_ratio > 0.70 and rr_variability > 120.0:
                arrhythmias.append("Arterial fibrillation 1")
            return arrhythmias

        # Basic Frequency Domain Check (Welch's Method) for 4-9Hz F-waves vs 4-6Hz Flutter
        try:
            # Mask out the QRS complexes to isolate the baseline
            baseline_mask = np.ones_like(analysis_sig, dtype=bool)
            for beat in beats:
                q_on = int(beat.get("q_onset") or beat.get("r_peak") or 0) - int(0.05 * self.fs)
                j_pt = int(beat.get("j_point") or beat.get("r_peak") or 0) + int(0.10 * self.fs)
                q_on = max(0, q_on)
                j_pt = min(analysis_sig.size, j_pt)
                baseline_mask[q_on:j_pt] = False

            baseline_only = analysis_sig[baseline_mask]
            
            if baseline_only.size > self.fs:
                freqs, psd = welch(baseline_only, fs=self.fs, nperseg=int(self.fs))
                
                # Fibrillatory range ~4-9 Hz. Flutter range ~4-6 Hz but with high regularity.
                fib_band = (freqs >= 4.0) & (freqs <= 9.0)
                total_power = np.sum(psd)
                fib_power = np.sum(psd[fib_band]) 
                
                fib_ratio = fib_power / max(total_power, 1e-9)

                # Pure AFib
                if p_absent_ratio > 0.6 and fib_ratio > 0.35 and rr_variability > 120.0:
                    # We output the user's specific string
                    arrhythmias.append("Arterial fibrillation 1")
                
                # Atrial Flutter: typically tight 250-350 bpm atrial rate, usually strict AV conduction ratio
                elif p_absent_ratio > 0.6 and fib_ratio > 0.4 and rr_variability <= 120.0:
                    arrhythmias.append("artrial flutter")

        except Exception as e:
            # Fallback
            if p_absent_ratio > 0.70 and rr_variability > 120.0:
                arrhythmias.append("Arterial fibrillation 1")
                
        return arrhythmias

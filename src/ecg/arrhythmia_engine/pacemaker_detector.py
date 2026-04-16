import numpy as np
from typing import Dict, List, Sequence

class PacemakerDetector:
    """
    Analyzes the ECG signal for artificial pacemaker spikes.
    Identifies if pacing is Atrial, Ventricular, or Sequential (AV) and 
    whether it is fixed-rate (asynchronous) or demand-based.
    """
    
    def __init__(self, fs: float):
        self.fs = fs

    def detect_pacemaker(self, signal_dict: Dict[str, Sequence[float]], beats: List[Dict[str, object]]) -> List[str]:
        arrhythmias = []
        if not beats:
            return arrhythmias
            
        # We need a lead to search for spikes. Usually Lead II or V1 shows pacer spikes well.
        sig = np.asarray(signal_dict.get("II", []))
        if sig.size == 0:
            return arrhythmias
            
        # Pacer spikes are extremely high-frequency (high slew rate).
        # We can detect them by taking the first derivative (np.diff) and looking for 
        # sudden massive jumps that exceed physiological limits.
        
        diff_sig = np.diff(sig)
        # Threshold: Assuming standard mV scale, a jump of > 2mV in 1-2 samples (at 500Hz) is usually a spike 
        # Or relative to the median absolute deviation of the derivative
        mad = float(np.median(np.abs(diff_sig - np.median(diff_sig))))
        if mad == 0:
            return arrhythmias
            
        # Empirical threshold for "huge sudden jump". Since we are on raw ADC scale (10mm/mV approx 200 counts/mV),
        # a real spike is virtually vertical and huge, easily > 300 counts in 1-2 samples.
        spike_threshold = max(mad * 20.0, 300.0) 
        spike_indices = np.where(np.abs(diff_sig) > spike_threshold)[0]
        
        if len(spike_indices) == 0:
            return arrhythmias # No pacing detected
            
        # Analyze relationship of spikes to P waves and QRS complexes
        atrial_pacing_count = 0
        ventricular_pacing_count = 0
        
        for beat in beats:
            q_on = int(beat.get("q_onset") or beat.get("r_peak") or 0)
            p_on = int(beat.get("p_onset") or beat.get("p_peak") or 0)
            
            # Check for spike in the 50ms (0.05 * fs) window preceding P wave
            if beat.get("p_present") and p_on > 0:
                p_window_start = max(0, p_on - int(0.05 * self.fs))
                p_spikes = [s for s in spike_indices if p_window_start <= s <= p_on]
                if p_spikes:
                    atrial_pacing_count += 1
                    
            # Check for spike in the 50ms window preceding QRS
            if q_on > 0:
                v_window_start = max(0, q_on - int(0.05 * self.fs))
                v_spikes = [s for s in spike_indices if v_window_start <= s <= q_on]
                if v_spikes:
                    ventricular_pacing_count += 1
                    
        total_beats = len(beats)
        
        if atrial_pacing_count > 0 and ventricular_pacing_count > 0:
            arrhythmias.append("atr vent sequential")
        elif atrial_pacing_count > total_beats * 0.8:
            arrhythmias.append("artrial 80bpm") # Usually mapped to the user string, though HR varies
        elif atrial_pacing_count > 0 or ventricular_pacing_count > 0:
            # Check if Demand or Asynchronous
            # If standard deviation of RR intervals is very low (fixed rate), it's Asynchronous
            rr_array = [float(b.get("rr_ms")) for b in beats if b.get("rr_ms") is not None]
            if rr_array and np.std(rr_array) < 10.0:
                arrhythmias.append("asynchronous 75 bpm")
            else:
                # If there are intrinsic beats mixed with paced beats, it's demand
                pacing_ratio = (atrial_pacing_count + ventricular_pacing_count) / total_beats
                if pacing_ratio > 0.8:
                    arrhythmias.append("demand Freq sinus")
                else:
                    arrhythmias.append("demand Occ Sinus")

        return arrhythmias

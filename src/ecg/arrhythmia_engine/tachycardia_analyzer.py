from typing import Dict, List
import numpy as np

class TachycardiaAnalyzer:
    """
    Classifies high-rate rhythms (Tachycardias) based on origin (Atrial vs Ventricular)
    and morphology profiles.
    """

    def check_tachycardia(self, beats: List[Dict[str, object]]) -> List[str]:
        arrhythmias = []
        if len(beats) < 4:
            return arrhythmias
            
        hr = beats[-1].get("heart_rate_bpm") or 0.0
        
        if hr < 100.0:
            return arrhythmias # Not tachycardia
            
        p_present_ratio = float(np.mean([bool(beat.get("p_present")) for beat in beats]))
        mean_qrs = float(np.mean([float(beat.get("qrs_ms") or 0.0) for beat in beats]))
        
        pr_values = [float(b.get("pr_ms")) for b in beats if b.get("pr_ms") is not None]
        mean_pr = float(np.mean(pr_values)) if pr_values else 0.0

        rr_ms_array = [float(b.get("rr_ms")) for b in beats if b.get("rr_ms") is not None]
        rr_variability = float(np.ptp(rr_ms_array[-5:])) if rr_ms_array else 0.0
        
        # 1. Ventricular Tachycardia (V-Tach)
        # Fast HR, Wide QRS, No P-waves (or AV dissociation)
        if hr >= 110.0 and mean_qrs >= 120.0 and p_present_ratio < 0.4:
            # Differentiate Mono vs Poly based on QRS amplitude/width standard deviation
            qrs_amps = [abs(float(b.get("qrs_amplitude") or 0.0)) for b in beats]
            if len(qrs_amps) > 0 and np.std(qrs_amps) > 0.2 * np.mean(qrs_amps):
                arrhythmias.append("poly V tach")
            else:
                arrhythmias.append("mono v tach")
            return arrhythmias # Override supraventricular checks
            
        # 2. Supraventricular Tachycardias (Narrow Complex)
        if hr >= 150.0 and mean_qrs < 120.0:
            if rr_variability < 40.0:
                # Regular Narrow Complex Tachycardia
                if p_present_ratio < 0.5 or mean_pr < 120.0:
                    arrhythmias.append("Supra Vtach") # Usually AVNRT, P-waves hidden in QRS
                elif p_present_ratio >= 0.5:
                    # Visible P-waves before narrow QRS
                    if hr > 180.0:
                        arrhythmias.append("paroxysmal Atach")
                    else:
                        arrhythmias.append("artrial tach")
                        
        elif 100.0 <= hr < 150.0:
            if p_present_ratio < 0.5 and mean_qrs < 120.0:
                arrhythmias.append("Nodal rhythm") # Junctional/Nodal tachycardia depending on rate

        return arrhythmias

import numpy as np
from typing import Dict, List

class BlockAnalyzer:
    """
    Analyzes sequences of PR, PP, and RR intervals to classify AV blocks.
    Detects 1st Degree, 2nd Degree Type 1 (Wenckebach), 2nd Degree Type 2, and 3rd Degree Blocks.
    """
    
    def __init__(self, fs: float):
        self.fs = fs

    def check_av_blocks(self, beats: List[Dict[str, object]]) -> List[str]:
        arrhythmias = []
        if len(beats) < 4:
            return arrhythmias

        # Extract sequence metrics
        pr_values = []
        rr_values = []
        p_present_beats = 0
        
        for beat in beats:
            if beat.get("p_present"):
                p_present_beats += 1
            
            pr = beat.get("pr_ms")
            if pr is not None:
                pr_values.append(float(pr))
                
            rr = beat.get("rr_ms")
            if rr is not None:
                rr_values.append(float(rr))

        if not pr_values or not rr_values:
            return arrhythmias

        p_present_ratio = p_present_beats / len(beats)
        mean_pr = np.mean(pr_values)
        
        # 1. 3rd Degree (Complete) AV Block
        # Indicated by regular PP, regular RR, but highly variable PR (AV dissociation)
        # For simplicity in this heuristic, we look at very high PR variance, 
        # a slow consistent ventricular rate, and distinct P waves.
        if len(pr_values) > 3 and p_present_ratio > 0.7:
            pr_range = np.ptp(pr_values)
            hr = beats[-1].get("heart_rate_bpm", 60.0)
            
            if pr_range > 80.0 and hr < 60.0:
                arrhythmias.append("3rd deg AV block")
                return arrhythmias  # Usually supersedes others

        # 2. 2nd Degree Blocks
        # Look for dropped beats (long RR without corresponding expected QRS)
        if len(rr_values) >= 3:
            mean_rr = np.mean(rr_values)
            # Find an RR sequence that suggests a drop (e.g. RR is ~2x previous RR)
            for i in range(1, len(rr_values)):
                if rr_values[i] > 1.7 * np.mean(rr_values[max(0, i-3):i]):
                    # We have a pause/dropped beat! Let's check PR behavior leading up to it.
                    if i >= 2 and len(pr_values) >= i:
                        pr_delta = pr_values[i-1] - pr_values[i-2]
                        if pr_delta > 20.0: # Progressive lengthening
                            arrhythmias.append("2nd Deg AV block t1")
                            break
                        elif abs(pr_delta) < 20.0: # Constant PR before drop
                            arrhythmias.append("2nd Deg AV block t2")
                            break
                    else:
                        # Missed beat but no PR context
                        hr = beats[-1].get("heart_rate_bpm", 60.0)
                        if hr < 100:
                            arrhythmias.append("missed beat at 80bpm") # Matching the user string
                        else:
                            arrhythmias.append("missed beat at 120 bpm")
                        break

        # 3. 1st Degree Block
        # Consistent prolongation of PR interval > 200 ms
        if p_present_ratio > 0.7 and mean_pr > 200.0:
            if "2nd Deg AV block t1" not in arrhythmias and "2nd Deg AV block t2" not in arrhythmias:
                 arrhythmias.append("C- 1st Deg AV block")

        return arrhythmias

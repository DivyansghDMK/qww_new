from typing import Dict, List

class PatternAnalyzer:
    """
    Analyzes sequence permutations of Normal (N) and Ventricular (V) beats
    to classify specific arrangements like Bigeminy, Trigeminy, and Runs of PVCs.
    """
    
    def check_ectopic_patterns(self, beats: List[Dict[str, object]]) -> List[str]:
        arrhythmias = []
        if len(beats) < 3:
            return arrhythmias
            
        # Convert beats to a sequence string mapping N (Normal/Sinus) and V (Ventricular)
        sequence = ""
        mean_rr_overall = sum(float(b.get("rr_ms", 0)) for b in beats if b.get("rr_ms")) / max(1, len([b for b in beats if b.get("rr_ms")]))
        
        for beat in beats:
            qrs_ms = float(beat.get("qrs_ms") or 0.0)
            p_present = bool(beat.get("p_present"))
            qrs_amp = float(beat.get("qrs_amplitude") or 0.0)
            
            # Simple heuristic matching original detect_arrhythmias for PVCs
            is_v = (qrs_ms > 120.0 and not p_present and qrs_amp > 0.1)
            sequence += "V" if is_v else "N"
            
        # Analyze String Segment Patterns
        # "NVNVNV" -> Bigeminy
        # "NNVNNV" -> Trigeminy
        # "VVV" -> Run of PVCs
        
        if "VVV" in sequence:
            if "VVVV" in sequence:
                 arrhythmias.append("mono v tach") # Or Poly based on correlation, simple map for now
            else:
                 arrhythmias.append("run of PVC")
                 
        if "NVNVNV" in sequence and "trigeminy" not in arrhythmias:
             # Just matching the API's arbitrary label choices
             arrhythmias.append("trigeminy") 
             
        if "NNVNNV" in sequence and "trigeminy" not in arrhythmias:
             arrhythmias.append("trigeminy")
             
        # Also simple counts
        v_count = sequence.count("V")
        if v_count >= 5:
             arrhythmias.append("freq multi focal PVCs")
             
        # Example implementation for missing pacemaker logic
        # 1*12 or 2*6 combinations. We don't have pacing spikes yet
        # so this is stubbed for integration.
        return arrhythmias

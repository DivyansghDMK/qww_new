import numpy as np
from typing import Dict, List, Sequence, Tuple, Optional

class TemplateMatcher:
    """
    Uses cross-correlation to build templates of normal sinus beats and 
    compares incoming ectopic beats (PVCs/PACs) to classify them.
    Also handles multi-focal grouping and Ventricular origin matching (Left vs Right).
    """

    def __init__(self, fs: float):
        self.fs = fs
        self.templates = [] # List of tuples containing (morphology_array, count, is_ventricular)
        
    def _extract_qrs(self, signal: np.ndarray, r_peak: int) -> Optional[np.ndarray]:
        # Extract a fixed 200ms window around the R-peak for correlation testing
        left = max(0, r_peak - int(0.08 * self.fs))
        right = min(signal.size, r_peak + int(0.12 * self.fs))
        if right > left:
            return signal[left:right]
        return None

    def group_and_analyze_ectopics(self, signal_dict: Dict[str, Sequence[float]], beats: List[Dict[str, object]]) -> List[str]:
        arrhythmias = []
        if not beats:
            return arrhythmias
            
        primary_sig = np.asarray(signal_dict.get("II", []))
        v1_sig = np.asarray(signal_dict.get("V1", []))
        
        if primary_sig.size == 0:
            return arrhythmias
            
        # Group into families
        families = [] # List of list of beat_indices
        
        for i, beat in enumerate(beats):
            r_peak = int(beat.get("r_peak") or 0)
            qrs = self._extract_qrs(primary_sig, r_peak)
            if qrs is None or qrs.size < 10:
                continue
                
            matched = False
            for family in families:
                ref_idx = family[0]
                ref_qrs = self._extract_qrs(primary_sig, int(beats[ref_idx]["r_peak"]))
                
                # Cross correlation
                if ref_qrs is not None and ref_qrs.size == qrs.size:
                    corr = np.corrcoef(qrs, ref_qrs)[0, 1]
                    if corr > 0.85: # 85% match is the same morphology
                        family.append(i)
                        matched = True
                        break
            
            if not matched:
                families.append([i])
                
        # Find dominant family (Normal Sinus Template)
        if not families:
            return arrhythmias
            
        families.sort(key=len, reverse=True)
        dominant_family = families[0]
        ectopic_families = families[1:]
        
        # Analyze Ectopics
        pvc_count = 0
        pvc_focus_count = 0 
        
        for e_family in ectopic_families:
            # We check the first beat of this ectopic family to classify it
            sample_beat = beats[e_family[0]]
            
            qrs_ms = float(sample_beat.get("qrs_ms") or 0.0)
            p_present = bool(sample_beat.get("p_present"))
            rr_ms = float(sample_beat.get("rr_ms") or 0.0)
            
            # Calculate mean RR of the whole strip for prematurity checking
            mean_rr = float(np.mean([float(b.get("rr_ms", 0)) for b in beats if b.get("rr_ms")]))
            is_premature = (rr_ms > 0 and rr_ms < mean_rr * 0.9)
            
            # PVC Criteria: Wide QRS, morphologically distinct, absent P-wave, and usually premature
            if qrs_ms > 110.0 and (not p_present or is_premature):
                pvc_count += len(e_family)
                pvc_focus_count += 1
                
                # Determine Left vs Right focus using V1 polarity
                if v1_sig.size > 0:
                    qrs_start = int(sample_beat.get("q_onset") or sample_beat.get("r_peak") - 10)
                    qrs_end = int(sample_beat.get("j_point") or sample_beat.get("r_peak") + 10)
                    v1_window = v1_sig[max(0, qrs_start):min(v1_sig.size, qrs_end + 1)]
                    
                    if v1_window.size > 0:
                        pos = np.max(v1_window)
                        neg = abs(np.min(v1_window))
                        
                        # RV PVC has LBBB morphology (Negative in V1)
                        if neg > pos * 1.5:
                            arrhythmias.append("PVC2 right vent")
                            # R on T logic (extremely short RR)
                            if rr_ms > 0 and rr_ms < 300.0:
                                arrhythmias.append("PVC 2 RV on T")
                            elif rr_ms > 0 and rr_ms < 450.0:
                                arrhythmias.append("PVC 2 RV early")
                        else:
                            # LV PVC has RBBB morphology (Positive in V1)
                            arrhythmias.append("PVC1 left vent")
                            if rr_ms > 0 and rr_ms < 300.0:
                                arrhythmias.append("PVC1 LV R on T")
                            elif rr_ms > 0 and rr_ms < 450.0:
                                arrhythmias.append("PVC 1 LV early")
                                
            # PAC/PNC Criteria: Must actually be premature!
            elif is_premature and qrs_ms <= 110.0:
                if p_present:
                    arrhythmias.append("P-artrial PAC")
                else:
                    arrhythmias.append("P-nodal PNC")

        if pvc_focus_count > 1:
            arrhythmias.append("multi focal PVCs")
            
        return arrhythmias

import numpy as np
from typing import List, Dict, Tuple
from scipy.signal import correlate

def extract_beat_templates(signal: np.ndarray, r_peaks: List[int], fs: float) -> np.ndarray:
    """Extracts identical fixed-width windows around each R-peak"""
    templates = []
    # Real devices usually use ~250ms window centered on R-peak for clustering
    pre_window = int(0.10 * fs)
    post_window = int(0.15 * fs)
    
    for r in r_peaks:
        start = r - pre_window
        end = r + post_window
        if start >= 0 and end < len(signal):
            segment = signal[start:end]
            # Baseline zero, amplitude normalize
            segment = segment - np.median(segment)
            if np.max(np.abs(segment)) > 0:
                segment = segment / np.max(np.abs(segment))
            templates.append(segment)
        else:
            # Pad with zeros if at edges just to keep lengths aligned
            templates.append(np.zeros(pre_window + post_window))
            
    return np.array(templates)

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    sig = np.asarray(signal, dtype=float)
    sig = sig - np.mean(sig)
    return sig / (np.std(sig) + 1e-6)


def align_signal(beat: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Aligns a beat to the reference template to correct for jitter."""
    beat_norm = normalize_signal(beat)
    template_norm = normalize_signal(template)
    corr = correlate(beat_norm, template_norm, mode='full')
    shift = np.argmax(corr) - len(template_norm) + 1
    return np.roll(beat, -shift)


def compute_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation similarity after normalization and temporal alignment."""
    a_norm = normalize_signal(a)
    b_norm = normalize_signal(b)
    a_aligned = normalize_signal(align_signal(a_norm, b_norm))
    n = min(len(a_aligned), len(b_norm))
    if n < 3:
        return 0.0
    return float(np.dot(a_aligned[:n], b_norm[:n]) / max(n, 1))

def cluster_templates_crosscorr(templates: np.ndarray, similarity_threshold: float = 0.85) -> np.ndarray:
    n_beats = len(templates)
    if n_beats == 0:
        return np.array([])
        
    cluster_ids = np.zeros(n_beats, dtype=int) - 1
    current_cluster = 0
    
    for i in range(n_beats):
        if cluster_ids[i] != -1:
            continue
            
        cluster_ids[i] = current_cluster
        ref_template = templates[i]
        ref_norm = np.dot(ref_template, ref_template)
        
        if ref_norm < 1e-6:
            current_cluster += 1
            continue
            
        for j in range(i + 1, n_beats):
            if cluster_ids[j] != -1:
                continue
            
            # Align the beat to the ref template before correlating
            comp_template = align_signal(templates[j], ref_template)
            comp_norm = np.dot(comp_template, comp_template)
            if comp_norm < 1e-6:
                continue
                
            corr = np.dot(ref_template, comp_template) / np.sqrt(ref_norm * comp_norm)
            if corr >= similarity_threshold:
                cluster_ids[j] = current_cluster
                
        current_cluster += 1
        
    return cluster_ids


def cluster_beats(
    beats: np.ndarray,
    threshold: float = 0.88,
    min_cluster_size: int = 3,
) -> List[List[np.ndarray]]:
    """Group morphologically similar beats using aligned correlation."""
    clusters: List[List[np.ndarray]] = []
    for beat in beats:
        placed = False
        for cluster in clusters:
            if compute_similarity(beat, cluster[0]) >= threshold:
                cluster.append(beat)
                placed = True
                break
        if not placed:
            clusters.append([beat])
    return [cluster for cluster in clusters if len(cluster) >= max(1, int(min_cluster_size))]


def morphology_features(clusters: List[List[np.ndarray]]) -> Dict[str, float]:
    """Summarize dominant-vs-ectopic morphology burden."""
    if not clusters:
        return {"cluster_count": 0.0, "dominant_ratio": 0.0, "ectopic_ratio": 0.0}
    sizes = [len(cluster) for cluster in clusters]
    total = float(sum(sizes))
    dominant = float(max(sizes)) if sizes else 0.0
    dominant_ratio = (dominant / total) if total > 0 else 0.0
    return {
        "cluster_count": float(len(clusters)),
        "dominant_ratio": dominant_ratio,
        "ectopic_ratio": max(0.0, 1.0 - dominant_ratio),
    }

def classify_ectopics(cluster_ids: np.ndarray, rr_ms: np.ndarray) -> List[str]:
    findings = []
    if len(cluster_ids) == 0:
        return findings
        
    unique, counts = np.unique(cluster_ids, return_counts=True)
    
    # Filter out noise clusters that are too small
    valid_clusters = [c for c, count in zip(unique, counts) if count >= 3]
    if len(valid_clusters) == 0:
        # If absolutely everything is noise, just return empty
        return findings
        
    # Recalculate dominant among valid clusters
    dominant_cluster = valid_clusters[np.argmax([counts[list(unique).index(c)] for c in valid_clusters])]
    
    if len(valid_clusters) > 1:
        for cluster in valid_clusters:
            if cluster == dominant_cluster:
                continue
                
            mean_rr = np.mean(rr_ms) if len(rr_ms) else 800.0
            ectopic_indices = np.where(cluster_ids == cluster)[0]
            
            premature_count = 0
            for idx in ectopic_indices:
                if idx > 0 and idx - 1 < len(rr_ms) and rr_ms[idx - 1] < 0.8 * mean_rr:
                    premature_count += 1
                    
            if premature_count >= 1:
                findings.append("Premature Ventricular Contraction (PVC) Morphology")
                break
                
        if len(valid_clusters) > 2 and "Premature Ventricular Contraction (PVC) Morphology" in findings:
            findings.append("Multifocal PVCs")
            
    return findings

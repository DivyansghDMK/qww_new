import numpy as np
from scipy.signal import find_peaks
import traceback


class ArrhythmiaDetector:
    """Detect various types of arrhythmias from ECG data"""
    
    def __init__(self, sampling_rate=500):
        self.fs = sampling_rate
    
    def detect_arrhythmias(self, signal, analysis, has_received_serial_data=False, min_serial_data_packets=50):
        """Detect various arrhythmias using peak analysis context
        
        Args:
            signal: ECG signal data
            analysis: Analysis results with peaks
            has_received_serial_data: True if serial data has actually started flowing (not just initial state)
            min_serial_data_packets: Minimum number of data packets received before checking for asystole
        """
        arrhythmias = []
        analysis = analysis or {}
        r_peaks = analysis.get('r_peaks', [])
        p_peaks = analysis.get('p_peaks', [])
        q_peaks = analysis.get('q_peaks', [])
        s_peaks = analysis.get('s_peaks', [])
        
        # IMPORTANT: Check for Asystole ONLY if serial data has actually started flowing
        # Don't detect asystole during initial application startup (no data yet)
        # Only detect when flatline occurs during active serial data acquisition
        try:
            # Convert signal to numpy array if needed
            signal_array = np.array(signal) if signal is not None else np.array([])
            
            # Only check for asystole if:
            # 1. We have received serial data (has_received_serial_data flag is True)
            # 2. We have signal data to analyze
            if has_received_serial_data and len(signal_array) > 0:
                # Calculate heart rate for asystole check (can be None if no R peaks)
                heart_rate = None
                if len(r_peaks) >= 2:
                    rr_intervals = np.diff(r_peaks) / self.fs * 1000
                    mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else None
                    heart_rate = 60000 / mean_rr if mean_rr and mean_rr > 0 else None
                
                # Check for Asystole (critical condition - absence of cardiac activity)
                # Only check if we've received enough serial data packets (not initial state)
                if self._is_asystole(signal_array, r_peaks, heart_rate, min_data_packets=min_serial_data_packets):
                    return ["Asystole (Cardiac Arrest)"]
        except Exception as e:
            print(f"Error in asystole detection: {e}")
            traceback.print_exc()
        
        # Now check for insufficient data (after asystole check)
        if len(r_peaks) < 3:
            return ["Insufficient data for arrhythmia detection."]
        
        # Calculate RR intervals (ms)
        rr_intervals = np.diff(r_peaks) / self.fs * 1000
        mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else None
        heart_rate = 60000 / mean_rr if mean_rr and mean_rr > 0 else None
        
        # Estimate PR and QRS durations to support junctional rhythm logic
        pr_interval = self._estimate_pr_interval(p_peaks, q_peaks)
        qrs_duration = self._estimate_qrs_duration(q_peaks, s_peaks)
        
        # Check for specific arrhythmias first
        # Wrap each detection in try-except to prevent one failure from breaking all detections
        
        try:
            af_detected = self._is_atrial_fibrillation(signal, r_peaks, p_peaks, rr_intervals, qrs_duration)
            if af_detected:
                arrhythmias.append("Atrial Fibrillation Detected")
        except Exception as e:
            print(f"Error in atrial fibrillation detection: {e}")
        
        try:
            if self._is_ventricular_fibrillation(signal, r_peaks, rr_intervals):
                arrhythmias.append("Ventricular Fibrillation Detected")
        except Exception as e:
            print(f"Error in ventricular fibrillation detection: {e}")
        
        try:
            if self._is_ventricular_tachycardia(rr_intervals, qrs_duration):
                arrhythmias.append("Possible Ventricular Tachycardia")
        except Exception as e:
            print(f"Error in ventricular tachycardia detection: {e}")
        
        try:
            if self._is_ventricular_ectopics(signal, r_peaks, qrs_duration, p_peaks, rr_intervals):
                arrhythmias.append("Ventricular Ectopics Detected")
        except Exception as e:
            print(f"Error in ventricular ectopics detection: {e}")
        
        bigeminy_detected = False
        try:
            if self._is_bigeminy(rr_intervals, qrs_duration, signal, r_peaks):
                arrhythmias.append("Bigeminy")
                bigeminy_detected = True
        except Exception as e:
            print(f"Error in bigeminy detection: {e}")
        
        try:
            if self._is_asynchronous_75_bpm(heart_rate, rr_intervals, p_peaks, r_peaks):
                arrhythmias.append("Asynchronous 75 bpm")
        except Exception as e:
            print(f"Error in asynchronous 75 bpm detection: {e}")
        
        try:
            if self._is_junctional_rhythm(heart_rate, qrs_duration, pr_interval, rr_intervals, p_peaks, r_peaks):
                arrhythmias.append("Possible Junctional Rhythm")
        except Exception as e:
            print(f"Error in junctional rhythm detection: {e}")
        
        try:
            if self._is_atrial_flutter(heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks):
                arrhythmias.append("Possible Atrial Flutter")
        except Exception as e:
            print(f"Error in atrial flutter detection: {e}")
        
        try:
            av_block_type = self._is_av_block(pr_interval, p_peaks, r_peaks, rr_intervals, heart_rate)
            if av_block_type:
                arrhythmias.append(av_block_type)
        except Exception as e:
            print(f"Error in AV block detection: {e}")
        
        try:
            if self._is_high_av_block(pr_interval, p_peaks, r_peaks, rr_intervals, heart_rate):
                arrhythmias.append("High AV-Block")
        except Exception as e:
            print(f"Error in high AV block detection: {e}")
        
        try:
            if self._is_wpw_syndrome(pr_interval, qrs_duration, signal, p_peaks, q_peaks, r_peaks):
                arrhythmias.append("WPW Syndrome (Wolff-Parkinson-White)")
        except Exception as e:
            print(f"Error in WPW syndrome detection: {e}")
        
        try:
            if self._is_left_bundle_branch_block(qrs_duration, pr_interval, rr_intervals, signal, q_peaks, r_peaks):
                arrhythmias.append("Left Bundle Branch Block (LBBB)")
        except Exception as e:
            print(f"Error in LBBB detection: {e}")
        
        try:
            if self._is_right_bundle_branch_block(qrs_duration, pr_interval, rr_intervals, signal, r_peaks):
                arrhythmias.append("Right Bundle Branch Block (RBBB)")
        except Exception as e:
            print(f"Error in RBBB detection: {e}")
        
        try:
            if self._is_left_anterior_fascicular_block(qrs_duration, heart_rate, signal, r_peaks, s_peaks):
                arrhythmias.append("Left Anterior Fascicular Block (LAFB)")
        except Exception as e:
            print(f"Error in LAFB detection: {e}")
        
        try:
            if self._is_left_posterior_fascicular_block(qrs_duration, heart_rate, signal, r_peaks, s_peaks):
                arrhythmias.append("Left Posterior Fascicular Block (LPFB)")
        except Exception as e:
            print(f"Error in LPFB detection: {e}")
        
        try:
            if self._is_atrial_tachycardia(heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks):
                arrhythmias.append("Atrial Tachycardia")
        except Exception as e:
            print(f"Error in atrial tachycardia detection: {e}")
        
        try:
            if self._is_supraventricular_tachycardia(heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks):
                arrhythmias.append("Supraventricular Tachycardia (SVT)")
        except Exception as e:
            print(f"Error in supraventricular tachycardia detection: {e}")
        
        # If no major arrhythmia, check rate-based conditions
        if not arrhythmias:
            try:
                if self._is_bradycardia(rr_intervals):
                    arrhythmias.append("Sinus Bradycardia")
                elif self._is_tachycardia(rr_intervals):
                    arrhythmias.append("Sinus Tachycardia")
            except Exception as e:
                print(f"Error in rate-based detection: {e}")

        # If still nothing, check for NSR
        if not arrhythmias and self._is_normal_sinus_rhythm(rr_intervals):
            return ["Normal Sinus Rhythm"]
            
        return arrhythmias if arrhythmias else ["Unspecified Irregular Rhythm"]

    def _estimate_pr_interval(self, p_peaks, q_peaks):
        """Approximate PR interval (ms) using detected P and Q peaks"""
        if not p_peaks or not q_peaks:
            return None
        intervals = []
        for p, q in zip(p_peaks, q_peaks):
            if q > p:
                intervals.append((q - p) / self.fs * 1000)
        return np.mean(intervals) if intervals else None

    def _estimate_qrs_duration(self, q_peaks, s_peaks):
        """Approximate QRS duration (ms) using detected Q and S peaks"""
        if not q_peaks or not s_peaks:
            return None
        durations = []
        for q, s in zip(q_peaks, s_peaks):
            if s > q:
                durations.append((s - q) / self.fs * 1000)
        return np.mean(durations) if durations else None
    
    def _is_normal_sinus_rhythm(self, rr_intervals):
        """Check if rhythm is normal sinus rhythm"""
        if len(rr_intervals) < 3: return False
        mean_hr = 60000 / np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        return 60 <= mean_hr <= 100 and std_rr < 120 # Variation less than 120ms
    
    def _is_asystole(self, signal, r_peaks, heart_rate, min_data_packets=50):
        """Detect Asystole - absence of cardiac electrical activity (flatline)
        
        Args:
            signal: ECG signal array
            r_peaks: Detected R peaks
            heart_rate: Calculated heart rate
            min_data_packets: Minimum data packets required before detecting asystole
                             This prevents false positives during initial startup
        """
        # Asystole characteristics:
        # 1. Very few or no R peaks (no QRS complexes)
        # 2. Very low or zero heart rate
        # 3. Flat or nearly flat signal (low amplitude variation)
        # 4. Minimal electrical activity
        # 5. Must have received substantial serial data (not just initial state)
        
        if len(signal) == 0:
            return False
        
        # Convert to numpy array if needed
        signal = np.array(signal)
        
        # Calculate signal statistics for flatline detection
        signal_amplitude = np.ptp(signal)  # Peak-to-peak amplitude
        signal_std = np.std(signal)
        signal_mean = np.mean(np.abs(signal))
        signal_max_abs = np.max(np.abs(signal))
        
        # Check signal duration - need enough data to confirm flatline
        signal_duration_sec = len(signal) / self.fs
        min_duration_for_asystole = 2.0  # Need at least 2 seconds of flatline data
        
        # Don't detect asystole if we don't have enough signal duration
        if signal_duration_sec < min_duration_for_asystole:
            return False
        
        # Check 1: Very few or no R peaks - PRIMARY INDICATOR
        # Asystole has no or very few QRS complexes
        if len(r_peaks) == 0:
            # No R peaks detected - check if signal is flat/zero (asystole condition)
            
            # Check for zero or near-zero signal (flatline)
            # Count values that are essentially zero (within small threshold)
            zero_threshold = 0.2  # Consider values < 0.2 as essentially zero
            near_zero_count = np.sum(np.abs(signal) < zero_threshold)
            near_zero_ratio = near_zero_count / len(signal) if len(signal) > 0 else 0
            
            # Check if signal is flatline (all or most values are zero/flat)
            is_zero_or_flat = (
                signal_max_abs < zero_threshold or  # Maximum absolute value is very small
                (signal_amplitude < 0.25 and signal_std < 0.1) or  # Very low variation
                (signal_mean < 0.15 and signal_std < 0.08) or  # Very low mean with low std
                near_zero_ratio > 0.75  # 75% of signal is near zero
            )
            
            if is_zero_or_flat:
                # Verify it's a true flatline (not just initial startup)
                # Need substantial data to confirm asystole
                if signal_duration_sec >= min_duration_for_asystole:
                    return True
            
            # If we have enough signal length and no R peaks with very low amplitude, likely asystole
            if len(signal) > 200 and signal_max_abs < 0.3 and signal_duration_sec >= min_duration_for_asystole:
                return True
            
            return False  # Not enough data to confirm
        
        # Check 2: Very few R peaks (1-2) with flat signal - likely asystole
        # If we have very few R peaks and signal is mostly flat, it's asystole
        if len(r_peaks) <= 2:  # Very few R peaks
            signal_duration_sec = len(signal) / self.fs
            
            # More lenient thresholds for flatline detection
            is_mostly_flat = (signal_amplitude < 0.25 and signal_std < 0.12) or (signal_mean < 0.2 and signal_std < 0.1)
            
            # If signal is mostly flat with very few peaks, likely asystole
            if is_mostly_flat:
                # Check if duration is long enough to confirm (not just start of signal)
                if signal_duration_sec > 2:  # At least 2 seconds
                    return True
                # Even with shorter duration, if signal is very flat, likely asystole
                if signal_amplitude < 0.15 and signal_std < 0.08:
                    return True
        
        # Check 3: Very low heart rate (near zero or extremely low)
        if heart_rate is not None:
            if heart_rate < 20:  # Extremely low heart rate suggests asystole
                # Also check signal amplitude - must be low to confirm asystole
                if signal_amplitude < 0.2 and signal_std < 0.1:
                    return True
        
        # Check 4: If we have very few R peaks relative to signal duration
        # Calculate expected number of beats based on signal duration
        signal_duration_sec = len(signal) / self.fs
        
        # If we have fewer than 20 beats per minute, it's likely asystole
        if signal_duration_sec > 3:  # At least 3 seconds of data
            beats_per_minute = (len(r_peaks) / signal_duration_sec) * 60
            if beats_per_minute < 20:  # Less than 20 bpm
                # Check signal amplitude - asystole has very low amplitude
                # More lenient thresholds
                if signal_amplitude < 0.25 and signal_std < 0.12:
                    return True
        
        return False
    
    def _is_atrial_fibrillation(self, signal, r_peaks, p_peaks, rr_intervals, qrs_duration):
        """Detect Atrial Fibrillation (AF)
        
        AF characteristics:
        - Highly irregular RR intervals (irregularly irregular pattern)
        - Absence of clear P waves before QRS complexes
        - Normal QRS complexes (narrow, <120ms)
        - Variable heart rate
        - High RR interval variability (coefficient of variation > 0.12-0.15)
        """
        # Reduced minimum requirement - need at least 5 beats to assess irregularity
        if len(r_peaks) < 5:
            return False
        
        # Calculate RR intervals from R peaks if not provided
        if len(rr_intervals) < 2:
            rr_intervals = np.diff(r_peaks) / self.fs * 1000
        
        if len(rr_intervals) < 2:
            return False
        
        # Check 1: Highly irregular RR intervals (irregularly irregular pattern)
        # This is the hallmark of AF
        mean_rr = np.mean(rr_intervals)
        if mean_rr <= 0:
            return False
        
        rr_cv = np.std(rr_intervals) / mean_rr  # Coefficient of variation
        rr_std = np.std(rr_intervals)
        
        
        # Lower threshold for better sensitivity - AF has high RR interval variability
        # CV > 0.10 is more sensitive for AF detection (reduced from 0.12)
        if rr_cv > 0.10:
            # Check 2: Normal QRS complexes (AF should have narrow QRS)
            # Wide QRS suggests ventricular origin, not AF
            if qrs_duration is None or qrs_duration <= 120:
                # Check 3: Absence or very few P waves relative to R peaks
                # AF has no organized P waves
                if p_peaks is None or len(p_peaks) == 0:
                    # No P waves detected - strong indicator of AF
                    print(f"[AF Detection] ✓ Detected: High RR CV ({rr_cv:.3f}) + No P waves + Narrow QRS")
                    return True
                
                # If P waves exist, check if they're significantly fewer than R peaks
                # In AF, P waves are absent or very irregular
                p_ratio = len(p_peaks) / len(r_peaks) if len(r_peaks) > 0 else 0
                if p_ratio < 0.7:  # Less than 70% of R peaks have P waves (more lenient)
                    print(f"[AF Detection] ✓ Detected: High RR CV ({rr_cv:.3f}) + Few P waves (ratio: {p_ratio:.2f}) + Narrow QRS")
                    return True
                
                # Check P wave regularity - AF has no regular P waves
                if len(p_peaks) >= 2:
                    p_intervals = np.diff(p_peaks) / self.fs * 1000
                    if len(p_intervals) > 0 and np.mean(p_intervals) > 0:
                        p_cv = np.std(p_intervals) / np.mean(p_intervals)
                        # High P wave interval variability suggests AF (no organized atrial activity)
                        if p_cv > 0.12:  # Lowered threshold
                            return True
                
                # If RR CV is very high (>0.15) with narrow QRS, likely AF even with some P waves
                if rr_cv > 0.15:
                    return True
        
        # Check 4: Very high RR variability (CV > 0.18) with narrow QRS
        # This is a strong indicator of AF even without P wave analysis
        if rr_cv > 0.18:
            if qrs_duration is None or qrs_duration <= 120:
                # Additional check: variability in consecutive RR intervals
                # AF shows "irregularly irregular" pattern
                if len(rr_intervals) >= 3:
                    # Check for alternating patterns (not regular)
                    differences = np.abs(np.diff(rr_intervals))
                    mean_diff = np.mean(differences)
                    # High variability in consecutive differences indicates irregularly irregular
                    if mean_diff > mean_rr * 0.10:  # Consecutive intervals vary by >10% of mean
                        return True
                # If we have very high CV and narrow QRS, likely AF
                return True
        
        # Check 5: Moderate RR variability with very irregular pattern and narrow QRS
        # This catches cases with moderate but consistent irregularity
        if rr_cv > 0.12 and len(rr_intervals) >= 3:
            if qrs_duration is None or qrs_duration <= 120:
                # Check for "irregularly irregular" - no pattern in the irregularity
                differences = np.abs(np.diff(rr_intervals))
                # High variability in differences between consecutive intervals
                if len(differences) > 0 and np.std(differences) > mean_rr * 0.08:
                    # Few or irregular P waves
                    if p_peaks is None or len(p_peaks) == 0 or (len(r_peaks) > 0 and len(p_peaks) < len(r_peaks) * 0.7):
                        return True
        
        return False
    
    def _is_ventricular_tachycardia(self, rr_intervals, qrs_duration):
        """Detect ventricular tachycardia (VT)
        
        VT characteristics (simplified):
        - Fast rate (>120 bpm)
        - Relatively regular rhythm
        - **Wide QRS complex** (typically >120 ms)
        """
        if len(rr_intervals) < 3:
            return False
        
        # Require wide QRS; narrow‑complex fast rhythms should be classified as SVT/atrial tachycardia
        if qrs_duration is None or qrs_duration <= 120:
            return False
        
        mean_rr = np.mean(rr_intervals)
        if mean_rr <= 0:
            return False
        
        mean_hr = 60000 / mean_rr  # rr_intervals are in ms
        rr_std = np.std(rr_intervals)
        
        # VT: fast and fairly regular
        return mean_hr > 120 and rr_std < 80
    
    def _is_ventricular_fibrillation(self, signal, r_peaks, rr_intervals):
        """Detect Ventricular Fibrillation (VF)
        
        VF characteristics:
        - Chaotic, irregular waveform with no organized QRS complexes
        - High variability in signal amplitude and RR intervals
        - Absence of clear R peaks or very irregular R peaks
        - Fast, irregular rate
        - High signal entropy/variability
        """
        if signal is None or len(signal) < 500:  # Need sufficient signal length
            return False
        
        signal_array = np.array(signal)
        
        # Calculate RR intervals from R peaks if not provided or insufficient
        if len(rr_intervals) < 3 and len(r_peaks) >= 3:
            rr_intervals = np.diff(r_peaks) / self.fs * 1000
        
        # Check 1: High variability in RR intervals (if we have R peaks)
        if len(rr_intervals) >= 3:
            rr_cv = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            # VF has very high RR interval variability (coefficient of variation > 0.3)
            if rr_cv > 0.3:
                # Also check signal characteristics
                signal_std = np.std(signal_array)
                signal_amplitude = np.ptp(signal_array)  # Peak-to-peak
                
                # VF has high signal variability
                if signal_std > 50 and signal_amplitude > 100:
                    return True
        
        # Check 2: Very few or irregular R peaks relative to signal length
        signal_duration_sec = len(signal_array) / self.fs
        if signal_duration_sec >= 2.0:  # Need at least 2 seconds
            # VF can have fast rate but with chaotic pattern
            # Check if we have very irregular R peaks or very few R peaks with high signal variability
            if len(r_peaks) >= 3:
                calculated_rr = np.diff(r_peaks) / self.fs * 1000
                if len(calculated_rr) >= 2:
                    rr_cv = np.std(calculated_rr) / np.mean(calculated_rr) if np.mean(calculated_rr) > 0 else 0
                    
                    # High variability in RR intervals combined with high signal entropy
                    if rr_cv > 0.25:
                        signal_std = np.std(signal_array)
                        signal_amplitude = np.ptp(signal_array)
                        
                        # High signal variability indicates chaotic pattern
                        if signal_std > 40 and signal_amplitude > 80:
                            return True
            
            # Check 3: Very few R peaks with high signal variability (chaotic pattern)
            if len(r_peaks) < 5 and signal_duration_sec >= 3.0:
                signal_std = np.std(signal_array)
                signal_amplitude = np.ptp(signal_array)
                signal_mean_abs = np.mean(np.abs(signal_array))
                
                # High variability with few organized beats suggests VF
                if signal_std > 50 and signal_amplitude > 100 and signal_mean_abs > 30:
                    return True
        
        # Check 4: Signal entropy analysis - VF has high entropy (chaotic)
        if len(signal_array) >= 1000:
            # Calculate approximate entropy or coefficient of variation
            signal_std = np.std(signal_array)
            signal_mean_abs = np.mean(np.abs(signal_array))
            
            # High relative variability indicates chaotic pattern
            if signal_mean_abs > 0:
                relative_variability = signal_std / signal_mean_abs
                
                # VF typically has high relative variability (> 1.0)
                if relative_variability > 1.0:
                    signal_amplitude = np.ptp(signal_array)
                    
                    # Combined with high amplitude variability
                    if signal_amplitude > 100:
                        # Check if R peaks are very irregular or few
                        calculated_rr = np.diff(r_peaks) / self.fs * 1000 if len(r_peaks) >= 3 else np.array([])
                        if len(r_peaks) < 8 or (len(calculated_rr) >= 3 and np.std(calculated_rr) / np.mean(calculated_rr) > 0.2 if np.mean(calculated_rr) > 0 else False):
                            return True
        
        return False
    
    def _is_bradycardia(self, rr_intervals):
        """Detect bradycardia"""
        if len(rr_intervals) < 3: return False
        mean_hr = 60000 / np.mean(rr_intervals)
        return mean_hr < 60
    
    def _is_tachycardia(self, rr_intervals):
        """Detect tachycardia"""
        if len(rr_intervals) < 3: return False
        mean_hr = 60000 / np.mean(rr_intervals)
        return mean_hr >= 100
    
    def _is_ventricular_ectopics(self, signal, r_peaks, qrs_duration, p_peaks, rr_intervals):
        """Detect Ventricular Ectopics (PVCs) - enhanced detection"""
        # Ventricular Ectopics characteristics:
        # 1. Wide QRS complexes (>120ms) - key feature
        # 2. Premature beats (shorter RR interval than expected)
        # 3. Absence of P wave before the ectopic beat
        # 4. Compensatory pause after the ectopic beat
        # 5. Bizarre QRS morphology (different from normal beats)
        
        if len(r_peaks) < 5:
            return False
        
        if qrs_duration is not None and qrs_duration > 120:
            # Wide QRS detected - check for premature beats
            rr_intervals_ms = rr_intervals if len(rr_intervals) > 0 else np.diff(r_peaks) / self.fs * 1000
            if len(rr_intervals_ms) < 2:
                return False
            
            mean_rr = np.mean(rr_intervals_ms)
            if mean_rr <= 0:
                return False
            
            premature_count = 0
            compensatory_pause_count = 0
            
            for i in range(len(rr_intervals_ms)):
                if rr_intervals_ms[i] < 0.85 * mean_rr:  # Premature beat
                    premature_count += 1
                    # Check for compensatory pause (next interval > 115% of mean)
                    if i + 1 < len(rr_intervals_ms) and rr_intervals_ms[i + 1] > 1.15 * mean_rr:
                        compensatory_pause_count += 1
            
            if premature_count >= 1 and compensatory_pause_count >= 1:
                return True
            if premature_count >= 2:  # Multiple premature beats
                return True
        
        rr_intervals_sec = np.diff(r_peaks) / self.fs
        if len(rr_intervals_sec) < 2:
            return False
        
        mean_rr = np.mean(rr_intervals_sec)
        if mean_rr <= 0:
            return False
        
        # Check for premature beats
        for i in range(len(rr_intervals_sec)):
            if rr_intervals_sec[i] < 0.8 * mean_rr:  # Premature beat
                # Check for compensatory pause
                if i + 1 < len(rr_intervals_sec) and rr_intervals_sec[i + 1] > 1.2 * mean_rr:
                    # Check if P wave is absent before this premature beat
                    premature_r_idx = i + 1  # Index of premature R peak
                    if premature_r_idx < len(r_peaks):
                        premature_r = r_peaks[premature_r_idx]
                        
                        # Look for P wave before this premature beat
                        # P wave should be 120-200ms before R peak
                        p_found = False
                        if p_peaks is not None:
                            for p in p_peaks:
                                pr_distance = (premature_r - p) / self.fs * 1000  # in ms
                                if 120 <= pr_distance <= 200:  # Normal PR interval
                                    p_found = True
                                    break
                        
                        # If no P wave found before premature beat, it's likely ventricular ectopic
                        if not p_found:
                            return True
        
        return False
    
    def _is_bigeminy(self, rr_intervals, qrs_duration, signal, r_peaks):
        """Detect Bigeminy - alternating pattern of normal beats and PVCs"""
        # Bigeminy characteristics:
        # 1. Alternating pattern: normal beat, PVC, normal beat, PVC
        # 2. RR intervals show pattern: long (normal), short (coupling interval), long, short, etc.
        # 3. The premature beats are typically wide QRS complexes
        # 4. The coupling interval (distance from normal R to PVC R) is usually consistent
        
        try:
            # Convert to numpy array if needed
            rr_intervals = np.array(rr_intervals) if not isinstance(rr_intervals, np.ndarray) else rr_intervals
            
            if len(rr_intervals) < 4:  # Reduced from 6 to 4 - need at least 4 intervals to detect pattern
                return False
            
            if len(r_peaks) < 5:  # Reduced from 7 to 5 - need at least 5 beats to see pattern
                return False
            
            # Convert RR intervals to milliseconds if needed
            if len(rr_intervals) > 0:
                max_rr = float(np.max(rr_intervals))
                if max_rr < 10:  # Likely in seconds, convert to ms
                    rr_intervals_ms = rr_intervals * 1000
                else:
                    rr_intervals_ms = rr_intervals.copy()
            else:
                return False
            
            # Calculate mean RR interval
            mean_rr = float(np.mean(rr_intervals_ms))
            if mean_rr <= 0:
                return False
            
            # In bigeminy, we expect alternating long and short intervals
            # Long interval = normal beat + compensatory pause
            # Short interval = coupling interval (normal R to PVC R)
            
            # Identify potential premature beats (short intervals)
            # More lenient thresholds for bigeminy detection
            short_threshold = 0.75 * mean_rr  # More lenient - Premature beats are < 75% of mean
            long_threshold = 1.03 * mean_rr   # More lenient - Normal/compensatory pauses are > 103% of mean
            
            # Check for alternating pattern
            alternating_pattern_count = 0
            consistent_coupling = True
            
            # Check if we have alternating short-long pattern
            for i in range(len(rr_intervals_ms) - 1):
                current = float(rr_intervals_ms[i])
                next_interval = float(rr_intervals_ms[i + 1])
                
                # Check for alternating pattern: short followed by long, or long followed by short
                is_short = current < short_threshold
                is_long = current > long_threshold
                next_is_short = next_interval < short_threshold
                next_is_long = next_interval > long_threshold
                
                # Alternating pattern: short-long or long-short
                if (is_short and next_is_long) or (is_long and next_is_short):
                    alternating_pattern_count += 1
            
            # More sensitive threshold - reduced to 25% for better detection
            min_alternating = max(2, int(len(rr_intervals_ms) * 0.25))  # At least 2 alternating pairs or 25%
            if alternating_pattern_count < min_alternating:
                return False
            
            # Check for consistent coupling intervals (short intervals should be similar)
            short_intervals = [float(rr) for rr in rr_intervals_ms if float(rr) < short_threshold]
            if len(short_intervals) >= 2:
                coupling_std = float(np.std(short_intervals))
                coupling_mean = float(np.mean(short_intervals))
                # Coupling intervals should be relatively consistent (CV < 0.25, very lenient)
                if coupling_mean > 0:
                    coupling_cv = coupling_std / coupling_mean
                    if coupling_cv > 0.25:  # Increased from 0.20 to 0.25 for more leniency
                        consistent_coupling = False
            
            # Check if we have wide QRS (ventricular origin for PVCs)
            # In bigeminy, the premature beats should be wide
            has_wide_qrs = qrs_duration is not None and qrs_duration > 120
            
            # Bigeminy diagnosis:
            # - Alternating pattern of intervals
            # - Consistent coupling intervals (if we can detect them)
            # - Wide QRS suggests ventricular origin (preferred but not always present)
            # Very lenient criteria - if we have alternating pattern, detect bigeminy
            if alternating_pattern_count >= min_alternating:
                # If we have wide QRS, it's more likely bigeminy
                if has_wide_qrs:
                    return True
                # Even without wide QRS, if pattern is strong enough, it could be bigeminy
                # Reduced threshold to 30% for more sensitive detection
                if alternating_pattern_count >= max(2, int(len(rr_intervals_ms) * 0.3)):
                    # If coupling is consistent, definitely bigeminy
                    if consistent_coupling:
                        return True
                    # Even without consistent coupling, if pattern is strong (50%+), detect it
                    if alternating_pattern_count >= max(2, int(len(rr_intervals_ms) * 0.5)):
                        return True
                    # For irregular patterns, if we have at least 3 alternating pairs, detect it
                    if alternating_pattern_count >= 3:
                        return True
            
            return False
        except Exception as e:
            print(f"Error in bigeminy detection details: {e}")
            traceback.print_exc()
            return False
    
    def _is_asynchronous_75_bpm(self, heart_rate, rr_intervals, p_peaks, r_peaks):
        """Detect Asynchronous 75 bpm - irregular rhythm pattern around 75 bpm"""
        # Asynchronous 75 bpm characteristics:
        # 1. Heart rate around 75 bpm (typically 65-85 bpm range)
        # 2. Irregular rhythm (asynchronous - not regular)
        # 3. Variable RR intervals
        # 4. Not other specific arrhythmias (like AFib, which is also irregular)
        
        try:
            if heart_rate is None:
                return False
            
            # Check if heart rate is around 75 bpm (70-80 bpm range for primary detection)
            is_around_75_bpm = 70 <= heart_rate <= 80
            
            # Convert to numpy array if needed
            rr_intervals = np.array(rr_intervals) if not isinstance(rr_intervals, np.ndarray) else rr_intervals
            
            if len(rr_intervals) < 3:  # Need at least 3 intervals
                return False
            
            # Convert RR intervals to milliseconds if needed
            if len(rr_intervals) > 0:
                max_rr = float(np.max(rr_intervals))
                if max_rr < 10:  # Likely in seconds, convert to ms
                    rr_intervals_ms = rr_intervals * 1000
                else:
                    rr_intervals_ms = rr_intervals.copy()
            else:
                return False
            
            # Calculate variability in RR intervals
            mean_rr = float(np.mean(rr_intervals_ms))
            std_rr = float(np.std(rr_intervals_ms))
            
            if mean_rr <= 0:
                return False
            
            # Coefficient of variation (CV) - measures irregularity
            cv = std_rr / mean_rr if mean_rr > 0 else 0
            
            # Special handling for heart rates around 75 bpm (70-80)
            if is_around_75_bpm:
                # For 70-80 bpm, be VERY lenient - this is the key characteristic
                # Allow even very regular rhythms (CV >= 0.005 = 0.5%)
                if cv < 0.005:  # Only reject if extremely regular (CV < 0.5%)
                    return False
                if cv > 0.25:  # Too irregular (might be AFib) - allow up to 25%
                    return False
                
                # Allow very small variations (std_rr >= 5ms) for 75 bpm range
                if std_rr < 5:  # Variation less than 5ms - extremely regular
                    return False
                if std_rr > 300:  # Variation more than 300ms - too irregular
                    return False
                
                # Very lenient P wave requirement for 75 bpm - allow even 5%
                p_count = len(p_peaks) if p_peaks is not None else 0
                r_count = len(r_peaks) if r_peaks is not None else 0
                if r_count > 0 and p_count < r_count * 0.05:  # Less than 5% P waves
                    return False
                
                # For 70-80 bpm range, if CV and std are in acceptable range, detect it
                # This is the primary use case - heart rate around 75 bpm
                if 0.005 <= cv <= 0.25 and 5 <= std_rr <= 300:
                    return True
            
            # For other heart rates (60-90 bpm but not 70-80), use stricter criteria
            if not (60 <= heart_rate <= 90):
                return False
            
            # Asynchronous rhythm should show some irregularity
            # CV between 0.03 and 0.15 suggests moderate irregularity
            if cv < 0.03:  # Too regular
                return False
            
            if cv > 0.15:  # Too irregular (might be AFib)
                return False
            
            # Check for variability in RR intervals
            if std_rr < 30:  # Variation less than 30ms - too regular
                return False
            
            if std_rr > 250:  # Variation more than 250ms - too irregular
                return False
            
            # Check if P waves are present
            p_count = len(p_peaks) if p_peaks is not None else 0
            r_count = len(r_peaks) if r_peaks is not None else 0
            
            # P wave requirement for other heart rates
            if r_count > 0 and p_count < r_count * 0.2:  # Less than 20% P waves
                if len(rr_intervals_ms) < 5:
                    if p_count < r_count * 0.1:  # At least 10% if very few beats
                        return False
                else:
                    return False
            
            # Additional check: Look for pattern of variability
            gradual_variation = True
            large_jumps = 0
            for i in range(len(rr_intervals_ms) - 1):
                diff = abs(float(rr_intervals_ms[i + 1]) - float(rr_intervals_ms[i]))
                if diff > 200:  # Large jump
                    large_jumps += 1
                    if large_jumps > 1 or (large_jumps == 1 and len(rr_intervals_ms) < 5):
                        gradual_variation = False
                        break
            
            # For other heart rates, require gradual variation
            if 0.03 <= cv <= 0.15 and 30 <= std_rr <= 250:
                if gradual_variation:
                    return True
            
            return False
        except Exception as e:
            print(f"Error in asynchronous 75 bpm detection details: {e}")
            traceback.print_exc()
            return False

    def _is_left_bundle_branch_block(self, qrs_duration, pr_interval, rr_intervals, signal, q_peaks, r_peaks):
        """Detect Left Bundle Branch Block (LBBB) heuristically"""
        if qrs_duration is None or qrs_duration < 130:
            return False
        
        # PR interval typically normal in LBBB (exclude first-degree block patterns)
        if pr_interval is not None and pr_interval > 220:
            return False
        
        if len(rr_intervals) < 3:
            return False
        
        # Normalize RR intervals to milliseconds if needed
        if np.max(rr_intervals) < 10:
            rr_ms = rr_intervals * 1000
        else:
            rr_ms = rr_intervals
        
        rr_mean = np.mean(rr_ms)
        if rr_mean <= 0:
            return False
        rr_cv = np.std(rr_ms) / rr_mean
        # LBBB usually occurs with relatively regular rhythm
        if rr_cv > 0.15:
            return False
        
        # LBBB often shows absent/very small Q waves in most beats
        q_count = len(q_peaks) if q_peaks is not None else 0
        r_count = len(r_peaks) if r_peaks is not None else 0
        if r_count == 0:
            return False
        if q_count > r_count * 0.6:  # too many Q waves -> unlikely LBBB
            return False
        
        # Look for notched/broad R waves indicating delayed depolarization
        notched_count = 0
        total_checked = 0
        for r in r_peaks[:min(6, len(r_peaks))]:
            start = max(0, r - int(0.02 * self.fs))
            end = min(len(signal), r + int(0.08 * self.fs))
            if end - start < 5:
                continue
            segment = signal[start:end]
            # Normalize segment
            seg = segment - np.min(segment)
            if np.max(seg) - np.min(seg) <= 0:
                continue
            # Find local peaks within window
            try:
                peaks, _ = find_peaks(seg, distance=max(2, int(0.01 * self.fs)))
            except Exception:
                continue
            # Notched R wave: at least two significant peaks
            if len(peaks) >= 2:
                notched_count += 1
            total_checked += 1
        
        if total_checked == 0:
            return False
        
        notched_ratio = notched_count / total_checked
        # Require at least 30% of inspected beats to show notching
        if notched_ratio < 0.3:
            return False
        
        return True

    def _is_right_bundle_branch_block(self, qrs_duration, pr_interval, rr_intervals, signal, r_peaks):
        """Detect Right Bundle Branch Block (RBBB) heuristically"""
        if qrs_duration is None or qrs_duration < 120:
            return False
        
        if pr_interval is not None and pr_interval > 220:
            return False
        
        if len(rr_intervals) < 3:
            return False
        
        if np.max(rr_intervals) < 10:
            rr_ms = rr_intervals * 1000
        else:
            rr_ms = rr_intervals
        rr_mean = np.mean(rr_ms)
        if rr_mean <= 0:
            return False
        rr_cv = np.std(rr_ms) / rr_mean
        if rr_cv > 0.18:
            return False
        
        if not r_peaks or len(r_peaks) < 3:
            return False
        
        double_spike_count = 0
        checked = 0
        for r in r_peaks[:min(6, len(r_peaks))]:
            start = max(0, r - int(0.015 * self.fs))
            end = min(len(signal), r + int(0.09 * self.fs))
            if end - start < 6:
                continue
            segment = signal[start:end]
            segment = segment - np.mean(segment)
            first_peak_val = np.max(segment)
            if first_peak_val <= 0:
                continue
            try:
                peaks, props = find_peaks(segment, distance=max(2, int(0.008 * self.fs)))
            except Exception:
                continue
            if len(peaks) < 2:
                continue
            # Find second peak that occurs after first by 15-70 ms
            for i in range(len(peaks) - 1):
                p1 = peaks[i]
                p2 = peaks[i + 1]
                delta = (p2 - p1) / self.fs * 1000
                if 15 <= delta <= 70:
                    amp_ratio = segment[p2] / first_peak_val if first_peak_val != 0 else 0
                    if amp_ratio >= 0.3:
                        double_spike_count += 1
                        break
            checked += 1
        
        if checked == 0:
            return False
        
        if double_spike_count / checked < 0.3:
            return False
        
        return True
    
    def _is_left_anterior_fascicular_block(self, qrs_duration, heart_rate, signal, r_peaks, s_peaks):
        """Detect Left Anterior Fascicular Block (LAFB) heuristically from a single lead"""
        if qrs_duration is None or qrs_duration > 130:
            return False
        
        if heart_rate is not None and not (45 <= heart_rate <= 120):
            return False
        
        if not r_peaks or not s_peaks:
            return False
        
        sample_count = min(len(r_peaks), len(s_peaks), 6)
        if sample_count < 3:
            return False
        
        r_amplitudes = []
        s_amplitudes = []
        for i in range(sample_count):
            r_idx = r_peaks[i]
            s_idx = s_peaks[i]
            if r_idx >= len(signal) or s_idx >= len(signal):
                continue
            r_amplitudes.append(abs(signal[r_idx]))
            s_amplitudes.append(abs(signal[s_idx]))
        
        if len(r_amplitudes) < 3 or len(s_amplitudes) < 3:
            return False
        
        avg_r = np.mean(r_amplitudes)
        avg_s = np.mean(s_amplitudes)
        
        if avg_r <= 0 or avg_s <= 0:
            return False
        
        # LAFB often shows small R waves with deep S waves in inferior leads
        if avg_s / avg_r < 1.6:
            return False
        
        # Check for gradual negative terminal deflection (slurred S wave)
        slurred_count = 0
        checked = 0
        for i in range(sample_count):
            r_idx = r_peaks[i]
            s_idx = s_peaks[i]
            if r_idx >= len(signal) or s_idx >= len(signal):
                continue
            checked += 1
            start = min(r_idx, s_idx)
            end = min(len(signal), s_idx + int(0.04 * self.fs))
            if end - start < 5:
                continue
            segment = signal[start:end]
            # Compute derivative to see if slope is gradual (absolute derivative small)
            diff = np.diff(segment)
            if len(diff) == 0:
                continue
            # If more than 60% of derivative magnitudes are below threshold, consider slurred
            threshold = 0.2 * np.max(np.abs(segment)) if np.max(np.abs(segment)) > 0 else 0.05
            if threshold == 0:
                continue
            # Convert boolean array comparison to scalar to avoid array truth value errors
            diff_abs = np.abs(diff)
            slow_portion = float(np.mean(diff_abs < threshold)) if len(diff_abs) > 0 else 0.0
            if slow_portion > 0.6:
                slurred_count += 1
        
        if checked == 0:
            return False
        
        slurred_ratio = slurred_count / checked
        if slurred_ratio < 0.4:
            return False
        
        return True
    
    def _is_left_posterior_fascicular_block(self, qrs_duration, heart_rate, signal, r_peaks, s_peaks):
        """Detect Left Posterior Fascicular Block (LPFB) heuristically from a single lead"""
        if qrs_duration is None or qrs_duration > 130:
            return False
        
        if heart_rate is not None and not (45 <= heart_rate <= 120):
            return False
        
        if not r_peaks or not s_peaks:
            return False
        
        sample_count = min(len(r_peaks), len(s_peaks), 6)
        if sample_count < 3:
            return False
        
        r_amplitudes = []
        s_amplitudes = []
        for i in range(sample_count):
            r_idx = r_peaks[i]
            s_idx = s_peaks[i]
            if r_idx >= len(signal) or s_idx >= len(signal):
                continue
            r_amplitudes.append(abs(signal[r_idx]))
            s_amplitudes.append(abs(signal[s_idx]))
        
        if len(r_amplitudes) < 3 or len(s_amplitudes) < 3:
            return False
        
        avg_r = np.mean(r_amplitudes)
        avg_s = np.mean(s_amplitudes)
        if avg_s <= 0 or avg_r <= 0:
            return False
        
        # LPFB shows tall R waves and small S waves in inferior leads
        if avg_r / avg_s < 1.6:
            return False
        
        # Check for terminal positive slope (upright tail) after S wave
        positive_tail_count = 0
        inspected = 0
        for i in range(sample_count):
            s_idx = s_peaks[i]
            if s_idx >= len(signal):
                continue
            inspected += 1
            start = s_idx
            end = min(len(signal), s_idx + int(0.05 * self.fs))
            if end - start < 4:
                continue
            segment = signal[start:end]
            diff = np.diff(segment)
            if len(diff) == 0:
                continue
            # Convert boolean array comparison to scalar to avoid array truth value errors
            positive_ratio = float(np.mean(diff > 0)) if len(diff) > 0 else 0.0
            if positive_ratio > 0.6:
                positive_tail_count += 1
        
        if inspected == 0:
            return False
        
        if positive_tail_count / inspected < 0.4:
            return False
        
        return True
    
    def _is_junctional_rhythm(self, heart_rate, qrs_duration, pr_interval, rr_intervals, p_peaks, r_peaks):
        """Detect Junctional Rhythm heuristically"""
        if heart_rate is None or qrs_duration is None:
            return False
        if not (40 <= heart_rate <= 60):
            return False
        if qrs_duration > 120:
            return False
        if len(rr_intervals) < 3 or np.std(rr_intervals) >= 120:
            return False
        p_count = len(p_peaks) if p_peaks is not None else 0
        r_count = len(r_peaks) if r_peaks is not None else max(len(rr_intervals) + 1, 1)
        p_ratio = p_count / max(r_count, 1)
        pr_short = pr_interval is not None and pr_interval <= 120
        return p_ratio < 0.4 or pr_short
    
    def _is_atrial_flutter(self, heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks):
        """Detect Atrial Flutter based on rapid atrial activity"""
        if heart_rate is None or qrs_duration is None:
            return False
        if not (130 <= heart_rate <= 180):
            return False
        if qrs_duration > 120:
            return False
        if len(rr_intervals) < 3 or np.std(rr_intervals) >= 120:
            return False
        if not p_peaks or not r_peaks:
            return False
        flutter_ratio = len(p_peaks) / max(len(r_peaks), 1)
        return flutter_ratio >= 1.5
    
    def _is_av_block(self, pr_interval, p_peaks, r_peaks, rr_intervals, heart_rate):
        """Detect AV Block (Atrioventricular Block) - different degrees"""
        if not p_peaks or not r_peaks or len(p_peaks) < 2 or len(r_peaks) < 2:
            return None
        
        if pr_interval is not None and pr_interval > 200:
            return "First-Degree AV Block"
        
        p_count = len(p_peaks)
        r_count = len(r_peaks)
        
        if p_count > r_count * 1.2:  # More than 20% more P waves than QRS complexes
            dropped_ratio = (p_count - r_count) / max(p_count, 1)
            
            if dropped_ratio > 0.5:  # More than 50% of P waves not conducted
                # Check if P-P intervals are regular (atrial rhythm) and R-R intervals are regular (ventricular escape)
                if len(p_peaks) >= 3 and len(r_peaks) >= 3:
                    p_intervals = np.diff(p_peaks) / self.fs * 1000  # in ms
                    r_intervals_ms = rr_intervals if len(rr_intervals) > 0 else np.diff(r_peaks) / self.fs * 1000
                    
                    # Convert to scalar booleans to avoid array comparison issues
                    p_std = np.std(p_intervals) if len(p_intervals) > 0 else float('inf')
                    r_std = np.std(r_intervals_ms) if len(r_intervals_ms) > 0 else float('inf')
                    p_regular = bool(p_std < 100)  # P waves regular
                    r_regular = bool(r_std < 100)  # R waves regular
                    
                    # If both are regular but independent, it's third-degree block
                    if p_regular and r_regular and heart_rate and heart_rate < 60:
                        return "Third-Degree AV Block (Complete Heart Block)"
            
            if dropped_ratio > 0.2:  # At least 20% dropped beats
                # Try to detect Wenckebach pattern (progressive PR prolongation)
                if pr_interval is not None:
                    # For simplicity, if we have prolonged PR and dropped beats, suggest Type I
                    if pr_interval > 180:
                        return "Second-Degree AV Block (Type I - Wenckebach)"
                    else:
                        return "Second-Degree AV Block (Type II)"
                return "Second-Degree AV Block"
        
        return None
    
    def _is_high_av_block(self, pr_interval, p_peaks, r_peaks, rr_intervals, heart_rate):
        """Detect High AV-Block - high-grade blocks (Type II second-degree and third-degree)"""
        # High AV-Block includes:
        # 1. Third-degree AV block (Complete Heart Block) - complete AV dissociation
        # 2. Second-degree AV block Type II (Mobitz Type II) - fixed PR with sudden dropped beats
        
        if not p_peaks or not r_peaks or len(p_peaks) < 3 or len(r_peaks) < 2:
            return False
        
        p_count = len(p_peaks)
        r_count = len(r_peaks)
        
        # Check for significant number of dropped beats (P waves without QRS)
        if p_count <= r_count * 1.1:  # Less than 10% more P waves than QRS - not high-grade block
            return False
        
        dropped_ratio = (p_count - r_count) / max(p_count, 1)
        
        # Third-degree AV Block (Complete Heart Block)
        # Characteristics:
        # - Complete AV dissociation (P waves and QRS complexes are independent)
        # - More than 50% of P waves not conducted
        # - Regular P-P intervals (atrial rhythm)
        # - Regular R-R intervals (ventricular escape rhythm, usually slower)
        if dropped_ratio > 0.5:  # More than 50% of P waves not conducted
            if len(p_peaks) >= 3 and len(r_peaks) >= 3:
                p_intervals = np.diff(p_peaks) / self.fs * 1000  # in ms
                r_intervals_ms = rr_intervals if len(rr_intervals) > 0 else np.diff(r_peaks) / self.fs * 1000
                
                # Convert to scalar booleans to avoid array comparison issues
                p_std = np.std(p_intervals) if len(p_intervals) > 0 else float('inf')
                r_std = np.std(r_intervals_ms) if len(r_intervals_ms) > 0 else float('inf')
                p_regular = bool(p_std < 100)  # P waves regular (atrial rhythm)
                r_regular = bool(r_std < 100)  # R waves regular (ventricular escape)
                
                # Complete heart block: both rhythms are regular but independent
                # Ventricular rate is usually slower (<60 bpm) in complete heart block
                if p_regular and r_regular:
                    if heart_rate is not None and heart_rate < 60:
                        return True  # Third-degree block with slow ventricular rate
                    # Even if ventricular rate is not slow, if both are regular and independent, it's likely third-degree
                    if len(p_intervals) > 0 and len(r_intervals_ms) > 0:
                        # Check if P rate and R rate are different (independent rhythms)
                        p_rate = 60000 / np.mean(p_intervals) if np.mean(p_intervals) > 0 else 0
                        r_rate = 60000 / np.mean(r_intervals_ms) if np.mean(r_intervals_ms) > 0 else 0
                        if abs(p_rate - r_rate) > 20:  # Different rates indicate independent rhythms
                            return True
        
        # Second-degree AV Block Type II (Mobitz Type II)
        # Characteristics:
        # - Fixed PR interval before dropped beats (unlike Type I which has progressive prolongation)
        # - Sudden dropped QRS complexes (no progressive PR prolongation)
        # - More serious than Type I, often progresses to third-degree block
        if dropped_ratio > 0.25:  # At least 25% dropped beats
            if pr_interval is not None:
                if pr_interval <= 250:  # PR is not extremely prolonged
                    return True
            else:
                # If we have significant dropped beats (>30%), it's likely high-grade block
                if dropped_ratio > 0.3:
                    return True
        
        return False
    
    def _is_wpw_syndrome(self, pr_interval, qrs_duration, signal, p_peaks, q_peaks, r_peaks):
        """Detect WPW Syndrome (Wolff-Parkinson-White)"""
        
        if pr_interval is None or qrs_duration is None:
            return False
        
        # Key diagnostic criteria: Short PR + Wide QRS
        short_pr = pr_interval < 120
        wide_qrs = qrs_duration > 120
        
        if not (short_pr and wide_qrs):
            return False
        
        if r_peaks and len(r_peaks) >= 2 and q_peaks and len(q_peaks) >= 1:
            delta_wave_detected = False
            
            # Check a few QRS complexes for delta wave pattern
            for i, r_peak in enumerate(r_peaks[:min(3, len(r_peaks))]):
                # Find corresponding Q peak
                q_peak = None
                for q in q_peaks:
                    if 0 < (r_peak - q) < int(0.15 * self.fs):  # Q should be within 150ms before R
                        q_peak = q
                        break
                
                if q_peak is not None and q_peak < r_peak:
                    segment_start = max(0, q_peak - int(0.02 * self.fs))  # 20ms before Q
                    segment_end = r_peak
                    
                    if segment_end > segment_start and segment_end < len(signal):
                        segment = signal[segment_start:segment_end]
                        if len(segment) > 5:
                            # Calculate slope/rate of rise
                            rise = np.max(segment) - np.min(segment)
                            duration_samples = len(segment)
                            
                            if duration_samples > int(0.08 * self.fs):  # More than 80ms for upstroke
                                delta_wave_detected = True
                                break
                            
                            # Alternative: Check if there's a gradual slope before sharp rise
                            # Delta wave creates a "slurred" appearance
                            if len(segment) > 10:
                                first_half = segment[:len(segment)//2]
                                second_half = segment[len(segment)//2:]
                                first_rise = np.max(first_half) - np.min(first_half) if len(first_half) > 0 else 0
                                second_rise = np.max(second_half) - np.min(second_half) if len(second_half) > 0 else 0
                                
                                # If first half has significant rise (delta), it's suggestive
                                if first_rise > 0.3 * rise and first_rise > 0:
                                    delta_wave_detected = True
                                    break
            
            return True
        
        # If we have short PR and wide QRS but can't check delta wave, still suggest WPW
        return True
    
    def _is_atrial_tachycardia(self, heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks):
        """Detect Atrial Tachycardia - fast regular rhythm with narrow QRS"""
        # Atrial Tachycardia characteristics:
        # 1. Fast heart rate (typically 150-250 bpm, but can be >100 bpm)
        # 2. Regular rhythm
        # 3. Narrow QRS complexes (supraventricular origin)
        # 4. P waves present but may be different morphology or hidden in T waves
        
        if heart_rate is None or qrs_duration is None:
            return False
        
        # Must be tachycardic (>100 bpm, typically 150-250 bpm for atrial tachycardia)
        if heart_rate < 100:
            return False
        
        # Narrow QRS (supraventricular origin)
        if qrs_duration > 120:
            return False
        
        # Regular rhythm (low variability in RR intervals)
        if len(rr_intervals) < 3:
            return False
        
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        
        # Regular rhythm: standard deviation < 120ms or < 10% of mean
        is_regular = rr_std < 120 or (rr_mean > 0 and rr_std / rr_mean < 0.1)
        
        if not is_regular:
            return False
        
        p_count = len(p_peaks) if p_peaks is not None else 0
        r_count = len(r_peaks) if r_peaks is not None else 0
        
        # If heart rate is very fast (>180), P waves may be hidden
        if heart_rate > 180:
            # At very fast rates, P waves are often hidden in T waves
            # Still consider it atrial tachycardia if narrow QRS and regular
            return True
        
        if heart_rate >= 150:
            return True
        
        if heart_rate >= 100 and p_count > 0:
            # Some P waves detected - could be atrial tachycardia
            # Atrial tachycardia often has P waves with different morphology
            return True
        
        return False
    
    def _is_supraventricular_tachycardia(self, heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks):
        """Detect Supraventricular Tachycardia (SVT)"""
        
        if heart_rate is None or qrs_duration is None:
            return False
        
        # SVT is usually a rapid rhythm >150 bpm (often 160–220 bpm)
        # For HR between 100–150, we classify as sinus/atrial tachycardia instead.
        if heart_rate < 150:
            return False
        
        # Narrow QRS (supraventricular origin) - essential for SVT
        if qrs_duration > 120:
            return False
        
        # Regular rhythm (low variability in RR intervals)
        if len(rr_intervals) < 3:
            return False
        
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        
        # Regular rhythm: standard deviation < 120ms or < 10% of mean
        is_regular = rr_std < 120 or (rr_mean > 0 and rr_std / rr_mean < 0.1)
        
        if not is_regular:
            return False
        
        # Check P wave characteristics
        p_count = len(p_peaks) if p_peaks is not None else 0
        r_count = len(r_peaks) if r_peaks is not None else 0
        
        # At high rates, SVT is favored when P waves are absent/hidden or retrograde
        # i.e. fewer P waves than R waves.
        if r_count > 0 and p_count < r_count * 0.8:
            return True
        
        # Even if P waves are seen for most beats, a very fast narrow‑complex regular rhythm
        # is still consistent with SVT.
        return True


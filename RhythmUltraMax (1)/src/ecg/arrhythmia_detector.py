"""
arrhythmia_detector.py — Complete Merged + Optimized
ALL original detections preserved + ALL new arrhythmias from clinical sheet.
"""

import numpy as np
from scipy.signal import find_peaks
import traceback


class ArrhythmiaDetector:
    def __init__(self, sampling_rate=500):
        self.fs = sampling_rate

    def detect_arrhythmias(self, signal, analysis,
                           has_received_serial_data=False,
                           min_serial_data_packets=50):
        analysis   = analysis or {}
        r_peaks    = np.array(analysis.get('r_peaks', []), dtype=int)
        p_peaks    = np.array(analysis.get('p_peaks', []), dtype=int)
        q_peaks    = np.array(analysis.get('q_peaks', []), dtype=int)
        s_peaks    = np.array(analysis.get('s_peaks', []), dtype=int)
        signal_arr = np.asarray(signal, dtype=float) if signal is not None else np.array([])

        try:
            if has_received_serial_data and len(signal_arr) > 0:
                hr = None
                if len(r_peaks) >= 2:
                    rr_ms = np.diff(r_peaks) / self.fs * 1000.0
                    m = float(np.mean(rr_ms))
                    hr = 60000.0 / m if m > 0 else None
                if self._is_asystole(signal_arr, r_peaks, hr,
                                     min_data_packets=min_serial_data_packets):
                    return ["Asystole (Cardiac Arrest)"]
        except Exception as e:
            print(f"Asystole check error: {e}")
            traceback.print_exc()

        if len(r_peaks) < 3:
            return ["Insufficient data for arrhythmia detection."]

        rr_ms        = np.diff(r_peaks) / self.fs * 1000.0
        mean_rr      = float(np.mean(rr_ms)) if len(rr_ms) > 0 else 0.0
        hr           = 60000.0 / mean_rr if mean_rr > 0 else 0.0
        pr_interval  = self._estimate_pr_interval(p_peaks, q_peaks)
        qrs_duration = self._estimate_qrs_duration(q_peaks, s_peaks)
        arrhythmias  = []

        def _try(label, fn, *args):
            try:
                result = fn(*args)
                if result:
                    arrhythmias.append(result if isinstance(result, str) else label)
            except Exception as e:
                print(f"Error detecting {label}: {e}")

        # Life-threatening ventricular
        _try("Ventricular Fibrillation Detected",
             self._is_ventricular_fibrillation, signal_arr, r_peaks, rr_ms)
        _try("Torsade de Pointes",
             self._is_torsade_de_pointes, signal_arr, r_peaks, rr_ms, qrs_duration)
        _try("Polymorphic Ventricular Tachycardia",
             self._is_poly_vtach, signal_arr, r_peaks, rr_ms, qrs_duration)
        _try("Possible Ventricular Tachycardia",
             self._is_ventricular_tachycardia, rr_ms, qrs_duration)

        # R-on-T
        if self._is_pvc_r_on_t(signal_arr, r_peaks, rr_ms, qrs_duration, 'LV'):
            arrhythmias.append("PVC1 LV R-on-T (Dangerous)")
        if self._is_pvc_r_on_t(signal_arr, r_peaks, rr_ms, qrs_duration, 'RV'):
            arrhythmias.append("PVC2 RV R-on-T (Dangerous)")

        # Atrial Fibrillation
        try:
            if self._is_atrial_fibrillation(signal_arr, r_peaks, p_peaks, rr_ms, qrs_duration):
                arrhythmias.append("Atrial Fibrillation Detected")
            elif self._is_afib_rvr(r_peaks, p_peaks, rr_ms, qrs_duration, hr):
                arrhythmias.append("Atrial Fibrillation 2 (with RVR)")
        except Exception as e:
            print(f"Error in AF detection: {e}")

        _try("Possible Atrial Flutter",
             self._is_atrial_flutter, hr, qrs_duration, rr_ms, p_peaks, r_peaks)

        # PVCs & Ectopics
        _try("Ventricular Ectopics Detected",
             self._is_ventricular_ectopics, signal_arr, r_peaks, qrs_duration, p_peaks, rr_ms)
        for label in self._classify_pvcs(signal_arr, r_peaks, rr_ms, qrs_duration, p_peaks, q_peaks, s_peaks):
            arrhythmias.append(label)

        # Bigeminy / Trigeminy / Run
        bigeminy_detected = False
        try:
            if self._is_bigeminy(rr_ms, qrs_duration, signal_arr, r_peaks):
                arrhythmias.append("Bigeminy")
                bigeminy_detected = True
        except Exception as e:
            print(f"Error in bigeminy detection: {e}")
        _try("Trigeminy", self._is_trigeminy, rr_ms, qrs_duration, signal_arr, r_peaks)
        _try("Run of PVCs (>=3 consecutive)",
             self._is_run_of_pvcs, signal_arr, r_peaks, rr_ms, qrs_duration)

        # AV Blocks
        try:
            av = self._is_av_block(pr_interval, p_peaks, r_peaks, rr_ms, hr)
            if av:
                arrhythmias.append(av)
        except Exception as e:
            print(f"Error in AV block detection: {e}")
        _try("High AV-Block",
             self._is_high_av_block, pr_interval, p_peaks, r_peaks, rr_ms, hr)

        # Bundle branch / WPW
        _try("WPW Syndrome (Wolff-Parkinson-White)",
             self._is_wpw_syndrome, pr_interval, qrs_duration, signal_arr, p_peaks, q_peaks, r_peaks)
        _try("Left Bundle Branch Block (LBBB)",
             self._is_left_bundle_branch_block, qrs_duration, pr_interval, rr_ms, signal_arr, q_peaks, r_peaks)
        _try("Right Bundle Branch Block (RBBB)",
             self._is_right_bundle_branch_block, qrs_duration, pr_interval, rr_ms, signal_arr, r_peaks)
        _try("Left Anterior Fascicular Block (LAFB)",
             self._is_left_anterior_fascicular_block, qrs_duration, hr, signal_arr, r_peaks, s_peaks)
        _try("Left Posterior Fascicular Block (LPFB)",
             self._is_left_posterior_fascicular_block, qrs_duration, hr, signal_arr, r_peaks, s_peaks)

        # Supraventricular
        _try("Supraventricular Tachycardia (SVT)",
             self._is_supraventricular_tachycardia, hr, qrs_duration, rr_ms, p_peaks, r_peaks)
        _try("Paroxysmal Atrial Tachycardia (PAT)",
             self._is_pat, hr, qrs_duration, rr_ms, p_peaks, r_peaks)
        _try("Atrial Tachycardia",
             self._is_atrial_tachycardia, hr, qrs_duration, rr_ms, p_peaks, r_peaks)
        _try("Atrial PAC (Premature Atrial Contraction)",
             self._is_pac, signal_arr, r_peaks, rr_ms, qrs_duration, p_peaks)
        _try("Nodal PNC (Premature Junctional Contraction)",
             self._is_pnc, signal_arr, r_peaks, rr_ms, qrs_duration, p_peaks, pr_interval)
        _try("Possible Junctional Rhythm",
             self._is_junctional_rhythm, hr, qrs_duration, pr_interval, rr_ms, p_peaks, r_peaks)

        # Sinus arrhythmias
        _try("Sinus Arrhythmia",
             self._is_sinus_arrhythmia, rr_ms, hr, p_peaks, r_peaks)
        missed = self._is_missed_beat(r_peaks, rr_ms, hr)
        if missed:
            arrhythmias.append(missed)

        # Rate-based fallback
        if not arrhythmias:
            try:
                if self._is_bradycardia(rr_ms):
                    arrhythmias.append("Sinus Bradycardia")
                elif self._is_tachycardia(rr_ms):
                    arrhythmias.append("Sinus Tachycardia")
            except Exception as e:
                print(f"Error in rate-based detection: {e}")

        if not arrhythmias and self._is_normal_sinus_rhythm(rr_ms):
            return ["Normal Sinus Rhythm"]

        return arrhythmias if arrhythmias else ["Unspecified Irregular Rhythm"]

    # ── Utilities ──────────────────────────────────────────────────────────

    def _estimate_pr_interval(self, p_peaks, q_peaks):
        p_arr = np.asarray(p_peaks, dtype=int)
        q_arr = np.asarray(q_peaks, dtype=int)
        if len(p_arr) == 0 or len(q_arr) == 0:
            return None
        intervals = []
        for p in p_arr:
            q_after = q_arr[q_arr > p]
            if len(q_after):
                v = (q_after[0] - p) / self.fs * 1000.0
                if v < 400:
                    intervals.append(v)
        return float(np.mean(intervals)) if intervals else None

    def _estimate_qrs_duration(self, q_peaks, s_peaks):
        q_arr = np.asarray(q_peaks, dtype=int)
        s_arr = np.asarray(s_peaks, dtype=int)
        if len(q_arr) == 0 or len(s_arr) == 0:
            return None
        durations = []
        for q in q_arr:
            s_after = s_arr[s_arr > q]
            if len(s_after):
                v = (s_after[0] - q) / self.fs * 1000.0
                if v < 200:
                    durations.append(v)
        return float(np.mean(durations)) if durations else None

    def _rr_cv(self, rr_ms):
        if len(rr_ms) < 2:
            return 0.0
        m = float(np.mean(rr_ms))
        return float(np.std(rr_ms) / m) if m > 0 else 0.0

    # ── Original preserved detections ──────────────────────────────────────

    def _is_normal_sinus_rhythm(self, rr_intervals):
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3: return False
        return 60 <= 60000.0/float(np.mean(rr)) <= 100 and float(np.std(rr)) < 120

    def _is_asystole(self, signal, r_peaks, heart_rate, min_data_packets=50):
        sig = np.asarray(signal, dtype=float)
        if len(sig) == 0 or len(sig)/self.fs < 2.0: return False
        amp = float(np.ptp(sig)); max_abs = float(np.max(np.abs(sig)))
        flat_amp = 50 if max_abs > 10 else 0.05
        flat_std = 20 if max_abs > 10 else 0.02
        if len(r_peaks) == 0:
            return amp < flat_amp or (amp < flat_amp*5 and float(np.std(sig)) < flat_std)
        if len(r_peaks) <= 2:
            return amp < flat_amp*5 and float(np.std(sig)) < flat_std*6
        if heart_rate is not None and heart_rate < 20:
            return amp < flat_amp*10 and float(np.std(sig)) < flat_std*5
        dur = len(sig)/self.fs
        if dur > 3 and (len(r_peaks)/dur)*60 < 20 and amp < flat_amp*12:
            return True
        return False

    def _is_atrial_fibrillation(self, signal, r_peaks, p_peaks, rr_intervals, qrs_duration):
        if len(r_peaks) < 5: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 2: rr = np.diff(r_peaks)/self.fs*1000.0
        if len(rr) < 2: return False
        mean_rr = float(np.mean(rr))
        if mean_rr <= 0: return False
        rr_cv = float(np.std(rr)/mean_rr)
        p_arr = np.asarray(p_peaks, dtype=int)
        if rr_cv > 0.18:
            if qrs_duration is None or qrs_duration <= 120:
                if len(p_arr) == 0: return True
                if len(p_arr)/max(len(r_peaks),1) < 0.7: return True
                if len(p_arr) >= 2:
                    p_iv = np.diff(p_arr)/self.fs*1000.0
                    if len(p_iv)>0 and float(np.mean(p_iv))>0 and float(np.std(p_iv)/np.mean(p_iv))>0.12: return True
                return True
        if rr_cv > 0.12 and len(rr) >= 3:
            if qrs_duration is None or qrs_duration <= 120:
                diffs = np.abs(np.diff(rr))
                if len(diffs)>0 and float(np.std(diffs))>mean_rr*0.08:
                    if len(p_arr)==0 or len(p_arr)<len(r_peaks)*0.7: return True
        return False

    def _is_afib_rvr(self, r_peaks, p_peaks, rr_ms, qrs_duration, hr):
        if len(rr_ms) < 4: return False
        return self._rr_cv(rr_ms) > 0.10 and hr > 110 and len(p_peaks) < len(r_peaks)*0.5

    def _is_ventricular_tachycardia(self, rr_intervals, qrs_duration):
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3 or qrs_duration is None or qrs_duration <= 120: return False
        mean_rr = float(np.mean(rr))
        if mean_rr <= 0: return False
        return 60000.0/mean_rr > 120 and float(np.std(rr)) < 80

    def _is_ventricular_fibrillation(self, signal, r_peaks, rr_intervals):
        if signal is None or len(signal) < 500: return False
        sig = np.asarray(signal, dtype=float)
        rr  = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3 and len(r_peaks) >= 3: rr = np.diff(r_peaks)/self.fs*1000.0
        if len(rr) >= 3:
            mean_rr = float(np.mean(rr))
            if mean_rr > 0 and float(np.std(rr)/mean_rr) > 0.3:
                if float(np.std(sig)) > 50 and float(np.ptp(sig)) > 100: return True
        dur = len(sig)/self.fs
        if dur >= 2.0 and len(r_peaks) >= 3:
            crr = np.diff(r_peaks)/self.fs*1000.0
            if len(crr) >= 2:
                m = float(np.mean(crr))
                if m > 0 and float(np.std(crr)/m) > 0.25 and float(np.std(sig)) > 40 and float(np.ptp(sig)) > 80: return True
        if dur >= 3.0 and len(r_peaks) < 5:
            if float(np.std(sig)) > 50 and float(np.ptp(sig)) > 100 and float(np.mean(np.abs(sig))) > 30: return True
        if len(sig) >= 1000:
            ma = float(np.mean(np.abs(sig)))
            if ma > 0 and float(np.std(sig))/ma > 1.0 and float(np.ptp(sig)) > 100:
                crr2 = np.diff(r_peaks)/self.fs*1000.0 if len(r_peaks) >= 3 else np.array([])
                if len(r_peaks) < 8 or (len(crr2)>=3 and float(np.mean(crr2))>0 and float(np.std(crr2)/np.mean(crr2))>0.2): return True
        return False

    def _is_bradycardia(self, rr_intervals):
        rr = np.asarray(rr_intervals, dtype=float)
        return len(rr) >= 3 and 60000.0/float(np.mean(rr)) < 60

    def _is_tachycardia(self, rr_intervals):
        rr = np.asarray(rr_intervals, dtype=float)
        return len(rr) >= 3 and 60000.0/float(np.mean(rr)) >= 100

    def _is_ventricular_ectopics(self, signal, r_peaks, qrs_duration, p_peaks, rr_intervals):
        if len(r_peaks) < 5: return False
        rr_ms = np.asarray(rr_intervals, dtype=float)
        if len(rr_ms) == 0: rr_ms = np.diff(r_peaks)/self.fs*1000.0
        if len(rr_ms) < 2: return False
        mean_rr = float(np.mean(rr_ms))
        if mean_rr <= 0: return False
        if qrs_duration is not None and qrs_duration > 120:
            prem = int(np.sum(rr_ms < 0.85*mean_rr))
            comp = sum(1 for i in range(len(rr_ms)-1) if rr_ms[i]<0.85*mean_rr and rr_ms[i+1]>1.15*mean_rr)
            if prem >= 1 and comp >= 1: return True
            if prem >= 2: return True
        rr_sec = np.diff(r_peaks)/self.fs
        if len(rr_sec) < 2: return False
        mean_s = float(np.mean(rr_sec))
        p_arr  = np.asarray(p_peaks, dtype=int)
        for i in range(len(rr_sec)):
            if rr_sec[i] < 0.8*mean_s and i+1 < len(rr_sec) and rr_sec[i+1] > 1.2*mean_s:
                pr = r_peaks[i+1] if i+1 < len(r_peaks) else None
                if pr is not None:
                    if not any(120 <= (pr-p)/self.fs*1000 <= 200 for p in p_arr): return True
        return False

    def _is_bigeminy(self, rr_intervals, qrs_duration, signal, r_peaks):
        try:
            rr = np.asarray(rr_intervals, dtype=float)
            if len(rr) < 4 or len(r_peaks) < 5: return False
            if float(np.max(rr)) < 10: rr = rr*1000.0
            mean_rr = float(np.mean(rr))
            if mean_rr <= 0: return False
            sh, lg = 0.75*mean_rr, 1.03*mean_rr
            alt = sum(1 for i in range(len(rr)-1)
                      if (bool(rr[i]<sh) and bool(rr[i+1]>lg)) or (bool(rr[i]>lg) and bool(rr[i+1]<sh)))
            min_alt = max(2, int(len(rr)*0.25))
            if alt < min_alt: return False
            short_ivs = [float(v) for v in rr if float(v) < sh]
            consistent = True
            if len(short_ivs) >= 2:
                cm = float(np.mean(short_ivs))
                consistent = (float(np.std(short_ivs)/cm) <= 0.25) if cm > 0 else False
            has_wide = qrs_duration is not None and qrs_duration > 120
            if alt >= min_alt:
                if has_wide: return True
                if alt >= max(2, int(len(rr)*0.3)):
                    if consistent: return True
                    if alt >= max(2, int(len(rr)*0.5)): return True
                    if alt >= 3: return True
            return False
        except Exception as e:
            print(f"Error in bigeminy: {e}")
            return False

    def _is_asynchronous_75_bpm(self, heart_rate, rr_intervals, p_peaks, r_peaks):
        try:
            if heart_rate is None: return False
            rr = np.asarray(rr_intervals, dtype=float)
            if len(rr) < 3: return False
            if float(np.max(rr)) < 10: rr = rr*1000.0
            mean_rr = float(np.mean(rr)); std_rr = float(np.std(rr))
            if mean_rr <= 0: return False
            cv = std_rr/mean_rr
            if 70 <= heart_rate <= 80:
                if cv < 0.005 or cv > 0.25 or std_rr < 5 or std_rr > 300: return False
                p_c = len(p_peaks) if p_peaks is not None else 0
                r_c = len(r_peaks) if r_peaks is not None else 0
                if r_c > 0 and p_c < r_c*0.05: return False
                return True
            if not (60 <= heart_rate <= 90) or not (0.03 <= cv <= 0.15 and 30 <= std_rr <= 250): return False
            p_c = len(p_peaks) if p_peaks is not None else 0
            r_c = len(r_peaks) if r_peaks is not None else 0
            if r_c > 0 and p_c < r_c*0.2: return False
            for i in range(len(rr)-1):
                if abs(float(rr[i+1])-float(rr[i])) > 200: return False
            return True
        except Exception: return False

    def _is_left_bundle_branch_block(self, qrs_duration, pr_interval, rr_intervals, signal, q_peaks, r_peaks):
        if qrs_duration is None or qrs_duration < 130: return False
        if pr_interval is not None and pr_interval > 220: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3: return False
        rr_ms = rr*1000.0 if float(np.max(rr)) < 10 else rr
        mean_rr = float(np.mean(rr_ms))
        if mean_rr <= 0 or float(np.std(rr_ms)/mean_rr) > 0.15: return False
        q_arr = np.asarray(q_peaks, dtype=int); r_arr = np.asarray(r_peaks, dtype=int)
        if len(r_arr) == 0 or len(q_arr) > len(r_arr)*0.6: return False
        sig = np.asarray(signal, dtype=float); notched = total = 0
        for r in r_arr[:min(6,len(r_arr))]:
            st = max(0,r-int(0.02*self.fs)); en = min(len(sig),r+int(0.08*self.fs))
            if en-st < 5: continue
            seg = sig[st:en]-sig[st:en].min()
            try: pks,_ = find_peaks(seg, distance=max(2,int(0.01*self.fs)))
            except: continue
            if len(pks) >= 2: notched += 1
            total += 1
        return total > 0 and notched/total >= 0.3

    def _is_right_bundle_branch_block(self, qrs_duration, pr_interval, rr_intervals, signal, r_peaks):
        if qrs_duration is None or qrs_duration < 120: return False
        if pr_interval is not None and pr_interval > 220: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3: return False
        rr_ms = rr*1000.0 if float(np.max(rr)) < 10 else rr
        mean_rr = float(np.mean(rr_ms))
        if mean_rr <= 0 or float(np.std(rr_ms)/mean_rr) > 0.18: return False
        r_arr = np.asarray(r_peaks, dtype=int)
        if len(r_arr) < 3: return False
        sig = np.asarray(signal, dtype=float); ds = checked = 0
        for r in r_arr[:min(6,len(r_arr))]:
            st = max(0,r-int(0.015*self.fs)); en = min(len(sig),r+int(0.09*self.fs))
            if en-st < 6: continue
            seg = sig[st:en]-float(np.mean(sig[st:en]))
            fpv = float(np.max(seg))
            if fpv <= 0: continue
            try: pks,_ = find_peaks(seg, distance=max(2,int(0.008*self.fs)))
            except: continue
            for i in range(len(pks)-1):
                d = (pks[i+1]-pks[i])/self.fs*1000.0
                if 15 <= d <= 70 and seg[pks[i+1]]/fpv >= 0.3: ds += 1; break
            checked += 1
        return checked > 0 and ds/checked >= 0.3

    def _is_left_anterior_fascicular_block(self, qrs_duration, heart_rate, signal, r_peaks, s_peaks):
        if qrs_duration is None or qrs_duration > 130: return False
        if heart_rate is not None and not (45 <= heart_rate <= 120): return False
        r_arr = np.asarray(r_peaks, dtype=int); s_arr = np.asarray(s_peaks, dtype=int)
        n = min(len(r_arr),len(s_arr),6)
        if n < 3: return False
        sig = np.asarray(signal, dtype=float)
        r_a = [abs(float(sig[r_arr[i]])) for i in range(n) if r_arr[i]<len(sig)]
        s_a = [abs(float(sig[s_arr[i]])) for i in range(n) if s_arr[i]<len(sig)]
        if len(r_a) < 3 or len(s_a) < 3: return False
        avg_r,avg_s = float(np.mean(r_a)),float(np.mean(s_a))
        if avg_r<=0 or avg_s<=0 or avg_s/avg_r < 1.6: return False
        sl=ch=0
        for i in range(n):
            if r_arr[i]>=len(sig) or s_arr[i]>=len(sig): continue
            ch+=1
            seg = sig[min(r_arr[i],s_arr[i]):min(len(sig),s_arr[i]+int(0.04*self.fs))]
            if len(seg)<5: continue
            diff = np.diff(seg); thr = 0.2*float(np.max(np.abs(seg))) if float(np.max(np.abs(seg)))>0 else 0.05
            if thr>0 and float(np.mean(np.abs(diff)<thr))>0.6: sl+=1
        return ch>0 and sl/ch >= 0.4

    def _is_left_posterior_fascicular_block(self, qrs_duration, heart_rate, signal, r_peaks, s_peaks):
        if qrs_duration is None or qrs_duration > 130: return False
        if heart_rate is not None and not (45 <= heart_rate <= 120): return False
        r_arr = np.asarray(r_peaks, dtype=int); s_arr = np.asarray(s_peaks, dtype=int)
        n = min(len(r_arr),len(s_arr),6)
        if n < 3: return False
        sig = np.asarray(signal, dtype=float)
        r_a = [abs(float(sig[r_arr[i]])) for i in range(n) if r_arr[i]<len(sig)]
        s_a = [abs(float(sig[s_arr[i]])) for i in range(n) if s_arr[i]<len(sig)]
        if len(r_a)<3 or len(s_a)<3: return False
        avg_r,avg_s = float(np.mean(r_a)),float(np.mean(s_a))
        if avg_r<=0 or avg_s<=0 or avg_r/avg_s < 1.6: return False
        pt=insp=0
        for i in range(n):
            if s_arr[i]>=len(sig): continue
            insp+=1
            seg = sig[s_arr[i]:min(len(sig),s_arr[i]+int(0.05*self.fs))]
            if len(seg)<4: continue
            if float(np.mean(np.diff(seg)>0))>0.6: pt+=1
        return insp>0 and pt/insp >= 0.4

    def _is_junctional_rhythm(self, heart_rate, qrs_duration, pr_interval, rr_intervals, p_peaks, r_peaks):
        if heart_rate is None or qrs_duration is None: return False
        if not (40 <= heart_rate <= 60) or qrs_duration > 120: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3 or float(np.std(rr)) >= 120: return False
        p_arr = np.asarray(p_peaks, dtype=int); r_arr = np.asarray(r_peaks, dtype=int)
        r_c = max(len(r_arr), len(rr)+1, 1)
        return len(p_arr)/r_c < 0.4 or (pr_interval is not None and pr_interval <= 120)

    def _is_atrial_flutter(self, heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks):
        if heart_rate is None or qrs_duration is None: return False
        if not (130 <= heart_rate <= 180) or qrs_duration > 120: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3 or float(np.std(rr)) >= 120: return False
        p_arr = np.asarray(p_peaks, dtype=int); r_arr = np.asarray(r_peaks, dtype=int)
        if len(p_arr)==0 or len(r_arr)==0: return False
        return len(p_arr)/max(len(r_arr),1) >= 1.5

    def _is_av_block(self, pr_interval, p_peaks, r_peaks, rr_intervals, heart_rate):
        p_arr = np.asarray(p_peaks, dtype=int); r_arr = np.asarray(r_peaks, dtype=int)
        if len(p_arr) < 2 or len(r_arr) < 2: return None
        if pr_interval is not None and pr_interval > 200: return "First-Degree AV Block"
        p_c,r_c = len(p_arr),len(r_arr)
        if p_c > r_c*1.2:
            dropped = (p_c-r_c)/max(p_c,1)
            if dropped > 0.5 and len(p_arr)>=3 and len(r_arr)>=3:
                p_iv = np.diff(p_arr)/self.fs*1000.0
                r_iv = np.asarray(rr_intervals,dtype=float)
                if len(r_iv)==0: r_iv = np.diff(r_arr)/self.fs*1000.0
                p_reg = bool(float(np.std(p_iv))<100) if len(p_iv)>0 else False
                r_reg = bool(float(np.std(r_iv))<100) if len(r_iv)>0 else False
                if p_reg and r_reg and heart_rate is not None and heart_rate < 60:
                    return "Third-Degree AV Block (Complete Heart Block)"
            if dropped > 0.2:
                if pr_interval is not None:
                    return ("Second-Degree AV Block (Type I - Wenckebach)"
                            if pr_interval > 180 else "Second-Degree AV Block (Type II)")
                return "Second-Degree AV Block"
        return None

    def _is_high_av_block(self, pr_interval, p_peaks, r_peaks, rr_intervals, heart_rate):
        p_arr = np.asarray(p_peaks, dtype=int); r_arr = np.asarray(r_peaks, dtype=int)
        if len(p_arr)<3 or len(r_arr)<2: return False
        p_c,r_c = len(p_arr),len(r_arr)
        if p_c <= r_c*1.1: return False
        dropped = (p_c-r_c)/max(p_c,1)
        if dropped > 0.5 and len(p_arr)>=3 and len(r_arr)>=3:
            p_iv = np.diff(p_arr)/self.fs*1000.0
            r_iv = np.asarray(rr_intervals,dtype=float)
            if len(r_iv)==0: r_iv = np.diff(r_arr)/self.fs*1000.0
            p_reg = bool(float(np.std(p_iv))<100) if len(p_iv)>0 else False
            r_reg = bool(float(np.std(r_iv))<100) if len(r_iv)>0 else False
            if p_reg and r_reg:
                if heart_rate is not None and heart_rate < 60: return True
                if len(p_iv)>0 and len(r_iv)>0:
                    pr = 60000.0/float(np.mean(p_iv)) if float(np.mean(p_iv))>0 else 0
                    rr2 = 60000.0/float(np.mean(r_iv)) if float(np.mean(r_iv))>0 else 0
                    if abs(pr-rr2)>20: return True
        if dropped>0.25 and pr_interval is not None and pr_interval<=250: return True
        if dropped>0.3: return True
        return False

    def _is_wpw_syndrome(self, pr_interval, qrs_duration, signal, p_peaks, q_peaks, r_peaks):
        if pr_interval is None or qrs_duration is None: return False
        if not (pr_interval < 120 and qrs_duration > 120): return False
        r_arr = np.asarray(r_peaks, dtype=int); q_arr = np.asarray(q_peaks, dtype=int)
        if len(r_arr) < 2 or len(q_arr) < 1: return True
        sig = np.asarray(signal, dtype=float)
        for r in r_arr[:min(3,len(r_arr))]:
            for q in q_arr:
                if 0 < (r-q) < int(0.15*self.fs):
                    seg = sig[max(0,q-int(0.02*self.fs)):r]
                    if len(seg)>int(0.08*self.fs): return True
                    if len(seg)>10:
                        h=len(seg)//2; rise=float(np.ptp(seg))
                        if rise>0 and float(np.ptp(seg[:h]))>0.3*rise: return True
                    break
        return True

    def _is_atrial_tachycardia(self, heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks):
        if heart_rate is None or qrs_duration is None: return False
        if heart_rate < 100 or qrs_duration > 120: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3: return False
        mean_rr = float(np.mean(rr))
        is_reg = float(np.std(rr))<120 or (mean_rr>0 and float(np.std(rr)/mean_rr)<0.1)
        if not is_reg: return False
        if heart_rate > 150: return True
        return len(np.asarray(p_peaks, dtype=int)) > 0

    def _is_supraventricular_tachycardia(self, heart_rate, qrs_duration, rr_intervals, p_peaks, r_peaks):
        if heart_rate is None or qrs_duration is None: return False
        if heart_rate < 150 or qrs_duration > 120: return False
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3: return False
        mean_rr = float(np.mean(rr))
        is_reg = float(np.std(rr))<120 or (mean_rr>0 and float(np.std(rr)/mean_rr)<0.1)
        return is_reg

    # ── New detections ──────────────────────────────────────────────────────

    def _is_poly_vtach(self, signal, r_peaks, rr_ms, qrs_duration):
        if len(rr_ms)<4 or qrs_duration is None: return False
        hr = 60000.0/float(np.mean(rr_ms)); cv = self._rr_cv(rr_ms)
        if hr<100 or qrs_duration<120 or cv<0.12: return False
        sig = np.asarray(signal,dtype=float); r_arr = np.asarray(r_peaks,dtype=int)
        win = int(self.fs*0.1)
        amps = [float(np.ptp(sig[max(0,p-win):min(len(sig),p+win)])) for p in r_arr if p-win>=0]
        if len(amps)<4: return False
        m = float(np.mean(amps))
        return float(np.std(amps)/m)>0.30 if m>0 else False

    def _is_torsade_de_pointes(self, signal, r_peaks, rr_ms, qrs_duration):
        if len(rr_ms)<6 or qrs_duration is None: return False
        hr = 60000.0/float(np.mean(rr_ms))
        if hr<150 or qrs_duration<120: return False
        sig = np.asarray(signal,dtype=float); r_arr = np.asarray(r_peaks,dtype=int)
        win = int(self.fs*0.12)
        amps = [float(np.ptp(sig[max(0,p-win):min(len(sig),p+win)])) for p in r_arr if p-win>=0]
        if len(amps)<6: return False
        amp_arr = np.array(amps); m = float(np.mean(amp_arr))
        amp_cv = float(np.std(amp_arr)/m) if m>0 else 0
        env_peaks,_ = find_peaks(amp_arr, distance=3)
        return amp_cv>0.35 and len(env_peaks)>=2

    def _is_pvc_r_on_t(self, signal, r_peaks, rr_ms, qrs_duration, side='LV'):
        if len(rr_ms)<4 or qrs_duration is None: return False
        mean_rr = float(np.mean(rr_ms)); qt_est = mean_rr*0.42
        return any(rr < qt_est*0.95 and rr < mean_rr*0.80 for rr in rr_ms)

    def _classify_pvcs(self, signal, r_peaks, rr_ms, qrs_duration, p_peaks, q_peaks, s_peaks):
        results = []
        if len(rr_ms)<4 or qrs_duration is None or qrs_duration<100: return results
        mean_rr = float(np.mean(rr_ms)); r_arr = np.asarray(r_peaks,dtype=int)
        sig = np.asarray(signal,dtype=float); win = int(self.fs*0.06)
        ectopic_idx = [i+1 for i,rr in enumerate(rr_ms) if rr < mean_rr*0.82]
        if not ectopic_idx: return results
        morphs=[]; early=0
        for idx in ectopic_idx:
            if idx>=len(r_arr): continue
            pos=r_arr[idx]; sl=sig[max(0,pos-win):min(len(sig),pos+win)]
            if len(sl)<4: continue
            morphs.append('LV' if float(np.max(sl))>abs(float(np.min(sl)))*1.2 else 'RV')
            if idx-1<len(rr_ms) and rr_ms[idx-1]<mean_rr*0.70: early+=1
        if not morphs: return results
        n_morphs=len(set(morphs)); freq=len(ectopic_idx)/max(len(rr_ms),1)
        lv=morphs.count('LV'); rv=morphs.count('RV')
        if n_morphs>=2:
            results.append("Frequent Multi-focal PVCs" if freq>0.2 else "Multi-focal PVCs")
        elif lv>=rv:
            results.append("PVC1 LV Early" if early>0 else "PVC1 Left Ventricle")
        else:
            results.append("PVC2 RV Early" if early>0 else "PVC2 Right Ventricle")
        return results

    def _is_trigeminy(self, rr_ms, qrs_duration, signal, r_peaks):
        if len(rr_ms)<6: return False
        thirds = list(range(2,len(rr_ms),3))
        if not thirds: return False
        other = [i for i in range(len(rr_ms)) if i not in thirds]
        if not other: return False
        mo = float(np.mean([rr_ms[i] for i in other]))
        if mo <= 0: return False
        return sum(1 for i in thirds if rr_ms[i]<mo*0.80)/len(thirds) > 0.60

    def _is_run_of_pvcs(self, signal, r_peaks, rr_ms, qrs_duration):
        if len(rr_ms)<3 or qrs_duration is None or qrs_duration<100: return False
        mean_rr = float(np.mean(rr_ms)); consec=mx=0
        for rr in rr_ms:
            if rr<mean_rr*0.82: consec+=1; mx=max(mx,consec)
            else: consec=0
        return mx>=3

    def _is_pat(self, heart_rate, qrs_duration, rr_ms, p_peaks, r_peaks):
        if heart_rate is None or len(rr_ms)<4: return False
        return (150<=heart_rate<=250 and self._rr_cv(rr_ms)<0.10
                and (qrs_duration is None or qrs_duration<120))

    def _is_pac(self, signal, r_peaks, rr_ms, qrs_duration, p_peaks):
        if len(rr_ms)<4: return False
        if not (qrs_duration is None or qrs_duration<120): return False
        mean_rr = float(np.mean(rr_ms)); n_prem = sum(1 for rr in rr_ms if rr<mean_rr*0.85)
        p_arr = np.asarray(p_peaks,dtype=int)
        return n_prem>=1 and n_prem<len(rr_ms)*0.30 and len(p_arr)>=len(r_peaks)*0.5

    def _is_pnc(self, signal, r_peaks, rr_ms, qrs_duration, p_peaks, pr_ms):
        if len(rr_ms)<4: return False
        narrow = qrs_duration is None or qrs_duration<120
        short_pr = pr_ms is not None and pr_ms<120
        p_arr = np.asarray(p_peaks,dtype=int)
        absent_p = len(p_arr)<len(r_peaks)*0.6
        mean_rr = float(np.mean(rr_ms))
        n_prem = sum(1 for rr in rr_ms if rr<mean_rr*0.88)
        return narrow and (short_pr or absent_p) and n_prem>=1

    def _is_sinus_arrhythmia(self, rr_ms, hr, p_peaks, r_peaks):
        if len(rr_ms)<4 or hr is None: return False
        cv = self._rr_cv(rr_ms); p_arr = np.asarray(p_peaks,dtype=int)
        return 50<=hr<=110 and 0.10<=cv<=0.25 and len(p_arr)>=len(r_peaks)*0.7

    def _is_missed_beat(self, r_peaks, rr_ms, hr):
        if len(rr_ms)<4: return None
        mean_rr = float(np.mean(rr_ms))
        for rr in rr_ms:
            if rr>mean_rr*1.8:
                if 65<=hr<=95: return "Missed Beat at ~80 BPM (SA Block / Sinus Pause)"
                if 105<=hr<=140: return "Missed Beat at ~120 BPM (SA Block / Sinus Pause)"
                return "Sinus Pause / Missed Beat"
        return None
"""
ArrhythmiaEngine — Priority-based clinical rhythm classifier.

Priority order (highest → lowest):
  1. Asystole
  2. Ventricular Fibrillation
  3. Ventricular Tachycardia
  4. Atrial Fibrillation
  5. Atrial Flutter
  6. AV Block (3rd-degree, then 1st-degree)
  7. Bundle Branch Block (LBBB / RBBB)
  8. Sinus rhythms (Bradycardia / NSR / Tachycardia)
  9. QT findings

Additional secondary findings (BBB, QT, etc.) are appended AFTER the
primary rhythm — never replacing it.  The first element of the returned
list is always the primary rhythm.
"""


class ArrhythmiaEngine:
    def __init__(self, features):
        self.f = features

    # ── helpers ───────────────────────────────────────────────────────────────

    def is_irregular(self):
        rr = self.f.get("rr_intervals", [])
        if len(rr) < 3:
            return False
        import numpy as np
        rr = np.array(rr)
        return float(np.std(rr)) > 80

    def _rr_variability(self):
        rr = self.f.get("rr_intervals", [])
        if len(rr) < 2:
            return 0
        import numpy as np
        rr = np.array(rr)
        return float(np.std(rr))

    # ── lethal / primary checks ───────────────────────────────────────────────

    def _is_asystole(self):
        """Signal amplitude below clinical threshold → no cardiac output.

        Three detection paths (highest → lowest priority):
        1. signal_std < 50 ADC counts   — raw ADC flat-line (device sends ~2060)
        2. signal_amplitude < 0.05 mV   — mV-normalised flat line
        3. HR effectively zero + no QRS — classic HR-based fallback
        """
        # Path 1: ADC-scale flat-line (std in raw counts)
        signal_std = self.f.get("signal_std", None)
        if signal_std is not None:
            det_mean = float(self.f.get("signal_amplitude", 0) or 0)
            if det_mean > 100.0 and float(signal_std) < 50.0:
                return True

        # Path 2: mV-scale amplitude check
        amplitude = self.f.get("signal_amplitude", None)
        if amplitude is not None:
            if float(amplitude) < 0.05:
                return True

        # Path 3: HR-based fallback
        hr = self.f.get("hr", 0)
        qrs = self.f.get("qrs", 0)
        return hr < 5 and qrs == 0

    def _is_vf(self):
        """
        Ventricular Fibrillation:
          - Chaotic signal (high relative variance OR no organised R-peaks)
          - NOT a flat line (that's asystole)
        Feature: vf_score > 0 (set by upstream signal processor), OR
                 no R-peaks detected with non-zero amplitude.
        """
        vf_score = self.f.get("vf_score", 0)
        if vf_score and float(vf_score) > 0.35:
            return True
        # Fallback: no organised QRS + signal present
        hr = self.f.get("hr", 0)
        qrs = self.f.get("qrs", 0)
        amplitude = self.f.get("signal_amplitude", None)
        has_signal = amplitude is None or float(amplitude) >= 0.05
        return has_signal and (hr == 0 or hr > 150) and not qrs

    def _is_vt(self):
        """
        Ventricular Tachycardia:
          - HR >= 100 bpm
          - Wide QRS >= 120 ms
          - No P waves (AV dissociation)
          - Relatively regular rhythm (not VF)
        """
        hr = self.f.get("hr", 0)
        qrs = self.f.get("qrs", 0)
        dominant_ratio = float(self.f.get("dominant_ratio", 0.0) or 0.0)
        return hr > 150 and qrs > 120 and dominant_ratio < 0.7
    def _is_atrial_flutter(self):
        flutter_flag = self.f.get("atrial_flutter", False)
        flutter_score = self.f.get("flutter_score", 0)
        rr_var = float(self._rr_variability() or 0.0)

        p = self.f.get("p_detected", True)
        
        # If any clear P waves are detected, it contradicts Atrial Flutter
        if p:
            return False

        # Flutter usually keeps a comparatively organised ventricular response.
        # If RR variability is already in the AF range, prefer AF.
        if rr_var > 80.0:
            return False

        # STRICT condition (Fluke-like)
        return (
            flutter_flag
            and float(flutter_score) > 0.18
        )

    # ── main detection ────────────────────────────────────────────────────────

    def detect(self):
        hr = self.f.get("hr", 0)
        pr = self.f.get("pr", 0)
        qrs = self.f.get("qrs", 0)
        qtc = self.f.get("qtc", 0)
        p = self.f.get("p_detected", True)
        indicator = self.f.get("lbbb_indicator", 0)
        dominant_ratio = float(self.f.get("dominant_ratio", 0.0) or 0.0)
        cluster_count = int(self.f.get("cluster_count", 0) or 0)
        ectopic_ratio = float(self.f.get("ectopic_ratio", 0.0) or 0.0)
        rr_var = self._rr_variability()

        # ─── STEP 1: Lethal / primary rhythm (ONE winner only) ────────────────
        if self._is_asystole():
            primary = "Asystole"
        elif self._is_vf():
            primary = "Ventricular Fibrillation"
        elif self._is_vt():
            primary = "Ventricular Tachycardia"
        elif self._is_atrial_flutter():
            primary = "Atrial Flutter"
        elif not p and self.is_irregular():
            primary = "Atrial Fibrillation"
        elif dominant_ratio and dominant_ratio < 0.6 and self.is_irregular():
            primary = "Atrial Fibrillation"
        elif self.f.get("av_dissociation") and hr < 60:
            primary = "Third-degree AV Block"
        elif hr < 40 and not p:
            primary = "Third-degree AV Block"
        elif hr < 40 and p:
            primary = "Sinus Bradycardia"
        elif self.f.get("pr_progression") and p:
            primary = "Second-degree AV Block (Mobitz I)"
        elif self.f.get("dropped_beats") and p:
            primary = "Second-degree AV Block (Mobitz II)"
        elif pr > 200 and p:
            primary = "First-degree AV Block (Prolonged PR)"
        elif 60 <= hr <= 100 and p:
            primary = "Normal Sinus Rhythm"
        elif hr > 100 and p:
            primary = "Sinus Tachycardia"
        elif hr < 60 and p:
            primary = "Sinus Bradycardia"
        elif hr < 60:
            primary = "Bradycardia (non-sinus)"
        elif hr > 100:
            primary = "Tachycardia (non-sinus)"
        else:
            primary = "Rhythm Undetermined"

        results = [primary]

        # ─── STEP 2: Secondary / morphology findings ─────────────────────────
        # >> When Asystole is primary, ALL parameters are 0; suppress every
        #    secondary finding so the UI shows only "Asystole".
        if primary == "Asystole":
            return results

        # These are ADDED to the primary, never replace it.
        LETHAL = {"Asystole", "Ventricular Fibrillation", "Ventricular Tachycardia"}

        # Bundle Branch Block — requires wide QRS + morphology indicator
        # (only add if NOT already captured in VT / primary)
        if qrs >= 110 and primary not in LETHAL:
            if indicator > 0.2:
                results.append("Complete Left Bundle Branch Block")
            elif indicator < -0.2:
                results.append("Complete Right Bundle Branch Block")
            # Wide QRS without clear BBB pattern → only flag if no lethal dx
            elif qrs >= 120 and "Wide QRS" not in results:
                results.append("Wide QRS (non-specific)")

        # QT prolongation — secondary finding regardless of primary
        if qtc > 500 and "Long QT Syndrome" not in results:
            results.append("Long QT Syndrome")
        elif 460 < qtc <= 500 and "Prolonged QTc" not in results:
            results.append("Prolonged QTc")

        if ectopic_ratio > 0.2 and "Frequent PVCs" not in results:
            results.append("Frequent PVCs")
        if cluster_count > 2 and "Multifocal PVCs" not in results:
            results.append("Multifocal PVCs")

        return results

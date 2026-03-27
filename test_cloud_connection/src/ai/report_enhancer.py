"""
AI-Powered Report Enhancement
Generates intelligent summaries, explanations, and recommendations for ECG reports
"""

import json
import os
from typing import Dict, List, Optional

# Try to import OpenAI (optional dependency)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print(" OpenAI not installed. Install with: pip install openai")

# Try to import Anthropic Claude (optional dependency)
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


class AIReportEnhancer:
    """
    Enhances ECG reports with AI-generated insights, summaries, and recommendations
    """
    
    def __init__(self, api_provider="openai", api_key=None):
        """
        Initialize AI Report Enhancer
        
        Args:
            api_provider: "openai" or "claude" or "local"
            api_key: API key for the provider (optional, can use env var)
        """
        self.api_provider = api_provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        
        if api_provider == "openai" and OPENAI_AVAILABLE and self.api_key:
            openai.api_key = self.api_key
            self.enabled = True
        elif api_provider == "claude" and CLAUDE_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.enabled = True
        else:
            self.enabled = False
            print(" AI features disabled. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
    
    def generate_executive_summary(self, metrics: Dict, arrhythmias: List[str], 
                                   patient_data: Optional[Dict] = None) -> str:
        """
        Generate AI-powered executive summary for ECG report
        
        Args:
            metrics: Dictionary with ECG metrics (HR, PR, QRS, QTc, ST, etc.)
            arrhythmias: List of detected arrhythmias
            patient_data: Optional patient information (age, gender, etc.)
        
        Returns:
            Executive summary string
        """
        if not self.enabled:
            return self._generate_fallback_summary(metrics, arrhythmias)
        
        # Build prompt
        prompt = self._build_summary_prompt(metrics, arrhythmias, patient_data)
        
        try:
            if self.api_provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a cardiology AI assistant. Provide clear, professional, and medically accurate ECG analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            
            elif self.api_provider == "claude":
                message = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=500,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text.strip()
        
        except Exception as e:
            print(f" AI API error: {e}")
            return self._generate_fallback_summary(metrics, arrhythmias)
    
    def generate_intelligent_findings(self, metrics: Dict, arrhythmias: List[str],
                                       patient_data: Optional[Dict] = None) -> List[Dict]:
        """
        Generate intelligent, context-aware findings
        
        Returns:
            List of finding dictionaries with 'finding', 'explanation', 'severity', 'recommendations'
        """
        if not self.enabled:
            return self._generate_fallback_findings(metrics, arrhythmias)
        
        prompt = f"""
        Analyze this ECG reading and provide intelligent findings:
        
        Metrics:
        - Heart Rate: {metrics.get('HR', 'N/A')} bpm
        - PR Interval: {metrics.get('PR', 'N/A')} ms
        - QRS Duration: {metrics.get('QRS', 'N/A')} ms
        - QTc Interval: {metrics.get('QTc', 'N/A')} ms
        - ST Segment: {metrics.get('ST', 'N/A')} mV
        
        Detected Arrhythmias: {', '.join(arrhythmias) if arrhythmias else 'None'}
        
        Patient Info: {f"Age: {patient_data.get('age', 'N/A')}, Gender: {patient_data.get('gender', 'N/A')}" if patient_data else "Not provided"}
        
        For each significant finding, provide:
        1. Finding name (brief)
        2. Explanation in plain language
        3. Clinical significance
        4. Severity (Low/Moderate/High)
        5. Specific recommendations
        
        Format as JSON array of objects with keys: finding, explanation, significance, severity, recommendations
        """
        
        try:
            if self.api_provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a cardiology AI. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                return result.get('findings', [])
        
        except Exception as e:
            print(f" AI API error: {e}")
            return self._generate_fallback_findings(metrics, arrhythmias)
    
    def calculate_risk_score(self, metrics: Dict, arrhythmias: List[str],
                           patient_data: Optional[Dict] = None) -> Dict:
        """
        Calculate comprehensive risk score
        
        Returns:
            Dictionary with risk_score (0-100), risk_level, risk_factors, recommendations
        """
        risk_score = 0
        risk_factors = []
        
        # Heart Rate Risk
        hr = metrics.get('HR', 70)
        if hr > 100:
            risk_score += 15
            risk_factors.append("Elevated heart rate")
        elif hr < 60:
            risk_score += 10
            risk_factors.append("Bradycardia")
        
        # QTc Prolongation Risk
        qtc = metrics.get('QTc', 400)
        if qtc > 500:
            risk_score += 30
            risk_factors.append("Severe QTc prolongation")
        elif qtc > 450:
            risk_score += 15
            risk_factors.append("QTc prolongation")
        
        # PR Interval Risk
        pr = metrics.get('PR', 160)
        if pr > 200:
            risk_score += 10
            risk_factors.append("Prolonged PR interval")
        
        # QRS Duration Risk
        qrs = metrics.get('QRS', 90)
        if qrs > 120:
            risk_score += 15
            risk_factors.append("Wide QRS complex")
        
        # Arrhythmia Risk
        if arrhythmias:
            for arr in arrhythmias:
                if "Atrial Fibrillation" in arr or "Ventricular Tachycardia" in arr:
                    risk_score += 25
                    risk_factors.append(f"Serious arrhythmia: {arr}")
                else:
                    risk_score += 10
                    risk_factors.append(f"Arrhythmia: {arr}")
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "High"
        elif risk_score >= 40:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        # Generate recommendations based on risk
        recommendations = self._generate_risk_recommendations(risk_score, risk_factors)
        
        return {
            'risk_score': min(100, risk_score),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }
    
    def generate_plain_language_explanation(self, technical_term: str, 
                                           context: Optional[Dict] = None) -> str:
        """
        Convert technical ECG terms to plain language
        """
        explanations = {
            'prolonged_pr_interval': 
                "Your heart's electrical signal is taking longer than normal to travel from the upper to lower chambers. This is often harmless but worth monitoring.",
            
            'qtc_prolongation':
                "The time it takes for your heart's lower chambers to reset between beats is longer than usual. This can sometimes be related to medications or heart conditions.",
            
            'atrial_fibrillation':
                "Your heart's upper chambers are beating irregularly. This can cause palpitations and may increase stroke risk. Treatment options are available.",
            
            'ventricular_tachycardia':
                "Your heart's lower chambers are beating too fast. This is a serious condition that requires immediate medical attention.",
            
            'bradycardia':
                "Your heart rate is slower than normal. This can be normal for athletes but may indicate issues in others.",
            
            'tachycardia':
                "Your heart rate is faster than normal. This can be due to stress, exercise, or medical conditions."
        }
        
        return explanations.get(technical_term.lower(), 
                              f"{technical_term} detected. Please consult with your healthcare provider for interpretation.")
    
    def _build_summary_prompt(self, metrics: Dict, arrhythmias: List[str],
                             patient_data: Optional[Dict]) -> str:
        """Build prompt for AI summary generation"""
        prompt = f"""
        Analyze this ECG reading and provide a professional executive summary:
        
        ECG Metrics:
        - Heart Rate: {metrics.get('HR', 'N/A')} bpm
        - PR Interval: {metrics.get('PR', 'N/A')} ms
        - QRS Duration: {metrics.get('QRS', 'N/A')} ms
        - QTc Interval: {metrics.get('QTc', 'N/A')} ms
        - ST Segment: {metrics.get('ST', 'N/A')} mV
        
        Detected Arrhythmias: {', '.join(arrhythmias) if arrhythmias else 'None detected'}
        
        """
        
        if patient_data:
            prompt += f"""
        Patient Information:
        - Age: {patient_data.get('age', 'N/A')}
        - Gender: {patient_data.get('gender', 'N/A')}
        """
        
        prompt += """
        
        Provide a 2-3 paragraph executive summary that includes:
        1. Overall cardiac health assessment
        2. Key findings in plain language
        3. Clinical significance
        4. Urgency level (Normal/Monitor/Urgent)
        
        Be professional, clear, and medically accurate.
        """
        
        return prompt
    
    def _generate_fallback_summary(self, metrics: Dict, arrhythmias: List[str]) -> str:
        """Generate summary without AI (fallback)"""
        hr = metrics.get('HR', 70)
        qtc = metrics.get('QTc', 400)
        
        summary = f"This ECG reading shows a heart rate of {hr} bpm. "
        
        if arrhythmias:
            summary += f"The following arrhythmias were detected: {', '.join(arrhythmias)}. "
        else:
            summary += "No significant arrhythmias were detected. "
        
        if qtc > 450:
            summary += "QTc interval is prolonged and may require medical attention. "
        elif 400 <= qtc <= 450:
            summary += "All measured intervals are within normal limits. "
        
        summary += "Please consult with your healthcare provider for complete interpretation."
        
        return summary
    
    def _generate_fallback_findings(self, metrics: Dict, arrhythmias: List[str]) -> List[Dict]:
        """Generate findings without AI (fallback)"""
        findings = []
        
        hr = metrics.get('HR', 70)
        if hr > 100:
            findings.append({
                'finding': 'Tachycardia',
                'explanation': f'Heart rate of {hr} bpm is above normal range',
                'severity': 'Moderate',
                'recommendations': ['Monitor heart rate', 'Consider stress management']
            })
        elif hr < 60:
            findings.append({
                'finding': 'Bradycardia',
                'explanation': f'Heart rate of {hr} bpm is below normal range',
                'severity': 'Low',
                'recommendations': ['May be normal for athletes', 'Monitor for symptoms']
            })
        
        if arrhythmias:
            for arr in arrhythmias:
                findings.append({
                    'finding': arr,
                    'explanation': f'{arr} detected in ECG reading',
                    'severity': 'High' if 'Tachycardia' in arr or 'Fibrillation' in arr else 'Moderate',
                    'recommendations': ['Consult cardiologist', 'Follow-up ECG recommended']
                })
        
        return findings
    
    def _generate_risk_recommendations(self, risk_score: int, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on risk score"""
        recommendations = []
        
        if risk_score >= 70:
            recommendations.extend([
                "Urgent cardiology consultation recommended",
                "Consider immediate medical evaluation",
                "Monitor symptoms closely"
            ])
        elif risk_score >= 40:
            recommendations.extend([
                "Cardiology follow-up recommended",
                "Monitor ECG parameters regularly",
                "Review medications with healthcare provider"
            ])
        else:
            recommendations.extend([
                "Continue routine monitoring",
                "Maintain healthy lifestyle",
                "Schedule routine follow-up as recommended"
            ])
        
        return recommendations


# Convenience function for easy integration
def enhance_report_with_ai(metrics: Dict, arrhythmias: List[str],
                          patient_data: Optional[Dict] = None,
                          api_provider: str = "openai") -> Dict:
    """
    Convenience function to enhance ECG report with AI
    
    Returns:
        Dictionary with:
        - executive_summary: str
        - intelligent_findings: List[Dict]
        - risk_assessment: Dict
        - plain_language_explanations: Dict[str, str]
    """
    enhancer = AIReportEnhancer(api_provider=api_provider)
    
    return {
        'executive_summary': enhancer.generate_executive_summary(metrics, arrhythmias, patient_data),
        'intelligent_findings': enhancer.generate_intelligent_findings(metrics, arrhythmias, patient_data),
        'risk_assessment': enhancer.calculate_risk_score(metrics, arrhythmias, patient_data),
        'plain_language_explanations': {
            arr: enhancer.generate_plain_language_explanation(arr)
            for arr in arrhythmias
        }
    }


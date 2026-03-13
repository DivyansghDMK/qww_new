"""
AI-Powered Dashboard Insights
Provides real-time intelligent insights, trends, and recommendations for the dashboard
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np


class DashboardInsightsEngine:
    """
    Generates intelligent insights for the dashboard
    """
    
    def __init__(self):
        self.insights_history = []
        self.trend_data = {}
    
    def generate_health_score(self, metrics: Dict, historical_data: Optional[List[Dict]] = None) -> Dict:
        """
        Calculate AI-powered health score (0-100)
        
        Returns:
            Dictionary with score, level, breakdown, trend
        """
        base_score = 100
        deductions = []
        
        # Heart Rate Score (0-20 points)
        hr = metrics.get('HR', 70)
        if 60 <= hr <= 100:
            hr_score = 20
        elif 50 <= hr < 60 or 100 < hr <= 110:
            hr_score = 15
            deductions.append("Heart rate slightly outside optimal range")
        else:
            hr_score = 5
            deductions.append("Heart rate significantly abnormal")
        
        # QTc Score (0-25 points)
        qtc = metrics.get('QTc', 400)
        if qtc <= 440:
            qtc_score = 25
        elif qtc <= 470:
            qtc_score = 15
            deductions.append("QTc slightly prolonged")
        else:
            qtc_score = 0
            deductions.append("QTc significantly prolonged - high risk")
        
        # PR Interval Score (0-15 points)
        pr = metrics.get('PR', 160)
        if 120 <= pr <= 200:
            pr_score = 15
        elif 200 < pr <= 220:
            pr_score = 10
            deductions.append("PR interval slightly prolonged")
        else:
            pr_score = 5
            deductions.append("PR interval abnormal")
        
        # QRS Duration Score (0-15 points)
        qrs = metrics.get('QRS', 90)
        if qrs <= 100:
            qrs_score = 15
        elif qrs <= 120:
            qrs_score = 10
            deductions.append("QRS duration borderline")
        else:
            qrs_score = 5
            deductions.append("Wide QRS complex detected")
        
        # Rhythm Score (0-25 points)
        arrhythmias = metrics.get('arrhythmias', [])
        if not arrhythmias:
            rhythm_score = 25
        elif any("Normal" in arr for arr in arrhythmias):
            rhythm_score = 20
        else:
            rhythm_score = 5
            deductions.append("Arrhythmia detected")
        
        # Calculate total score
        total_score = hr_score + qtc_score + pr_score + qrs_score + rhythm_score
        
        # Determine level
        if total_score >= 85:
            level = "Excellent"
            color = "#00C853"  # Green
        elif total_score >= 70:
            level = "Good"
            color = "#64DD17"  # Light green
        elif total_score >= 55:
            level = "Fair"
            color = "#FFD600"  # Yellow
        elif total_score >= 40:
            level = "Moderate"
            color = "#FF6D00"  # Orange
        else:
            level = "Poor"
            color = "#D32F2F"  # Red
        
        # Trend analysis
        trend = "stable"
        if historical_data and len(historical_data) >= 2:
            recent_scores = [self._calculate_simple_score(d) for d in historical_data[-3:]]
            if len(recent_scores) >= 2:
                if recent_scores[-1] > recent_scores[0] + 5:
                    trend = "improving"
                elif recent_scores[-1] < recent_scores[0] - 5:
                    trend = "declining"
        
        return {
            'score': total_score,
            'level': level,
            'color': color,
            'trend': trend,
            'breakdown': {
                'heart_rate': hr_score,
                'qtc_interval': qtc_score,
                'pr_interval': pr_score,
                'qrs_duration': qrs_score,
                'rhythm': rhythm_score
            },
            'deductions': deductions
        }
    
    def generate_trend_insights(self, current_metrics: Dict, 
                               historical_data: List[Dict]) -> List[Dict]:
        """
        Generate trend-based insights
        
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        if len(historical_data) < 2:
            return insights
        
        # Heart Rate Trend
        hr_values = [d.get('HR', 70) for d in historical_data[-7:]]  # Last 7 readings
        if len(hr_values) >= 2:
            hr_trend = np.mean(hr_values[-3:]) - np.mean(hr_values[:3])
            if abs(hr_trend) > 5:
                insights.append({
                    'type': 'trend',
                    'metric': 'Heart Rate',
                    'title': 'Heart Rate Trend',
                    'message': f"Heart rate showing {'increasing' if hr_trend > 0 else 'decreasing'} trend ({abs(hr_trend):.1f} bpm change). This may indicate changes in activity, stress, or medication effects.",
                    'severity': 'moderate',
                    'icon': '📈' if hr_trend > 0 else '📉'
                })
        
        # QTc Trend
        qtc_values = [d.get('QTc', 400) for d in historical_data[-7:]]
        if len(qtc_values) >= 2:
            qtc_trend = np.mean(qtc_values[-3:]) - np.mean(qtc_values[:3])
            if qtc_trend > 10:
                insights.append({
                    'type': 'alert',
                    'metric': 'QTc Interval',
                    'title': 'QTc Prolongation Trend',
                    'message': f"QTc interval is increasing ({qtc_trend:.0f}ms over recent readings). This may increase arrhythmia risk. Consider medication review.",
                    'severity': 'high',
                    'icon': '⚠️',
                    'action': 'Consult cardiologist'
                })
        
        return insights
    
    def generate_smart_alerts(self, metrics: Dict, historical_data: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Generate context-aware smart alerts
        """
        alerts = []
        
        # Context-aware QTc alert
        qtc = metrics.get('QTc', 400)
        if qtc > 500:
            alerts.append({
                'type': 'critical',
                'title': 'Severe QTc Prolongation',
                'message': f'QTc interval is {qtc}ms (normal <440ms). This significantly increases risk of dangerous arrhythmias. Immediate medical evaluation recommended.',
                'priority': 1,
                'action': 'Seek immediate medical attention'
            })
        elif qtc > 470:
            alerts.append({
                'type': 'high',
                'title': 'QTc Prolongation',
                'message': f'QTc interval is {qtc}ms. Monitor closely and review medications that may affect QT interval.',
                'priority': 2,
                'action': 'Schedule cardiology consultation'
            })
        
        # Arrhythmia alerts
        arrhythmias = metrics.get('arrhythmias', [])
        for arr in arrhythmias:
            if "Ventricular Tachycardia" in arr:
                alerts.append({
                    'type': 'critical',
                    'title': 'Serious Arrhythmia Detected',
                    'message': f'{arr} detected. This is a potentially life-threatening condition requiring immediate medical attention.',
                    'priority': 1,
                    'action': 'Seek emergency medical care'
                })
            elif "Atrial Fibrillation" in arr:
                alerts.append({
                    'type': 'high',
                    'title': 'Atrial Fibrillation',
                    'message': f'{arr} detected. This increases stroke risk. Consult cardiologist for treatment options.',
                    'priority': 2,
                    'action': 'Consult cardiologist'
                })
        
        # Sort by priority
        alerts.sort(key=lambda x: x['priority'])
        
        return alerts
    
    def generate_personalized_recommendations(self, metrics: Dict, 
                                            patient_data: Optional[Dict] = None,
                                            historical_data: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Generate personalized recommendations
        """
        recommendations = []
        
        # HRV-based stress management
        hr = metrics.get('HR', 70)
        if hr > 90 and patient_data:
            recommendations.append({
                'category': 'Lifestyle',
                'title': 'Stress Management',
                'description': 'Your heart rate suggests elevated stress levels.',
                'actions': [
                    'Practice deep breathing exercises (5-10 min daily)',
                    'Ensure 7-9 hours of sleep',
                    'Consider meditation or yoga',
                    'Review work-life balance'
                ],
                'priority': 'medium'
            })
        
        # QTc medication review
        qtc = metrics.get('QTc', 400)
        if qtc > 450:
            recommendations.append({
                'category': 'Medical',
                'title': 'Medication Review',
                'description': 'QTc prolongation may be related to medications.',
                'actions': [
                    'Review current medications with healthcare provider',
                    'Check for drugs that prolong QT interval',
                    'Consider medication adjustments if appropriate'
                ],
                'priority': 'high'
            })
        
        # Exercise recommendations
        if 60 <= hr <= 100 and not metrics.get('arrhythmias'):
            recommendations.append({
                'category': 'Lifestyle',
                'title': 'Maintain Current Activity',
                'description': 'Your cardiovascular parameters are healthy.',
                'actions': [
                    'Continue regular exercise routine',
                    'Maintain healthy diet',
                    'Stay hydrated'
                ],
                'priority': 'low'
            })
        
        return recommendations
    
    def _calculate_simple_score(self, metrics: Dict) -> int:
        """Calculate simple health score for trend comparison"""
        score = 100
        
        hr = metrics.get('HR', 70)
        if not (60 <= hr <= 100):
            score -= 15
        
        qtc = metrics.get('QTc', 400)
        if qtc > 450:
            score -= 20
        
        if metrics.get('arrhythmias'):
            score -= 25
        
        return max(0, score)


# Convenience function
def generate_dashboard_insights(current_metrics: Dict,
                               historical_data: Optional[List[Dict]] = None,
                               patient_data: Optional[Dict] = None) -> Dict:
    """
    Generate all dashboard insights
    
    Returns:
        Dictionary with:
        - health_score: Dict
        - trend_insights: List[Dict]
        - smart_alerts: List[Dict]
        - recommendations: List[Dict]
    """
    engine = DashboardInsightsEngine()
    
    return {
        'health_score': engine.generate_health_score(current_metrics, historical_data),
        'trend_insights': engine.generate_trend_insights(current_metrics, historical_data or []),
        'smart_alerts': engine.generate_smart_alerts(current_metrics, historical_data),
        'recommendations': engine.generate_personalized_recommendations(
            current_metrics, patient_data, historical_data
        )
    }


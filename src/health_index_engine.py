import numpy as np

class HealthIndexEngine:
    """
    Computes Motor Health Index (HI 0-100%), degradation state, and projected RUL/maintenance alert.
    """
    
    BASE_HEALTH_SCORES = {
        'Normal_1': 100.0,
        'Ball_007_1': 78.0,
        'IR_007_1': 72.0,
        'OR_007_6_1': 70.0,
        'Ball_014_1': 52.0,
        'IR_014_1': 45.0,
        'OR_014_6_1': 42.0,
        'Ball_021_1': 22.0,
        'IR_021_1': 15.0,
        'OR_021_6_1': 10.0
    }
    
    FAULT_METADATA = {
        'Normal_1': {'type': 'Healthy Motor', 'location': 'None', 'severity': '0.000" (None)', 'status': 'OPTIMAL'},
        'Ball_007_1': {'type': 'Ball Defect', 'location': 'Rolling Element', 'severity': '0.007" (Minor)', 'status': 'WARNING'},
        'Ball_014_1': {'type': 'Ball Defect', 'location': 'Rolling Element', 'severity': '0.014" (Moderate)', 'status': 'ALERT'},
        'Ball_021_1': {'type': 'Ball Defect', 'location': 'Rolling Element', 'severity': '0.021" (Severe)', 'status': 'CRITICAL'},
        'IR_007_1': {'type': 'Inner Race Defect', 'location': 'Inner Raceway', 'severity': '0.007" (Minor)', 'status': 'WARNING'},
        'IR_014_1': {'type': 'Inner Race Defect', 'location': 'Inner Raceway', 'severity': '0.014" (Moderate)', 'status': 'ALERT'},
        'IR_021_1': {'type': 'Inner Race Defect', 'location': 'Inner Raceway', 'severity': '0.021" (Severe)', 'status': 'CRITICAL'},
        'OR_007_6_1': {'type': 'Outer Race Defect', 'location': "Outer Raceway @ 6 o'clock", 'severity': '0.007" (Minor)', 'status': 'WARNING'},
        'OR_014_6_1': {'type': 'Outer Race Defect', 'location': "Outer Raceway @ 6 o'clock", 'severity': '0.014" (Moderate)', 'status': 'ALERT'},
        'OR_021_6_1': {'type': 'Outer Race Defect', 'location': "Outer Raceway @ 6 o'clock", 'severity': '0.021" (Severe)', 'status': 'CRITICAL'}
    }

    @classmethod
    def calculate_health_index(cls, predicted_class: str, class_probabilities: dict, rms: float = None) -> dict:
        weighted_hi = 0.0
        total_prob = 0.0
        for c, prob in class_probabilities.items():
            score = cls.BASE_HEALTH_SCORES.get(c, 50.0)
            weighted_hi += prob * score
            total_prob += prob
            
        if total_prob > 0:
            hi = weighted_hi / total_prob
        else:
            hi = cls.BASE_HEALTH_SCORES.get(predicted_class, 50.0)
            
        if rms is not None and rms > 0.15:
            rms_penalty = min(20.0, (rms - 0.15) * 15.0)
            hi = max(5.0, hi - rms_penalty)
            
        hi = round(float(np.clip(hi, 0.0, 100.0)), 2)
        
        if hi >= 85.0:
            status = 'NORMAL (Healthy)'
            alert_level = 'LOW'
            color = '#22c55e'
            action = 'Standard routine maintenance interval.'
        elif hi >= 65.0:
            status = 'EARLY DEGRADATION (Warning)'
            alert_level = 'MODERATE'
            color = '#eab308'
            action = 'Schedule non-urgent bearing inspection during next planned downtime.'
        elif hi >= 35.0:
            status = 'MODERATE FAULT (Alert)'
            alert_level = 'HIGH'
            color = '#f97316'
            action = 'Immediate lubrication check and prepare replacement bearing.'
        else:
            status = 'CRITICAL FAILURE RISK'
            alert_level = 'CRITICAL'
            color = '#ef4444'
            action = 'Urgent shutdown recommended. Risk of catastrophic motor stall or stator rub.'
            
        meta = cls.FAULT_METADATA.get(predicted_class, {
            'type': predicted_class, 'location': 'Unknown', 'severity': 'Unknown', 'status': 'UNKNOWN'
        })
        
        margin_pct = max(0.0, hi - 20.0)
        est_cycles_remaining = int(margin_pct * 50)
        
        return {
            'health_index': hi,
            'status': status,
            'alert_level': alert_level,
            'color': color,
            'recommended_action': action,
            'fault_type': meta['type'],
            'fault_location': meta['location'],
            'fault_severity': meta['severity'],
            'est_cycles_remaining': est_cycles_remaining
        }

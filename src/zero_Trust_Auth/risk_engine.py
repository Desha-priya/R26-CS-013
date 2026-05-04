# risk_engine.py
# THE NOVELTY - Context-aware adaptive risk thresholding
# This class is imported by main.py and used for every scoring decision

import numpy as np
import joblib
import os
from datetime import datetime
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)

BASE_DIR =Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models" / "zero_trust_auth" 
 
class RiskEngine:
    """
    Combines Isolation Forest + One-Class SVM scores into a single
    risk score (0.0 to 1.0), then applies adaptive thresholds based on:
      1. Session context  (what the user is currently doing)
      2. Cross-layer alerts (threats detected by other NeuraShield layers)

    This is the KEY NOVELTY of the Zero-Trust Auth layer.
    """

    # ** Context modes and their base thresholds **********─
    # Lower threshold = stricter = easier to trigger step-up auth
    # Higher threshold = more relaxed = harder to trigger

    CONTEXT_THRESHOLDS = {
        'normal_browsing':  0.65,   # relaxed - user just browsing
        'sensitive_access': 0.40,   # strict  - user accessing sensitive files/admin
        'financial':        0.35,   # very strict - financial operations
        'under_attack':     0.25,   # maximum sensitivity - other layers raised alarm
    }
#--------------------------------------****-------------
    # How much each cross-layer alert lowers the threshold
    # More severe alert = bigger reduction
    ALERT_REDUCTIONS = {
        'network_guardian':          0.10,   # intrusion detected on network
        'ransomware_killer':         0.15,   # ransomware behaviour detected
        'content_threat_detection':  0.08,   # deepfake/phishing detected
    }

    def __init__(self):
        logging.info("Loading models into Risk Engine...")
        
        self.scaler    = self._safe_load("scaler.pkl")
        self.iso_forest = self._safe_load("isolation_forest.pkl")
        self.oc_svm     = self._safe_load("oneclass_svm.pkl")
        self.user_profiles = self._safe_load("user_profiles.pkl")

        # Active alerts from other layers - stored in memory
        # { 'network_guardian': timestamp, ... }
        self.active_alerts = {}

        # Alert expiry - alerts older than this are ignored (seconds)
        self.alert_expiry_seconds = 300   # 5 minutes

        print(f"Risk Engine ready. {len(self.user_profiles)} user profiles loaded.")

    # Safe loading with error handling
    def _safe_load(self, filename):
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file missing: {path}")
        return joblib.load(path)
    

    # ** Public method 1: Score a live session ************─
    def score_session(self, live_features: list, user_id: int,
                      context: str = 'normal_browsing') -> dict:
        """
        Main scoring function. Call this every 30 seconds during a live session.

        live_features : list of 48 feature values in same order as training
        user_id       : which user is currently logged in
        context       : current activity context (see CONTEXT_THRESHOLDS)

        Returns a dict with everything the UI needs to display.
        """

        # Step 1: Scale the live features using the SAME scaler from training
        # Without this step the model scores would be meaningless
        features_array = np.array(live_features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)

        # Step 2: Isolation Forest score
        # score_samples returns a raw score - more negative = more anomalous
        # We normalise it to 0-1 where 1 = most anomalous
        if_raw   = float(self.iso_forest.score_samples(features_scaled)[0])
        if_score = self._normalise_if_score(if_raw)

        # Step 3: One-Class SVM score
        # Only run SVM if IF already suspects something (score > 0.3)
        # This is the layered approach - saves computation, mirrors the architecture
        svm_score = 0.0
        svm_used  = False
        if if_score > 0.3:
            svm_raw   = float(self.oc_svm.score_samples(features_scaled)[0])
            svm_score = self._normalise_svm_score(svm_raw)
            svm_used  = True

        # Step 4: Combine both scores
        # IF catches big anomalies (weight 60%), SVM catches subtle ones (weight 40%)
        # If SVM wasn't used, IF carries full weight
        if svm_used:
            combined_score = (0.60 * if_score) + (0.40 * svm_score)
        else:
            combined_score = if_score

        # Step 5: Compare against user's own enrolled profile
        # If we have their personal baseline, factor in personal deviation too
        personal_deviation = self._personal_deviation(
            features_scaled[0], user_id
        )

        # Blend: 70% model score + 30% personal deviation
        final_score = (0.70 * combined_score) + (0.30 * personal_deviation)
        final_score = float(np.clip(final_score, 0.0, 1.0))

#*******************************************************************--------------------*******************
#*****************************************************************
        # Step 6: Get adaptive threshold
        # This is where the NOVELTY happens
        threshold = self._get_adaptive_threshold(context)

        # Step 7: Make decision
        decision = self._make_decision(final_score, threshold)

        return {
            'user_id':            user_id,
            'timestamp':          datetime.now().isoformat(),
            'context':            context,
            'if_score':           round(if_score, 4),
            'svm_score':          round(svm_score, 4) if svm_used else None,
            'svm_used':           svm_used,
            'combined_score':     round(combined_score, 4),
            'personal_deviation': round(personal_deviation, 4),
            'final_risk_score':   round(final_score, 4),
            'adaptive_threshold': round(threshold, 4),
            'active_alerts':      list(self._get_active_alerts().keys()),
            'decision':           decision,
            # UI display helpers
            'risk_level':         self._risk_level(final_score),
            'risk_percent':       int(final_score * 100),
        }

    # ** Public method 2: Receive alert from another layer ─
    def receive_alert(self, source: str, severity: str = 'medium') -> dict:
        """
        Called when another NeuraShield layer sends a threat alert.
        source   : 'network_guardian' | 'ransomware_killer' | 'content_threat_detection'
        severity : 'low' | 'medium' | 'high'
        """
        if source not in self.ALERT_REDUCTIONS:
            return {'status': 'error', 'message': f'Unknown source: {source}'}

        self.active_alerts[source] = {
            'timestamp': datetime.now(),
            'severity':  severity,
        }

        print(f"[ALERT] Received from {source} - severity: {severity}")
        print(f"[ALERT] Threshold now tightened. Active alerts: {list(self.active_alerts.keys())}")

        return {
            'status':   'alert_received',
            'source':   source,
            'severity': severity,
            'effect':   f'Threshold reduced by {self.ALERT_REDUCTIONS[source]:.2f}',
        }

    # ** Public method 3: Clear expired alerts ************─
    def clear_expired_alerts(self):
        now = datetime.now()
        expired = [
            src for src, data in self.active_alerts.items()
            if (now - data['timestamp']).seconds > self.alert_expiry_seconds
        ]
        for src in expired:
            del self.active_alerts[src]
            print(f"[ALERT] Expired and cleared: {src}")

    # ** Private helpers **********************************─

    def _get_adaptive_threshold(self, context: str) -> float:
        """
        NOVELTY CORE: Calculate the threshold dynamically.
        Base threshold from context, then reduce for each active alert.
        """
        # Start with context-based threshold
        base = self.CONTEXT_THRESHOLDS.get(context, 0.65)

        # Clear any expired alerts first
        self.clear_expired_alerts()

        # Reduce threshold for each active alert from other layers
        alert_reduction = sum(
            self.ALERT_REDUCTIONS.get(src, 0.0)
            for src in self._get_active_alerts()
        )

        # Threshold can never go below 0.15 (would flag everyone)
        adaptive = max(0.15, base - alert_reduction)

        return adaptive

    def _get_active_alerts(self) -> dict:
        """Return only non-expired alerts."""
        now = datetime.now()
        return {
            src: data for src, data in self.active_alerts.items()
            if (now - data['timestamp']).seconds <= self.alert_expiry_seconds
        }

    def _personal_deviation(self, scaled_features: np.ndarray,
                             user_id: int) -> float:
        """
        Compare live features to THIS user's enrolled profile.
        Returns 0.0 (identical to profile) to 1.0 (very different).
        """
        if user_id not in self.user_profiles:
            return 0.8   # unknown user - high suspicion 0.5 modarete)

        enrolled = np.array(self.user_profiles[user_id]['features'])
        # Euclidean distance between live and enrolled feature vectors
        distance = float(np.linalg.norm(scaled_features - enrolled))

        # dynamic distance Normalise
        max_possible_distance = len(scaled_features) * 3.0   # safe upper bound
        normalised = np.clip(distance / max_possible_distance, 0.0, 1.0)
        
        return float(normalised)

    def _normalise_if_score(self, raw_score: float) -> float:
        """
        Convert IF raw score (-0.662 to -0.383 in our data) to 0-1 anomaly scale.
        0 = most normal, 1 = most anomalous.
        We use the actual min/max from training for accurate normalisation.
        """
        # From training output: range was -0.662 to -0.383
        min_score = -0.70
        max_score = -0.35
        # Invert: lower raw score = higher anomaly score
        normalised = (max_score - raw_score) / (max_score - min_score)
        return float(np.clip(normalised, 0.0, 1.0))

    def _normalise_svm_score(self, raw_score: float) -> float:
        """
        Convert SVM raw score (0.536 to 0.795 in our data) to 0-1 anomaly scale.
        0 = most normal, 1 = most anomalous.
        """
        # From training output: range was 0.536 to 0.795
        min_score = 0.50
        max_score = 0.80
        # Invert: lower SVM score = higher anomaly
        normalised = (raw_score - max_score) / (min_score - max_score)
        return float(np.clip(normalised, 0.0, 1.0))

    def _make_decision(self, score: float, threshold: float) -> str:
        """Three-level decision based on score vs adaptive threshold."""
        if score < threshold * 0.6:
            return 'allow'            # comfortably below threshold
        elif score < threshold:
            return 'warn'             # approaching threshold - log it
        else:
            return 'step_up_auth'     # exceeded threshold - challenge user

    def _risk_level(self, score: float) -> str:
        if score < 0.35:   return 'low'
        elif score < 0.60: return 'medium'
        else:              return 'high'


# ** Quick test when run directly ************************─
if __name__ == "__main__":
    engine = RiskEngine()

    print("\n--- TEST 1: Normal session, normal browsing ---")
    # Simulate user 1's normal features (load from their profile)
    user1_features = engine.user_profiles[1]['raw_features']
    result = engine.score_session(user1_features, user_id=1,
                                  context='normal_browsing')
    print(f"Risk score : {result['risk_percent']}%")
    print(f"Risk level : {result['risk_level']}")
    print(f"Decision   : {result['decision']}")
    print(f"Threshold  : {result['adaptive_threshold']}")

    print("\n--- TEST 2: Same user, sensitive file access ---")
    result2 = engine.score_session(user1_features, user_id=1,
                                   context='sensitive_access')
    print(f"Risk score : {result2['risk_percent']}%")
    print(f"Threshold  : {result2['adaptive_threshold']}  ← stricter now")
    print(f"Decision   : {result2['decision']}")

    print("\n--- TEST 3: Alert from Network Guardian arrives ---")
    engine.receive_alert('network_guardian', severity='high')
    result3 = engine.score_session(user1_features, user_id=1,
                                   context='sensitive_access')
    print(f"Risk score : {result3['risk_percent']}%")
    print(f"Threshold  : {result3['adaptive_threshold']}  ← even stricter after alert")
    print(f"Active alerts: {result3['active_alerts']}")
    print(f"Decision   : {result3['decision']}")

    print("\n--- TEST 4: Simulated attacker (random features) ---")
    fake_features = list(np.random.uniform(0, 5, 48))
    result4 = engine.score_session(fake_features, user_id=1,
                                   context='normal_browsing')
    print(f"Risk score : {result4['risk_percent']}%")
    print(f"Risk level : {result4['risk_level']}")
    print(f"Decision   : {result4['decision']}")
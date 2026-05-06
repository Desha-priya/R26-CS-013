# risk_engine.py - IMPROVED more robust, accurate, and personalised risk engine 

# Changes from v1:
#   - Loads feature_weights.pkl and applies them to live features
#   - Sliding window: keeps last 3 scores and averages (reduces false positives)
#   - Per-user personal baseline comparison (tighter personal threshold)
#   - SVM now triggers at IF > 0.1 (much earlier) : because SVM is stronger in v2
#   - Combined score weights flipped: SVM 70%, IF 30% (SVM more reliable in v2)
#   - Time-of-day risk factor (login at 3am = stricter threshold)
#--------------------------------------------------------------------------------

import numpy as np
import joblib
import os
from pathlib import Path
from datetime import datetime
from collections import deque

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "zero_trust_auth"
OUTPUT_FILE = ROOT_DIR / "output" / "risk_engine_result.txt"

def log_to_file(text):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "a") as f:
        f.write(text + "\n")


class RiskEngine:

    CONTEXT_THRESHOLDS = {
        'normal_browsing':  0.65,
        'sensitive_access': 0.40,
        'financial':        0.35,
        'under_attack':     0.25,
    }

    ALERT_REDUCTIONS = {
        'network_guardian':         0.10,
        'ransomware_killer':        0.15,
        'content_threat_detection': 0.08,
    }

    # Alert severity also multiplies the risk score directly
    # This is correct behaviour : if network is compromised,
    # even a slightly suspicious user should be flagged harder
    ALERT_SCORE_BOOST = {
        'network_guardian':         0.08,
        'ransomware_killer':        0.12,
        'content_threat_detection': 0.06,
    }

    IF_SCORE_MIN  = -0.70
    IF_SCORE_MAX  = -0.38
    SVM_SCORE_MIN =  0.00
    SVM_SCORE_MAX =  0.42

    

    def __init__(self):
        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading models into Risk Engine...")

        self.scaler        = joblib.load(MODELS_DIR / "scaler_v2.pkl")
        self.iso_forest    = joblib.load(MODELS_DIR / "isolation_forest_v2.pkl")
        self.oc_svm        = joblib.load(MODELS_DIR / "oneclass_svm_v2.pkl")
        self.user_profiles = joblib.load(MODELS_DIR / "user_profiles_v2.pkl")

        weights_path = MODELS_DIR / "feature_weights_v2.pkl"
        self.feature_weights = (
            joblib.load(weights_path)
            if os.path.exists(weights_path) else None
        )

        self.active_alerts        = {}
        self.alert_expiry_seconds = 300
        self.score_history        = {}

        
        self.feature_names = list(self.user_profiles[1]['feature_names'])

        self.logger.info(f"Risk Engine ready. {len(self.user_profiles)} user profiles loaded.")

    #******************************************************
    def score_session(self, live_features: dict, user_id: int,
                      context: str = 'normal_browsing',
                      use_window: bool = True) -> dict:
        """
        use_window=True  → live monitoring (sliding window smoothing ON)
        use_window=False → single replay demo (no history, fresh score each time)
        """

        # Step 1: Build feature array from dict (STRICT mode)

        if not isinstance(live_features, dict):
            raise TypeError("live_features must be a dict with feature names")

        missing = [f for f in self.feature_names if f not in live_features]

        if missing:
            raise ValueError(f"Missing features: {missing[:5]}...")

        features_array = np.array([
            live_features[f] for f in self.feature_names
        ], dtype=float)

        if self.feature_weights is not None:
            fw             = np.array(self.feature_weights)
            min_len        = min(len(features_array), len(fw))
            features_array = features_array[:min_len] * fw[:min_len]

            if len(features_array) < len(fw):
                features_array = np.pad(
                    features_array, (0, len(fw) - len(features_array))
                )

        # Step 2: Scale
        features_scaled = self.scaler.transform(features_array.reshape(1, -1))

        # Step 3: Isolation Forest score
        if_raw   = float(self.iso_forest.score_samples(features_scaled)[0])
        if_score = self._normalise_if_score(if_raw)
        if_score = if_score ** 1.5

        # Step 4: SVM : triggers at IF > 0.10
        svm_score = 0.0
        svm_used  = False
        svm_raw_val = None
        if if_score > 0.10:
            svm_raw_val = float(self.oc_svm.score_samples(features_scaled)[0])
            svm_score   = self._normalise_svm_score(svm_raw_val)
            svm_score   = svm_score ** 1.5
            svm_used    = True

        # Step 5: Combine : SVM 70% + IF 30%
        if svm_used:
            combined_score = (0.30 * if_score) + (0.70 * svm_score)
        else:
            combined_score = if_score

        # Step 6: Personal deviation from enrolled profile
        personal_deviation = self._personal_deviation(features_scaled[0], user_id)

        # Step 7: Blend model score + personal deviation
        instant_score = (0.75 * combined_score) + (0.25 * personal_deviation)

        #** NEW: Step 7b : Alert score boost**************
        # Active alerts from other layers add directly to the risk score
        # This means: if network is compromised, even a borderline user
        # gets pushed over the threshold : which is correct security behaviour
        active = self._get_active_alerts()
        alert_boost = sum(
            self.ALERT_SCORE_BOOST.get(src, 0.0)
            for src in active
        )
        instant_score = instant_score + alert_boost
        instant_score = float(np.clip(instant_score, 0.0, 1.0))

        # Step 8: Sliding window : ONLY for live sessions
        # For replay demo, skip window so each replay is independent
        if use_window:
            if user_id not in self.score_history:
                self.score_history[user_id] = deque(maxlen=3)
            history = self.score_history[user_id]
            prev    = history[-1] if len(history) > 0 else instant_score
            # Retain spikes, decay old risk slowly
            final_score = max(instant_score, prev * 0.85)
            # Context baseline floor
            baseline = {
                'normal_browsing':  0.03,
                'sensitive_access': 0.06,
                'financial':        0.08,
                'under_attack':     0.10,
            }
            final_score = max(final_score, baseline.get(context, 0.05))
            history.append(final_score)
        else:
            # Replay mode : use instant score directly, no history influence
            final_score = instant_score

        final_score = float(np.clip(final_score, 0.0, 1.0))

        # Step 9: Adaptive threshold (alerts lower this)
        threshold = self._get_adaptive_threshold(context)

        # Step 10: Decision
        decision = self._make_decision(final_score, threshold)

        return {
            'user_id':            user_id,
            'timestamp':          datetime.now().isoformat(),
            'context':            context,
            'if_raw':             round(if_raw, 4),
            'if_score':           round(if_score, 4),
            'svm_raw':            round(svm_raw_val, 4) if svm_used else None,
            'svm_score':          round(svm_score, 4) if svm_used else None,
            'svm_used':           svm_used,
            'combined_score':     round(combined_score, 4),
            'personal_deviation': round(personal_deviation, 4),
            'alert_boost':        round(alert_boost, 4),
            'instant_score':      round(instant_score, 4),
            'final_risk_score':   round(final_score, 4),
            'adaptive_threshold': round(threshold, 4),
            'active_alerts':      list(active.keys()),
            'decision':           decision,
            'risk_level':         self._risk_level(final_score),
            'risk_percent':       int(final_score * 100),
            'window_size':        len(self.score_history.get(user_id, [])),
            'mode':               'live' if use_window else 'replay',
        }

    #******************************************************
    def receive_alert(self, source: str, severity: str = 'medium') -> dict:
        if source not in self.ALERT_REDUCTIONS:
            return {'status': 'error', 'message': f'Unknown source: {source}'}
        self.active_alerts[source] = {
            'timestamp': datetime.now(),
            'severity':  severity,
        }
        self.logger.info(f"[ALERT] {source} : severity: {severity}")
        return {
            'status':        'alert_received',
            'source':        source,
            'severity':      severity,
            'threshold_drop': self.ALERT_REDUCTIONS[source],
            'score_boost':   self.ALERT_SCORE_BOOST[source],
            'effect':        f'Threshold -{self.ALERT_REDUCTIONS[source]:.2f} | Score +{self.ALERT_SCORE_BOOST[source]:.2f}',
        }

    def clear_expired_alerts(self):
        now     = datetime.now()
        expired = [
            src for src, data in self.active_alerts.items()
            if (now - data['timestamp']).seconds > self.alert_expiry_seconds
        ]
        for src in expired:
            del self.active_alerts[src]

    def reset_user_window(self, user_id: int):
        if user_id in self.score_history:
            del self.score_history[user_id]

    #******************************************************
    def _get_adaptive_threshold(self, context: str) -> float:
        base            = self.CONTEXT_THRESHOLDS.get(context, 0.65)
        self.clear_expired_alerts()
        alert_reduction = sum(
            self.ALERT_REDUCTIONS.get(src, 0.0)
            for src in self._get_active_alerts()
        )
        hour           = datetime.now().hour
        time_reduction = 0.08 if (hour < 7 or hour > 22) else 0.0
        return max(0.15, base - alert_reduction - time_reduction)

    def _get_active_alerts(self) -> dict:
        now = datetime.now()
        return {
            src: data for src, data in self.active_alerts.items()
            if (now - data['timestamp']).seconds <= self.alert_expiry_seconds
        }

    def _personal_deviation(self, scaled_features: np.ndarray,
                             user_id: int) -> float:
        if user_id not in self.user_profiles:
            return 0.5
        enrolled = np.array(self.user_profiles[user_id]['features'])
        min_len  = min(len(scaled_features), len(enrolled))
        distance = float(np.linalg.norm(scaled_features[:min_len] - enrolled[:min_len]))
        return float(np.clip(distance / 15.0, 0.0, 1.0))

    def _normalise_if_score(self, raw: float) -> float:
        raw   = np.clip(raw, self.IF_SCORE_MIN, self.IF_SCORE_MAX)
        score = (self.IF_SCORE_MAX - raw) / (self.IF_SCORE_MAX - self.IF_SCORE_MIN)
        return float(np.clip(score, 0.0, 1.0))

    def _normalise_svm_score(self, raw: float) -> float:
        raw   = np.clip(raw, self.SVM_SCORE_MIN, self.SVM_SCORE_MAX)
        score = (self.SVM_SCORE_MAX - raw) / (self.SVM_SCORE_MAX - self.SVM_SCORE_MIN)
        return float(np.clip(score, 0.0, 1.0))

    def _make_decision(self, score: float, threshold: float) -> str:
        if score < threshold * 0.6:  return 'allow'
        elif score < threshold:       return 'warn'
        else:                         return 'step_up_auth'

    def _risk_level(self, score: float) -> str:
        if score < 0.35:   return 'low'
        elif score < 0.60: return 'medium'
        else:              return 'high'


#** TEST**************************************************
if __name__ == "__main__":
    engine = RiskEngine()

    feature_names = engine.feature_names

    def to_feature_dict(feature_list):
        return dict(zip(feature_names, feature_list))

    # Clear previous log
    open(OUTPUT_FILE, "w").close()

    print("Running Risk Engine tests...")

    #****************************************************
    print("Test 1 running...")
    log_to_file("\n--- TEST 1: Normal user, no alerts ---")

    user1_list = engine.user_profiles[1]['raw_features']
    user1 = to_feature_dict(user1_list)

    r = engine.score_session(user1, 1, 'normal_browsing', use_window=False)
    log_to_file(f"Risk: {r['risk_percent']}% | Decision: {r['decision']} | Threshold: {r['adaptive_threshold']}")

    #****************************************************
    print("Test 2 running...")
    log_to_file("\n--- TEST 2: All 3 alerts active, normal user ---")

    engine.receive_alert('network_guardian', 'high')
    engine.receive_alert('ransomware_killer', 'high')
    engine.receive_alert('content_threat_detection', 'high')

    r = engine.score_session(user1, 1, 'normal_browsing', use_window=False)
    log_to_file(f"Risk: {r['risk_percent']}% | Alert boost: {r['alert_boost']} | Threshold: {r['adaptive_threshold']} | Decision: {r['decision']}")

    #****************************************************
    print("Test 3 running...")
    log_to_file("\n--- TEST 3: All alerts, anomaly user, sensitive context ---")

    anomaly_uid = None
    for uid, data in engine.user_profiles.items():
        features_scaled = np.array(data['features']).reshape(1, -1)
        if engine.iso_forest.predict(features_scaled)[0] == -1:
            anomaly_uid = uid
            break

    if anomaly_uid:
        anomaly_list = engine.user_profiles[anomaly_uid]['raw_features']
        anomaly_features = to_feature_dict(anomaly_list)

        r = engine.score_session(anomaly_features, anomaly_uid, 'sensitive_access', use_window=False)
        log_to_file(f"Anomaly user {anomaly_uid} | Risk: {r['risk_percent']}% | Decision: {r['decision']} | Threshold: {r['adaptive_threshold']}")

    #****************************************************
    print("Test 4 running...")
    log_to_file("\n--- TEST 4: Simulated attacker, all alerts ---")

    np.random.seed(42)
    fake_list = list(np.random.uniform(0, 5, len(feature_names)))
    fake_features = to_feature_dict(fake_list)

    r = engine.score_session(fake_features, 1, 'normal_browsing', use_window=False)
    log_to_file(f"Risk: {r['risk_percent']}% | Decision: {r['decision']}")

    #****************************************************
    print("Test 5 running...")
    log_to_file("\n--- TEST 5: Live mode window test (3 consecutive calls) ---")

    engine.active_alerts = {}

    user_live = to_feature_dict(user1_list)

    for i in range(3):
        r = engine.score_session(user_live, 999, 'normal_browsing', use_window=True)
        log_to_file(f"Call {i+1}: Risk: {r['risk_percent']}% | Window size: {r['window_size']}")

    #****************************************************
    print("Final sanity check...")
    log_to_file("\n--- SANITY CHECK ---")
    log_to_file(f"Total features expected: {len(feature_names)}")
    log_to_file(f"User feature count: {len(user1)}")

    print("Done. Results saved to output/risk_engine_result.txt")
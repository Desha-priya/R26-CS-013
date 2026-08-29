

import numpy as np
import joblib
import os
from datetime import datetime
from collections import deque
import pathlib

# Define root path for the zero trust module
ROOT_PATH = pathlib.Path(__file__).parent.parent.parent

MODELS_DIR = ROOT_PATH / "models" / "zero_trust_auth"

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

    # Dual effect: alerts lower threshold AND boost risk score
    ALERT_SCORE_BOOST = {
        'network_guardian':         0.10,
        'ransomware_killer':        0.15,
        'content_threat_detection': 0.08,
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
        self.alert_expiry_seconds = 600   # 10 min — long enough for demo
        self.score_history        = {}

        self.logger.info(f"Risk Engine ready. {len(self.user_profiles)} user profiles loaded.")

    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--
    def score_session(self, live_features: list, user_id: int,
                      context: str = 'normal_browsing',
                      use_window: bool = True) -> dict:

        # -*- LIVE USER PATH (id >= 9000) -*--*--*--*--*--*--*--*--*--*--*--
        # Browser-captured features are in ms/seconds from JS.
        # They do NOT match the BB-MAS training distribution.
        # Running them through IF/SVM gives meaningless scores.
        # Instead: compare ONLY to their own enrolled profile.
        # This is the correct zero-trust approach for live sessions.
        if user_id >= 9000:
            return self._score_live_user(live_features, user_id, context, use_window)

        # -*- DATASET USER PATH (id 1-116) -*--*--*--*--*--*--*--*--*--*--*-
        # Step 1: Feature weights
        features_array = np.array(live_features, dtype=float)
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

        # Step 3: Isolation Forest
        if_raw   = float(self.iso_forest.score_samples(features_scaled)[0])
        if_score = self._normalise_if_score(if_raw)
        if_score = if_score ** 1.5

        # Step 4: One-Class SVM (activates when IF > 0.10)
        svm_score   = 0.0
        svm_used    = False
        svm_raw_val = None
        if if_score > 0.10:
            svm_raw_val = float(self.oc_svm.score_samples(features_scaled)[0])
            svm_score   = self._normalise_svm_score(svm_raw_val)
            svm_score   = svm_score ** 1.5
            svm_used    = True

        # Step 5: Combine IF 30% + SVM 70%
        combined_score = ((0.30 * if_score) + (0.70 * svm_score)) if svm_used else if_score

        # Step 6: Personal deviation
        personal_deviation = self._personal_deviation_dataset(features_scaled[0], user_id)

        # Step 7: Blend
        instant_score = (0.75 * combined_score) + (0.25 * personal_deviation)
        instant_score = float(np.clip(instant_score, 0.0, 1.0))

        # Step 7b: Alert boost
        active_now  = self._get_active_alerts()
        alert_boost = sum(self.ALERT_SCORE_BOOST.get(src, 0.0) for src in active_now)
        instant_score = float(np.clip(instant_score + alert_boost, 0.0, 1.0))

        # Step 8: No sliding window for dataset replay
        final_score = instant_score
        final_score = float(np.clip(final_score, 0.0, 1.0))

        threshold = self._get_adaptive_threshold(context)
        decision  = self._make_decision(final_score, threshold)

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
            'active_alerts':      list(active_now.keys()),
            'decision':           decision,
            'risk_level':         self._risk_level(final_score),
            'risk_percent':       int(final_score * 100),
            'window_size':        0,
            'mode':               'replay',
        }

    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--
    def _score_live_user(self, live_features: list, user_id: int,
                         context: str, use_window: bool) -> dict:
        """
        Score a live enrolled user (id >= 9000).
        Uses personal deviation only — compares current typing
        to their OWN enrolled profile. No IF/SVM (wrong scale).

        Deviation score interpretation:
          0.0 - 0.2 = typing just like enrolled session → low risk
          0.2 - 0.5 = some difference → medium risk
          0.5 - 1.0 = very different → high risk (different person?)
        """
        if user_id not in self.user_profiles:
            # No enrolled profile — return medium risk by default
            return self._empty_live_result(user_id, context, 0.4)

        enrolled_raw = self.user_profiles[user_id].get('raw_features', [])
        if not enrolled_raw:
            return self._empty_live_result(user_id, context, 0.4)

        # Compare raw keystroke features (first 15) directly
        # Using raw values avoids scaler distortion for live browser data
        live    = np.array(live_features[:15],    dtype=float)
        enr     = np.array(enrolled_raw[:15],     dtype=float)
        min_len = min(len(live), len(enr))

        if min_len < 3:
            return self._empty_live_result(user_id, context, 0.3)

        live = live[:min_len]
        enr  = enr[:min_len]

        # Normalise each feature relative to enrolled value
        # This makes comparison scale-independent
        # Avoid division by zero for zero-valued features
        safe_enr = np.where(np.abs(enr) < 0.001, 0.001, enr)
        relative_diff = np.abs((live - enr) / safe_enr)

        # Cap outliers — single extreme feature shouldn't dominate
        relative_diff = np.clip(relative_diff, 0.0, 3.0)

        # Mean relative difference across all features
        # 0.0 = identical, 1.0 = 100% different on average, 3.0 = extreme
        mean_diff = float(np.mean(relative_diff))

        # Convert to 0-1 risk score
        # 0.0 diff → 0.0 risk, 0.5 diff → 0.5 risk, 1.0+ diff → 1.0 risk
        personal_deviation = float(np.clip(mean_diff / 1.5, 0.0, 1.0))

        # Alert boost still applies to live users
        active_now  = self._get_active_alerts()
        alert_boost = sum(self.ALERT_SCORE_BOOST.get(src, 0.0) for src in active_now)

        instant_score = float(np.clip(personal_deviation + alert_boost, 0.0, 1.0))

        # Sliding window for live sessions — smooths out typing bursts
        if use_window:
            if user_id not in self.score_history:
                self.score_history[user_id] = deque(maxlen=5)
            history = self.score_history[user_id]
            prev    = history[-1] if len(history) > 0 else instant_score
            # Decay previous score slowly, keep spikes
            final_score = max(instant_score, prev * 0.75)
            # Small baseline floor — gauge never shows 0%
            baseline = {
                'normal_browsing':  0.02,
                'sensitive_access': 0.04,
                'financial':        0.05,
                'under_attack':     0.07,  # NOT 0.7 — was a typo
            }
            final_score = max(final_score, baseline.get(context, 0.02))
            history.append(final_score)
        else:
            final_score = instant_score

        final_score = float(np.clip(final_score, 0.0, 1.0))
        threshold   = self._get_adaptive_threshold(context)
        decision    = self._make_decision(final_score, threshold)

        return {
            'user_id':            user_id,
            'timestamp':          datetime.now().isoformat(),
            'context':            context,
            'if_raw':             None,
            'if_score':           None,
            'svm_raw':            None,
            'svm_score':          None,
            'svm_used':           False,
            'combined_score':     None,
            'personal_deviation': round(personal_deviation, 4),
            'alert_boost':        round(alert_boost, 4),
            'instant_score':      round(instant_score, 4),
            'final_risk_score':   round(final_score, 4),
            'adaptive_threshold': round(threshold, 4),
            'active_alerts':      list(active_now.keys()),
            'decision':           decision,
            'risk_level':         self._risk_level(final_score),
            'risk_percent':       int(final_score * 100),
            'window_size':        len(self.score_history.get(user_id, [])),
            'mode':               'live',
        }

    def _empty_live_result(self, user_id, context, score):
        """Fallback result when live user has no enrolled profile."""
        threshold = self._get_adaptive_threshold(context)
        return {
            'user_id': user_id, 'timestamp': datetime.now().isoformat(),
            'context': context, 'if_raw': None, 'if_score': None,
            'svm_raw': None, 'svm_score': None, 'svm_used': False,
            'combined_score': None, 'personal_deviation': score,
            'alert_boost': 0.0, 'instant_score': score,
            'final_risk_score': score, 'adaptive_threshold': threshold,
            'active_alerts': [], 'decision': self._make_decision(score, threshold),
            'risk_level': self._risk_level(score), 'risk_percent': int(score*100),
            'window_size': 0, 'mode': 'live',
        }

    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--
    def receive_alert(self, source: str, severity: str = 'medium') -> dict:
        if source not in self.ALERT_REDUCTIONS:
            return {'status': 'error', 'message': f'Unknown source: {source}'}
        self.active_alerts[source] = {'timestamp': datetime.now(), 'severity': severity}
        self.logger.info(f"[ALERT] {source} — severity: {severity}")
        return {
            'status':         'alert_received',
            'source':         source,
            'severity':       severity,
            'threshold_drop': self.ALERT_REDUCTIONS[source],
            'score_boost':    self.ALERT_SCORE_BOOST[source],
            'effect':         f'Threshold -{self.ALERT_REDUCTIONS[source]:.2f} | Score +{self.ALERT_SCORE_BOOST[source]:.2f}',
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

    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--
    def _get_adaptive_threshold(self, context: str) -> float:
        base            = self.CONTEXT_THRESHOLDS.get(context, 0.65)
        self.clear_expired_alerts()
        alert_reduction = sum(self.ALERT_REDUCTIONS.get(src, 0.0)
                              for src in self._get_active_alerts())
        hour            = datetime.now().hour
        time_reduction  = 0.08 if (hour < 7 or hour > 22) else 0.0
        return max(0.15, base - alert_reduction - time_reduction)

    def _get_active_alerts(self) -> dict:
        now = datetime.now()
        return {
            src: data for src, data in self.active_alerts.items()
            if (now - data['timestamp']).seconds <= self.alert_expiry_seconds
        }

    def _personal_deviation_dataset(self, scaled_features, user_id: int) -> float:
        """Personal deviation for dataset users — uses scaled feature space."""
        if user_id not in self.user_profiles:
            return 0.5
        enrolled    = np.array(self.user_profiles[user_id]['features'])
        compare_len = min(len(scaled_features), len(enrolled))
        distance    = float(np.linalg.norm(
            scaled_features[:compare_len] - enrolled[:compare_len]
        ))
        return float(np.clip(distance / 15.0, 0.0, 1.0))

    def _normalise_if_score(self, raw: float) -> float:
        raw = np.clip(raw, self.IF_SCORE_MIN, self.IF_SCORE_MAX)
        return float(np.clip(
            (self.IF_SCORE_MAX - raw) / (self.IF_SCORE_MAX - self.IF_SCORE_MIN),
            0.0, 1.0
        ))

    def _normalise_svm_score(self, raw: float) -> float:
        raw = np.clip(raw, self.SVM_SCORE_MIN, self.SVM_SCORE_MAX)
        return float(np.clip(
            (self.SVM_SCORE_MAX - raw) / (self.SVM_SCORE_MAX - self.SVM_SCORE_MIN),
            0.0, 1.0
        ))

    def _make_decision(self, score: float, threshold: float) -> str:
        if score < threshold * 0.6:  return 'allow'
        elif score < threshold:       return 'warn'
        else:                         return 'step_up_auth'

    def _risk_level(self, score: float) -> str:
        if score < 0.35:   return 'low'
        elif score < 0.60: return 'medium'
        else:              return 'high'


# -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--
if __name__ == "__main__":
    engine = RiskEngine()
    user1  = engine.user_profiles[1]['raw_features']
#'''
    print("\n--- TEST 1: Normal user, no alerts ---")
    r = engine.score_session(user1, 1, 'normal_browsing', use_window=False)
    print(f"Risk: {r['risk_percent']}% | Decision: {r['decision']} | Threshold: {r['adaptive_threshold']}")

    print("\n--- TEST 2: All alerts ---")
    engine.receive_alert('network_guardian', 'high')
    engine.receive_alert('ransomware_killer', 'high')
    engine.receive_alert('content_threat_detection', 'high')
    r = engine.score_session(user1, 1, 'normal_browsing', use_window=False)
    print(f"Risk: {r['risk_percent']}% | Boost: {r['alert_boost']} | Threshold: {r['adaptive_threshold']} | Decision: {r['decision']}")

    print("\n--- TEST 3: Anomaly user + all alerts + sensitive ---")
    anomaly_uid = next(
        uid for uid, data in engine.user_profiles.items()
        if engine.iso_forest.predict(np.array(data['features']).reshape(1,-1))[0] == -1
    )
    r = engine.score_session(engine.user_profiles[anomaly_uid]['raw_features'],
                             anomaly_uid, 'sensitive_access', use_window=False)
    print(f"User {anomaly_uid} | Risk: {r['risk_percent']}% | Decision: {r['decision']}")

    print("\n--- TEST 4: Simulated attacker ---")
    np.random.seed(42)
    r = engine.score_session(list(np.random.uniform(0,5,48)), 1, 'normal_browsing', use_window=False)
    print(f"Risk: {r['risk_percent']}% | Decision: {r['decision']}")

    print("\n--- TEST 5: Live user simulation (same features = low risk) ---")
    engine.active_alerts = {}

    # Simulate enrollment

    sample_ks = [0.13, 0.05, 0.05, 0.3, 0.11, 0.43, 0.5, 0.0, 2.0, 0.18,
                 0.05, 0.01, 80, 0.38, 1.16]
    
    engine.user_profiles[9999] = {'raw_features': sample_ks + [0.0]*33}

    # Score same features = should be low risk
    r = engine.score_session(sample_ks, 9999, 'normal_browsing', use_window=True)
    print(f"Same typing: Risk: {r['risk_percent']}% | Decision: {r['decision']}")

    # Score very different features = should be high risk
    different = [x * 5 for x in sample_ks]
    r = engine.score_session(different, 9999, 'normal_browsing', use_window=False)

    print(f"Different typing: Risk: {r['risk_percent']}% | Decision: {r['decision']}") 

   # '''
"""
predict.py
-----------
Unified inference wrapper: combines Isolation Forest (fast anomaly
screen) with the Random Forest classifier (confidence + category) into
a single confidence score, matching the "hybrid detector" story in the
proposal (Stage 1 = anomaly screen, Stage 2 = confirmatory classifier).

Also attaches a SHAP-based explanation to every prediction (see
explain.py) - this is the project's explainability contribution: every
verdict comes with WHICH features drove it, not just a confidence
number.

Usage:
    from src.detection.predict import HybridDetector
    detector = HybridDetector()
    result = detector.predict(feature_dict)
    # result = {"is_threat": bool, "confidence": float, "stage1_anomaly": bool,
    #           "explanation": {...}, ...}
"""

import joblib
import numpy as np

from src.detection.model_integrity import verified_path
from src.utils import paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_model(relative_path: str):
    """Loads a model file ONLY after its fingerprint was checked (see
    model_integrity.py): a pickle can run code, so a changed file is refused."""
    return joblib.load(verified_path(paths.resource(relative_path)))


class HybridDetector:
    def __init__(self, config_path=None):
        self.cfg = paths.load_config(config_path)

        iso_cfg = self.cfg["isolation_forest"]
        rf_cfg = self.cfg["random_forest"]

        self.iso_model = _load_model(iso_cfg["model_path"])
        self.iso_scaler = _load_model(self.cfg["scaler_path"])
        self.iso_features = _load_model(self.cfg["scaler_path"].replace(".pkl", "_features.pkl"))

        self.rf_model = _load_model(rf_cfg["model_path"])
        self.rf_scaler = _load_model(self.cfg["scaler_path"].replace(".pkl", "_rf.pkl"))
        self.rf_features = _load_model(rf_cfg["model_path"].replace(".pkl", "_features.pkl"))

        self.alert_threshold = self.cfg["live_monitor"]["confidence_alert_threshold"]

        # SHAP explainer wraps the already-trained Random Forest (no retraining).
        # It is imported only here because the shap library is large and slow to
        # load. If it is not installed, predictions still work, just without the
        # "why" explanation.
        self.explainer = None
        try:
            from src.detection.explain import ShapExplainer
            self.explainer = ShapExplainer(self.rf_model, self.rf_features)
        except ImportError:
            logger.warning("[predict] The 'shap' library is not installed - predictions will "
                           "have no SHAP explanation.")

    def _row_from_dict(self, feature_dict: dict, feature_list: list) -> np.ndarray:
        # Missing features default to 0 - keeps this robust to feed from
        # either the CIC-MalMem-2022 schema or the live monitor's schema.
        return np.array([[feature_dict.get(f, 0.0) for f in feature_list]])

    def predict(self, feature_dict: dict, compute_explanation: bool = True) -> dict:
        # ---- Stage 1: Isolation Forest anomaly screen ----
        X_iso = self._row_from_dict(feature_dict, self.iso_features)
        X_iso_scaled = self.iso_scaler.transform(X_iso)
        anomaly_flag = self.iso_model.predict(X_iso_scaled)[0] == -1
        anomaly_raw_score = self.iso_model.decision_function(X_iso_scaled)[0]

        # ---- Stage 2: Random Forest confirmatory classification ----
        X_rf = self._row_from_dict(feature_dict, self.rf_features)
        X_rf_scaled = self.rf_scaler.transform(X_rf)
        rf_proba = self.rf_model.predict_proba(X_rf_scaled)[0][1]  # P(malware)

        # Combine: only escalate to "confirmed threat" if BOTH stages agree,
        # matching the two-stage design in the proposal (reduces false positives).
        is_threat = bool(anomaly_flag and rf_proba >= self.alert_threshold)

        result = {
            "is_threat": is_threat,
            "confidence": round(float(rf_proba), 4),
            "stage1_anomaly_detected": bool(anomaly_flag),
            "stage1_anomaly_score": round(float(anomaly_raw_score), 4),
            "stage2_malware_probability": round(float(rf_proba), 4),
        }

        # Only compute the SHAP explanation when it's actually needed
        # (confirmed threats, or on request) - SHAP is fast for tree
        # models but there's no reason to spend it on routine benign
        # traffic in a live monitoring loop. compute_explanation=False
        # also enables fast bulk evaluation (e.g. scoring an entire
        # dataset for accuracy/FPR) without paying SHAP's per-call cost
        # when only the verdict, not the explanation, is needed.
        if is_threat and compute_explanation and self.explainer is not None:
            result["explanation"] = self.explainer.explain(X_rf_scaled, top_n=3)

        return result


if __name__ == "__main__":
    # Quick smoke test using a dummy feature dict (all zeros -> should look benign)
    detector = HybridDetector()
    dummy = {f: 0.0 for f in detector.rf_features}
    print(detector.predict(dummy))

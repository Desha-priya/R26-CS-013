"""
explain.py
------------
Adds explainability to the Random Forest classifier using SHAP
(SHapley Additive exPlanations). This is the project's novelty
contribution: instead of just "malware, 97% confidence", every
prediction comes with WHICH features drove that verdict and by how
much - turning the "Brain Explanation" panel in the proposal's UI
mockup (Appendix D) from a hardcoded string into something generated
from the actual model.

Uses shap.TreeExplainer, a fast SHAP variant built specifically for
tree-based models (Random Forest, XGBoost, etc.) - it computes exact
Shapley values by walking the tree structure directly, rather than
the much slower brute-force sampling used for generic models.

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model
Predictions" (NeurIPS 2017).
"""

import numpy as np
import shap


class ShapExplainer:
    def __init__(self, rf_model, feature_names: list):
        self.explainer = shap.TreeExplainer(rf_model)
        self.feature_names = feature_names
        # For binary classifiers, shap.TreeExplainer returns per-class
        # contributions. We care about class index 1 = "malware".
        self.malware_class_index = 1

    def explain(self, X_row: np.ndarray, top_n: int = 3) -> dict:
        """
        X_row: a single-row, scaled feature array (shape (1, n_features)),
        same input that was passed to the Random Forest's predict_proba.

        Returns a dict with the top contributing features (both toward
        and away from "malware") and a plain-English summary sentence.
        """
        shap_values = self.explainer.shap_values(X_row)

        # shap.TreeExplainer output shape varies by version/model type:
        # newer versions return shape (samples, features, classes);
        # older versions return a list of per-class arrays. Handle both.
        if isinstance(shap_values, list):
            row_values = shap_values[self.malware_class_index][0]
        elif shap_values.ndim == 3:
            row_values = shap_values[0, :, self.malware_class_index]
        else:
            row_values = shap_values[0]

        contributions = list(zip(self.feature_names, row_values))
        # Sort by absolute contribution - biggest drivers of the
        # decision first, regardless of direction.
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top_pushing_malware = [c for c in contributions if c[1] > 0][:top_n]
        top_pushing_benign = [c for c in contributions if c[1] < 0][:top_n]

        summary = self._build_summary(top_pushing_malware, top_pushing_benign)

        return {
            "method": "shap_tree_explainer",
            "top_features": [
                {"feature": name, "contribution": round(float(val), 4),
                 "direction": "toward_malware" if val > 0 else "toward_benign"}
                for name, val in contributions[:top_n]
            ],
            "summary": summary,
        }

    @staticmethod
    def _humanize_feature_name(name: str) -> str:
        """Turns 'malfind.ninjections' into 'code injections detected'
        style plain-English phrases for the alert/dashboard. Falls back
        to a lightly cleaned-up version of the raw name if not mapped."""
        mapping = {
            "malfind.ninjections": "abnormal code injection count",
            "malfind.uniqueInjections": "unique code injection sites",
            "handles.nsection": "unusual memory section handles",
            "ldrmodules.not_in_load": "modules hidden from the loader",
            "ldrmodules.not_in_init": "modules hidden from initialization list",
            "ldrmodules.not_in_mem": "modules hidden from memory list",
            "ldrmodules_anomaly_score": "hidden/unlinked module activity",
            "psxview.not_in_ethread_pool": "hidden process indicators",
            "psxview.not_in_pslist": "process hidden from process list",
            "callbacks.nanonymous": "anonymous kernel callbacks",
            "handles.nhandles": "abnormal total open handles",
            "dlllist.ndlls": "unusual number of loaded DLLs",
            "injections_per_process": "high injection-to-process ratio",
            "handles_per_process": "abnormal handles-per-process ratio",
        }
        if name in mapping:
            return mapping[name]
        return name.replace("_", " ").replace(".", " ").lower()

    def _build_summary(self, top_malware: list, top_benign: list) -> str:
        if not top_malware and not top_benign:
            return "No individual feature stood out - verdict driven by combined weak signals."

        parts = []
        if top_malware:
            phrases = [self._humanize_feature_name(name) for name, _ in top_malware]
            parts.append("driven mainly by " + ", ".join(phrases))
        if top_benign:
            phrases = [self._humanize_feature_name(name) for name, _ in top_benign]
            parts.append("offset slightly by " + ", ".join(phrases))

        return "; ".join(parts).capitalize() + "."


if __name__ == "__main__":
    # Self-contained smoke test with a small synthetic Random Forest,
    # so this can be verified without the real trained model/dataset.
    from sklearn.ensemble import RandomForestClassifier

    feature_names = [
        "malfind.ninjections", "handles.nsection", "ldrmodules_anomaly_score",
        "dlllist.ndlls", "psxview.not_in_ethread_pool",
    ]
    X = np.random.rand(200, len(feature_names))
    # Make injections + handle anomalies the real drivers of the label,
    # so we can sanity-check the explainer picks them out correctly.
    y = ((X[:, 0] * 3 + X[:, 1] * 2) > 2.5).astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    explainer = ShapExplainer(model, feature_names)

    # A row engineered to look like ransomware: high injections, high handles
    suspicious_row = np.array([[0.95, 0.9, 0.3, 0.4, 0.5]])
    print("Prediction probability (malware):", model.predict_proba(suspicious_row)[0][1])
    result = explainer.explain(suspicious_row, top_n=3)
    print("\nExplanation result:")
    for f in result["top_features"]:
        print(f"  {f['feature']:35s} contribution={f['contribution']:+.4f}  ({f['direction']})")
    print("\nSummary:", result["summary"])

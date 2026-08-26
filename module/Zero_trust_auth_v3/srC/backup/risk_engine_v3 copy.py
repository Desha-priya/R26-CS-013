# ============================================================
# risk_engine_v3.py
# NeuraShield Zero-Trust Risk Engine V3
#
# Behavioral source:
#     Siamese BiLSTM V3 cosine similarity
#
# Responsibilities:
#     - behavioral biometric risk
#     - contextual risk
#     - cross-layer alerts
#     - adaptive authentication threshold
#     - action sensitivity
#     - continuous authentication decisions
#
# IMPORTANT:
#     No training.
#     No weight updates.
#     No Isolation Forest.
#     No One-Class SVM.
#     No old behavioral models.
# ============================================================

import uuid
from datetime import datetime, timezone


class RiskEngineV3:

    # ========================================================
    # VALIDATED MODEL OPERATING POINT
    # ========================================================

    # Frozen from the V3 validation analysis.
    #
    # Siamese BiLSTM embeddings are L2-normalized and compared
    # using cosine similarity.
    #
    # Validation accuracy-optimal threshold:
    #     0.847201
    #
    # Validation cosine EER:
    #     0.149622
    #
    # Test cosine AUC:
    #     0.926613
    #
    # Test cosine EER:
    #     0.145381
    #
    # This value must only be changed after retraining and
    # performing a new validation-only threshold selection.

    VALIDATED_SIMILARITY_THRESHOLD = 0.847201

    # Similarity below threshold but close to it:
    # monitor instead of immediately interrupting.
    MONITOR_BAND = 0.05

    # ========================================================
    # CONTEXT RISK
    # ========================================================

    CONTEXT_RISK = {
        "normal": 0.05,
        "normal_browsing": 0.05,

        "sensitive_access": 0.30,

        "financial": 0.50,
        "financial_transaction": 0.50,

        "privileged": 0.60,

        "under_attack": 0.75,
        "critical": 0.90,
    }

    # ========================================================
    # CROSS-LAYER ALERT RISK
    # ========================================================

    ALERT_RISK = {
        "low": 0.20,
        "medium": 0.45,
        "high": 0.75,
        "critical": 0.95,
    }

    # ========================================================
    # ACTION CLASSIFICATION
    # ========================================================

    FINANCIAL_ACTIONS = {
        "financial_transaction",
        "transaction",
        "transfer",
        "transfer_funds",
        "bank_transfer",
        "payment",
        "withdraw",
        "deposit",
    }

    PRIVILEGED_ACTIONS = {
        "admin",
        "administrator",
        "change_permissions",
        "privilege_change",
        "security_settings",
        "account_settings",
    }

    HIGH_SENSITIVITY_ACTIONS = {
        "password_change",
        "change_password",
        "enable_mfa",
        "disable_mfa",
        "delete_account",
        "security_change",
    }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        self.active_alerts = {}

        self.alert_expiry_seconds = 600

        self.last_decisions = {}

        print("[RiskEngineV3] Initialized")
        print("[RiskEngineV3] Behavioral model: Siamese BiLSTM V3")
        print("[RiskEngineV3] Old behavioral models: DISABLED")
        print("[RiskEngineV3] Adaptive threshold: ENABLED")
        print("[RiskEngineV3] Cross-layer alerts: ENABLED")
        print("[RiskEngineV3] Continuous authentication: ENABLED")

    # ========================================================
    # ALERT MANAGEMENT
    # ========================================================

    def receive_alert(
        self,
        source: str,
        severity: str = "medium"
    ):

        source = str(source).strip()

        severity = str(
            severity or "medium"
        ).lower().strip()

        if severity not in self.ALERT_RISK:
            severity = "medium"

        self.active_alerts[source] = {
            "severity": severity,
            "timestamp": datetime.now(timezone.utc),
        }

        return {
            "status": "alert_received",
            "source": source,
            "severity": severity,
        }

    # --------------------------------------------------------

    def clear_alert(self, source: str):

        self.active_alerts.pop(
            str(source),
            None
        )

        return {
            "status": "cleared",
            "source": source,
        }

    # --------------------------------------------------------

    def clear_expired_alerts(self):

        now = datetime.now(timezone.utc)

        expired = []

        for source, alert in self.active_alerts.items():

            age = (
                now - alert["timestamp"]
            ).total_seconds()

            if age > self.alert_expiry_seconds:
                expired.append(source)

        for source in expired:
            self.active_alerts.pop(
                source,
                None
            )

    # ========================================================
    # CROSS-LAYER RISK
    # ========================================================

    def calculate_alert_risk(self):

        self.clear_expired_alerts()

        if not self.active_alerts:
            return 0.0

        risks = []

        for alert in self.active_alerts.values():

            severity = alert["severity"]

            risks.append(
                self.ALERT_RISK.get(
                    severity,
                    0.45
                )
            )

        strongest = max(risks)

        # Multiple independent active alerts increase
        # confidence that the session is under attack.

        if len(risks) >= 3:

            strongest = min(
                1.0,
                strongest + 0.15
            )

        elif len(risks) == 2:

            strongest = min(
                1.0,
                strongest + 0.05
            )

        return round(
            strongest,
            4
        )

    # ========================================================
    # CONTEXT
    # ========================================================

    def calculate_context_risk(
        self,
        context: str
    ):

        context = str(
            context or "normal_browsing"
        ).lower().strip()

        return self.CONTEXT_RISK.get(
            context,
            0.05
        )

    # ========================================================
    # ACTION SENSITIVITY
    # ========================================================

    def classify_action(
        self,
        action: str
    ):

        action = str(
            action or ""
        ).lower().strip()

        if action in self.FINANCIAL_ACTIONS:
            return "CRITICAL"

        if action in self.PRIVILEGED_ACTIONS:
            return "CRITICAL"

        if action in self.HIGH_SENSITIVITY_ACTIONS:
            return "HIGH"

        # Handle compound application actions.

        if any(
            word in action
            for word in (
                "transfer",
                "payment",
                "transaction",
                "withdraw",
                "deposit",
                "bank_transfer",
            )
        ):
            return "CRITICAL"

        if any(
            word in action
            for word in (
                "admin",
                "permission",
                "privilege",
                "security",
            )
        ):
            return "HIGH"

        if any(
            word in action
            for word in (
                "password",
                "settings",
                "mfa",
            )
        ):
            return "HIGH"

        return "NORMAL"

    # ========================================================
    # ADAPTIVE AUTHENTICATION THRESHOLD
    # ========================================================

    def calculate_threshold(
        self,
        behavioral_risk,
        alert_risk,
        context_risk,
        action_sensitivity
    ):

        # Base threshold comes from the validated Siamese
        # BiLSTM operating point.

        threshold = (
            self.VALIDATED_SIMILARITY_THRESHOLD
        )

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        if context_risk >= 0.90:

            threshold = max(
                threshold,
                0.97
            )

        elif context_risk >= 0.75:

            threshold = max(
                threshold,
                0.95
            )

        elif context_risk >= 0.50:

            threshold = max(
                threshold,
                0.92
            )

        elif context_risk >= 0.30:

            threshold = max(
                threshold,
                0.88
            )

        # ----------------------------------------------------
        # CROSS-LAYER ALERTS
        # ----------------------------------------------------

        if alert_risk >= 0.90:

            threshold = max(
                threshold,
                0.97
            )

        elif alert_risk >= 0.70:

            threshold = max(
                threshold,
                0.93
            )

        elif alert_risk >= 0.40:

            threshold = max(
                threshold,
                0.89
            )

        # ----------------------------------------------------
        # BEHAVIORAL RISK
        # ----------------------------------------------------

        if behavioral_risk >= 0.70:

            threshold = max(
                threshold,
                0.90
            )

        # ----------------------------------------------------
        # ACTION SENSITIVITY
        # ----------------------------------------------------

        if action_sensitivity == "CRITICAL":

            threshold = max(
                threshold,
                0.93
            )

        elif action_sensitivity == "HIGH":

            threshold = max(
                threshold,
                0.89
            )

        # ----------------------------------------------------
        # EXTREME COMBINATION
        # ----------------------------------------------------

        if (
            alert_risk >= 0.70
            and context_risk >= 0.50
            and action_sensitivity == "CRITICAL"
        ):

            threshold = 0.98

        return round(
            min(
                threshold,
                0.995
            ),
            4
        )

    # ========================================================
    # OVERALL RISK
    # ========================================================

    def calculate_overall_risk(
        self,
        behavioral_risk,
        alert_risk,
        context_risk
    ):

        # Behavioral identity remains the primary signal.

        overall = (
            0.50 * behavioral_risk
            + 0.30 * alert_risk
            + 0.20 * context_risk
        )

        return round(
            min(
                1.0,
                max(
                    0.0,
                    overall
                )
            ),
            4
        )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    def risk_level(
        self,
        overall_risk
    ):

        if overall_risk >= 0.75:
            return "CRITICAL"

        if overall_risk >= 0.50:
            return "HIGH"

        if overall_risk >= 0.25:
            return "MEDIUM"

        return "LOW"

    # ========================================================
    # MAIN DECISION
    # ========================================================

    def evaluate(
        self,
        behavioral_similarity: float,
        context: str = "normal_browsing",
        action: str = "",
        user_id=None,
        session_id=None
    ):

        # ----------------------------------------------------
        # SANITIZE SIMILARITY
        # ----------------------------------------------------

        try:

            similarity = float(
                behavioral_similarity
            )

        except (
            TypeError,
            ValueError
        ):

            similarity = 0.0

        similarity = max(
            0.0,
            min(
                1.0,
                similarity
            )
        )

        # ----------------------------------------------------
        # RISK SIGNALS
        # ----------------------------------------------------

        behavioral_risk = round(
            1.0 - similarity,
            4
        )

        alert_risk = round(
            self.calculate_alert_risk(),
            4
        )

        context_risk = round(
            self.calculate_context_risk(
                context
            ),
            4
        )

        action_sensitivity = (
            self.classify_action(
                action
            )
        )

        # ----------------------------------------------------
        # OVERALL RISK
        # ----------------------------------------------------

        overall_risk = (
            self.calculate_overall_risk(
                behavioral_risk,
                alert_risk,
                context_risk
            )
        )

        # ----------------------------------------------------
        # ADAPTIVE THRESHOLD
        # ----------------------------------------------------

        adaptive_threshold = (
            self.calculate_threshold(
                behavioral_risk,
                alert_risk,
                context_risk,
                action_sensitivity
            )
        )

        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        reauth_boundary = (
            adaptive_threshold
            - self.MONITOR_BAND
        )

        if similarity < reauth_boundary:

            decision = "REAUTHENTICATE"

            requires_reauthentication = True
            requires_monitoring = False

        elif similarity < adaptive_threshold:

            decision = "MONITOR"

            requires_reauthentication = False
            requires_monitoring = True

        else:

            decision = "ALLOW"

            requires_reauthentication = False
            requires_monitoring = False

        # ----------------------------------------------------
        # REASONS
        # ----------------------------------------------------

        reasons = []

        if requires_reauthentication:

            reasons.append(
                "Behavioral similarity is below "
                "the adaptive authentication boundary."
            )

        elif requires_monitoring:

            reasons.append(
                "Behavioral similarity is below "
                "the adaptive threshold; session "
                "requires increased monitoring."
            )

        if alert_risk >= 0.70:

            reasons.append(
                "Cross-layer security risk is high."
            )

        elif alert_risk >= 0.40:

            reasons.append(
                "Cross-layer security alert is active."
            )

        if context_risk >= 0.75:

            reasons.append(
                "Contextual risk is critical."
            )

        elif context_risk >= 0.50:

            reasons.append(
                "Sensitive or financial context detected."
            )

        if action_sensitivity == "CRITICAL":

            reasons.append(
                "Critical financial or privileged action "
                "detected; stricter behavioral verification applied."
            )

        elif action_sensitivity == "HIGH":

            reasons.append(
                "High-sensitivity action detected; "
                "stricter behavioral verification applied."
            )

        if not reasons:

            reasons.append(
                "Behavioral identity is consistent with "
                "the authenticated user and contextual "
                "risk is acceptable."
            )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = {

            "decision_id":
                str(uuid.uuid4()),

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "decision":
                decision,

            "risk_level":
                self.risk_level(
                    overall_risk
                ),

            "overall_risk":
                overall_risk,

            "adaptive_threshold":
                adaptive_threshold,

            "behavioral_similarity":
                round(
                    similarity,
                    6
                ),

            "behavioral_risk":
                behavioral_risk,

            "alert_risk":
                alert_risk,

            "context_risk":
                context_risk,

            "action_sensitivity":
                action_sensitivity,

            "requires_reauthentication":
                requires_reauthentication,

            "requires_monitoring":
                requires_monitoring,

            "reasons":
                reasons,

            "controls": {

                "training_enabled":
                    False,

                "weight_updates":
                    False,

                "old_behavioral_models":
                    False,

                "adaptive_threshold":
                    True,

                "cross_layer_alerts":
                    True,

                "continuous_authentication":
                    True,

                "financial_action_escalation":
                    action_sensitivity == "CRITICAL",
            },

            "metadata": {

                "behavioral_model":
                    "siamese_bilstm_v3",

                "user_id":
                    user_id,

                "session_id":
                    session_id,

                "context":
                    context,

                "action":
                    action,

                "alert_count":
                    len(
                        self.active_alerts
                    ),
            }
        }

        if user_id is not None:

            self.last_decisions[
                str(user_id)
            ] = result

        return result


# ============================================================
# SELF TEST
# ============================================================

def self_test():

    print("=" * 70)
    print("ZERO TRUST RISK ENGINE V3")
    print("SELF TEST")
    print("=" * 70)

    engine = RiskEngineV3()

    # --------------------------------------------------------
    # TEST 1
    # --------------------------------------------------------

    print("\n[TEST 1] NORMAL ACTIVITY")

    result = engine.evaluate(
        behavioral_similarity=0.94,
        context="normal_browsing",
        action="VIEW_PROFILE",
        user_id="USER-001",
        session_id="TEST-001"
    )

    print(result)

    # --------------------------------------------------------
    # TEST 2
    # --------------------------------------------------------

    print("\n[TEST 2] BEHAVIORAL ANOMALY")

    result = engine.evaluate(
        behavioral_similarity=0.70,
        context="normal_browsing",
        action="VIEW_ACCOUNT",
        user_id="USER-001",
        session_id="TEST-002"
    )

    print(result)

    # --------------------------------------------------------
    # TEST 3
    # --------------------------------------------------------

    print(
        "\n[TEST 3] BEHAVIORAL ANOMALY + "
        "CROSS-LAYER ALERT"
    )

    engine.receive_alert(
        "network_guardian",
        "high"
    )

    result = engine.evaluate(
        behavioral_similarity=0.82,
        context="sensitive_access",
        action="ACCOUNT_SETTINGS",
        user_id="USER-001",
        session_id="TEST-003"
    )

    print(result)

    # --------------------------------------------------------
    # TEST 4
    # --------------------------------------------------------

    print(
        "\n[TEST 4] NORMAL FINANCIAL ACTION"
    )

    engine.active_alerts.clear()

    result = engine.evaluate(
        behavioral_similarity=0.95,
        context="financial",
        action="FINANCIAL_TRANSACTION",
        user_id="USER-001",
        session_id="TEST-004"
    )

    print(result)

    # --------------------------------------------------------
    # TEST 5
    # --------------------------------------------------------

    print(
        "\n[TEST 5] HIGH-RISK FINANCIAL ACTION"
    )

    engine.receive_alert(
        "network_guardian",
        "high"
    )

    engine.receive_alert(
        "content_threat_detection",
        "critical"
    )

    result = engine.evaluate(
        behavioral_similarity=0.82,
        context="financial",
        action="TRANSFER_FUNDS",
        user_id="USER-001",
        session_id="TEST-005"
    )

    print(result)

    print(
        "\n" + "=" * 70
    )

    print(
        "SELF TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    self_test()
# ============================================================
# risk_engine_v3.py
# NeuraShield Zero-Trust Risk Engine V3
#
# Behavioral source:
#     Siamese BiLSTM V3

#     behavioral biometric risk
#     contextual risk
#     cross-layer alerts
#     adaptive authentication threshold
#     financial-action escalation
#     continuous authentication
# ============================================================

import uuid
from datetime import datetime, timezone


class RiskEngineV3:

    # --------------------------------------------------------
    # BASE ADAPTIVE THRESHOLDS
    # --------------------------------------------------------

    BASE_THRESHOLDS = {
        "NORMAL": 0.70,
        "ELEVATED": 0.55,
        "HIGH": 0.45,
        "CRITICAL": 0.30,
    }

    # --------------------------------------------------------
    # CONTEXT RISK
    # --------------------------------------------------------

    CONTEXT_RISK = {
        "normal_browsing": 0.05,
        "normal": 0.05,

        "sensitive_access": 0.30,

        "financial": 0.50,
        "financial_transaction": 0.50,

        "privileged": 0.60,

        "under_attack": 0.75,
        "critical": 0.90,
    }

    # --------------------------------------------------------
    # ALERT RISK
    # --------------------------------------------------------

    ALERT_RISK = {
        "low": 0.20,
        "medium": 0.45,
        "high": 0.75,
        "critical": 0.95,
    }

    # --------------------------------------------------------
    # ACTION CLASSIFICATION
    # --------------------------------------------------------

    FINANCIAL_ACTIONS = {
        "financial_transaction",
        "transfer",
        "transfer_funds",
        "payment",
        "withdraw",
        "deposit",
        "bank_transfer",
        "transaction",
    }

    PRIVILEGED_ACTIONS = {
        "admin",
        "administrator",
        "change_permissions",
        "privilege_change",
        "security_settings",
        "account_settings",
    }

    def __init__(self):

        self.active_alerts = {}

        self.alert_expiry_seconds = 600

        self.last_decisions = {}

        print("[RiskEngineV3] Initialized")
        print("[RiskEngineV3] Old behavioral ML models: DISABLED")
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

        severity = str(severity).lower()

        self.active_alerts[source] = {
            "severity": severity,
            "timestamp": datetime.now(timezone.utc),
        }

        return {
            "status": "alert_received",
            "source": source,
            "severity": severity,
        }

    def clear_alert(self, source: str):

        self.active_alerts.pop(source, None)

        return {
            "status": "cleared",
            "source": source,
        }

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
            self.active_alerts.pop(source, None)

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

        # Multiple independent alerts increase confidence
        # that the session is under attack.
        if len(risks) >= 3:

            combined = max(risks)

            combined = min(
                1.0,
                combined + 0.15
            )

            return combined

        return max(risks)

    # ========================================================
    # CONTEXT
    # ========================================================

    def calculate_context_risk(
        self,
        context: str
    ):

        context = str(
            context or "normal_browsing"
        ).lower()

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

        if (
            "transfer" in action
            or "payment" in action
            or "transaction" in action
            or "withdraw" in action
            or "deposit" in action
        ):

            return "CRITICAL"

        if (
            "admin" in action
            or "permission" in action
            or "security" in action
        ):

            return "HIGH"

        if (
            "account" in action
            or "password" in action
            or "settings" in action
        ):

            return "HIGH"

        return "NORMAL"

    # ========================================================
    # ADAPTIVE THRESHOLD
    # ========================================================

    def calculate_threshold(
        self,
        behavioral_risk,
        alert_risk,
        context_risk,
        action_sensitivity
    ):

        # Start from normal threshold.
        threshold = 0.70

        # ----------------------------------------------------
        # Context pressure
        # ----------------------------------------------------

        if context_risk >= 0.75:
            threshold = min(
                threshold,
                0.30
            )

        elif context_risk >= 0.50:
            threshold = min(
                threshold,
                0.40
            )

        elif context_risk >= 0.30:
            threshold = min(
                threshold,
                0.55
            )

        # ----------------------------------------------------
        # Cross-layer alerts
        # ----------------------------------------------------

        if alert_risk >= 0.90:

            threshold = min(
                threshold,
                0.20
            )

        elif alert_risk >= 0.70:

            threshold = min(
                threshold,
                0.45
            )

        elif alert_risk >= 0.40:

            threshold = min(
                threshold,
                0.55
            )

        # ----------------------------------------------------
        # Behavioral anomaly
        # ----------------------------------------------------

        if behavioral_risk >= 0.70:

            threshold = min(
                threshold,
                0.45
            )

        elif behavioral_risk >= 0.50:

            threshold = min(
                threshold,
                0.55
            )

        # ----------------------------------------------------
        # Critical actions
        # ----------------------------------------------------

        if action_sensitivity == "CRITICAL":

            threshold = min(
                threshold,
                0.40
            )

        elif action_sensitivity == "HIGH":

            threshold = min(
                threshold,
                0.50
            )

        # ----------------------------------------------------
        # Combined extreme situation
        # ----------------------------------------------------

        if (
            alert_risk >= 0.70
            and context_risk >= 0.50
            and action_sensitivity == "CRITICAL"
        ):

            threshold = 0.15

        return round(
            threshold,
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

        # Behavioral identity is the primary signal.
        #
        # Context and cross-layer security telemetry
        # modify the trust decision.

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
        # Similarity -> behavioral risk
        #
        # Siamese model:
        #     1.0 = very similar
        #     0.0 = very different
        #
        # Therefore:
        #     behavioral risk = 1 - similarity
        # ----------------------------------------------------

        similarity = float(
            max(
                0.0,
                min(
                    1.0,
                    behavioral_similarity
                )
            )
        )

        behavioral_risk = round(
            1.0 - similarity,
            4
        )

        # ----------------------------------------------------
        # Other signals
        # ----------------------------------------------------

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
            self.classify_action(action)
        )

        # ----------------------------------------------------
        # Overall risk
        # ----------------------------------------------------

        overall_risk = (
            self.calculate_overall_risk(
                behavioral_risk,
                alert_risk,
                context_risk
            )
        )

        # ----------------------------------------------------
        # Adaptive threshold
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
        # Decision
        #
        # IMPORTANT:
        # The threshold is a similarity threshold.
        #
        # Authentication is allowed when:
        #
        #     similarity >= threshold
        #
        # Otherwise:
        #
        #     reauthentication
        # ----------------------------------------------------

        requires_reauthentication = (
            similarity < adaptive_threshold
        )

        if requires_reauthentication:

            decision = "REAUTHENTICATE"

        else:

            decision = "ALLOW"

        # ----------------------------------------------------
        # Reasons
        # ----------------------------------------------------

        reasons = []

        if similarity < adaptive_threshold:

            reasons.append(
                "Behavioral similarity is below "
                "the adaptive authentication threshold."
            )

        if alert_risk >= 0.70:

            reasons.append(
                "Cross-layer security risk is high."
            )

        if context_risk >= 0.70:

            reasons.append(
                "Overall contextual risk is critical."
            )

        if action_sensitivity == "CRITICAL":

            reasons.append(
                "Financial or privileged action detected; "
                "stricter authentication policy applied."
            )

        if not reasons:

            reasons.append(
                "Behavioral identity is consistent with "
                "the authenticated user and contextual "
                "risk is acceptable."
            )

        # ----------------------------------------------------
        # Result
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
                    4
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
                    action_sensitivity
                    == "CRITICAL",
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

        # Save most recent decision.
        if user_id is not None:

            self.last_decisions[
                user_id
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

    print("\n[1] NORMAL USER")

    r = engine.evaluate(
        behavioral_similarity=0.94,
        context="normal_browsing",
        action="VIEW_PROFILE",
        user_id="USER-001",
        session_id="TEST-001"
    )

    print(r)

    print("\n[2] BEHAVIORAL ANOMALY")

    r = engine.evaluate(
        behavioral_similarity=0.31,
        context="normal_browsing",
        action="VIEW_ACCOUNT",
        user_id="USER-001",
        session_id="TEST-002"
    )

    print(r)

    print("\n[3] ANOMALY + CROSS-LAYER ALERT")

    engine.receive_alert(
        "network_guardian",
        "high"
    )

    r = engine.evaluate(
        behavioral_similarity=0.40,
        context="sensitive_access",
        action="ACCOUNT_SETTINGS",
        user_id="USER-001",
        session_id="TEST-003"
    )

    print(r)

    print("\n[4] NORMAL FINANCIAL ACTION")

    engine.active_alerts.clear()

    r = engine.evaluate(
        behavioral_similarity=0.92,
        context="financial",
        action="FINANCIAL_TRANSACTION",
        user_id="USER-001",
        session_id="TEST-004"
    )

    print(r)

    print("\n[5] HIGH-RISK FINANCIAL ACTION")

    engine.receive_alert(
        "network_guardian",
        "high"
    )

    engine.receive_alert(
        "content_threat_detection",
        "critical"
    )

    r = engine.evaluate(
        behavioral_similarity=0.42,
        context="financial",
        action="TRANSFER_FUNDS",
        user_id="USER-001",
        session_id="TEST-005"
    )

    print(r)

    print("\n" + "=" * 70)
    print("SELF TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    self_test()
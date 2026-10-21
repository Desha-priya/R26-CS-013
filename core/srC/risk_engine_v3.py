# -*-*-*-*-*-*-*-*-*---***-*-*--**--====
# risk_engine_v3.py
# NeuraShield Zero-Trust Risk Engine V3   security add on

import uuid
from datetime import datetime, timezone


class RiskEngineV3:

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # In Here VALIDATED MODEL OPERATING POINT  ---
    # Frozen from the V3 validation analysis.
    # Siamese BiLSTM embeddings are L2-normalized and compared using cosine similarity.
    #
    # Validation accuracy-optimal threshold:  0.847201
    # Validation cosine EER: 0.149622
    # Test cosine AUC: 0.926613
    # Test cosine EER: 0.145381

    # This value have to only be changed after retraining and performing a new validation-only threshold selection.
    VALIDATED_SIMILARITY_THRESHOLD = 0.847201

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # NORMAL-BROWSING OPERATIONAL FLOOR
    #
    # This is a deliberate UX/product decision, NOT a re-
    # validation of the model. VALIDATED_SIMILARITY_THRESHOLD
    # above remains the citable, EER-optimal number for the
    # model's research validation and is unchanged.
    #
    # In real-world use, requiring the EER-optimal boundary even
    # during ordinary, low-stakes browsing produced too many false
    # interruptions (e.g. typing slower than usual). Since a false
    # accept during normal browsing is far less costly than one
    # during a financial/sensitive action, this lower floor only
    # applies when context_risk < 0.30 (normal browsing, no active
    # alerts). The moment context, cross-layer alerts, or the
    # behavioral signal itself escalate, the same upward rules in
    # calculate_threshold() below still apply from here — this
    # only lowers the floor, never the ceiling.
    NORMAL_BROWSING_THRESHOLD = 0.70

    # Similarity below threshold but close to it:
    # monitor instead of immediately interrupting.
    MONITOR_BAND = 0.05

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

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

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # CROSS-LAYER ALERT RISK
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

    ALERT_RISK = {
        "low": 0.20,
        "medium": 0.45,
        "high": 0.75,
        "critical": 0.95,
    }

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # ACTION CLASSIFICATION
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

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

    CONSECUTIVE_REAUTH_REQUIRED = 2  # default / sensitive contexts

    CONSECUTIVE_REAUTH_REQUIRED_NORMAL = 4  # normal_browsing only

    def _consecutive_required_for(self, context_risk):
        """
        context_risk < 0.30 means normal/low-stakes browsing (see
        CONTEXT_RISK mapping) — give it more tolerance. Anything
        at or above that (sensitive_access, financial, privileged,
        under_attack, critical) keeps the stricter default...
        """

        if context_risk < 0.30:

            return self.CONSECUTIVE_REAUTH_REQUIRED_NORMAL

        return self.CONSECUTIVE_REAUTH_REQUIRED

    # After a successful, identity-confirmed step-up, suppress re-triggering REAUTHENTICATE for this long in NON-sensitive
    VERIFIED_GRACE_SECONDS = 120

    # Shorter grace if the step-up passed liveness but had no enrolled face reference (presence proven, identity not).
    UNVERIFIED_GRACE_SECONDS = 30

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # INITIALIZATION
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

    def __init__(self):

        self.active_alerts = {}

        self.alert_expiry_seconds = 600

        self.last_decisions = {}

        # Per-user/session count of consecutive windows that crossed the REAUTHENTICATE boundary.
        self._consecutive_low = {}

        # Per-user grace-period state after a passed step-up.
        # {user_id: (expires_at_ts, identity_confirmed_bool)}
        self._verified_until = {}

        print("[RiskEngineV3] Initialized")
        print("[RiskEngineV3] Behavioral model: Siamese BiLSTM V3")
        print("[RiskEngineV3] Old behavioral models: DISABLED")
        print("[RiskEngineV3] Adaptive threshold: ENABLED")
        print("[RiskEngineV3] Cross-layer alerts: ENABLED")
        print("[RiskEngineV3] Continuous authentication: ENABLED")
        print(
            "[RiskEngineV3] Decision smoothing: "
            f"{self.CONSECUTIVE_REAUTH_REQUIRED_NORMAL} consecutive "
            "windows in normal browsing / "
            f"{self.CONSECUTIVE_REAUTH_REQUIRED} in sensitive/"
            "financial/attack contexts, required for step-up"
        )
        print(
            "[RiskEngineV3] Post-verification grace: "
            f"{self.VERIFIED_GRACE_SECONDS}s verified / "
            f"{self.UNVERIFIED_GRACE_SECONDS}s liveness-only, "
            "suspended in sensitive contexts"
        )

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # SMOOTHING + VERIFICATION STATE
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

    def reset_consecutive(self, user_id):
        """
        Call after a successful step-up or fresh enrollment, so
        the next window starts clean instead of carrying over a
        stale anomaly streak.
        """

        self._consecutive_low.pop(
            str(user_id),
            None
        )

    def mark_verified(self, user_id, identity_confirmed=False):
        """
        Call after a PASSED step-up (face, PIN, or liveness).
        Opens a grace window during which normal-context
        REAUTHENTICATE escalation is suppressed (downgraded to
        MONITOR). Never applies once context_risk >= 0.30.
        """

        grace_seconds = (
            self.VERIFIED_GRACE_SECONDS
            if identity_confirmed
            else self.UNVERIFIED_GRACE_SECONDS
        )

        self._verified_until[str(user_id)] = (
            datetime.now().timestamp() + grace_seconds,
            identity_confirmed,
        )

    def clear_verified(self, user_id):
        """Explicitly end a grace period (e.g. on session lock)."""

        self._verified_until.pop(
            str(user_id),
            None
        )

    def get_verification_status(self, user_id):
        """
        Grace status for the dashboard / decision logic.
        """

        entry = self._verified_until.get(str(user_id))

        if entry is None:

            return {
                "grace_active": False,
                "grace_remaining_seconds": 0,
                "identity_confirmed": False,
            }

        expires_at, identity_confirmed = entry

        remaining = expires_at - datetime.now().timestamp()

        return {
            "grace_active": remaining > 0,
            "grace_remaining_seconds": max(0, int(remaining)),
            "identity_confirmed": identity_confirmed,
        }

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # ALERT MANAGEMENT
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

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

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # CROSS-LAYER RISK
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

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

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # CONTEXT
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

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

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # ACTION SENSITIVITY
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

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

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # ADAPTIVE AUTHENTICATION THRESHOLD
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

    def calculate_threshold(
        self,
        behavioral_risk,
        alert_risk,
        context_risk,
        action_sensitivity
    ):

        # Normal, low-stakes browsing starts from the more
        # forgiving operational floor; anything else starts from
        # the validated EER-optimal boundary as before. Every
        # upward escalation rule below still applies from
        # whichever starting point applies.
        threshold = (
            self.NORMAL_BROWSING_THRESHOLD
            if context_risk < 0.30
            else self.VALIDATED_SIMILARITY_THRESHOLD
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

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # OVERALL RISK
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

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

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # RISK LEVEL
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

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

    # -*-*-*-*-*-*-*-*-*---***-*-*--**--
    # MAIN DECISION
    # -*-*-*-*-*-*-*-*-*---***-*-*--**--

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
        # CONTEXT (computed early — needed for the safe-score
        # check right below)
        # ----------------------------------------------------

        context_risk = round(
            self.calculate_context_risk(
                context
            ),
            4
        )

        # ----------------------------------------------------
        # VERIFICATION / GRACE STATUS (checked early so it can
        # inform the DISPLAYED risk score itself, not just the
        # decision further below)
        # ----------------------------------------------------

        grace_key = (
            str(user_id)
            if user_id is not None
            else "global"
        )

        verification = self.get_verification_status(grace_key)

        # ----------------------------------------------------
        # RISK SIGNALS
        # ----------------------------------------------------

        raw_behavioral_risk = round(
            1.0 - similarity,
            4
        )

        # If a recent, IDENTITY-CONFIRMED step-up (face match or
        # PIN) is still within its grace window, and we're not in
        # a sensitive/financial/attack context, the person has
        # already proven who they are via a stronger factor than
        # typing rhythm alone. Showing a high behavioral-risk
        # number next to an ALLOW/MONITOR decision would be
        # confusing and would undercut the trust that
        # verification just established — so the DISPLAYED
        # behavioral/overall risk reflects that trust.
        #
        # This is cosmetic/reporting only: raw_behavioral_risk
        # below still drives the actual threshold/decision/
        # smoothing logic unchanged, so a renewed, sustained
        # anomaly is still caught exactly as before.
        face_verified_safe = (
            verification["grace_active"]
            and verification["identity_confirmed"]
            and context_risk < 0.30
        )

        behavioral_risk = (
            min(raw_behavioral_risk, 0.08)
            if face_verified_safe
            else raw_behavioral_risk
        )

        alert_risk = round(
            self.calculate_alert_risk(),
            4
        )

        action_sensitivity = (
            self.classify_action(
                action
            )
        )

        # ----------------------------------------------------
        # OVERALL RISK (uses the possibly-capped behavioral_risk
        # above, so the gauge reflects the same trust the
        # decision does)
        # ----------------------------------------------------

        overall_risk = (
            self.calculate_overall_risk(
                behavioral_risk,
                alert_risk,
                context_risk
            )
        )

        # ----------------------------------------------------
        # ADAPTIVE THRESHOLD (always uses the RAW behavioral
        # risk — security escalation must never be softened by
        # the cosmetic display cap above)
        # ----------------------------------------------------

        adaptive_threshold = (
            self.calculate_threshold(
                raw_behavioral_risk,
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

        # ----------------------------------------------------
        # RAW TIER (before smoothing)
        # ----------------------------------------------------

        raw_reauth_candidate = similarity < reauth_boundary

        raw_monitor_candidate = (
            not raw_reauth_candidate
            and similarity < adaptive_threshold
        )

        # ----------------------------------------------------
        # POST-VERIFICATION GRACE PERIOD
        #
        # grace_key/verification were already computed earlier
        # (needed for the safe-score display cap above) — reused
        # here for the decision-suppression logic.

        grace_suppressed_reauth = False

        if (
            raw_reauth_candidate
            and verification["grace_active"]
            and context_risk < 0.30
        ):

            raw_reauth_candidate = False
            raw_monitor_candidate = True
            grace_suppressed_reauth = True

        # ----------------------------------------------------
        # SMOOTHING

        smoothing_key = (
            str(user_id)
            if user_id is not None
            else "global"
        )

        required_streak = self._consecutive_required_for(context_risk)

        if raw_reauth_candidate:

            streak = (
                self._consecutive_low.get(smoothing_key, 0) + 1
            )

            self._consecutive_low[smoothing_key] = streak

            if streak >= required_streak:

                decision = "REAUTHENTICATE"
                requires_reauthentication = True
                requires_monitoring = False

            else:

                decision = "MONITOR"
                requires_reauthentication = False
                requires_monitoring = True

        elif raw_monitor_candidate:

            self._consecutive_low[smoothing_key] = 0

            decision = "MONITOR"
            requires_reauthentication = False
            requires_monitoring = True

        else:

            self._consecutive_low[smoothing_key] = 0

            decision = "ALLOW"
            requires_reauthentication = False
            requires_monitoring = False

        consecutive_low_count = self._consecutive_low.get(
            smoothing_key,
            0
        )

        # ----------------------------------------------------
        # REASONS
        # ----------------------------------------------------

        reasons = []

        if requires_reauthentication:

            reasons.append(
                f"Behavioral similarity has been below the "
                f"adaptive boundary for {consecutive_low_count} "
                f"consecutive windows — step-up required."
            )

        elif requires_monitoring:

            if raw_reauth_candidate:

                reasons.append(
                    "Behavioral similarity is below the "
                    "reauthentication boundary "
                    f"({consecutive_low_count}/"
                    f"{required_streak} "
                    "consecutive windows); monitoring before "
                    "forcing step-up."
                )

            else:
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
            "timestamp":  datetime.now(  timezone.utc ).isoformat(),

            "decision": decision,
            "risk_level": self.risk_level( overall_risk ),
            "overall_risk": overall_risk,
            "adaptive_threshold": adaptive_threshold,
            "behavioral_similarity": round( similarity, 6 ),
            "behavioral_risk":behavioral_risk,
            "alert_risk": alert_risk,
            "context_risk":
                context_risk,
            "action_sensitivity":
                action_sensitivity,
            "requires_reauthentication":
                requires_reauthentication,
            "requires_monitoring":
                requires_monitoring,

            "consecutive_low_count":
                consecutive_low_count,
            "grace_active":
                verification["grace_active"],
            "grace_remaining_seconds":
                verification["grace_remaining_seconds"],
            "grace_suppressed_reauth":
                grace_suppressed_reauth,
            "identity_confirmed":
                verification["identity_confirmed"],

            "reasons":
                reasons,
            "controls": {
                "training_enabled": False,
                "weight_updates": False,
                "old_behavioral_models": False,
                "adaptive_threshold": True,
                "cross_layer_alerts": True,
                "continuous_authentication": True,
                "financial_action_escalation": action_sensitivity == "CRITICAL",
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


# -*-*-*-*-*-*-*-*-*---***-*-*--**--====
# SELF TEST - - - - - 
# -*-*-*-*-*-*-*-*-*---***-*-*--**--====
def self_test():
    # openpyxl is only needed by this developer self-test, so it is
    # imported here and not at module level. Importing it at the top
    # would make it a hard run-time dependency of the packaged program.
    import os
    from openpyxl import Workbook

    os.makedirs("test", exist_ok=True)

    # --------------------------------------------------------
    # OUTPUT FILES
    # --------------------------------------------------------
    txt_file = "test/risk_engine_self_test_output.txt"
    excel_file = "test/risk_engine_self_test_results.xlsx"

    terminal_output = []
    test_results = []

    def log_output(text=""):
        print(text)
        terminal_output.append(str(text))

    log_output("=" * 70)
    log_output("ZERO TRUST RISK ENGINE V3")
    log_output("SELF TEST")
    log_output("=" * 70)

    engine = RiskEngineV3()

    # --------------------------------------------------------
    # TEST 1
    # --------------------------------------------------------

    log_output("\n[TEST 1] NORMAL ACTIVITY")

    result = engine.evaluate(
        behavioral_similarity=0.94,
        context="normal_browsing",
        action="VIEW_PROFILE",
        user_id="USER-001",
        session_id="TEST-001"
    )

    log_output(result)

    test_results.append({
        "test": "TEST 1",
        "description": "NORMAL ACTIVITY",
        "session_id": "TEST-001",
        "result": result
    })

    # --------------------------------------------------------
    # TEST 2
    # --------------------------------------------------------

    log_output("\n[TEST 2] BEHAVIORAL ANOMALY")

    result = engine.evaluate(
        behavioral_similarity=0.70,
        context="normal_browsing",
        action="VIEW_ACCOUNT",
        user_id="USER-001",
        session_id="TEST-002"
    )

    log_output(result)

    test_results.append({
        "test": "TEST 2",
        "description": "BEHAVIORAL ANOMALY",
        "session_id": "TEST-002",
        "result": result
    })

    # --------------------------------------------------------
    # TEST 3
    # --------------------------------------------------------

    log_output(
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

    log_output(result)

    test_results.append({
        "test": "TEST 3",
        "description": "BEHAVIORAL ANOMALY + CROSS-LAYER ALERT",
        "session_id": "TEST-003",
        "result": result
    })

    # --------------------------------------------------------
    # TEST 4
    # --------------------------------------------------------

    log_output(
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

    log_output(result)

    test_results.append({
        "test": "TEST 4",
        "description": "NORMAL FINANCIAL ACTION",
        "session_id": "TEST-004",
        "result": result
    })

    # --------------------------------------------------------
    # TEST 5
    # --------------------------------------------------------

    log_output(
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

    log_output(result)

    test_results.append({
        "test": "TEST 5",
        "description": "HIGH-RISK FINANCIAL ACTION",
        "session_id": "TEST-005",
        "result": result
    })

    # --------------------------------------------------------
    # END
    # --------------------------------------------------------

    log_output(
        "\n" + "=" * 70
    )

    log_output(
        "SELF TEST COMPLETE"
    )

    log_output(
        "=" * 70
    )

    # --------------------------------------------------------
    # SAVE COMPLETE TERMINAL OUTPUT TO TXT
    # --------------------------------------------------------

    with open(
        txt_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(terminal_output)
        )

    # --------------------------------------------------------
    # SAVE RESULTS TO EXCEL
    # --------------------------------------------------------

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Self Test Results"

    # Headers
    worksheet.append([
        "Test",
        "Description",
        "Session ID",
        "Result"
    ])

    # Results
    for item in test_results:

        worksheet.append([
            item["test"],
            item["description"],
            item["session_id"],
            str(item["result"])
        ])

    # Adjust column widths
    worksheet.column_dimensions["A"].width = 15
    worksheet.column_dimensions["B"].width = 45
    worksheet.column_dimensions["C"].width = 20
    worksheet.column_dimensions["D"].width = 100

    # Wrap result text
    for row in worksheet.iter_rows():

        for cell in row:
            cell.alignment = cell.alignment.copy(
                wrap_text=True,
                vertical="top"
            )

    workbook.save(excel_file)

    # --------------------------------------------------------
    # CONFIRM FILES
    # --------------------------------------------------------

    print("\nOutput files created:")
    print(f"TXT   : {txt_file}")
    print(f"Excel : {excel_file}")


    
if __name__ == "__main__":
    self_test()
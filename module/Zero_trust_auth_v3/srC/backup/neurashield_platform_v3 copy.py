# ============================================================
# neurashield_platform_v3.py
#
# NeuraShield Unified Zero-Trust Platform V3
#
# Layer 3:
#     Siamese BiLSTM V3 behavioral biometrics
#
# Risk:
#     RiskEngineV3
#
# Architecture:
#
#   KeystrokeAgentV3
#          |
#          | behavioral evidence
#          v
#   NeuraShield Platform
#          |
#          +---- Session Context
#          |
#          +---- Cross-Layer Alerts
#          |
#          +---- Financial Context
#          |
#          v
#   RiskEngineV3
#          |
#          v
#   ALLOW / MONITOR / REAUTHENTICATE
#
# Old IF/SVM behavioral models:
#     DISABLED
#
# ============================================================

from fastapi import (
    FastAPI,
    Request,
    HTTPException
)

from fastapi.responses import HTMLResponse

from pydantic import BaseModel

from typing import Optional

import os
import sys
import csv
import pathlib
import threading
import hashlib

from datetime import datetime

from collections import deque


# ============================================================
# LOCAL IMPORTS
# ============================================================

sys.path.append(
    os.path.dirname(__file__)
)

from risk_engine_v3 import RiskEngineV3

from session_monitor import SessionMonitor

from keystroke_agent_v3 import (
    KeystrokeAgentV3
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="NeuraShield Platform V3 — Siamese BiLSTM"
)


# ============================================================
# RISK ENGINE
# ============================================================

engine = RiskEngineV3()


# ============================================================
# SESSION MONITOR
# ============================================================

monitor = SessionMonitor()

monitor.start()


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_USER_ID = "9001"

AUDIT_LOG = "audit_log.csv"


# ============================================================
# V3 KEYSTROKE AGENT
#
# IMPORTANT:
# The platform does NOT create its own keyboard listener.
#
# KeystrokeAgentV3 owns:
#     keyboard capture
#     feature extraction
#     normalization
#     Siamese inference
#
# Platform owns:
#     context
#     risk
#     alerts
#     decisions
# ============================================================

_agent = KeystrokeAgentV3(
    user_id=DEFAULT_USER_ID,
    platform_url="http://127.0.0.1:8000"
)


# ============================================================
# PLATFORM STATE
# ============================================================

_latest_agent_score = {}

platform_alerts = {}

recent_events = deque(
    maxlen=50
)


layer_states = {

    1: {
        "name":
            "Network Guardian",
        "status":
            "monitoring",
        "threat_count":
            0,
        "color":
            "teal",
    },

    2: {
        "name":
            "Ransomware Killer",
        "status":
            "monitoring",
        "threat_count":
            0,
        "color":
            "red",
    },

    3: {
        "name":
            "Zero-Trust Auth",
        "status":
            "active",
        "threat_count":
            0,
        "color":
            "purple",
    },

    4: {
        "name":
            "Content Threat Det.",
        "status":
            "monitoring",
        "threat_count":
            0,
        "color":
            "amber",
    },

}


# ============================================================
# AUDIT LOG
# ============================================================

AUDIT_HDRS = [

    "timestamp",

    "event_type",

    "user_id",

    "layer",

    "risk_percent",

    "decision",

    "context",

    "active_alerts",

    "details",

]

_audit_lock = threading.Lock()


def write_audit(
    event_type,
    user_id=None,
    layer=None,
    risk_percent=None,
    decision=None,
    context=None,
    active_alerts=None,
    details=None,
):

    row = {

        "timestamp":
            datetime.now().isoformat(),

        "event_type":
            event_type or "",

        "user_id":
            str(user_id)
            if user_id is not None
            else "",

        "layer":
            str(layer)
            if layer is not None
            else "",

        "risk_percent":
            str(risk_percent)
            if risk_percent is not None
            else "",

        "decision":
            decision or "",

        "context":
            context or "",

        "active_alerts":
            ",".join(
                active_alerts
            )
            if active_alerts
            else "",

        "details":
            details or "",

    }


    exists = os.path.exists(
        AUDIT_LOG
    )


    with _audit_lock:

        with open(
            AUDIT_LOG,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=AUDIT_HDRS
            )

            if not exists:

                writer.writeheader()

            writer.writerow(row)


# ============================================================
# JSON CLEANING
# ============================================================

def clean_for_json(obj):

    import math

    try:

        import numpy as np

        if isinstance(
            obj,
            np.integer
        ):
            return int(obj)

        if isinstance(
            obj,
            np.floating
        ):

            if (
                np.isnan(obj)
                or np.isinf(obj)
            ):

                return None

            return float(obj)

        if isinstance(
            obj,
            np.ndarray
        ):

            return clean_for_json(
                obj.tolist()
            )

        if isinstance(
            obj,
            np.bool_
        ):

            return bool(obj)

    except Exception:
        pass


    if isinstance(
        obj,
        dict
    ):

        return {
            str(k):
                clean_for_json(v)
            for k, v in obj.items()
        }


    if isinstance(
        obj,
        (list, tuple)
    ):

        return [
            clean_for_json(v)
            for v in obj
        ]


    if isinstance(
        obj,
        float
    ):

        if (
            math.isnan(obj)
            or math.isinf(obj)
        ):

            return None

        return obj


    if not isinstance(
        obj,
        (
            str,
            int,
            bool,
            type(None)
        )
    ):

        return str(obj)


    return obj


# ============================================================
# EVENT SYSTEM
# ============================================================

def add_event(
    layer_num,
    event_type,
    message,
    severity="info"
):

    if layer_num not in layer_states:

        layer_num = 3


    event = {

        "timestamp":
            datetime.now().isoformat(),

        "layer":
            layer_num,

        "type":
            event_type,

        "message":
            message,

        "severity":
            severity,

    }


    recent_events.appendleft(
        event
    )


    layer_states[
        layer_num
    ][
        "last_event"
    ] = datetime.now().isoformat()


    return event


# ============================================================
# ACTION DERIVATION
#
# The agent supplies behavioral evidence.
#
# Context/action intelligence belongs to
# the platform, not the biometric model.
# ============================================================

def derive_action_from_context(
    context
):

    context = (
        str(context or "")
        .strip()
        .lower()
    )


    if context in {
        "financial",
        "financial_transaction",
    }:

        return "FINANCIAL_TRANSACTION"


    if context in {
        "privileged",
        "admin",
        "administrator",
    }:

        return "ADMIN"


    if context in {
        "sensitive_access",
    }:

        return "ACCOUNT_SETTINGS"


    return "VIEW_ACCOUNT"


# ============================================================
# BEHAVIORAL SCORE PROCESSOR
# ============================================================

def process_agent_result(
    result: dict
):

    global _latest_agent_score


    status = result.get(
        "status"
    )


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    if status == "warming_up":
        _latest_agent_score = result
        return result

# --------------------------------------------------------
    # Enrollment
    # --------------------------------------------------------

    if status == "enrolled":
        _latest_agent_score = result
        add_event(3, "enrolled", f"User {DEFAULT_USER_ID} enrolled (Siamese V3)", "info")
        write_audit("enroll", user_id=DEFAULT_USER_ID, layer=3, details="siamese_bilstm_v3")
        return result

    # Accept inference even if status missing (old payloads)
    has_similarity = (
        "similarity" in result
        or "cosine_similarity" in result
    )

    if status != "inference_complete" and not has_similarity:
        _latest_agent_score = result
        return result

    # force status for downstream
    if status != "inference_complete":
        result = {**result, "status": "inference_complete"}

    similarity = float(
        result.get(
            "similarity",
            result.get(
                "cosine_similarity",
                0.0
            )
        )
    )


    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context = getattr(
        monitor,
        "current_context",
        "normal_browsing"
    )


    # --------------------------------------------------------
    # Action
    # --------------------------------------------------------

    action = derive_action_from_context(
        context
    )


    # --------------------------------------------------------
    # ZERO TRUST DECISION
    #
    # This is the only place where the
    # behavioral score becomes a risk decision.
    # --------------------------------------------------------

    decision = engine.evaluate(

        behavioral_similarity=
            similarity,

        context=
            context,

        action=
            action,

        user_id=
            DEFAULT_USER_ID,

        session_id=
            "live",

    )


    # --------------------------------------------------------
    # Combine biometric + risk evidence
    # --------------------------------------------------------

    combined = {

        **result,

        "decision":
            decision.get(
                "decision"
            ),

        "risk_decision":
            decision.get(
                "decision"
            ),

        "risk_percent":
            int(
                round(
                    float(
                        decision.get(
                            "overall_risk",
                            0
                        )
                    )
                    * 100
                )
            ),

        "overall_risk":
            decision.get(
                "overall_risk"
            ),

        "risk_level":
            decision.get(
                "risk_level"
            ),

        "adaptive_threshold":
            decision.get(
                "adaptive_threshold"
            ),

        "requires_monitoring":
            decision.get(
                "requires_monitoring"
            ),

        "behavioral_similarity":
            similarity,

        "behavioral_risk":
            decision.get(
                "behavioral_risk"
            ),

        "alert_risk":
            decision.get(
                "alert_risk"
            ),

        "context_risk":
            decision.get(
                "context_risk"
            ),

        "action":
            action,

        "context":
            context,

        "received_at":
            datetime.now().isoformat(),

    }


    _latest_agent_score = combined


    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    dec = decision.get(
        "decision",
        "ALLOW"
    )


    risk_pct = int(
        float(
            decision.get(
                "overall_risk",
                0
            )
        )
        * 100
    )


    write_audit(

        "agent_score",

        user_id=
            DEFAULT_USER_ID,

        layer=3,

        risk_percent=
            risk_pct,

        decision=
            dec,

        context=
            context,

        active_alerts=
            list(
                engine.active_alerts.keys()
            ),

        details=(
            f"similarity="
            f"{similarity:.4f};"
            f"action={action}"
        ),

    )


    # --------------------------------------------------------
    # REAUTHENTICATION
    # --------------------------------------------------------

    if dec == "REAUTHENTICATE":

        add_event(

            3,

            "reauthenticate",

            (
                "Behavioral/contextual risk "
                "requires step-up authentication"
            ),

            "high"

        )

        layer_states[
            3
        ][
            "threat_count"
        ] += 1


    # --------------------------------------------------------
    # MONITOR
    # --------------------------------------------------------

    elif dec == "MONITOR":

        add_event(

            3,

            "monitor",

            (
                "Behavioral/contextual risk "
                "requires increased monitoring"
            ),

            "medium"

        )


    return combined


# ============================================================
# START V3 AGENT
# ============================================================

@app.on_event("startup")
async def startup_event():

    print(
        "[Platform] Starting "
        "Siamese BiLSTM V3 agent..."
    )

    if not _agent.running:

        _agent.start()

    print(
        "[Platform] V3 behavioral agent started"
    )


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown_event():

    print(
        "[Platform] Stopping V3 agent..."
    )

    _agent.stop()


# ============================================================
# CENTRAL DASHBOARD
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def central_dashboard(
    request: Request
):

    path = pathlib.Path(
        "templates"
    ) / "central_dashboard.html"


    if not path.exists():

        return HTMLResponse(
            "<h2>"
            "central_dashboard.html not found"
            "</h2>"
        )


    return HTMLResponse(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# LAYER 3 DASHBOARD
# ============================================================

@app.get(
    "/layer3",
    response_class=HTMLResponse
)
async def layer3_dashboard(
    request: Request
):

    path = pathlib.Path(
        "templates"
    ) / "index.html"


    if not path.exists():

        return HTMLResponse(
            "<h2>"
            "index.html not found"
            "</h2>"
        )


    return HTMLResponse(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# PLATFORM STATUS
# ============================================================

@app.get(
    "/api/platform/status"
)
async def platform_status():

    try:

        engine.clear_expired_alerts()

    except Exception:
        pass


    now = datetime.now()


    active = {

        source: {

            "timestamp":
                value[
                    "timestamp"
                ].isoformat(),

            "severity":
                value[
                    "severity"
                ],

            "layer":
                value.get(
                    "layer",
                    0
                ),

            "message":
                value.get(
                    "message",
                    ""
                ),

        }

        for source, value
        in platform_alerts.items()

        if (
            now -
            value["timestamp"]
        ).total_seconds()
        < 600

    }


    count = len(
        active
    )


    threat_level = (

        "critical"
        if count >= 3

        else "high"
        if count >= 2

        else "medium"
        if count >= 1

        else "low"

    )


    return clean_for_json({

        "platform_threat_level":
            threat_level,

        "active_alerts":
            active,

        "n_active_alerts":
            count,

        "layer_states":
            layer_states,

        "session_monitor":
            monitor.get_status(),

        "latest_agent_score":
            _latest_agent_score,

        "agent_status":
            _agent.get_status(),

        "model":
            "Siamese_BiLSTM_V3",

        "old_if_svm":
            False,

        "behavioral_model":
            "siamese_bilstm_v3",

        "training_enabled":
            False,

        "weight_updates":
            False,

        "timestamp":
            datetime.now().isoformat(),

    })


# ============================================================
# EVENTS
# ============================================================

@app.get(
    "/api/platform/events"
)
async def platform_events():

    return clean_for_json({

        "events":
            list(
                recent_events
            )

    })


# ============================================================
# AUDIT LOG
# ============================================================

@app.get(
    "/api/platform/audit-log"
)
async def get_audit_log(
    limit: int = 50
):

    if not os.path.exists(
        AUDIT_LOG
    ):

        return {
            "entries": [],
            "total": 0
        }


    try:

        import pandas as pd

        df = pd.read_csv(
            AUDIT_LOG
        ).fillna("")


        entries = (
            df.tail(limit)
            .to_dict(
                orient="records"
            )
        )


        return {

            "entries":
                list(
                    reversed(
                        entries
                    )
                ),

            "total":
                len(df),

        }


    except Exception as exc:

        return {

            "entries": [],

            "error":
                str(exc),

        }


# ============================================================
# SESSION ACTIVITY
# ============================================================

@app.get(
    "/api/session-activity"
)
async def session_activity():

    return clean_for_json(
        monitor.get_status()
    )


# ============================================================
# BEHAVIORAL SCORE
#
# THIS IS THE MAIN V3 AGENT -> PLATFORM BRIDGE
# ============================================================

@app.post(
    "/api/behavioral-score"
)
async def behavioral_score(
    result: dict
):

    combined = (
        process_agent_result(
            result
        )
    )

    return clean_for_json(
        combined
    )


# ============================================================
# LATEST AGENT SCORE
# ============================================================

@app.get(
    "/api/agent-score"
)
async def get_agent_score():

    return clean_for_json(

        _latest_agent_score

        or {

            "status":
                "no_score_yet"

        }

    )


# ============================================================
# MANUAL KEY EVENT
#
# Useful when pynput is unavailable.
# Times must be milliseconds because the V3
# feature pipeline expects milliseconds.
# ============================================================

class KeyEventRequest(
    BaseModel
):

    key: str

    press_time: float

    release_time: float


@app.post(
    "/api/agent-event"
)
async def agent_event(
    req: KeyEventRequest
):

    # Convert manually supplied event
    # directly into the agent's internal
    # event representation.

    key_id = str(
        req.key
    )


    dwell = (
        req.release_time -
        req.press_time
    )


    if dwell <= 0:

        raise HTTPException(
            400,
            "release_time must be "
            "greater than press_time"
        )


    _agent.events.append({

        "PRESS_TIME":
            float(
                req.press_time
            ),

        "RELEASE_TIME":
            float(
                req.release_time
            ),

        "KEYSTROKE_ID":
            len(
                _agent.events
            ),

    })


    return {
        "status":
            "received",

        "timing_unit":
            "milliseconds",

        "key":
            key_id,

    }


# ============================================================
# KEYSTROKE DECISION BRIDGE
#
# Retained for compatibility with the
# separate integration client.
#
# It does NOT perform inference.
# It only forwards an already computed
# similarity into RiskEngineV3.
# ============================================================

class KeystrokeDecisionRequest(
    BaseModel
):

    user_id: str

    session_id: str

    behavioral_similarity: float

    context: str = (
        "normal_browsing"
    )

    action: str = (
        "VIEW_ACCOUNT"
    )


@app.post(
    "/api/platform/keystroke/decision"
)
async def keystroke_decision(
    req: KeystrokeDecisionRequest
):

    similarity = float(
        req.behavioral_similarity
    )


    if not (
        0.0 <= similarity <= 1.0
    ):

        raise HTTPException(
            400,
            "behavioral_similarity "
            "must be between 0 and 1"
        )


    decision = engine.evaluate(

        behavioral_similarity=
            similarity,

        context=
            req.context,

        action=
            req.action,

        user_id=
            req.user_id,

        session_id=
            req.session_id,

    )


    write_audit(

        "keystroke_decision",

        user_id=
            req.user_id,

        layer=3,

        risk_percent=
            int(
                decision.get(
                    "overall_risk",
                    0
                )
                * 100
            ),

        decision=
            decision.get(
                "decision"
            ),

        context=
            req.context,

        details=(
            f"similarity="
            f"{similarity:.4f}"
        ),

    )


    return clean_for_json(
        decision
    )


# ============================================================
# RESET PROFILE
# ============================================================

PIN_FILE = os.path.join(
    "models",
    "reset_pin.hash"
)


class ResetProfileRequest(
    BaseModel
):

    user_id: str = "9001"

    pin: str


def _hash_pin(
    pin: str
):

    return hashlib.sha256(
        pin.encode()
    ).hexdigest()


@app.post(
    "/api/reset-profile"
)
async def reset_profile(
    req: ResetProfileRequest
):

    if not (
        req.pin.isdigit()
        and len(req.pin) == 4
    ):

        raise HTTPException(
            400,
            "PIN must be exactly 4 digits"
        )


    pin_hash = _hash_pin(
        req.pin
    )


    os.makedirs(
        "models",
        exist_ok=True
    )


    if os.path.exists(
        PIN_FILE
    ):

        with open(
            PIN_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            stored = f.read().strip()


        if stored != pin_hash:

            write_audit(
                "profile_reset_denied",
                user_id=req.user_id,
                layer=3,
                details="bad PIN"
            )

            raise HTTPException(
                403,
                "Incorrect PIN"
            )


    else:

        with open(
            PIN_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                pin_hash
            )


    _agent.clear_reference()


    write_audit(
        "profile_reset",
        user_id=req.user_id,
        layer=3,
        details=(
            "siamese reference cleared"
        ),
    )


    add_event(
        3,
        "profile_reset",
        (
            f"User {req.user_id} "
            "reference cleared"
        ),
        "info"
    )


    return {

        "status":
            "reset",

        "user_id":
            req.user_id,

        "model":
            "Siamese_BiLSTM_V3",

    }


# ============================================================
# CROSS-LAYER ALERTS
# ============================================================

class PlatformAlertRequest(
    BaseModel
):

    source: str

    severity: str = "medium"

    layer: int = 0

    message: str = ""


@app.post(
    "/api/platform/alert"
)
async def platform_alert(
    req: PlatformAlertRequest
):

    platform_alerts[
        req.source
    ] = {

        "timestamp":
            datetime.now(),

        "severity":
            req.severity,

        "layer":
            req.layer,

        "message":
            req.message
            or
            f"Threat: {req.source}",

    }


    # Feed alert into RiskEngineV3.

    engine.receive_alert(
        req.source,
        req.severity
    )


    # Escalate Layer 3 visibility.

    if (
        req.severity
        in (
            "critical",
            "high"
        )
        or
        len(platform_alerts) >= 2
    ):

        if hasattr(
            monitor,
            "set_alert_override"
        ):

            monitor.set_alert_override(
                True
            )


        layer_states[
            3
        ][
            "status"
        ] = "elevated"


    if 1 <= req.layer <= 4:

        layer_states[
            req.layer
        ][
            "threat_count"
        ] += 1

        layer_states[
            req.layer
        ][
            "status"
        ] = "alert"


    add_event(

        req.layer or 3,

        "alert",

        req.message
        or
        f"Alert: {req.source}",

        req.severity

    )


    write_audit(

        "platform_alert",

        layer=
            req.layer,

        context=
            req.source,

        active_alerts=
            list(
                platform_alerts.keys()
            ),

        details=
            req.severity,

    )


    return {

        "status":
            "alert_received",

        "source":
            req.source,

        "platform_alerts":
            len(
                platform_alerts
            ),

        "layer3_notified":
            True,

    }


# ============================================================
# CLEAR SINGLE ALERT
# ============================================================

@app.delete(
    "/api/platform/alert/{source}"
)
async def clear_platform_alert(
    source: str
):

    platform_alerts.pop(
        source,
        None
    )


    engine.clear_alert(
        source
    )


    if (
        not platform_alerts
    ):

        if hasattr(
            monitor,
            "set_alert_override"
        ):

            monitor.set_alert_override(
                False
            )


        layer_states[
            3
        ][
            "status"
        ] = "active"


    return {

        "status":
            "cleared",

        "source":
            source,

    }


# ============================================================
# CLEAR ALL ALERTS
# ============================================================

@app.delete(
    "/api/platform/alerts"
)
async def clear_all_platform_alerts():

    platform_alerts.clear()

    engine.active_alerts.clear()


    if hasattr(
        monitor,
        "set_alert_override"
    ):

        monitor.set_alert_override(
            False
        )


    layer_states[
        3
    ][
        "status"
    ] = "active"


    for layer in (
        1,
        2,
        4
    ):

        layer_states[
            layer
        ][
            "status"
        ] = "monitoring"


    return {

        "status":
            "all_cleared"

    }


# ============================================================
# LIVENESS
# ============================================================

@app.post(
    "/api/liveness-check"
)
async def liveness_check():

    try:

        from face_liveness import (
            run_from_api
        )


        result = (
            run_from_api(
                timeout_seconds=25
            )
        )


        write_audit(

            "liveness",

            layer=3,

            decision=
                (
                    "passed"
                    if result.get(
                        "passed"
                    )
                    else "failed"
                ),

            details=
                result.get(
                    "reason"
                ),

        )


        add_event(

            3,

            "liveness",

            (
                "Liveness "
                +
                (
                    "PASSED"
                    if result.get(
                        "passed"
                    )
                    else "FAILED"
                )
            ),

            (
                "low"
                if result.get(
                    "passed"
                )
                else "high"
            ),

        )


        return result


    except Exception as exc:

        return {

            "passed":
                False,

            "reason":
                str(exc),

            "duration":
                0,

            "blink_count":
                0,

        }


# ============================================================
# LAYER 3 STATUS
# ============================================================

@app.get(
    "/api/status"
)
async def layer3_status():

    try:

        engine.clear_expired_alerts()

    except Exception:
        pass


    return clean_for_json({

        "active_alerts":
            list(
                engine.active_alerts.keys()
            ),

        "session_context":
            getattr(
                monitor,
                "current_context",
                "normal_browsing"
            ),

        "agent":
            _agent.get_status(),

        "latest_agent_score":
            _latest_agent_score,

        "model":
            "Siamese_BiLSTM_V3",

        "old_if_svm":
            False,

    })


# ============================================================
# OTHER LAYERS
# ============================================================

class LayerEventRequest(
    BaseModel
):

    event_type: str

    severity: str = "medium"

    details: str = ""

    source_ip: str = ""

    file_path: str = ""


# ------------------------------------------------------------
# LAYER 1
# ------------------------------------------------------------

@app.post(
    "/api/layer1/threat"
)
async def layer1_threat(
    req: LayerEventRequest
):

    layer_states[
        1
    ][
        "status"
    ] = "alert"


    layer_states[
        1
    ][
        "threat_count"
    ] += 1


    message = (
        f"Network Guardian: "
        f"{req.event_type}"
    )


    platform_alerts[
        "network_guardian"
    ] = {

        "timestamp":
            datetime.now(),

        "severity":
            req.severity,

        "layer":
            1,

        "message":
            message,

    }


    engine.receive_alert(
        "network_guardian",
        req.severity
    )


    add_event(
        1,
        req.event_type,
        message,
        req.severity
    )


    return {

        "status":
            "threat_raised",

        "message":
            message,

        "layer3_notified":
            True,

    }


# ------------------------------------------------------------
# LAYER 2
# ------------------------------------------------------------

@app.post(
    "/api/layer2/threat"
)
async def layer2_threat(
    req: LayerEventRequest
):

    layer_states[
        2
    ][
        "status"
    ] = "alert"


    layer_states[
        2
    ][
        "threat_count"
    ] += 1


    message = (
        f"Ransomware Killer: "
        f"{req.event_type}"
    )


    platform_alerts[
        "ransomware_killer"
    ] = {

        "timestamp":
            datetime.now(),

        "severity":
            req.severity,

        "layer":
            2,

        "message":
            message,

    }


    engine.receive_alert(
        "ransomware_killer",
        req.severity
    )


    add_event(
        2,
        req.event_type,
        message,
        req.severity
    )


    return {

        "status":
            "threat_raised",

        "message":
            message,

        "layer3_notified":
            True,

    }


# ------------------------------------------------------------
# LAYER 4
# ------------------------------------------------------------

@app.post(
    "/api/layer4/threat"
)
async def layer4_threat(
    req: LayerEventRequest
):

    layer_states[
        4
    ][
        "status"
    ] = "alert"


    layer_states[
        4
    ][
        "threat_count"
    ] += 1


    message = (
        f"Content Threat: "
        f"{req.event_type}"
    )


    platform_alerts[
        "content_threat_detection"
    ] = {

        "timestamp":
            datetime.now(),

        "severity":
            req.severity,

        "layer":
            4,

        "message":
            message,

    }


    engine.receive_alert(
        "content_threat_detection",
        req.severity
    )


    add_event(
        4,
        req.event_type,
        message,
        req.severity
    )


    return {

        "status":
            "threat_raised",

        "message":
            message,

        "layer3_notified":
            True,

    }


# ============================================================
# CONTEXT UPDATE
# ============================================================

class ContextUpdate(
    BaseModel
):

    url: str

    title: str = ""

    timestamp: str = ""


@app.post(
    "/api/context-update"
)
async def context_update(
    data: ContextUpdate
):

    try:

        monitor.update_from_url(
            data.url,
            data.title
        )


        return {

            "status":
                "received",

            "context":
                monitor.current_context,

        }


    except Exception as exc:

        return {

            "status":
                "error",

            "message":
                str(exc),

        }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn


    print(
        "\n" +
        "=" * 65
    )

    print(
        "NeuraShield Platform V3"
    )

    print(
        "Zero-Trust + Siamese BiLSTM"
    )

    print(
        "=" * 65
    )

    print(
        "Dashboard : "
        "http://localhost:8000"
    )

    print(
        "Layer 3   : "
        "http://localhost:8000/layer3"
    )

    print(
        "Status    : "
        "http://localhost:8000/api/platform/status"
    )

    print(
        "Model     : "
        "Siamese_BiLSTM_V3"
    )

    print(
        "Old IF/SVM: DISABLED"
    )

    print(
        "=" * 65
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
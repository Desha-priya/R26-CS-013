# ============================================================
# neurashield_platform_v3.py
#
# NeuraShield Unified Zero-Trust Platform V3 security add on
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
    HTTPException,
    Header,
    Depends
)

from fastapi.responses import HTMLResponse

from pydantic import BaseModel

from typing import Optional

import os
import json
import sys
import csv
import pathlib
import threading
import hashlib
import secrets

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
# API AUTHENTICATION
#
# Shared-secret API key for a single-operator local security
# console. This is NOT per-user auth or enterprise SSO — it is
# the minimum viable control to stop an unauthenticated actor
# on the network from injecting fake alerts, resetting the
# enrolled biometric profile, or forging risk decisions.
#
# Set NEURASHIELD_API_KEY before starting the platform in any
# non-throwaway environment. The default below is for local
# development only.
# ============================================================

NEURASHIELD_API_KEY = os.environ.get(
    "NEURASHIELD_API_KEY",
    "nue-ras-hie-ld"
)

if NEURASHIELD_API_KEY == "dev-only-change-me":

    print(
        "[SECURITY WARNING] NEURASHIELD_API_KEY is not set — "
        "using the default development key. Every state-changing "
        "endpoint is effectively unprotected. Set the "
        "NEURASHIELD_API_KEY environment variable before any "
        "real demo or deployment."
    )

else:

    print(
        "[SECURITY] API key loaded from NEURASHIELD_API_KEY "
        "environment variable."
    )


async def verify_api_key(
    x_api_key: str = Header(
        default=None,
        alias="X-API-Key"
    )
):

    if x_api_key != NEURASHIELD_API_KEY:

        raise HTTPException(
            status_code=401,
            detail=(
                "Missing or invalid X-API-Key header."
            )
        )

    return True


require_api_key = Depends(
    verify_api_key
)


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

_liveness_lock = threading.Lock()
_liveness_in_progress = False


def trigger_liveness_async(user_id):
    """
    Fires the webcam liveness challenge in a background thread
    when a REAUTHENTICATE decision occurs.

    Must NOT run inline inside the request handler that produced
    the decision: face_liveness.run_from_api() blocks on a
    subprocess for up to ~35s, and /api/behavioral-score is
    called synchronously by the agent every SCORE_INTERVAL
    seconds — blocking it would stall the whole FastAPI event
    loop (including the dashboard's own polling) for the
    duration of the challenge.

    Guarded so a REAUTHENTICATE decision repeated every 5s while
    the user hasn't cleared it yet doesn't spawn a second webcam
    window on top of the first.
    """

    global _liveness_in_progress

    with _liveness_lock:

        if _liveness_in_progress:
            return

        _liveness_in_progress = True

    def _run():

        global _liveness_in_progress

        try:

            from face_liveness import run_from_api

            live = run_from_api(
                timeout_seconds=25
            )

            passed = bool(live.get("passed"))
            identity_checked = bool(live.get("identity_checked"))
            face_similarity = live.get("face_similarity")

            write_audit(
                "liveness",
                user_id=user_id,
                layer=3,
                decision="passed" if passed else "failed",
                details=(
                    live.get("reason", "")
                    + (
                        f" | face_similarity={face_similarity}"
                        if face_similarity is not None
                        else ""
                    )
                ),
            )

            if passed:

                add_event(
                    3, "liveness", "Liveness PASSED — identity confirmed"
                    if identity_checked else
                    "Liveness PASSED (no face reference enrolled — identity not verified)",
                    "low",
                )

                # A genuine, confirmed step-up closes out the
                # elevated-risk episode this challenge was raised
                # for — reset the streak AND the visible threat
                # indicator so it doesn't sit elevated forever.
                engine.reset_consecutive(user_id)

                # Opens the post-verification grace period so the
                # user isn't re-challenged every ~10 seconds while
                # their typing sits near the threshold in a normal
                # context. Automatically suspended the moment
                # context becomes sensitive/financial — see
                # RiskEngineV3.evaluate().
                engine.mark_verified(
                    user_id,
                    identity_confirmed=identity_checked,
                )

                layer_states[3]["threat_count"] = max(
                    0,
                    layer_states[3]["threat_count"] - 1,
                )

            elif identity_checked:

                # Liveness was fine (a live human responded) but
                # the face did NOT match the enrolled reference —
                # a materially stronger signal than a mere
                # timeout, since it suggests a different person
                # is physically at the keyboard.
                add_event(
                    3, "liveness",
                    f"IDENTITY MISMATCH — face does not match enrolled reference "
                    f"(similarity={face_similarity})",
                    "high",
                )

            else:

                add_event(
                    3, "liveness",
                    f"Liveness FAILED — {live.get('reason', '')}",
                    "high",
                )

        except Exception as exc:

            write_audit(
                "liveness_error",
                user_id=user_id,
                layer=3,
                details=str(exc),
            )

        finally:

            with _liveness_lock:

                _liveness_in_progress = False

    threading.Thread(
        target=_run,
        daemon=True,
        name="LivenessTrigger",
    ).start()

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

    "prev_hash",

    "row_hash",

]

_audit_lock = threading.Lock()

_AUDIT_GENESIS_HASH = "0" * 64


def _load_last_audit_hash():
    """
    On startup, continue the existing hash chain instead of
    resetting it — otherwise every restart would silently break
    chain continuity even though nothing was tampered with.
    """

    if not os.path.exists(AUDIT_LOG):
        return _AUDIT_GENESIS_HASH

    try:

        with open(
            AUDIT_LOG,
            "r",
            newline="",
            encoding="utf-8"
        ) as f:

            rows = list(csv.DictReader(f))

        if not rows:
            return _AUDIT_GENESIS_HASH

        return rows[-1].get(
            "row_hash",
            _AUDIT_GENESIS_HASH
        ) or _AUDIT_GENESIS_HASH

    except Exception:

        return _AUDIT_GENESIS_HASH


_last_audit_hash = _load_last_audit_hash()


def _hash_row(prev_hash, row_data):
    """
    row_data must be the ordered, non-hash fields only.
    Canonical (sorted-key) JSON keeps the hash deterministic
    regardless of dict ordering.
    """

    canonical = json.dumps(
        row_data,
        sort_keys=True,
        separators=(",", ":")
    )

    payload = (prev_hash + canonical).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


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

    global _last_audit_hash

    core = {

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

        prev_hash = _last_audit_hash

        row_hash = _hash_row(prev_hash, core)

        row = {
            **core,
            "prev_hash": prev_hash,
            "row_hash": row_hash,
        }

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

        _last_audit_hash = row_hash


def verify_audit_chain():
    """
    Recomputes the hash chain from scratch and reports whether
    the on-disk audit log has been tampered with, and if so,
    the first row where the chain breaks.
    """

    if not os.path.exists(AUDIT_LOG):

        return {
            "verified": True,
            "rows_checked": 0,
            "message": "No audit log yet.",
        }

    with open(
        AUDIT_LOG,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        rows = list(csv.DictReader(f))

    expected_prev = _AUDIT_GENESIS_HASH

    for i, row in enumerate(rows):

        core = {
            k: row.get(k, "")
            for k in AUDIT_HDRS
            if k not in ("prev_hash", "row_hash")
        }

        if row.get("prev_hash", "") != expected_prev:

            return {
                "verified": False,
                "rows_checked": i,
                "message": (
                    f"Chain broken at row {i + 1}: "
                    "prev_hash does not match the previous "
                    "row's stored hash."
                ),
            }

        recomputed = _hash_row(expected_prev, core)

        if recomputed != row.get("row_hash", ""):

            return {
                "verified": False,
                "rows_checked": i,
                "message": (
                    f"Chain broken at row {i + 1}: "
                    "row content does not match its stored hash "
                    "— this row was likely edited after being "
                    "written."
                ),
            }

        expected_prev = recomputed

    return {
        "verified": True,
        "rows_checked": len(rows),
        "message": "Audit log integrity verified — no tampering detected.",
    }


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

    layer_states[
        layer_num
    ][
        "last_event_message"
    ] = message

    layer_states[
        layer_num
    ][
        "last_event_severity"
    ] = severity


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
    #
    # Merge rather than overwrite: a transient status must not
    # wipe out the last real decision/threshold the dashboard
    # is displaying. Only "status"/"buffer_size" change here.
    # --------------------------------------------------------

    if status == "warming_up":

        _latest_agent_score = {
            **_latest_agent_score,
            **result,
        }

        return result


    # --------------------------------------------------------
    # Enrollment
    # --------------------------------------------------------

    if status == "enrolled":

        _latest_agent_score = {
            **_latest_agent_score,
            **result,
        }

        add_event(
            3,
            "enrolled",
            (
                f"User {DEFAULT_USER_ID} "
                "enrolled with Siamese BiLSTM V3"
            ),
            "info"
        )

        write_audit(
            "enroll",
            user_id=DEFAULT_USER_ID,
            layer=3,
            details=(
                "siamese_bilstm_v3"
            ),
        )

        return result


    # --------------------------------------------------------
    # Only inference results continue.
    # --------------------------------------------------------

    if status != "inference_complete":

        _latest_agent_score = {
            **_latest_agent_score,
            **result,
        }

        return result


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

        "grace_active":
            decision.get(
                "grace_active"
            ),

        "grace_remaining_seconds":
            decision.get(
                "grace_remaining_seconds"
            ),

        "grace_suppressed_reauth":
            decision.get(
                "grace_suppressed_reauth"
            ),

        "identity_confirmed":
            decision.get(
                "identity_confirmed"
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

        trigger_liveness_async(
            DEFAULT_USER_ID
        )


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
    "/api/platform/status",
    dependencies=[require_api_key]
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
    "/api/platform/events",
    dependencies=[require_api_key]
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
    "/api/platform/audit-log",
    dependencies=[require_api_key]
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
            AUDIT_LOG,
            dtype=str,
            keep_default_na=False,
            on_bad_lines="skip",
            engine="python",
        )

        entries = (
            df.tail(limit)
            .to_dict(
                orient="records"
            )
        )

        # Normalize risk_percent to a real int for the frontend
        # (trend chart / gauge) instead of leaving it as a string
        # or letting pandas dtype-infer it inconsistently across
        # rows that mix numeric and empty values.
        for entry in entries:

            rp = entry.get("risk_percent", "")

            if rp not in (None, ""):

                try:

                    entry["risk_percent"] = int(float(rp))

                except (ValueError, TypeError):

                    pass

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
# AUDIT LOG INTEGRITY VERIFICATION
#
# Recomputes the SHA-256 hash chain from row 1 and reports
# whether the on-disk log matches what was actually written.
# This is what makes "tamper-evident" a true claim rather than
# just a label.
# ============================================================

@app.get(
    "/api/platform/audit-verify",
    dependencies=[require_api_key]
)
async def audit_verify():

    return verify_audit_chain()


# ============================================================
# SESSION ACTIVITY
# ============================================================

@app.get(
    "/api/session-activity",
    dependencies=[require_api_key]
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
    "/api/behavioral-score",
    dependencies=[require_api_key]
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
    "/api/agent-score",
    dependencies=[require_api_key]
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
    "/api/agent-event",
    dependencies=[require_api_key]
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
    "/api/platform/keystroke/decision",
    dependencies=[require_api_key]
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
    "/api/reset-profile",
    dependencies=[require_api_key]
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
    "/api/platform/alert",
    dependencies=[require_api_key]
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
    "/api/platform/alert/{source}",
    dependencies=[require_api_key]
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
    "/api/platform/alerts",
    dependencies=[require_api_key]
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
    "/api/liveness-check",
    dependencies=[require_api_key]
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


        if result.get("passed"):

            engine.reset_consecutive(
                DEFAULT_USER_ID
            )

            engine.mark_verified(
                DEFAULT_USER_ID,
                identity_confirmed=bool(
                    result.get("identity_checked")
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
# FACE ENROLLMENT
#
# One-time (or re-run any time) capture of the reference face
# used for identity matching during REAUTHENTICATE step-ups.
# Without this, liveness passes on ANY face — this endpoint is
# what makes it an identity check rather than just presence.
# ============================================================

@app.post(
    "/api/enroll-face",
    dependencies=[require_api_key]
)
async def enroll_face_endpoint():

    try:

        from face_liveness import (
            run_face_enrollment_from_api
        )

        result = run_face_enrollment_from_api(
            timeout_seconds=20
        )

        write_audit(
            "face_enroll",
            user_id=DEFAULT_USER_ID,
            layer=3,
            decision="saved" if result.get("saved") else "failed",
            details=result.get("reason", ""),
        )

        add_event(
            3,
            "face_enroll",
            (
                "Face reference saved"
                if result.get("saved")
                else f"Face enrollment failed — {result.get('reason', '')}"
            ),
            "info" if result.get("saved") else "medium",
        )

        return result

    except Exception as exc:

        return {
            "saved": False,
            "reason": str(exc),
        }


@app.get(
    "/api/face-reference-status",
    dependencies=[require_api_key]
)
async def face_reference_status():

    try:

        from face_liveness import has_face_reference

        return {"enrolled": has_face_reference()}

    except Exception as exc:

        return {"enrolled": False, "error": str(exc)}


# ============================================================
# LAYER 3 STATUS
# ============================================================

@app.get(
    "/api/status",
    dependencies=[require_api_key]
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
    "/api/layer1/threat",
    dependencies=[require_api_key]
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
    "/api/layer2/threat",
    dependencies=[require_api_key]
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
    "/api/layer4/threat",
    dependencies=[require_api_key]
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
    "/api/context-update",
    dependencies=[require_api_key]
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
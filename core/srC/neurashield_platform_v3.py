# NeuraShield Unified Zero-Trust Platform V3 security add on 
from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    Header,
    Depends
)

from fastapi.responses import HTMLResponse

from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from typing import Optional, Union

import os
import json
import sys
import csv
import pathlib
import threading
import time
import hashlib
import secrets
import re

from datetime import datetime, timedelta

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

import logging
from logging.handlers import RotatingFileHandler

from ns_paths import (
    data_path,
    resource_path,
    get_api_key,
    api_key_file,
)


# ============================================================
# PERSISTENT LOG FILE
#
# Everything printed to the console is also written to
# <data folder>/neurashield.log (1 MB x 3 rotating files), so a
# windowed exe (which has no console) still leaves diagnostics.
# Window titles and full URLs are never logged (privacy).
# ============================================================

class _Tee:
    def __init__(self, stream, logger):
        self._stream = stream
        self._logger = logger
        self._buf = ""

    def write(self, text):
        try:
            if self._stream is not None:
                self._stream.write(text)
        except Exception:
            pass
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._logger.info(line.rstrip())
        return len(text)

    def flush(self):
        try:
            if self._stream is not None:
                self._stream.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _setup_file_logging():
    try:
        handler = RotatingFileHandler(
            data_path("neurashield.log"),
            maxBytes=1_000_000, backupCount=3, encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s")
        )
        file_logger = logging.getLogger("neurashield.file")
        file_logger.setLevel(logging.INFO)
        file_logger.propagate = False
        if not file_logger.handlers:
            file_logger.addHandler(handler)
        if not isinstance(sys.stdout, _Tee):
            sys.stdout = _Tee(sys.stdout, file_logger)
            sys.stderr = _Tee(sys.stderr, file_logger)
    except Exception as exc:
        print(f"[Platform] File logging unavailable: {exc}")


_setup_file_logging()


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="NeuraShield Platform V3 — Siamese BiLSTM"
)

# ============================================================
# CORS
#
# DELETE is allowed because the alert-clearing endpoints use it. Extra
# origins (for example the core dashboard once it has its own port) can
# be added without editing the code:
#   NEURASHIELD_CORS_ORIGINS=http://localhost:9000,http://127.0.0.1:9000

_EXTRA_CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("NEURASHIELD_CORS_ORIGINS", "").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:5050",
        "http://127.0.0.1:5050",
    ] + _EXTRA_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.middleware("http")
async def no_cache_api_responses(request, call_next):
    """
    The dashboard polls the same URLs (e.g. /api/platform/audit-log
    ?limit=100) every few seconds. Without an explicit no-store
    directive, some browsers can serve a cached copy of a GET
    response instead of hitting the server — showing stale audit
    logs / status even though the backend has newer data. This
    forces every /api/ response to be treated as always-fresh.
    """

    response = await call_next(request)

    if request.url.path.startswith("/api/"):

        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"

    return response


# ============================================================
# RATE LIMITING

RATE_LIMIT_WINDOW_SECONDS = 10
# Four layer dashboards and the core dashboard all poll from 127.0.0.1, so
# they share one client address. The earlier limit of 60 requests per 10 s
# could be used up by dashboard polling alone, and the other layers' alert
# posts then received 429. Alert ingestion now has its own bucket, and the
# general limit is higher. It still stops a runaway local client.
RATE_LIMIT_MAX_REQUESTS = 200
RATE_LIMIT_INGEST_MAX_REQUESTS = 100

_rate_limit_buckets = {}  # {client_ip: [timestamp, timestamp, ...]}
_rate_limit_lock = threading.Lock()


@app.middleware("http")
async def rate_limit(request, call_next):

    # Exempt static/doc routes - only /api/ endpoints do real work
    # (writes, model inference, alert processing) worth protecting.
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    path = request.url.path
    is_ingest = (
        path.startswith("/api/platform/alert")
        or re.match(r"^/api/layer[1-4]/threat$", path) is not None
    )
    limit = (
        RATE_LIMIT_INGEST_MAX_REQUESTS
        if is_ingest
        else RATE_LIMIT_MAX_REQUESTS
    )

    with _rate_limit_lock:
        bucket = _rate_limit_buckets.setdefault(
            (client_ip, "ingest" if is_ingest else "general"), []
        )
        # Drop timestamps outside the current window before counting -
        # keeps the bucket small and the check accurate as time passes.
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        bucket[:] = [t for t in bucket if t > cutoff]

        if len(bucket) >= limit:
            # NOTE: raising HTTPException directly from @app.middleware is not reliably converted into a clean JSON response by
            # FastAPI's exception handlers (that mechanism is built for route/dependency-level exceptions) - it can surface as an
            # unhandled 500 instead of the intended 429. Returning a JSONResponse directly here is the reliable way to short-
            # circuit a request from inside HTTP middleware.
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: max {limit} "
                        f"requests per {RATE_LIMIT_WINDOW_SECONDS}s. Try again shortly."
                    )
                },
            )

        bucket.append(now)

    return await call_next(request)


# ============================================================
# RISK ENGINE
# ============================================================

engine = RiskEngineV3()


# ============================================================
# API AUTHENTICATION

# No built-in default key. Order: NEURASHIELD_API_KEY, then the key file in
# the data folder, otherwise a random key is generated once and stored.
# Other layers, the browser extension and the dashboards must use the same
# value (set NEURASHIELD_API_KEY for all of them, or read the key file).
NEURASHIELD_API_KEY = get_api_key()

# ============================================================
# BUILD MARKER
#
# Prints loudly at startup so you can immediately confirm which
# version of this file is actually running, just by looking at
# the console. Bump the date whenever you replace this file.
# ============================================================

PLATFORM_BUILD = "2026-10-12-integration-prep"

print("=" * 60)
print(f"[NeuraShield Platform] Build: {PLATFORM_BUILD}")
print("=" * 60)

if os.environ.get("NEURASHIELD_API_KEY"):

    print(
        "[SECURITY] API key loaded from NEURASHIELD_API_KEY "
        "environment variable."
    )

else:

    print(
        "[SECURITY] API key loaded from the key file: "
        f"{api_key_file()}"
    )

# Debug switch for developer tests. When off (the default), behavioural
# evidence is accepted only from the agent that this process started, and
# the manual key-event endpoint is disabled.
DEBUG_ENDPOINTS = os.environ.get(
    "NEURASHIELD_DEBUG_ENDPOINTS", "0"
).lower() in ("1", "true", "on", "yes")

# Secret shared only between this process and the agent it creates.
AGENT_TOKEN = secrets.token_urlsafe(24)

PORT = int(os.environ.get("NEURASHIELD_PORT", "8000"))


async def verify_api_key(
    x_api_key: str = Header(
        default=None,
        alias="X-API-Key"
    )
):

    if x_api_key is None or not secrets.compare_digest(x_api_key, NEURASHIELD_API_KEY):

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


async def verify_agent_token(
    x_agent_token: str = Header(
        default=None,
        alias="X-Agent-Token"
    )
):
    """
    Endpoints that accept behavioural evidence require the token that only
    the in-process agent knows. Knowing the API key (which the dashboard
    page exposes to any local client) is not enough to post a forged
    similarity score. Developer tests can set NEURASHIELD_DEBUG_ENDPOINTS=1.
    """

    if DEBUG_ENDPOINTS:
        return True

    if x_agent_token is None or not secrets.compare_digest(
        x_agent_token, AGENT_TOKEN
    ):

        raise HTTPException(
            status_code=403,
            detail="Behavioural evidence is accepted only from the local agent."
        )

    return True


require_agent_token = Depends(
    verify_agent_token
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

# Written at run time, so it lives in the data folder (see ns_paths.py).
AUDIT_LOG = data_path("audit_log.csv")


# ============================================================
# PIN / PASSKEY — PERSISTENT STORAGE (SQLite)
#============

import sqlite3

AUTH_DB_PATH = data_path("neurashield_auth.db")

_DEFAULT_PIN = "246810"

MIN_PIN_LENGTH = 6


def _pin_problem(pin):
    """Return a reason string if a NEW pin is unacceptable, else None."""
    if not pin or not pin.isdigit() or not (MIN_PIN_LENGTH <= len(pin) <= 8):
        return f"PIN must be {MIN_PIN_LENGTH} to 8 digits."
    if len(set(pin)) == 1:
        return "PIN is too simple (repeated digit)."
    if pin in "0123456789" or pin in "9876543210":
        return "PIN is too simple (sequence)."
    if pin == _DEFAULT_PIN:
        return "Choose a PIN different from the default."
    return None

# PINs are stored as salted PBKDF2-HMAC-SHA256. A stored value in the old
# format (plain SHA-256, 64 hex characters) is still accepted and is
# upgraded to the new format after its first successful verification.
_PBKDF2_ITERATIONS = 200_000


def _hash_pin_secure(pin, salt=None, iterations=_PBKDF2_ITERATIONS):

    salt = salt or secrets.token_bytes(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        str(pin).encode("utf-8"),
        salt,
        iterations,
    )

    return f"pbkdf2${iterations}${salt.hex()}${digest.hex()}"


def _verify_pin_hash(pin, stored):

    if not stored:
        return False

    if stored.startswith("pbkdf2$"):

        try:

            _, iterations, salt_hex, digest_hex = stored.split("$")

            digest = hashlib.pbkdf2_hmac(
                "sha256",
                str(pin).encode("utf-8"),
                bytes.fromhex(salt_hex),
                int(iterations),
            )

        except (ValueError, TypeError):

            return False

        return secrets.compare_digest(digest.hex(), digest_hex)

    legacy = hashlib.sha256(
        str(pin).encode("utf-8")
    ).hexdigest()

    return secrets.compare_digest(legacy, stored)


def _db_conn():
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """
    Creates the auth_config table if it doesn't exist, and seeds
    it with the default PIN (246810) on first run — the same
    documented default as before, just now persisted and
    changeable rather than baked into an env var.
    """

    conn = _db_conn()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            pin_hash TEXT NOT NULL,
            is_first_launch INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    existing = conn.execute(
        "SELECT id FROM auth_config WHERE id = 1"
    ).fetchone()

    if existing is None:

        now = datetime.now().isoformat()

        conn.execute(
            "INSERT INTO auth_config (id, pin_hash, is_first_launch, "
            "created_at, updated_at) VALUES (1, ?, 1, ?, ?)",
            (_hash_pin_secure(_DEFAULT_PIN), now, now),
        )

        conn.commit()

    conn.close()


# ============================================================
# PRIVACY CONSENT (GDPR / PDPA)
#
# Keystroke dynamics and the enrolled face reference are both
# biometric data — "special category" data under GDPR Article 9,
# and personal data under PDPA-style regimes. Processing it
# requires an explicit, informed basis (consent here), a clear
# notice of what's collected and why, and a way to withdraw.
# Tracked in the same local SQLite DB — no data ever leaves this
# machine, which is itself part of the compliance story
# (data minimisation / on-premises processing).
# ============================================================

PRIVACY_NOTICE_VERSION = "1.0"


def init_privacy_db():

    conn = _db_conn()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS privacy_consent (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            accepted INTEGER NOT NULL DEFAULT 0,
            accepted_at TEXT,
            withdrawn_at TEXT,
            notice_version TEXT
        )
        """
    )

    existing = conn.execute(
        "SELECT id FROM privacy_consent WHERE id = 1"
    ).fetchone()

    if existing is None:

        conn.execute(
            "INSERT INTO privacy_consent (id, accepted, notice_version) "
            "VALUES (1, 0, ?)",
            (PRIVACY_NOTICE_VERSION,),
        )

        conn.commit()

    conn.close()


def get_privacy_status():

    conn = _db_conn()

    row = conn.execute(
        "SELECT accepted, accepted_at, withdrawn_at, notice_version "
        "FROM privacy_consent WHERE id = 1"
    ).fetchone()

    conn.close()

    if row is None:

        return {
            "accepted": False,
            "accepted_at": None,
            "notice_version": PRIVACY_NOTICE_VERSION,
        }

    # If the notice text has been updated since they last accepted
    # (version mismatch), treat consent as needing to be re-given —
    # standard practice when the terms of processing change.
    accepted = bool(row["accepted"]) and (
        row["notice_version"] == PRIVACY_NOTICE_VERSION
    )

    return {
        "accepted": accepted,
        "accepted_at": row["accepted_at"],
        "withdrawn_at": row["withdrawn_at"],
        "notice_version": PRIVACY_NOTICE_VERSION,
    }


def set_privacy_accepted():

    conn = _db_conn()

    conn.execute(
        "UPDATE privacy_consent SET accepted = 1, accepted_at = ?, "
        "withdrawn_at = NULL, notice_version = ? WHERE id = 1",
        (datetime.now().isoformat(), PRIVACY_NOTICE_VERSION),
    )

    conn.commit()
    conn.close()


def set_privacy_withdrawn():

    conn = _db_conn()

    conn.execute(
        "UPDATE privacy_consent SET accepted = 0, withdrawn_at = ? "
        "WHERE id = 1",
        (datetime.now().isoformat(),),
    )

    conn.commit()
    conn.close()


init_privacy_db()


def get_stored_pin_hash():

    conn = _db_conn()

    row = conn.execute(
        "SELECT pin_hash FROM auth_config WHERE id = 1"
    ).fetchone()

    conn.close()

    return row["pin_hash"] if row else _hash_pin_secure(_DEFAULT_PIN)


def is_first_launch():

    conn = _db_conn()

    row = conn.execute(
        "SELECT is_first_launch FROM auth_config WHERE id = 1"
    ).fetchone()

    conn.close()

    return bool(row["is_first_launch"]) if row else True


def is_pin_still_default():

    return _verify_pin_hash(_DEFAULT_PIN, get_stored_pin_hash())


def set_pin_hash(new_hash, clear_first_launch=True):

    conn = _db_conn()

    conn.execute(
        "UPDATE auth_config SET pin_hash = ?, "
        + ("is_first_launch = 0, " if clear_first_launch else "")
        + "updated_at = ? WHERE id = 1",
        (new_hash, datetime.now().isoformat()),
    )

    conn.commit()
    conn.close()


init_auth_db()


def check_pin(submitted_pin):
    """
    True if the submitted PIN matches the stored PIN. A stored hash in the
    old SHA-256 format is upgraded to PBKDF2 on the first success.
    """

    stored = get_stored_pin_hash()

    if not _verify_pin_hash(submitted_pin, stored):
        return False

    if not stored.startswith("pbkdf2$"):

        set_pin_hash(
            _hash_pin_secure(submitted_pin),
            clear_first_launch=False,
        )

    return True


# ============================================================
# STEP-UP / SESSION-LOCK CONFIGURATION
# ============================================================

MAX_STEPUP_FAILURES = 3

POST_FACE_PIN_RISK_THRESHOLD = 0.60

_session_lock_state = {
    "enabled": os.environ.get(
        "NEURASHIELD_SESSION_LOCK", "off"
    ).lower() in ("1", "true", "on", "yes")
}


def session_lock_enabled():
    return _session_lock_state["enabled"]


# ============================================================
# V3 KEYSTROKE AGENT

_agent = KeystrokeAgentV3(
    user_id=DEFAULT_USER_ID,
    platform_url=f"http://127.0.0.1:{PORT}",
    agent_token=AGENT_TOKEN,
)


def _start_capture():
    """Starts keystroke capture. Called only once consent is on record."""

    try:

        if not _agent.running:

            _agent.start()

        return True

    except Exception as exc:

        write_audit(
            "capture_error",
            user_id=DEFAULT_USER_ID,
            layer=3,
            details=str(exc),
        )

        return False


def _stop_capture():
    """Stops capture and discards buffered typing events."""

    if _agent.running:

        _agent.stop()

    _agent.clear_buffers()


# ============================================================
# PLATFORM STATE
# ============================================================

_latest_agent_score = {}

# Decision of the previous scored window, used to log a new episode once
# and not once per 5-second window.
_last_decision = {"value": None}

_liveness_lock = threading.Lock()
_liveness_in_progress = False

# Live authentication state for the dashboard.
_auth_state = {
    "state": "authenticated",   # authenticated | pin_required | challenge | pending | locked
    "method": None,             # face | pin | liveness
    "since": None,
    "consecutive_failures": 0,
}

# Whether this user has already had a face-confirmed pass THIS
# session (i.e. since the platform started, or since their last
# lock/reset). Used so a RENEWED anomaly after an already-
# verified session asks for the lighter-weight PIN instead of
# re-opening the webcam every time.
_face_verified_this_session = {}

# Throttles brute-force PIN guessing against /api/unlock, which
# previously had no rate limiting at all — unlike the normal
# step-up path, failed unlock attempts don't go through
# _register_stepup_failure (the session is already locked; that
# function itself is now a no-op while locked), so they need
# their own counter and cooldown.
_unlock_attempts = {"count": 0, "locked_until": None}

UNLOCK_THROTTLE_AFTER = 5
UNLOCK_THROTTLE_SECONDS = 30


# The same counter now also covers PIN guesses made through
# /api/reset-profile and /api/auth/change-pin.

def _pin_throttle_remaining():

    until = _unlock_attempts["locked_until"]

    if until and datetime.now() < until:

        return int((until - datetime.now()).total_seconds()) + 1

    return 0


def _register_pin_failure():

    _unlock_attempts["count"] += 1

    if _unlock_attempts["count"] >= UNLOCK_THROTTLE_AFTER:

        _unlock_attempts["locked_until"] = datetime.now() + timedelta(
            seconds=UNLOCK_THROTTLE_SECONDS
        )
        _unlock_attempts["count"] = 0


def _clear_pin_failures():

    _unlock_attempts["count"] = 0
    _unlock_attempts["locked_until"] = None


def _set_auth_state(state, method=None):
    _auth_state["state"] = state
    if method is not None:
        _auth_state["method"] = method
    _auth_state["since"] = datetime.now().isoformat()


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

                current_risk = float(
                    _latest_agent_score.get("overall_risk", 0.0) or 0.0
                )

                needs_second_factor = (
                    current_risk >= POST_FACE_PIN_RISK_THRESHOLD
                )

                add_event(
                    3, "liveness", "Liveness PASSED — identity confirmed"
                    if identity_checked else
                    "Liveness PASSED (no face reference enrolled — identity not verified)",
                    "low",
                )

                engine.reset_consecutive(user_id)

                layer_states[3]["threat_count"] = max(
                    0,
                    layer_states[3]["threat_count"] - 1,
                )

                _auth_state["consecutive_failures"] = 0

                if identity_checked:
                    _face_verified_this_session[str(user_id)] = True

                if needs_second_factor:

                    # Face proved identity, but risk is still high
                    # enough that a second factor is required
                    # before fully trusting the session. Grace is
                    # NOT opened yet — only a passed PIN will.
                    _set_auth_state("pin_required", method="face")

                    add_event(
                        3, "auth",
                        f"Face verified, but risk still elevated "
                        f"({int(current_risk*100)}%) — second factor (PIN) required",
                        "medium",
                    )

                else:

                    engine.mark_verified(
                        user_id,
                        identity_confirmed=identity_checked,
                    )

                    _set_auth_state(
                        "authenticated",
                        method="face" if identity_checked else "liveness",
                    )

            elif identity_checked:

                add_event(
                    3, "liveness",
                    f"IDENTITY MISMATCH — face does not match enrolled reference "
                    f"(similarity={face_similarity})",
                    "high",
                )

                _register_stepup_failure(user_id)

            else:

                add_event(
                    3, "liveness",
                    f"Liveness FAILED — {live.get('reason', '')}",
                    "high",
                )

                _register_stepup_failure(user_id)

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


def lock_workstation(reason="Repeated authentication failure"):
    """
    Locks the session. If session_lock_enabled() and running on
    Windows, locks the real OS session via LockWorkStation();
    otherwise performs an in-app lock only (auth_state -> locked).

    Idempotent: calling this while already locked is a no-op, so
    no call path can trigger a rapid loop of LockWorkStation()
    calls even if something upstream fails to check state first.
    """

    if _auth_state["state"] == "locked":
        return

    _set_auth_state("locked")
    engine.clear_verified(DEFAULT_USER_ID)
    _face_verified_this_session.pop(str(DEFAULT_USER_ID), None)

    add_event(3, "session_lock", f"SESSION LOCKED — {reason}", "high")
    write_audit(
        "session_lock",
        user_id=DEFAULT_USER_ID,
        layer=3,
        decision="locked",
        details=reason,
    )

    if session_lock_enabled() and sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.user32.LockWorkStation()
        except Exception as exc:
            write_audit(
                "session_lock_error",
                user_id=DEFAULT_USER_ID,
                layer=3,
                details=str(exc),
            )


def _register_stepup_failure(user_id):
    """
    Increments the consecutive-failure counter after a failed
    step-up. Once MAX_STEPUP_FAILURES is reached, locks the
    session. Otherwise leaves auth_state in 'challenge'.
    """

    if _auth_state["state"] == "locked":
        return

    _auth_state["consecutive_failures"] += 1
    fails = _auth_state["consecutive_failures"]

    if fails >= MAX_STEPUP_FAILURES:

        lock_workstation(reason=f"{fails} consecutive failed step-ups")

    else:

        _set_auth_state("challenge")

        add_event(
            3, "auth",
            f"Step-up failed ({fails}/{MAX_STEPUP_FAILURES}) — "
            "retry or use PIN fallback",
            "medium",
        )


def verify_pin(submitted_pin, is_renewed_check=False):
    """
    Verifies a submitted PIN against the stored (SQLite-backed)
    hash. On success, opens a grace period (identity-confirmed)
    and clears the failure streak. On failure, registers a
    step-up failure. is_renewed_check marks that this PIN was
    requested because behavior drifted again during an already-
    verified session (not a fresh, never-verified challenge).
    """

    if check_pin(submitted_pin):

        engine.reset_consecutive(DEFAULT_USER_ID)
        engine.mark_verified(DEFAULT_USER_ID, identity_confirmed=True)

        _auth_state["consecutive_failures"] = 0
        _set_auth_state("authenticated", method="pin")

        layer_states[3]["threat_count"] = max(
            0, layer_states[3]["threat_count"] - 1
        )

        add_event(
            3, "auth",
            "PIN verification PASSED"
            + (" (renewed drift check)" if is_renewed_check else ""),
            "low",
        )
        write_audit(
            "pin_auth", user_id=DEFAULT_USER_ID, layer=3,
            decision="passed", details="PIN fallback",
        )

        return {"passed": True, "reason": "PIN accepted"}

    add_event(3, "auth", "PIN verification FAILED", "high")
    write_audit(
        "pin_auth", user_id=DEFAULT_USER_ID, layer=3,
        decision="failed", details="Incorrect PIN",
    )

    _register_stepup_failure(DEFAULT_USER_ID)

    return {"passed": False, "reason": "Incorrect PIN"}


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

    # The agent is scored every 5 s. While the decision stays MONITOR or
    # REAUTHENTICATE the same message would be added every cycle and push
    # the other layers' events out of the 50-event list on the dashboard.
    # Repeats of these two event types within 60 s are dropped.
    if event_type in ("monitor", "reauthenticate") and recent_events:

        last = recent_events[0]

        if (
            last.get("type") == event_type
            and last.get("message") == message
            and last.get("layer") == layer_num
        ):

            try:

                age = (
                    datetime.now()
                    - datetime.fromisoformat(last["timestamp"])
                ).total_seconds()

            except Exception:

                age = 999

            if age < 60:

                return last


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

    # Drop expired alerts (and the under_attack override that depended on
    # them) before this window is evaluated.
    prune_expired_alerts()


    # --------------------------------------------------------
    # Warm-up
    #
    # Merge rather than overwrite: a transient status must not wipe out the last real decision/threshold the dashboard
    # is displaying. Only "status"/"buffer_size" change here.

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

    context = getattr(
        monitor,
        "current_context",
        "normal_browsing"
    )

    # --------------------------------------------------------
    # Action

    action = derive_action_from_context(
        context
    )

    # --------------------------------------------------------
    # ZERO TRUST DECISION
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

        if _auth_state["state"] == "locked":

            return combined

        add_event(

            3,

            "reauthenticate",

            (
                "Behavioral/contextual risk "
                "requires step-up authentication"
            ),

            "high"

        )

        if _last_decision["value"] != "REAUTHENTICATE":

            layer_states[
                3
            ][
                "threat_count"
            ] += 1

        if _face_verified_this_session.get(str(DEFAULT_USER_ID)):

            # This session already proved identity via face once.
            # A renewed drift doesn't need the full webcam dance
            # again — a lighter-weight PIN re-check is enough
            # second proof, and is faster/less intrusive for the
            # user. Only a hard lock (repeated PIN failures) would
            # ever fall back to requiring the camera again, via
            # the normal unlock flow.
            _set_auth_state("pin_required", method="pin")

            add_event(
                3, "auth",
                "Behavioral pattern drifted again after prior "
                "face verification — requesting PIN re-check "
                "instead of re-opening face verification",
                "medium",
            )

        else:

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


    _last_decision["value"] = dec

    return combined


# ============================================================
# START V3 AGENT
# ============================================================

@app.on_event("startup")
async def startup_event():

    # Keystroke capture (and automatic enrolment) starts only after the
    # privacy notice has been accepted. Before, the agent started at launch
    # regardless of consent.
    if get_privacy_status().get("accepted"):

        print("[Platform] Consent on record - starting V3 agent...")

        if _start_capture():

            print("[Platform] V3 behavioral agent started")

    else:

        print(
            "[Platform] Waiting for privacy consent - "
            "keystroke capture is OFF until the notice is accepted."
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


@app.get(
    "/api/version",
    dependencies=[require_api_key]
)
async def platform_version():
    """
    Quick sanity check: open this URL directly in a browser (with
    the API key set as a header via an extension, or just check
    the console log at startup) to confirm which build is
    actually running, without guessing.
    """

    return {
        "build": PLATFORM_BUILD,
        "features": [
            "csv_dictreader_audit_read",
            "pin_lock_backend",
            "sqlite_pin_storage",
            "gdpr_pdpa_privacy_notice",
            "normal_browsing_floor_070",
            "safe_score_after_face_verify",
            "renewed_drift_asks_pin",
            "unified_alert_ingestion",
            "alert_expiry_clears_override",
            "consent_gated_capture",
            "pin_pbkdf2",
            "agent_token",
        ],
    }


# ============================================================
# DASHBOARD KEY INJECTION
#
# The dashboards are served by this platform, so the key is put into the
# page when it is served. Pages may use the placeholder, or still contain
# the old development key as a literal; both are replaced, so a page that
# was written before the key became configurable keeps working.
# ============================================================

_LEGACY_DASHBOARD_KEY = "nue-ras-hie-ld"


def _inject_api_key(html):
    """
    INTEGRATION SECURITY FIX: the page used to be served to ANY local caller
    with the real platform key written into it. The dashboards are now only
    reachable through the NeuraShield console, whose proxy adds X-API-Key to
    every upstream call itself, so the key is no longer put into the page at
    all (the placeholder and the legacy literal are blanked).
    """

    return (
        html
        .replace("__NEURASHIELD_API_KEY__", "")
        .replace(_LEGACY_DASHBOARD_KEY, "")
    )


def _console_gate(request, layer_id):
    """Dashboard pages need the platform key. A person who opens this address
    directly in a browser has no key (and must not get one): send them to the
    NeuraShield console, which requires login and then shows this page with the
    key added server-side. Programmatic callers without a key get a plain 401."""

    key = request.headers.get("x-api-key")

    if key is not None and secrets.compare_digest(key, NEURASHIELD_API_KEY):
        return None

    if "text/html" in (request.headers.get("accept") or ""):

        from fastapi.responses import RedirectResponse

        base = os.environ.get(
            "NEURASHIELD_CONSOLE_URL", "http://127.0.0.1:9000"
        ).rstrip("/")

        return RedirectResponse(f"{base}/embed/{layer_id}", status_code=303)

    raise HTTPException(
        status_code=401,
        detail="Missing or invalid X-API-Key header."
    )


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

    gate = _console_gate(request, "zta")
    if gate is not None:
        return gate

    path = pathlib.Path(
        resource_path("templates")
    ) / "central_dashboard.html"


    if not path.exists():

        return HTMLResponse(
            "<h2>"
            "central_dashboard.html not found"
            "</h2>"
        )

    return HTMLResponse(
        _inject_api_key(
            path.read_text(encoding="utf-8")
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

    gate = _console_gate(request, "zta")
    if gate is not None:
        return gate

    path = pathlib.Path(
        resource_path("templates")
    ) / "index.html"


    if not path.exists():

        return HTMLResponse(
            "<h2>"
            "index.html not found"
            "</h2>"
        )


    return HTMLResponse(
        _inject_api_key(
            path.read_text(encoding="utf-8")
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

    prune_expired_alerts()


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
            {
                **_agent.get_status(),
                "consent_pending":
                    not get_privacy_status()["accepted"],
            },

        # The dashboard's alert feed reads this key. It was missing, so the
        # feed stayed empty even though /api/platform/events had the events.
        "recent_events":
            list(recent_events),

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

        with open(
            AUDIT_LOG,
            "r",
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(f)

            total_rows = 0
            bad_rows = 0
            recent = deque(maxlen=limit)

            for row in reader:

                total_rows += 1

                # A malformed/garbage row (e.g. a stray duplicate
                # header line, or a hand-edited line) won't have a
                # real ISO timestamp — skip it but count it, so any
                # silent data loss is visible via the response
                # instead of just vanishing or corrupting ordering.
                ts = row.get("timestamp")

                if not ts or "T" not in ts:

                    bad_rows += 1
                    continue

                recent.append(row)

        entries = list(recent)

        # Normalize risk_percent to a real int for the frontend
        # (trend chart / gauge) instead of leaving it as a string.
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
                total_rows,

            "skipped_rows":
                bad_rows,

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
    dependencies=[require_api_key, require_agent_token]
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
    dependencies=[require_api_key, require_agent_token]
)
async def agent_event(
    req: KeyEventRequest
):

    # Developer-only: lets a caller inject typing events, so it is
    # disabled unless NEURASHIELD_DEBUG_ENDPOINTS=1.
    if not DEBUG_ENDPOINTS:

        raise HTTPException(404, "Not found")

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


    _agent.add_manual_event(
        req.press_time,
        req.release_time
    )


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

class ResetProfileRequest(
    BaseModel
):

    user_id: Union[str, int] = "9001"

    pin: str


# Resetting the behavioural profile lets the next typist become the enrolled
# user, so it must be at least as hard as unlocking the session. The earlier
# version kept a separate 4-digit PIN in models/reset_pin.hash that the first
# caller set to any value, and it did not check the session PIN at all.
@app.post(
    "/api/reset-profile",
    dependencies=[require_api_key]
)
async def reset_profile(
    req: ResetProfileRequest
):

    if _auth_state["state"] == "locked":

        raise HTTPException(
            423,
            "Session is locked. Unlock it first."
        )

    wait = _pin_throttle_remaining()

    if wait:

        raise HTTPException(
            429,
            f"Too many failed attempts. Try again in {wait}s."
        )

    if not check_pin(req.pin):

        _register_pin_failure()

        write_audit(
            "profile_reset_denied",
            user_id=req.user_id,
            layer=3,
            details="bad PIN"
        )

        add_event(
            3,
            "profile_reset",
            "Profile reset refused (wrong PIN)",
            "high"
        )

        raise HTTPException(
            403,
            "Incorrect PIN"
        )

    _clear_pin_failures()

    _agent.clear_reference()

    # Enrolment after a reset must use typing from after the reset, not
    # whatever was already in the buffer.
    _agent.clear_buffers()

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


# ------------------------------------------------------------
# ONE INGESTION PATH FOR ALL ALERTS
#
# /api/platform/alert and /api/layer1|2|4/threat used to be three separate
# implementations that behaved differently: only the first raised the
# under_attack override and wrote an audit row, so the same alert changed
# the required similarity depending on which URL a layer happened to use.
# All of them now call ingest_alert().
# ------------------------------------------------------------

ALERT_EXPIRY_SECONDS = 600

VALID_SEVERITIES = ("low", "medium", "high", "critical")

# Canonical alert sources and the layer number each one belongs to.
SOURCE_LAYER = {
    "network_guardian": 1,
    "ransomware_killer": 2,
    "content_threat_detection": 4,
}

SOURCE_BY_LAYER = {v: k for k, v in SOURCE_LAYER.items()}

_SOURCE_ALIASES = {
    "network": "network_guardian",
    "networkguardian": "network_guardian",
    "layer1": "network_guardian",
    "layer_1": "network_guardian",
    "ransomware": "ransomware_killer",
    "ransomwarekiller": "ransomware_killer",
    "layer2": "ransomware_killer",
    "layer_2": "ransomware_killer",
    "content": "content_threat_detection",
    "content_threat": "content_threat_detection",
    "contentthreatdetection": "content_threat_detection",
    "multimodal": "content_threat_detection",
    "multimodal_content_threat_detection": "content_threat_detection",
    "layer4": "content_threat_detection",
    "layer_4": "content_threat_detection",
}


def canonical_source(source):
    """
    Lower-case, punctuation to underscores, known aliases mapped to the
    canonical name. The risk engine keeps one alert per source, so two
    spellings of the same layer ("Network Guardian" and "network_guardian")
    would have been counted as two alerts and raised the alert risk.
    """

    name = re.sub(
        r"[^a-z0-9]+", "_", str(source or "").strip().lower()
    ).strip("_") or "unknown"

    return _SOURCE_ALIASES.get(name, name)


def _refresh_layer_states():

    layers_with_alert = {
        value.get("layer")
        for value in platform_alerts.values()
    }

    for number in (1, 2, 4):

        layer_states[number]["status"] = (
            "alert" if number in layers_with_alert else "monitoring"
        )

    layer_states[3]["status"] = (
        "elevated" if getattr(monitor, "alert_override", False) else "active"
    )


def prune_expired_alerts():
    """
    Removes alerts older than ALERT_EXPIRY_SECONDS from the platform view as
    well as from the risk engine, and clears the under_attack override once
    no alert is left. Before, the engine forgot an alert after 600 s but
    the platform kept it, so the override and the layer status stayed on
    until someone cleared them by hand.
    """

    now = datetime.now()

    expired = [
        source
        for source, value in list(platform_alerts.items())
        if (now - value["timestamp"]).total_seconds() >= ALERT_EXPIRY_SECONDS
    ]

    try:

        engine.clear_expired_alerts()

    except Exception:
        pass

    if not expired:
        return

    for source in expired:

        platform_alerts.pop(source, None)
        engine.clear_alert(source)

        add_event(
            3, "alert_expired",
            f"Alert from {source} expired after {ALERT_EXPIRY_SECONDS}s",
            "info",
        )
        write_audit(
            "alert_expired", layer=3, context=source,
            details=f"expired after {ALERT_EXPIRY_SECONDS}s",
        )

    if not platform_alerts:

        monitor.set_alert_override(False)

    _refresh_layer_states()


def ingest_alert(source, severity, layer, message, event_type="alert"):

    prune_expired_alerts()

    src = canonical_source(source)

    sev = str(severity or "medium").strip().lower()

    if sev not in VALID_SEVERITIES:
        sev = "medium"

    # For the three known layers the layer number follows from the source,
    # so a wrong or missing "layer" value cannot attribute an alert to the
    # Authentication Layer (layer 3) by mistake.
    lyr = SOURCE_LAYER.get(
        src,
        layer if isinstance(layer, int) else 0
    )

    text = message or f"Threat: {src}"

    platform_alerts[src] = {
        "timestamp": datetime.now(),
        "severity": sev,
        "layer": lyr,
        "message": text,
    }

    # Feed the alert into the risk engine.
    engine.receive_alert(src, sev)

    # Escalate: a high/critical alert, or two or more active alerts, puts
    # the session context into under_attack.
    if sev in ("critical", "high") or len(platform_alerts) >= 2:

        monitor.set_alert_override(True)

    if 1 <= lyr <= 4:

        layer_states[lyr]["threat_count"] += 1

    _refresh_layer_states()

    add_event(lyr or 3, event_type, text, sev)

    write_audit(
        "platform_alert",
        layer=lyr,
        context=src,
        active_alerts=list(platform_alerts.keys()),
        details=sev,
    )

    return {"source": src, "severity": sev, "layer": lyr}


@app.post(
    "/api/platform/alert",
    dependencies=[require_api_key]
)
async def platform_alert(
    req: PlatformAlertRequest
):

    result = ingest_alert(
        req.source,
        req.severity,
        req.layer,
        req.message,
    )

    return {

        "status":
            "alert_received",

        "source":
            result["source"],

        "severity":
            result["severity"],

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

    src = canonical_source(source)

    for key in {src, source}:

        platform_alerts.pop(key, None)
        engine.clear_alert(key)

    if not platform_alerts:

        monitor.set_alert_override(False)

    _refresh_layer_states()

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

    monitor.set_alert_override(False)

    _refresh_layer_states()

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
def liveness_check():

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
# ============================================================

@app.post(
    "/api/enroll-face",
    dependencies=[require_api_key]
)
def enroll_face_endpoint():

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
# PIN / PASSKEY FALLBACK
# ============================================================

class PinRequest(BaseModel):
    pin: str


@app.post(
    "/api/verify-pin",
    dependencies=[require_api_key]
)
async def verify_pin_endpoint(req: PinRequest):

    if _auth_state["state"] == "locked":

        return {
            "passed": False,
            "reason": "Session is locked. Unlock required first.",
            "locked": True,
        }

    is_renewed = _auth_state.get("method") == "pin" and _auth_state["state"] == "pin_required"

    return verify_pin(req.pin, is_renewed_check=is_renewed)


# ============================================================
# LIVE AUTH STATE  (drives the dashboard badge)
# ============================================================

@app.get(
    "/api/auth-state",
    dependencies=[require_api_key]
)
async def auth_state():

    verification = engine.get_verification_status(DEFAULT_USER_ID)

    return {
        **_auth_state,
        "grace_active": verification["grace_active"],
        "grace_remaining_seconds": verification["grace_remaining_seconds"],
        "identity_confirmed": verification["identity_confirmed"],
        "session_lock_enabled": session_lock_enabled(),
        "max_failures": MAX_STEPUP_FAILURES,
        "face_verified_this_session": bool(
            _face_verified_this_session.get(str(DEFAULT_USER_ID))
        ),
    }


# ============================================================
# UNLOCK
# ============================================================

class UnlockRequest(BaseModel):
    pin: str


@app.post(
    "/api/unlock",
    dependencies=[require_api_key]
)
async def unlock_endpoint(req: UnlockRequest):

    now = datetime.now()

    if _unlock_attempts["locked_until"] and now < _unlock_attempts["locked_until"]:

        remaining = int(
            (_unlock_attempts["locked_until"] - now).total_seconds()
        )

        return {
            "unlocked": False,
            "reason": f"Too many failed attempts. Try again in {remaining}s.",
            "throttled": True,
        }

    if not check_pin(req.pin):

        _unlock_attempts["count"] += 1

        add_event(
            3, "auth",
            f"Unlock attempt FAILED (wrong PIN) "
            f"({_unlock_attempts['count']}/{UNLOCK_THROTTLE_AFTER})",
            "high",
        )
        write_audit(
            "unlock", user_id=DEFAULT_USER_ID, layer=3,
            decision="failed", details="Incorrect PIN on unlock attempt",
        )

        if _unlock_attempts["count"] >= UNLOCK_THROTTLE_AFTER:

            _unlock_attempts["locked_until"] = now + timedelta(
                seconds=UNLOCK_THROTTLE_SECONDS
            )
            _unlock_attempts["count"] = 0

            return {
                "unlocked": False,
                "reason": (
                    f"Too many failed attempts. "
                    f"Try again in {UNLOCK_THROTTLE_SECONDS}s."
                ),
                "throttled": True,
            }

        return {"unlocked": False, "reason": "Incorrect PIN"}

    _unlock_attempts["count"] = 0
    _unlock_attempts["locked_until"] = None

    _auth_state["consecutive_failures"] = 0
    engine.reset_consecutive(DEFAULT_USER_ID)
    engine.mark_verified(DEFAULT_USER_ID, identity_confirmed=True)
    _set_auth_state("authenticated", method="pin")
    _face_verified_this_session[str(DEFAULT_USER_ID)] = False

    add_event(3, "auth", "Session UNLOCKED via PIN", "low")
    write_audit(
        "unlock", user_id=DEFAULT_USER_ID, layer=3,
        decision="unlocked", details="PIN unlock",
    )

    return {"unlocked": True}


# ============================================================
# SESSION-LOCK ON/OFF SETTING  (live, no restart needed)
# ============================================================

class SessionLockSetting(BaseModel):
    enabled: bool


@app.get(
    "/api/settings/session-lock",
    dependencies=[require_api_key]
)
async def get_session_lock_setting():

    return {"enabled": session_lock_enabled()}


@app.post(
    "/api/settings/session-lock",
    dependencies=[require_api_key]
)
async def set_session_lock_setting(req: SessionLockSetting):

    _session_lock_state["enabled"] = bool(req.enabled)

    add_event(
        3, "settings",
        f"Windows session lock {'ENABLED' if req.enabled else 'disabled'}",
        "info",
    )
    write_audit(
        "settings_change", user_id=DEFAULT_USER_ID, layer=3,
        details=f"session_lock_enabled={req.enabled}",
    )

    return {"enabled": session_lock_enabled()}


# ============================================================
# PIN SETUP / CHANGE  (SQLite-backed, persists across restarts)
# ============================================================

@app.get(
    "/api/auth/pin-status",
    dependencies=[require_api_key]
)
async def pin_status():
    """
    Tells the dashboard whether the PIN is still the documented
    default (246810) — used to show a first-launch "set up your
    PIN" prompt without hard-blocking the rest of the app.
    """

    return {
        "is_default": is_pin_still_default(),
        "is_first_launch": is_first_launch(),
    }


class SetPinRequest(BaseModel):
    new_pin: str


@app.post(
    "/api/auth/set-pin",
    dependencies=[require_api_key]
)
async def set_pin_endpoint(req: SetPinRequest):
    """
    First-time PIN setup — only meaningful while the PIN is still
    the default. To CHANGE an already-customized PIN, use
    /api/auth/change-pin instead, which requires the current PIN.
    """

    # First-time setup only. The original code did not check this, so the
    # PIN could be replaced at any time without knowing the current one.
    if not is_pin_still_default():

        add_event(3, "auth", "PIN setup refused: PIN already configured", "high")
        write_audit(
            "pin_setup", user_id=DEFAULT_USER_ID, layer=3,
            decision="refused",
            details="PIN already configured; change-pin requires the current PIN",
        )

        return {
            "set": False,
            "reason": "PIN is already configured. Use change-pin with the current PIN.",
        }

    problem = _pin_problem(req.new_pin)

    if problem:

        return {"set": False, "reason": problem}

    new_hash = _hash_pin_secure(req.new_pin)
    set_pin_hash(new_hash, clear_first_launch=True)

    add_event(3, "auth", "PIN set for the first time", "info")
    write_audit(
        "pin_setup", user_id=DEFAULT_USER_ID, layer=3,
        details="Initial PIN configured",
    )

    return {"set": True}


class ChangePinRequest(BaseModel):
    current_pin: str
    new_pin: str


@app.post(
    "/api/auth/change-pin",
    dependencies=[require_api_key]
)
async def change_pin_endpoint(req: ChangePinRequest):
    """
    Changes an existing PIN. Always requires the CURRENT PIN,
    even if it's still the default — this is the one operation
    that should never be doable without proving you already know
    the current value.
    """

    if _pin_throttle_remaining():

        return {
            "changed": False,
            "reason": (
                "Too many failed attempts. Try again in "
                f"{_pin_throttle_remaining()}s."
            ),
            "throttled": True,
        }

    if not check_pin(req.current_pin):

        _register_pin_failure()

        add_event(3, "auth", "PIN change attempt FAILED (wrong current PIN)", "high")
        write_audit(
            "pin_change", user_id=DEFAULT_USER_ID, layer=3,
            decision="failed", details="Incorrect current PIN",
        )
        return {"changed": False, "reason": "Current PIN is incorrect."}

    _clear_pin_failures()

    problem = _pin_problem(req.new_pin)

    if problem:

        return {"changed": False, "reason": problem}

    if req.new_pin == req.current_pin:

        return {"changed": False, "reason": "New PIN must be different from the current one."}

    new_hash = _hash_pin_secure(req.new_pin)
    set_pin_hash(new_hash, clear_first_launch=True)

    add_event(3, "auth", "PIN changed successfully", "low")
    write_audit(
        "pin_change", user_id=DEFAULT_USER_ID, layer=3,
        decision="passed", details="PIN changed by user",
    )

    return {"changed": True}


# ============================================================
# PRIVACY CONSENT ENDPOINTS (GDPR / PDPA)
# ============================================================

@app.get(
    "/api/privacy/status",
    dependencies=[require_api_key]
)
async def privacy_status():

    return get_privacy_status()


@app.post(
    "/api/privacy/accept",
    dependencies=[require_api_key]
)
async def privacy_accept():

    set_privacy_accepted()

    # Recorded in the tamper-evident audit trail too — real
    write_audit(
        "privacy_consent", user_id=DEFAULT_USER_ID, layer=3,
        decision="accepted",
        details=f"Privacy notice v{PRIVACY_NOTICE_VERSION} accepted",
    )
    add_event(3, "privacy", "Privacy notice accepted", "info")

    if _start_capture():

        add_event(3, "privacy", "Keystroke capture started", "info")
        write_audit(
            "capture_started", user_id=DEFAULT_USER_ID, layer=3,
            details="Started after consent",
        )

    return get_privacy_status()


@app.post(
    "/api/privacy/withdraw",
    dependencies=[require_api_key]
)
async def privacy_withdraw():
    """
    Right to withdraw consent (GDPR Art. 7(3) / equivalent PDPA
    provisions). Stops keystroke capture. Does not automatically delete the enrolled
    profile — that remains a separate, explicit action (Settings
    → Reset behavioral profile) — but is recorded here and
    surfaced clearly in the UI so the person understands what
    withdrawing means for continued use.
    """

    set_privacy_withdrawn()

    write_audit(
        "privacy_consent", user_id=DEFAULT_USER_ID, layer=3,
        decision="withdrawn",
        details="User withdrew privacy consent",
    )
    add_event(3, "privacy", "Privacy consent WITHDRAWN", "medium")

    # Withdrawal now also stops capture and discards the buffered events.
    # Stored references are not deleted automatically (see the privacy
    # notice); that stays an explicit administrator action.
    _stop_capture()

    add_event(3, "privacy", "Keystroke capture stopped", "medium")
    write_audit(
        "capture_stopped", user_id=DEFAULT_USER_ID, layer=3,
        details="Stopped after consent withdrawal",
    )

    return get_privacy_status()


# ============================================================
# LAYER 3 STATUS
# ============================================================

@app.get(
    "/api/status",
    dependencies=[require_api_key]
)
async def layer3_status():

    prune_expired_alerts()


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

    message: str = ""

    source_ip: str = ""

    file_path: str = ""


def _layer_threat(layer, name, req):

    message = (req.message or "").strip()[:300] or f"{name}: {req.event_type}"

    ingest_alert(
        SOURCE_BY_LAYER[layer],
        req.severity,
        layer,
        message,
        event_type=req.event_type,
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
# LAYER 1
# ------------------------------------------------------------

@app.post(
    "/api/layer1/threat",
    dependencies=[require_api_key]
)
async def layer1_threat(
    req: LayerEventRequest
):

    return _layer_threat(1, "Network Guardian", req)


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

    return _layer_threat(2, "Ransomware Killer", req)


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

    return _layer_threat(4, "Content Threat", req)


# ------------------------------------------------------------
# PER-LAYER STATUS  (dashboard "View status" button)
# ------------------------------------------------------------

@app.get(
    "/api/layer{layer_num}/status",
    dependencies=[require_api_key]
)
async def layer_status(layer_num: int):

    if layer_num not in layer_states:

        raise HTTPException(404, "Unknown layer")

    prune_expired_alerts()

    source = SOURCE_BY_LAYER.get(layer_num)

    return clean_for_json({
        "layer": layer_num,
        "state": layer_states[layer_num],
        "active_alert": (
            {
                "source": source,
                "severity": platform_alerts[source]["severity"],
                "message": platform_alerts[source].get("message", ""),
                "timestamp": platform_alerts[source]["timestamp"].isoformat(),
            }
            if source in platform_alerts else None
        ),
        "recent_events": [
            e for e in recent_events
            if e.get("layer") == layer_num
        ][:10],
    })


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
        "=" * 65
    )
    uvicorn.run(
        app,
        host="127.0.0.1", port=PORT, reload=False
    )
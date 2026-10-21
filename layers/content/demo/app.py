# =============================================================
# NeuraShield — Multimodal Content Threat Detection Layer
# =============================================================

import sys as _sys0
# Status lines use symbols (check marks, box drawing). When stdout/stderr are a
# file, a pipe or a packaged exe's console, Windows defaults to cp1252 and the
# first such print raises UnicodeEncodeError - which used to abort the model
# loading block. Never let a log line kill the layer.
for _stream in (_sys0.stdout, _sys0.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import sys
import os
import hmac
from collections import defaultdict, deque
from fastapi import Depends, Header
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Add project root to Python path so src/ imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Packaged exe: the install folder is read-only for a normal user, but the layer
# writes alerts/outputs relative to its working directory. Run from the layer's
# data folder (set by the supervisor); read-only resources (models, template)
# are resolved by absolute path from the bundle, see _resource() below.
if getattr(sys, "frozen", False):
    _data = os.environ.get("NEURASHIELD_DATA_DIR")
    if _data:
        os.makedirs(_data, exist_ok=True)
        os.chdir(_data)


def _resource(*parts):
    """Absolute path of a read-only layer file, from source or from the bundle."""
    roots = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots += [os.path.join(meipass, "layers", "content"), meipass]
    for r in roots:
        cand = os.path.join(r, *parts)
        if os.path.exists(cand):
            return cand
    return os.path.join(roots[0], *parts)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import prediction functions and model objects from fusion engine
from src.fusion.multimodal_fusion_v5 import (
    predict_text, predict_video, predict_audio,
    fuse_scores, save_alert, verify_audit_chain,
    acknowledge_alert, list_unacknowledged_alerts,
    text_model, video_model, audio_model
)
from src.common.security import get_api_key, key_fingerprint

# ── Security hardening (required by default, matches platform) ─
# .neurashield/api_key.local

_API_KEY = get_api_key()
print(f"[app] API key fingerprint: {key_fingerprint(_API_KEY)}")

# Debug mode: controls whether raw exception text is returned to the
# client. Default OFF so production doesn't leak internals; the demo
# can set NEURASHIELD_DEBUG=1 locally if verbose errors help.
_DEBUG = os.environ.get("NEURASHIELD_DEBUG", "0") == "1"


def require_api_key(x_api_key: str = Header(default=None)):
    """FastAPI dependency — always enforced (no opt-out via unset env var).
    Uses constant-time comparison (hmac.compare_digest) instead of `!=`
    so response timing can't be used to guess the key character-by-
    character — a standard hardening step for any real auth check."""
    if not x_api_key or not hmac.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return True


def safe_detail(e: Exception, generic: str = "Internal error") -> str:
    """Return the real error in debug mode, a generic message otherwise."""
    return str(e) if _DEBUG else generic


# ── Simple in-memory rate limiter ───────────────────────────────

_RATE_LIMIT = int(os.environ.get("NEURASHIELD_RATE_LIMIT_PER_MIN", "30"))
_rate_buckets: dict = defaultdict(deque)


def rate_limit(request_ip: str = "default"):
    now = time.time()
    bucket = _rate_buckets[request_ip]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= _RATE_LIMIT:
        raise HTTPException(status_code=429,
                            detail=f"Rate limit exceeded ({_RATE_LIMIT}/min). Try again shortly.")
    bucket.append(now)


# ── Application setup ─────────────────────────────────────────
app = FastAPI(
    title       = "NeuraShield — Content Threat Detection API",
    description = (
        "Multimodal AI threat detection layer for the NeuraShield autonomous "
        "cyber defence platform. Detects phishing emails, deepfake video, "
        "and cloned audio in real time using trained ML models. "
        "Part of a 4-layer autonomous cyber defence system."
    ),
    version  = "1.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc"
)

from src.watchers.email_routes import router as email_sources_router
app.include_router(email_sources_router)

# ── Automatic watchers (mail + call monitor) ────────────────────

from src.watchers.watcher_manager import (
    start_all_watchers, get_watcher_status,
    start_mail_watcher, stop_mail_watcher,
    start_call_monitor, stop_call_monitor,
)
from src.common import credential_store
from src.common.guidance import get_guidance

@app.on_event("startup")
async def _launch_watchers():
    try:
        start_all_watchers()
    except Exception as e:
       
        print(f"[WARN] watcher startup encountered an unexpected error: {e}")

_cors_env = os.environ.get("NEURASHIELD_CORS_ORIGINS", "*")
_cors_origins = ["*"] if _cors_env == "*" else [o.strip() for o in _cors_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory statistics tracker ─────────────────────────────

_stats = {
    "total_analyses"   : 0,
    "high_risk_count"  : 0,
    "medium_risk_count": 0,
    "low_risk_count"   : 0,
    "text_analyses"    : 0,
    "video_analyses"   : 0,
    "audio_analyses"   : 0,
    "server_start"     : datetime.now().isoformat()
}

def _make_request_id() -> str:
    """Generate a short unique ID for each incoming request."""
    return f"NS-{uuid4().hex[:8].upper()}"


# ── Upload validation (hardening) ──────────────────────────────
# Unvalidated uploads are a real production risk: unbounded file size
# can exhaust disk/memory, and unexpected extensions can be used to
# probe the server. Both are checked before any model touches the file.
MAX_VIDEO_MB = 200
MAX_AUDIO_MB = 25
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_AUDIO_EXT = {".flac", ".wav", ".mp3", ".m4a"}


async def _save_upload_validated(upload_file: UploadFile, allowed_ext: set,
                                 max_mb: int) -> str:
    """
    Save an UploadFile to a temp path after validating extension and size.
    Raises HTTPException(400) on any validation failure.
    Caller is responsible for deleting the returned temp path.
    """
    ext = os.path.splitext(upload_file.filename or "")[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed_ext)}"
        )

    max_bytes = max_mb * 1024 * 1024
    tmp = None
    written = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t:
            tmp = t.name
            while True:
                chunk = await upload_file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File exceeds {max_mb}MB limit."
                    )
                t.write(chunk)
        return tmp
    except HTTPException:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        raise
    except Exception as e:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

# ── GET / — Dashboard UI ──────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard(request: Request, x_api_key: str = Header(default=None)):
    """
    Serve the main NeuraShield threat detection dashboard.

    INTEGRATION SECURITY FIX: this page used to be handed to any local caller
    with the real X-API-Key written into it (window.NEURASHIELD_API_KEY). It now
    requires the key itself and no longer embeds it: the dashboard is reached
    through the NeuraShield console, whose proxy attaches X-API-Key to every
    request server-side, so the browser never needs the key. (/health stays
    open for the supervisor.)
    """
    if not x_api_key or not hmac.compare_digest(x_api_key, _API_KEY):
        # Opened directly in a browser: no key (and none is ever handed out).
        # Send the person to the NeuraShield console, which requires login.
        if "text/html" in (request.headers.get("accept") or ""):
            from fastapi.responses import RedirectResponse
            base = os.environ.get("NEURASHIELD_CONSOLE_URL", "http://127.0.0.1:9000").rstrip("/")
            return RedirectResponse(f"{base}/embed/content", status_code=303)
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    html_path = Path(_resource("template", "index.html"))
    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found in demo/ folder."
        )
    html = html_path.read_text(encoding="utf-8")
    # index.html carries the placeholder  window.NEURASHIELD_API_KEY = "__NEURASHIELD_API_KEY__";
    # Blank it: the page then sends no key of its own (AUTH_HEADERS = {}).
    html = html.replace("__NEURASHIELD_API_KEY__", "")
    return HTMLResponse(content=html)


# ── GET /health — System Health Check ─────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """
    System health check endpoint.
    Reports whether each ML model is loaded and ready.
    Other NeuraShield layers (Network Guardian, Zero-Trust Auth)
    can poll this endpoint before sending data for analysis.
    """
    return {
        "status"    : "operational",
        "timestamp" : datetime.now().isoformat(),
        "models_loaded": {
            "text_phishing_classifier" : text_model  is not None,
            "video_deepfake_classifier": video_model is not None,
            "audio_clone_classifier"   : audio_model is not None,
        },
        "system_info": {
            "api_version" : "1.0.0",
            "layer"       : "MultimodalContentThreatDetection",
            "project"     : "NeuraShield",
            "uptime_since": _stats["server_start"]
        }
    }


# ── GET /api/watchers/status — Automatic Monitor Status ────────
@app.get("/api/watchers/status", tags=["System"])
async def watchers_status():
    """
    Live status of the automatic watchers (mail watcher, real-time
    call monitor). Each watcher's state is one of:
      disabled  — not configured / not opted into, not an error
      starting  — thread launched, connecting/initializing
      running   — actively monitoring, stats reflect real activity
      error     — failed to start or lost connection; see 'detail'
    Open (no auth) — same pattern as /health and /api/stats, since
    this exposes only operational status, never credentials or content.
    """
    return get_watcher_status()


# ── Mail watcher control (dashboard-driven) ─────────────────────
class MailConfigRequest(BaseModel):
    host: str
    user: str
    password: str
    remember: bool = True   # persist encrypted for future starts


@app.get("/api/watchers/mail/config", tags=["System"])
async def mail_config_summary(_auth=Depends(require_api_key)):
    """
    Non-secret summary of the currently-configured mailbox (host,
    user — never the password), for the dashboard form to pre-fill.
    Requires auth since even the host/user is meaningful information.
    """
    return credential_store.mail_credentials_summary()


@app.post("/api/watchers/mail/start", tags=["System"])
async def mail_watcher_start(body: MailConfigRequest = None,
                             _auth=Depends(require_api_key)):
    """
    Start the mail watcher. Body is optional — omit it to (re)start
    using whatever mailbox is already configured (env vars or a
    previously-saved dashboard entry). Include it to configure and
    start a mailbox in one step; the password is encrypted at rest
    (see src/common/credential_store.py) and never echoed back.
    """
    if body:
        result = start_mail_watcher(body.host, body.user, body.password, body.remember)
    else:
        result = start_mail_watcher()
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["detail"])
    return result


@app.post("/api/watchers/mail/stop", tags=["System"])
async def mail_watcher_stop(_auth=Depends(require_api_key)):
    """Stop the mail watcher. Configured credentials are left in place
    (encrypted) so it can be restarted without re-entering them."""
    return stop_mail_watcher()


@app.delete("/api/watchers/mail/config", tags=["System"])
async def mail_config_clear(_auth=Depends(require_api_key)):
    """Stop the watcher (if running) and forget the stored mailbox
    credentials entirely."""
    stop_mail_watcher()
    removed = credential_store.clear_mail_credentials()
    return {"ok": True, "credentials_removed": removed}


# ── Call monitor control (dashboard-driven) ─────────────────────
@app.post("/api/watchers/call/start", tags=["System"])
async def call_monitor_start(_auth=Depends(require_api_key)):
    """
    Start the real-time call monitor (audio + video). No credentials
    needed — this captures local system audio/screen content, so
    it's a pure on/off toggle. Requires auth since it's a more
    invasive capability than mail polling.
    """
    result = start_call_monitor()
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["detail"])
    return result


@app.post("/api/watchers/call/stop", tags=["System"])
async def call_monitor_stop(_auth=Depends(require_api_key)):
    return stop_call_monitor()


# ── GET /api/audit/verify — GRC: Tamper-Evident Audit Log Check ─
@app.get("/api/audit/verify", tags=["System"])
async def audit_verify(_auth=Depends(require_api_key)):
    """
    Verifies the HMAC hash-chain across every saved alert file.
    Every alert is signed with a signature that includes the
    previous alert's signature, so editing, deleting, or reordering
    any past alert breaks the chain from that point forward — this
    is detectable, not just claimed. Requires auth (unlike /health)
    since it's a compliance-sensitive check, not routine status.
    """
    return verify_audit_chain()


# ── GET /api/alerts/unacknowledged — Live Notification Feed ─────
@app.get("/api/alerts/unacknowledged", tags=["Alerts"])
async def unacknowledged_alerts():
    """
    Recent HIGH/MEDIUM-risk alerts that haven't been acknowledged
    yet, each with plain-language guidance attached — this is what
    the dashboard's live notification banner polls. Open (no auth):
    the whole point is that a threat notification should be visible
    to whoever is using the dashboard, same reasoning as /health.
    """
    pending = list_unacknowledged_alerts()
    for a in pending:
        a["guidance"] = get_guidance(a)
    return {"count": len(pending), "alerts": pending}


class AcknowledgeRequest(BaseModel):
    note: str = ""


@app.post("/api/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge(alert_id: str, body: AcknowledgeRequest = None,
                      _auth=Depends(require_api_key)):
    """
    Records that the user has seen and acknowledged a specific alert.
    This becomes its own signed entry in the tamper-evident audit
    chain (see acknowledge_alert() in the fusion engine) — it does
    not modify the original alert. Requires auth: acknowledging is a
    user action worth attributing, not passive status viewing.
    """
    note = body.note if body else ""
    result = acknowledge_alert(alert_id, note=note)
    if not result["ok"]:
        raise HTTPException(status_code=404, detail=result["detail"])
    return result


# ── GET /api/stats — Live Statistics ──────────────────────────
@app.get("/api/stats", tags=["System"])
async def get_statistics():
    """
    Returns live threat detection statistics since server start.
    Dashboard polls this every few seconds to update the counters.
    """
    total = _stats["total_analyses"]
    return {
        "total_analyses"  : total,
        "threats_detected": _stats["high_risk_count"] + _stats["medium_risk_count"],
        "high_risk"       : _stats["high_risk_count"],
        "medium_risk"     : _stats["medium_risk_count"],
        "low_risk"        : _stats["low_risk_count"],
        "by_modality"     : {
            "text" : _stats["text_analyses"],
            "video": _stats["video_analyses"],
            "audio": _stats["audio_analyses"],
        },
        "threat_rate_pct" : round(
            (_stats["high_risk_count"] / total * 100) if total > 0 else 0, 1
        ),
        "server_start": _stats["server_start"]
    }


# ── GET /api/alerts — Alert History ───────────────────────────
@app.get("/api/alerts", tags=["Alerts"])
async def get_alert_history(limit: int = 10):
    """
    Returns recent saved alert JSON files from disk.
    Default: last 10 alerts. Maximum: 50.

    This endpoint allows other NeuraShield layers to query
    our recent threat decisions and adjust their own behaviour.
    For example: if we flag a video as HIGH_RISK, the Zero-Trust
    Authentication layer can trigger re-authentication.
    """
    limit     = min(limit, 50)
    alert_dir = Path("outputs/alerts")
    if not alert_dir.exists():
        return {"alerts": [], "total": 0}

    # Sort by modification time so newest alerts appear first
    files = sorted(
        alert_dir.glob("alert_*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )

    # FIX: this glob also matches acknowledgment records (event_type
    # 'alert_acknowledged') — they live in the same file-naming scheme
    # by design, so they're part of the same tamper-evident chain, but
    # they have no final_decision/final_risk_score/modality_scores.
    # The dashboard's alert-history view expects every entry to be a
    # full detection record (it crashed entirely — "Could not load
    # alerts" — the moment one acknowledgment record was mixed in, since
    # calling .replace() on a missing final_decision throws). Filter to
    # actual threat-detection records only; acknowledgments are still
    # fully visible via /api/audit/verify and on disk, just not here.
    alerts = []
    for f in files:
        if len(alerts) >= limit:
            break
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if data.get("event_type") == "alert_acknowledged":
            continue
        if "final_decision" not in data:
            continue
        alerts.append(data)

    total_on_disk = len(list(alert_dir.glob("alert_*.json")))
    return {"alerts": alerts, "total": total_on_disk}


# ── POST /api/analyse — Full Multimodal Analysis ──────────────
@app.post("/api/analyse", tags=["Analysis"])
async def analyse_multimodal(
    email_text : str        = Form(default=""),
    video_file : UploadFile = File(default=None),
    audio_file : UploadFile = File(default=None),
    _auth       = Depends(require_api_key),
    _rl         = Depends(rate_limit),
):
    """
    **Primary endpoint — Full Multimodal Threat Analysis.**

    Accepts any combination of text, video, and audio.
    Each provided modality is analysed by its trained model.
    All scores are fused into one final risk score (0.0–1.0).

    Risk thresholds:
    - ≥ 0.65 → HIGH_RISK   (immediate alert to other layers)
    - ≥ 0.40 → MEDIUM_RISK (flag for monitoring)
    - < 0.40 → LOW_RISK    (safe)

    The alert JSON is saved to outputs/alerts/ and also returned.
    """
    t0         = time.time()
    request_id = _make_request_id()
    used       = []

    # Default results — used when a modality is not provided
    text_res  = {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}
    video_res = {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}
    audio_res = {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}

    # ── Text ──────────────────────────────────────────────────
    if email_text and email_text.strip():
        try:
            text_res = predict_text(email_text)
            used.append("text")
            _stats["text_analyses"] += 1
        except Exception as e:
            text_res["error"] = str(e)

    # ── Video ─────────────────────────────────────────────────
    if video_file and video_file.filename:
        tmp = None
        try:
            tmp = await _save_upload_validated(
                video_file, ALLOWED_VIDEO_EXT, MAX_VIDEO_MB)
            video_res = predict_video(tmp)
            used.append("video")
            _stats["video_analyses"] += 1
        except HTTPException as e:
            # Bad video shouldn't kill the whole multimodal request —
            # record it as a soft error and keep going with text/audio.
            video_res["error"] = e.detail
        except Exception as e:
            video_res["error"] = str(e)
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    # ── Audio ─────────────────────────────────────────────────
    if audio_file and audio_file.filename:
        tmp = None
        try:
            tmp = await _save_upload_validated(
                audio_file, ALLOWED_AUDIO_EXT, MAX_AUDIO_MB)
            audio_res = predict_audio(tmp)
            used.append("audio")
            _stats["audio_analyses"] += 1
        except HTTPException as e:
            audio_res["error"] = e.detail
        except Exception as e:
            audio_res["error"] = str(e)
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    # ── Fuse and save ─────────────────────────────────────────
    alert      = fuse_scores(text_res, video_res, audio_res)
    saved_path = save_alert(alert)

    # Update session stats
    _stats["total_analyses"] += 1
    dec = alert["final_decision"]
    if   dec == "HIGH_RISK":   _stats["high_risk_count"]   += 1
    elif dec == "MEDIUM_RISK": _stats["medium_risk_count"] += 1
    else:                      _stats["low_risk_count"]    += 1

    return {
        "request_id"        : request_id,
        "timestamp"         : alert["timestamp"],
        "layer"             : alert["layer"],
        "final_risk_score"  : alert["final_risk_score"],
        "final_decision"    : alert["final_decision"],
        "recommendation"    : alert["recommendation"],
        "modalities_used"   : used,
        "modality_scores"   : alert["modality_scores"],
        "alert_saved_path"  : str(saved_path),
        "processing_time_ms": round((time.time() - t0) * 1000, 1)
    }


# ── POST /api/analyse/text ────────────────────────────────────
@app.post("/api/analyse/text", tags=["Analysis"])
async def analyse_text(email_text: str = Form(...),
                       _auth = Depends(require_api_key),
                       _rl = Depends(rate_limit)):
    """
    **Text-only — Phishing Email Detection.**
    Useful for quick email screening without video/audio context.
    """
    if not email_text.strip():
        raise HTTPException(status_code=400, detail="email_text cannot be empty.")
    try:
        result = predict_text(email_text)
        _stats["text_analyses"]  += 1
        _stats["total_analyses"] += 1
        return {"request_id": _make_request_id(), "modality": "text", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=safe_detail(e))


# ── POST /api/analyse/video ───────────────────────────────────
@app.post("/api/analyse/video", tags=["Analysis"])
async def analyse_video(video_file: UploadFile = File(...),
                        _auth = Depends(require_api_key),
                        _rl = Depends(rate_limit)):
    """
    **Video-only — Deepfake Video Detection.**
    Extracts visual features from the uploaded video (.mp4 / .avi)
    and classifies it as real or deepfake.
    """
    tmp = None
    try:
        tmp = await _save_upload_validated(
            video_file, ALLOWED_VIDEO_EXT, MAX_VIDEO_MB)
        result = predict_video(tmp)
        _stats["video_analyses"] += 1
        _stats["total_analyses"] += 1
        return {"request_id": _make_request_id(), "modality": "video",
                "filename": video_file.filename, **result}
    except HTTPException:
        raise   # preserve the real status code (e.g. 400 for bad upload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=safe_detail(e))
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ── POST /api/analyse/audio ───────────────────────────────────
@app.post("/api/analyse/audio", tags=["Analysis"])
async def analyse_audio(audio_file: UploadFile = File(...),
                        _auth = Depends(require_api_key),
                        _rl = Depends(rate_limit)):
    """
    **Audio-only — Voice Clone / Synthetic Speech Detection.**
    Extracts MFCC features from the uploaded audio (.flac / .wav / .mp3)
    and classifies it as real human speech or cloned/synthetic voice.
    """
    tmp = None
    try:
        tmp = await _save_upload_validated(
            audio_file, ALLOWED_AUDIO_EXT, MAX_AUDIO_MB)
        result = predict_audio(tmp)
        _stats["audio_analyses"] += 1
        _stats["total_analyses"] += 1
        return {"request_id": _make_request_id(), "modality": "audio",
                "filename": audio_file.filename, **result}
    except HTTPException:
        raise   # preserve the real status code (e.g. 400 for bad upload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=safe_detail(e))
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NeuraShield — Content Threat Detection API")
    print("=" * 60)
    print("  Dashboard : http://127.0.0.1:8001")
    print("  API Docs  : http://127.0.0.1:8001/docs")
    print("  Health    : http://127.0.0.1:8001/health")
    print("  Ctrl+C to stop")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)
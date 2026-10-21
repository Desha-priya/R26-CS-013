# =============================================================
# NeuraShield — Multimodal Content Threat Detection Layer
# Script  : demo/app.py
# Purpose : Production-grade FastAPI backend for the NeuraShield
#           Content Threat Detection Layer.

# API Endpoints:
#   GET  /                  → Serves the dashboard UI (index.html)
#   GET  /health            → System health + model status check
#   GET  /api/stats         → Live threat detection statistics
#   GET  /api/alerts        → Recent alert history (for other layers)
#   POST /api/analyse       → Full multimodal threat analysis
#   POST /api/analyse/text  → Text-only phishing detection
#   POST /api/analyse/video → Video-only deepfake detection
#   POST /api/analyse/audio → Audio-only voice clone detection
#   GET  /docs              → Auto-generated Swagger API docs
#
# Architecture note:
#   This file handles ONLY routing and request/response logic.
#   ML prediction logic lives in src/fusion/multimodal_fusion.py.
#   The UI lives in demo/index.html (served as a static file).
#   Keeping these separate makes each part independently testable.
#
# Run from project root:
#   pip install fastapi uvicorn python-multipart
#   python demo/app.py

# =============================================================

import sys
import os
import json
import tempfile
import shutil
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import prediction functions and model objects from fusion engine
from src.fusion.multimodal_fusion import (
    predict_text, predict_video, predict_audio,
    fuse_scores, save_alert,
    text_model, video_model, audio_model
)

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

# Allow browser requests from any origin (needed for local demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory statistics tracker ─────────────────────────────
# Counts every analysis run during this session.
# Displayed live on the dashboard. Resets on server restart.
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


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

# ── GET / — Dashboard UI ──────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Serve the main NeuraShield threat detection dashboard."""
    html_path = Path(__file__).parent.parent / "template" /"index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found in demo/ folder."
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


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
    )[:limit]

    alerts = []
    for f in files:
        try:
            alerts.append(json.loads(f.read_text()))
        except Exception:
            pass

    total_on_disk = len(list(alert_dir.glob("alert_*.json")))
    return {"alerts": alerts, "total": total_on_disk}


# ── POST /api/analyse — Full Multimodal Analysis ──────────────
@app.post("/api/analyse", tags=["Analysis"])
async def analyse_multimodal(
    email_text : str        = Form(default=""),
    video_file : UploadFile = File(default=None),
    audio_file : UploadFile = File(default=None)
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
            ext = os.path.splitext(video_file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t:
                shutil.copyfileobj(video_file.file, t)
                tmp = t.name
            video_res = predict_video(tmp)
            used.append("video")
            _stats["video_analyses"] += 1
        except Exception as e:
            video_res["error"] = str(e)
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    # ── Audio ─────────────────────────────────────────────────
    if audio_file and audio_file.filename:
        tmp = None
        try:
            ext = os.path.splitext(audio_file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t:
                shutil.copyfileobj(audio_file.file, t)
                tmp = t.name
            audio_res = predict_audio(tmp)
            used.append("audio")
            _stats["audio_analyses"] += 1
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
async def analyse_text(email_text: str = Form(...)):
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
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/analyse/video ───────────────────────────────────
@app.post("/api/analyse/video", tags=["Analysis"])
async def analyse_video(video_file: UploadFile = File(...)):
    """
    **Video-only — Deepfake Video Detection.**
    Extracts visual features from the uploaded video (.mp4 / .avi)
    and classifies it as real or deepfake.
    """
    tmp = None
    try:
        ext = os.path.splitext(video_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t:
            shutil.copyfileobj(video_file.file, t)
            tmp = t.name
        result = predict_video(tmp)
        _stats["video_analyses"] += 1
        _stats["total_analyses"] += 1
        return {"request_id": _make_request_id(), "modality": "video",
                "filename": video_file.filename, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ── POST /api/analyse/audio ───────────────────────────────────
@app.post("/api/analyse/audio", tags=["Analysis"])
async def analyse_audio(audio_file: UploadFile = File(...)):
    """
    **Audio-only — Voice Clone / Synthetic Speech Detection.**
    Extracts MFCC features from the uploaded audio (.flac / .wav / .mp3)
    and classifies it as real human speech or cloned/synthetic voice.
    """
    tmp = None
    try:
        ext = os.path.splitext(audio_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t:
            shutil.copyfileobj(audio_file.file, t)
            tmp = t.name
        result = predict_audio(tmp)
        _stats["audio_analyses"] += 1
        _stats["total_analyses"] += 1
        return {"request_id": _make_request_id(), "modality": "audio",
                "filename": audio_file.filename, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NeuraShield — Content Threat Detection API")
    print("=" * 60)
    print("  Dashboard : http://127.0.0.1:8000")
    print("  API Docs  : http://127.0.0.1:8000/docs")
    print("  Health    : http://127.0.0.1:8000/health")
    print("  Ctrl+C to stop")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
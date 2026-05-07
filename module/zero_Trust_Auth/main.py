# main.py - 
# NeuraShield - Zero-Trust Auth Layer
# FastAPI application - Python 3.14 compatible:
#   1. /api/enroll accepts ANY number of features (pads to 48) - safe for browser enrollment
#   2. /api/score accepts any feature length (pads to 48) - fixes live scoring
#   3. /api/replay uses use_window=False - fresh score each replay
#   4. /api/enroll-combined accepts keystroke + mouse features separately
#   5. /api/score-combined scores keystroke + mouse combined
#   6. /api/liveness-check triggers OpenCV face liveness on server webcam

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import pathlib
import sys
import os

# Define root path for the zero trust module
ROOT_PATH = pathlib.Path(__file__).parent.parent.parent

sys.path.append(os.path.dirname(__file__))
from risk_engine import RiskEngine

# -*- App setup -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-
app    = FastAPI(title="NeuraShield Zero-Trust Auth Layer")
engine = RiskEngine()

# -*- Load combined profiles for feature lookup -*--*--*--*--*--*--
PROFILES_FILE = ROOT_PATH / "module" / "zero_Trust_Auth" / "data_processing" / "user_behavioral_profiles_combined.csv"
profiles_df   = pd.read_csv(PROFILES_FILE)
FEATURE_COLS  = [c for c in profiles_df.columns if c != 'user']
N_FEATURES    = len(FEATURE_COLS)   # 48

# -*- Load feature weights once at startup -*--*--*--*--*--*--*--*--*-*
WEIGHTS_PATH    = ROOT_PATH / "models" / "zero_trust_auth" / "feature_weights_v2.pkl"
feature_weights = (
    joblib.load(WEIGHTS_PATH)
    if os.path.exists(WEIGHTS_PATH)
    else np.ones(N_FEATURES)
)
print(f"Features: {N_FEATURES} | Weights loaded: {len(feature_weights)}")


# -*- Helper: pad/trim any feature list to N_FEATURES -*--*--*--
def pad_features(raw: list) -> list:
    """Pad or trim to exactly N_FEATURES. Safe for any input length."""
    f = list(raw)[:N_FEATURES]
    f = f + [0.0] * (N_FEATURES - len(f))
    return f


# -*- Request / Response models -*--*--*--*--*--*--*--*--*--*--*--*--*--*--
class ScoreRequest(BaseModel):
    user_id:  int
    features: list[float]           # any length - padded internally
    context:  str = 'normal_browsing'

class AlertRequest(BaseModel):
    source:   str
    severity: str = 'medium'

class ReplayRequest(BaseModel):
    user_id:   int
    context:   str = 'normal_browsing'
    n_samples: int = 100

class EnrollRequest(BaseModel):
    user_id:  int
    features: list[float]           # browser keystroke features (15)

class CombinedEnrollRequest(BaseModel):
    user_id:        int
    ks_features:    list[float]     # 15 keystroke features from browser
    mouse_features: list[float]     # 33 mouse features from browser


# -*- Feature builder for dataset replay -*--*--*--*--*--*--*--*--*--*-
def get_user_features(user_id: int) -> Optional[list]:
    row = profiles_df[profiles_df['user'] == user_id]
    if row.empty:
        return None
    return row[FEATURE_COLS].values[0].tolist()


# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    html_path = ROOT_PATH / "module" / "zero_Trust_Auth" / "templates" / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            f"<h2 style='color:red;padding:40px'>templates/index.html not found"
            f"<br><small>Looking in: {html_path}</small></h2>"
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/users")
async def get_users():
    users = sorted(engine.user_profiles.keys())
    return {"users": users, "total": len(users)}


@app.post("/api/score")
async def score_session(req: ScoreRequest):
    """
    Score a live session.
    Accepts ANY number of features - pads/trims to 48 internally.
    Browser sends 15 keystroke features → safe.
    Dataset replay sends 48 → safe.
    use_window=True for live session smoothing.
    """
    features = pad_features(req.features)
    result   = engine.score_session(features, req.user_id, req.context,
                                    use_window=True)
    return result


@app.post("/api/replay")
async def replay_user(req: ReplayRequest):
    """
    Dataset replay - scores a real user from the combined CSV.
    use_window=False so each replay click gives a fresh independent score.
    """
    features = get_user_features(req.user_id)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"No profile data for user {req.user_id}"
        )
    result         = engine.score_session(features, req.user_id, req.context,
                                          use_window=False)
    result['mode'] = 'dataset_replay'
    return result


@app.post("/api/enroll")
async def enroll_user(req: EnrollRequest):
    """
    Enroll a live user from browser-captured KEYSTROKE features only.
    Pads to 48 features (mouse features set to zero).
    Use /api/enroll-combined when mouse data is also available.
    """
    features  = np.array(pad_features(req.features))
    fw        = np.array(feature_weights)[:N_FEATURES]
    weighted  = features * fw
    scaled    = engine.scaler.transform(weighted.reshape(1, -1))[0]

    engine.user_profiles[req.user_id] = {
        'features':      scaled.tolist(),
        'raw_features':  pad_features(req.features),
        'feature_names': FEATURE_COLS,
        'if_score':      float(engine.iso_forest.score_samples(scaled.reshape(1,-1))[0]),
        'svm_score':     float(engine.oc_svm.score_samples(scaled.reshape(1,-1))[0]),
    }
    engine.reset_user_window(req.user_id)

    if_s  = engine.user_profiles[req.user_id]['if_score']
    svm_s = engine.user_profiles[req.user_id]['svm_score']
    print(f"[ENROLL] User {req.user_id} | IF={if_s:.4f} | SVM={svm_s:.4f} | ks-only")

    return {
        "status":    "enrolled",
        "user_id":   req.user_id,
        "mode":      "keystroke_only",
        "if_score":  round(if_s, 4),
        "svm_score": round(svm_s, 4),
        "message":   "Profile built from keystroke features"
    }


@app.post("/api/enroll-combined")
async def enroll_combined(req: CombinedEnrollRequest):
    """
    Enroll a live user from BOTH keystroke + mouse features.
    Keystroke: 15 features (first block, matches training order)
    Mouse:     33 features (second block, matches training order)
    Combined:  48 features total - best possible live enrollment.
    """
    ks  = list(req.ks_features[:15])
    ms  = list(req.mouse_features[:33])
    ks  = ks  + [0.0] * (15 - len(ks))
    ms  = ms  + [0.0] * (33 - len(ms))
    combined = ks + ms   # exactly 48

    features  = np.array(combined)
    fw        = np.array(feature_weights)[:N_FEATURES]
    weighted  = features * fw
    scaled    = engine.scaler.transform(weighted.reshape(1, -1))[0]

    engine.user_profiles[req.user_id] = {
        'features':      scaled.tolist(),
        'raw_features':  combined,
        'feature_names': FEATURE_COLS,
        'if_score':      float(engine.iso_forest.score_samples(scaled.reshape(1,-1))[0]),
        'svm_score':     float(engine.oc_svm.score_samples(scaled.reshape(1,-1))[0]),
    }
    engine.reset_user_window(req.user_id)

    if_s  = engine.user_profiles[req.user_id]['if_score']
    svm_s = engine.user_profiles[req.user_id]['svm_score']
    print(f"[ENROLL] User {req.user_id} | IF={if_s:.4f} | SVM={svm_s:.4f} | combined")

    return {
        "status":         "enrolled",
        "user_id":        req.user_id,
        "mode":           "keystroke_and_mouse",
        "ks_features":    len(ks),
        "mouse_features": len(ms),
        "total_features": len(combined),
        "if_score":       round(if_s, 4),
        "svm_score":      round(svm_s, 4),
        "message":        "Profile built from keystroke + mouse features"
    }


@app.post("/api/score-combined")
async def score_combined(req: CombinedEnrollRequest):
    """
    Score a live session using BOTH keystroke + mouse features.
    Same logic as /api/score but accepts split input.
    """
    ks       = list(req.ks_features[:15])  + [0.0] * max(0, 15-len(req.ks_features))
    ms       = list(req.mouse_features[:33]) + [0.0] * max(0, 33-len(req.mouse_features))
    combined = ks + ms
    result   = engine.score_session(combined, req.user_id,
                                    getattr(req, 'context', 'normal_browsing'),
                                    use_window=True)
    return result


@app.post("/api/alert")
async def receive_alert(req: AlertRequest):
    result = engine.receive_alert(req.source, req.severity)
    return result


@app.delete("/api/alert/{source}")
async def clear_alert(source: str):
    if source in engine.active_alerts:
        del engine.active_alerts[source]
        return {"status": "cleared", "source": source}
    return {"status": "not_found", "source": source}


@app.get("/api/status")
async def system_status():
    engine.clear_expired_alerts()
    return {
        "active_alerts":      list(engine.active_alerts.keys()),
        "n_enrolled_users":   len(engine.user_profiles),
        "context_thresholds": engine.CONTEXT_THRESHOLDS,
        "alert_reductions":   engine.ALERT_REDUCTIONS,
        "alert_score_boosts": engine.ALERT_SCORE_BOOST,
    }


@app.get("/api/anomaly-users")
async def get_anomaly_users():
    profiles_data = joblib.load(ROOT_PATH / "models" / "zero_trust_auth" / "user_profiles_v2.pkl")
    results       = []
    for uid, data in profiles_data.items():
        features = np.array(data['features']).reshape(1, -1)
        pred     = engine.iso_forest.predict(features)[0]
        score    = float(engine.iso_forest.score_samples(features)[0])
        if pred == -1:
            results.append({"user_id": uid, "if_score": round(score, 4)})
    results.sort(key=lambda x: x['if_score'])
    return {"anomaly_users": results, "total": len(results)}


@app.post("/api/liveness-check")
async def liveness_check():
    """
    Triggers webcam face liveness check on the server machine.
    Requires face_liveness.py in the same folder.
    Webcam must be connected to the machine running this server.
    For demo: server and browser on same machine - works perfectly.
    """
    try:
        from face_liveness import detect_liveness
        result = detect_liveness(timeout_seconds=12)
        return {
            "passed":      result['passed'],
            "reason":      result['reason'],
            "duration":    result.get('duration', 0),
            "blink_count": result.get('blink_count', 0),
        }
    except ImportError:
        return {
            "passed":  False,
            "reason":  "face_liveness.py not found - place it in the same folder as main_v2.py",
            "duration": 0,
            "blink_count": 0,
        }
    except Exception as e:
        return {
            "passed":  False,
            "reason":  f"Liveness error: {str(e)}",
            "duration": 0,
            "blink_count": 0,
        }


# -*- Run -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--
if __name__ == "__main__":
    import uvicorn
    print("\nNeuraShield Zero-Trust Auth Layer starting...")
    print("Dashboard : http://localhost:8000")
    print("API docs  : http://localhost:8000/docs\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
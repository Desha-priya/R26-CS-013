# main.py
# NeuraShield - Zero-Trust Auth Layer

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os
import sys
import pathlib
from pathlib import Path


sys.path.append(os.path.dirname(__file__))
from risk_engine import RiskEngine

# ** App setup ********************************************-
app    = FastAPI(title="NeuraShield Zero-Trust Auth Layer")
engine = RiskEngine()

# ** Load aggregated keystroke profiles ********************

k_features = Path(__file__).resolve().parent / "zero_Trust_Auth" / "data_processing" / "keyboard_set" / "user_keystroke_profiles.csv"

KS_AGG_FILE = k_features

ks_agg_df   = None
if os.path.exists(KS_AGG_FILE):
    ks_agg_df = pd.read_csv(KS_AGG_FILE)
    print(f"Keystroke profiles loaded: {ks_agg_df.shape}")

# ** Load combined profiles for feature lookup ************-
PROFILES_FILE = "user_behavioral_profiles_combined.csv"
profiles_df   = pd.read_csv(PROFILES_FILE)
FEATURE_COLS  = [c for c in profiles_df.columns if c != 'user']

# ** Request models ****************************************
class ScoreRequest(BaseModel):
    user_id:  int
    features: list[float]
    context:  str = 'normal_browsing'

class AlertRequest(BaseModel):
    source:   str
    severity: str = 'medium'

class ReplayRequest(BaseModel):
    user_id:   int
    context:   str = 'normal_browsing'
    n_samples: int = 100

# ** Feature builder **************************************-
def build_features_from_raw(user_id: int, n_samples: int = 100) -> Optional[list]:
    user_row = profiles_df[profiles_df['user'] == user_id]
    if user_row.empty:
        return None
    return user_row[FEATURE_COLS].values[0].tolist()

# ** Routes ************************************************

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    html_path = pathlib.Path(__file__).parent / "templates" / "index.html"
    if not html_path.exists():
        return HTMLResponse(f"""
        <h2 style='font-family:sans-serif;color:red;padding:40px'>
        templates/index.html not found<br>
        <small style='color:#666'>Looking in: {html_path}</small>
        </h2>
        """)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/users")
async def get_users():
    users = sorted(engine.user_profiles.keys())
    return {"users": users, "total": len(users)}


@app.post("/api/score")
async def score_session(req: ScoreRequest):
    if len(req.features) != len(FEATURE_COLS):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_COLS)} features, got {len(req.features)}"
        )
    result = engine.score_session(req.features, req.user_id, req.context)
    return result


@app.post("/api/replay")
async def replay_user(req: ReplayRequest):
    features = build_features_from_raw(req.user_id, req.n_samples)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data for user {req.user_id}"
        )
    result = engine.score_session(features, req.user_id, req.context)
    result['mode'] = 'dataset_replay'
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
        "active_alerts":        list(engine.active_alerts.keys()),
        "n_enrolled_users":     len(engine.user_profiles),
        "context_thresholds":   engine.CONTEXT_THRESHOLDS,
        "alert_reductions":     engine.ALERT_REDUCTIONS,
    }


@app.get("/api/anomaly-users")
async def get_anomaly_users():
    profiles_data = joblib.load("models/user_profiles.pkl")
    results = []
    for uid, data in profiles_data.items():
        features = np.array(data['features']).reshape(1, -1)
        pred     = engine.iso_forest.predict(features)[0]
        score    = float(engine.iso_forest.score_samples(features)[0])
        if pred == -1:
            results.append({"user_id": uid, "if_score": round(score, 4)})
    results.sort(key=lambda x: x['if_score'])
    return {"anomaly_users": results, "total": len(results)}


# ** Run **************************************************-
if __name__ == "__main__":
    import uvicorn
    print("\nNeuraShield Zero-Trust Auth Layer starting...")
    print("Dashboard: http://localhost:8000")
    print("API docs:  http://localhost:8000/docs\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
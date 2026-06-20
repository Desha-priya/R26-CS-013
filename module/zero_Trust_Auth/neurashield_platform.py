# neurashield_platform.py
# NeuraShield — Unified Platform API
# Replaces main_v2.py as the main entry point for PP2
#
# Architecture:
#   - All 4 layers under one FastAPI app
#   - Layer 3 (Zero-Trust Auth) is fully implemented
#   - Layers 1, 2, 4 are simulated with realistic endpoints
#   - Central dashboard shows all layers in one view
#   - Session monitor auto-detects context from active window
#   - Per-user models used alongside global model
#   - Full audit log written to disk

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import os, sys, csv, pathlib, threading
from datetime import datetime
from collections import deque

sys.path.append(os.path.dirname(__file__))
from risk_engine import RiskEngine
from session_monitor import SessionMonitor

# ── App setup ─────────────────────────────────────────────
app    = FastAPI(title="NeuraShield Platform", version="2.0")
engine = RiskEngine()

# ── Session monitor ────────────────────────────────────────
monitor = SessionMonitor()
monitor.start()

# ── Load data ──────────────────────────────────────────────
PROFILES_FILE = "user_behavioral_profiles_combined.csv"
profiles_df   = pd.read_csv(PROFILES_FILE)
FEATURE_COLS  = [c for c in profiles_df.columns if c != 'user']
N_FEATURES    = len(FEATURE_COLS)

WEIGHTS_PATH    = "models/feature_weights_v2.pkl"
feature_weights = (
    joblib.load(WEIGHTS_PATH)
    if os.path.exists(WEIGHTS_PATH) else np.ones(N_FEATURES)
)

# ── Per-user models ────────────────────────────────────────
USER_MODELS_DIR = os.path.join("models", "user_models")
per_user_models = {}   # {uid: {'model': IF, 'scaler': scaler, 'stats': stats}}

def load_per_user_models():
    if not os.path.exists(USER_MODELS_DIR):
        print("[Platform] No per-user models found — run train_per_user_models.py")
        return
    count = 0
    for fname in os.listdir(USER_MODELS_DIR):
        if fname.endswith("_if.pkl"):
            uid = int(fname.replace("user_", "").replace("_if.pkl", ""))
            model_path  = os.path.join(USER_MODELS_DIR, f"user_{uid}_if.pkl")
            scaler_path = os.path.join(USER_MODELS_DIR, f"user_{uid}_scaler.pkl")
            stats_path  = os.path.join(USER_MODELS_DIR, f"user_{uid}_stats.pkl")
            try:
                per_user_models[uid] = {
                    'model':  joblib.load(model_path),
                    'scaler': joblib.load(scaler_path),
                    'stats':  joblib.load(stats_path) if os.path.exists(stats_path) else {},
                }
                count += 1
            except Exception as e:
                print(f"[Platform] Could not load model for user {uid}: {e}")
    print(f"[Platform] Loaded {count} per-user models")

load_per_user_models()

# ── Audit log ──────────────────────────────────────────────
AUDIT_LOG_FILE = "audit_log.csv"
AUDIT_HEADERS  = [
    'timestamp', 'event_type', 'user_id', 'layer',
    'risk_percent', 'decision', 'context', 'active_alerts', 'details'
]
_audit_lock = threading.Lock()

def write_audit(event_type, user_id=None, layer=None,
                risk_percent=None, decision=None,
                context=None, active_alerts=None, details=None):
    row = {
        'timestamp':     datetime.now().isoformat(),
        'event_type':    event_type,
        'user_id':       user_id or '',
        'layer':         layer or '',
        'risk_percent':  risk_percent or '',
        'decision':      decision or '',
        'context':       context or '',
        'active_alerts': ','.join(active_alerts) if active_alerts else '',
        'details':       details or '',
    }
    file_exists = os.path.exists(AUDIT_LOG_FILE)
    with _audit_lock:
        with open(AUDIT_LOG_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=AUDIT_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

# ── Platform-wide alert state ──────────────────────────────
# Tracks alerts from ALL layers
platform_alerts = {}   # {source: {timestamp, severity, layer, message}}

# ── Simulated layer states ─────────────────────────────────
layer_states = {
    1: {'name': 'Network Guardian',   'status': 'monitoring', 'last_event': None,
        'threat_count': 0, 'color': 'teal'},
    2: {'name': 'Ransomware Killer',  'status': 'monitoring', 'last_event': None,
        'threat_count': 0, 'color': 'red'},
    3: {'name': 'Zero-Trust Auth',    'status': 'active',     'last_event': None,
        'threat_count': 0, 'color': 'purple'},
    4: {'name': 'Content Threat Det.','status': 'monitoring', 'last_event': None,
        'threat_count': 0, 'color': 'amber'},
}

# Recent events feed for dashboard
recent_events = deque(maxlen=50)

def add_event(layer_num, event_type, message, severity='info'):
    event = {
        'timestamp': datetime.now().isoformat(),
        'layer':     layer_num,
        'type':      event_type,
        'message':   message,
        'severity':  severity,
    }
    recent_events.appendleft(event)
    layer_states[layer_num]['last_event'] = datetime.now().isoformat()
    return event


# ── Helpers ────────────────────────────────────────────────
def pad_features(raw):
    f = list(raw)[:N_FEATURES]
    return f + [0.0] * (N_FEATURES - len(f))

def get_user_features(user_id):
    row = profiles_df[profiles_df['user'] == user_id]
    return None if row.empty else row[FEATURE_COLS].values[0].tolist()

def get_per_user_score(user_id: int, raw_features: list) -> Optional[float]:
    """Score using per-user model if available. Returns 0-1 anomaly score."""
    if user_id not in per_user_models:
        return None
    try:
        m       = per_user_models[user_id]
        # Use first 13 features (window features) — same as training
        feats   = np.array(raw_features[:13]).reshape(1, -1)
        scaled  = m['scaler'].transform(feats)
        raw_sc  = float(m['model'].score_samples(scaled)[0])
        stats   = m['stats']
        mn, mx  = stats.get('min', -0.7), stats.get('max', -0.3)
        score   = float(np.clip((mx - raw_sc) / (mx - mn + 1e-6), 0.0, 1.0))
        return score
    except Exception:
        return None

def compliance_mode(context: str) -> dict:
    """Map context to compliance framework."""
    mapping = {
        'financial':        {'standard': 'PSD2',          'article': 'Article 97 — SCA',
                             'requirement': 'Strong Customer Authentication required'},
        'sensitive_access': {'standard': 'GDPR + NIST',   'article': 'Art.25 + SP800-207',
                             'requirement': 'Privacy by design + Zero Trust verification'},
        'normal_browsing':  {'standard': 'NIST SP800-207','article': 'Zero Trust Architecture',
                             'requirement': 'Continuous verification enforced'},
        'under_attack':     {'standard': 'ISO 27001',     'article': 'A.16 — Incident Response',
                             'requirement': 'Elevated authentication during active incident'},
    }
    return mapping.get(context, mapping['normal_browsing'])


# ══════════════════════════════════════════════════════════
# ROUTES — DASHBOARDS
# ══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def central_dashboard(request: Request):
    """Central platform dashboard — shows all 4 layers."""
    html_path = pathlib.Path(__file__).parent / "templates" / "central_dashboard.html"
    if not html_path.exists():
        return HTMLResponse("<h2 style='color:red;padding:40px'>central_dashboard.html not found</h2>")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/layer3", response_class=HTMLResponse)
async def layer3_dashboard(request: Request):
    """Layer 3 detailed dashboard — existing index.html."""
    html_path = pathlib.Path(__file__).parent / "templates" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h2 style='color:red;padding:40px'>index.html not found</h2>")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════
# ROUTES — PLATFORM STATUS
# ══════════════════════════════════════════════════════════

@app.get("/api/platform/status")
async def platform_status():
    """Full platform status — all layers + alerts + monitor."""
    engine.clear_expired_alerts()

    active = {k: {
        'timestamp': v['timestamp'].isoformat(),
        'severity':  v['severity'],
        'layer':     v.get('layer', 'unknown'),
        'message':   v.get('message', ''),
    } for k, v in platform_alerts.items()
      if (datetime.now() - v['timestamp']).seconds < 600}

    threat_level = 'low'
    if len(active) >= 3:         threat_level = 'critical'
    elif len(active) >= 2:       threat_level = 'high'
    elif len(active) >= 1:       threat_level = 'medium'

    return {
        'platform_threat_level': threat_level,
        'active_alerts':         active,
        'n_active_alerts':       len(active),
        'layer_states':          layer_states,
        'session_monitor':       monitor.get_status(),
        'n_enrolled_users':      len(engine.user_profiles),
        'n_per_user_models':     len(per_user_models),
        'recent_events':         list(recent_events)[:10],
        'timestamp':             datetime.now().isoformat(),
    }

@app.get("/api/platform/events")
async def platform_events():
    """Recent events feed for live dashboard updates."""
    return {'events': list(recent_events)}

@app.get("/api/platform/audit-log")
async def get_audit_log(limit: int = 50):
    """Return last N audit log entries."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return {'entries': [], 'total': 0}
    try:
        df      = pd.read_csv(AUDIT_LOG_FILE)
        entries = df.tail(limit).to_dict(orient='records')
        return {'entries': list(reversed(entries)), 'total': len(df)}
    except Exception as e:
        return {'entries': [], 'error': str(e)}

@app.get("/api/session-activity")
async def session_activity():
    """Current session context from Windows monitor."""
    return monitor.get_status()


# ══════════════════════════════════════════════════════════
# ROUTES — PLATFORM ALERTS (cross-layer)
# ══════════════════════════════════════════════════════════

class PlatformAlertRequest(BaseModel):
    source:   str
    severity: str   = 'medium'
    layer:    int   = 0
    message:  str   = ''

@app.post("/api/platform/alert")
async def platform_alert(req: PlatformAlertRequest):
    """
    Receive alert from any layer.
    Notifies Layer 3 risk engine AND updates platform state.
    Other layers call this when they detect a threat.
    """
    # Store in platform alerts
    platform_alerts[req.source] = {
        'timestamp': datetime.now(),
        'severity':  req.severity,
        'layer':     req.layer,
        'message':   req.message or f"Threat detected by {req.source}",
    }

    # Notify Layer 3 risk engine
    engine.receive_alert(req.source, req.severity)

    # Update session monitor — switch to under_attack if critical
    if req.severity == 'critical' or len(platform_alerts) >= 2:
        monitor.set_alert_override(True)
        layer_states[3]['status'] = 'elevated'

    # Update layer state
    if 1 <= req.layer <= 4:
        layer_states[req.layer]['threat_count'] += 1
        layer_states[req.layer]['status']        = 'alert'

    # Add to event feed
    add_event(req.layer or 3, 'alert', req.message or f"Alert: {req.source}", req.severity)

    write_audit('platform_alert', layer=req.layer, context=req.source,
                active_alerts=list(platform_alerts.keys()),
                details=f"severity={req.severity} msg={req.message}")

    return {
        'status':          'alert_received',
        'source':          req.source,
        'platform_alerts': len(platform_alerts),
        'layer3_notified': True,
        'effect':          f'Threshold -{engine.ALERT_REDUCTIONS.get(req.source,0.08):.2f} '
                           f'| Score +{engine.ALERT_SCORE_BOOST.get(req.source,0.08):.2f}',
    }

@app.delete("/api/platform/alert/{source}")
async def clear_platform_alert(source: str):
    if source in platform_alerts:
        del platform_alerts[source]
    if source in engine.active_alerts:
        del engine.active_alerts[source]
    if not platform_alerts:
        monitor.set_alert_override(False)
        for ls in layer_states.values():
            if ls['status'] == 'alert':
                ls['status'] = 'monitoring'
        layer_states[3]['status'] = 'active'
    add_event(3, 'alert_cleared', f"Alert cleared: {source}", 'info')
    return {'status': 'cleared', 'source': source}

@app.delete("/api/platform/alerts")
async def clear_all_platform_alerts():
    platform_alerts.clear()
    engine.active_alerts.clear()
    monitor.set_alert_override(False)
    layer_states[3]['status'] = 'active'
    for i in [1,2,4]:
        layer_states[i]['status'] = 'monitoring'
    add_event(3, 'all_alerts_cleared', "All platform alerts cleared", 'info')
    return {'status': 'all_cleared'}


# ══════════════════════════════════════════════════════════
# ROUTES — LAYER 3 (Zero-Trust Auth) — all /layer3/* endpoints
# ══════════════════════════════════════════════════════════

class ScoreRequest(BaseModel):
    user_id:  int
    features: list[float]
    context:  str = 'normal_browsing'

class EnrollRequest(BaseModel):
    user_id:  int
    features: list[float]

class CombinedEnrollRequest(BaseModel):
    user_id:        int
    ks_features:    list[float]
    mouse_features: list[float]

class ReplayRequest(BaseModel):
    user_id:   int
    context:   str = 'normal_browsing'
    n_samples: int = 100

@app.get("/api/users")
async def get_users():
    return {"users": sorted(engine.user_profiles.keys()),
            "total": len(engine.user_profiles)}

@app.post("/api/score")
async def score_session(req: ScoreRequest):
    # Use auto-detected context if not overridden
    context  = req.context
    features = pad_features(req.features)

    result   = engine.score_session(features, req.user_id, context, use_window=True)

    # Add per-user model score if available
    per_user_score = get_per_user_score(req.user_id, req.features[:13])
    if per_user_score is not None:
        result['per_user_score']       = round(per_user_score, 4)
        result['per_user_model_used']  = True
        # Blend: 70% personal model + 30% global
        blended = 0.70 * per_user_score + 0.30 * result['final_risk_score']
        result['final_risk_score']     = round(float(np.clip(blended, 0, 1)), 4)
        result['risk_percent']         = int(result['final_risk_score'] * 100)
        result['risk_level']           = engine._risk_level(result['final_risk_score'])
        result['decision']             = engine._make_decision(
            result['final_risk_score'], result['adaptive_threshold']
        )
    else:
        result['per_user_score']      = None
        result['per_user_model_used'] = False

    # Add compliance info
    result['compliance'] = compliance_mode(context)

    # Audit
    write_audit('score', user_id=req.user_id, layer=3,
                risk_percent=result['risk_percent'],
                decision=result['decision'], context=context,
                active_alerts=result.get('active_alerts', []))

    # Add to event feed if suspicious
    if result['decision'] != 'allow':
        add_event(3, result['decision'],
                  f"User {req.user_id}: {result['risk_percent']}% risk — {result['decision']}",
                  'high' if result['decision'] == 'step_up_auth' else 'medium')
        layer_states[3]['threat_count'] += 1

    return result

@app.post("/api/replay")
async def replay_user(req: ReplayRequest):
    features = get_user_features(req.user_id)
    if features is None:
        raise HTTPException(404, f"No profile for user {req.user_id}")

    # Use session monitor context if not manually set
    context = req.context
    result  = engine.score_session(features, req.user_id, context, use_window=False)

    per_user_score = get_per_user_score(req.user_id, features[:13])
    if per_user_score is not None:
        result['per_user_score']      = round(per_user_score, 4)
        result['per_user_model_used'] = True
    else:
        result['per_user_score']      = None
        result['per_user_model_used'] = False

    result['mode']       = 'dataset_replay'
    result['compliance'] = compliance_mode(context)

    write_audit('replay', user_id=req.user_id, layer=3,
                risk_percent=result['risk_percent'],
                decision=result['decision'], context=context,
                active_alerts=result.get('active_alerts',[]))

    if result['decision'] != 'allow':
        add_event(3, result['decision'],
                  f"Replay User {req.user_id}: {result['risk_percent']}% — {result['decision']}",
                  'high' if result['decision'] == 'step_up_auth' else 'medium')

    return result

@app.post("/api/enroll")
async def enroll_user(req: EnrollRequest):
    features = np.array(pad_features(req.features))
    fw       = np.array(feature_weights)[:N_FEATURES]
    weighted = features * fw
    scaled   = engine.scaler.transform(weighted.reshape(1,-1))[0]
    engine.user_profiles[req.user_id] = {
        'features':     scaled.tolist(),
        'raw_features': pad_features(req.features),
        'feature_names': FEATURE_COLS,
        'if_score':     float(engine.iso_forest.score_samples(scaled.reshape(1,-1))[0]),
        'svm_score':    float(engine.oc_svm.score_samples(scaled.reshape(1,-1))[0]),
        'enrolled_at':  datetime.now().isoformat(),
    }
    engine.reset_user_window(req.user_id)
    write_audit('enroll', user_id=req.user_id, layer=3, details='keystroke_only')
    add_event(3, 'enrollment', f"User {req.user_id} enrolled (keystroke)", 'info')
    return {"status":"enrolled","user_id":req.user_id,"mode":"keystroke_only"}

@app.post("/api/enroll-combined")
async def enroll_combined(req: CombinedEnrollRequest):
    ks  = list(req.ks_features[:15])    + [0.0]*(15-len(req.ks_features[:15]))
    ms  = list(req.mouse_features[:33]) + [0.0]*(33-len(req.mouse_features[:33]))
    combined = ks + ms
    features = np.array(combined)
    fw       = np.array(feature_weights)[:N_FEATURES]
    weighted = features * fw
    scaled   = engine.scaler.transform(weighted.reshape(1,-1))[0]
    engine.user_profiles[req.user_id] = {
        'features':     scaled.tolist(),
        'raw_features': combined,
        'feature_names': FEATURE_COLS,
        'if_score':     float(engine.iso_forest.score_samples(scaled.reshape(1,-1))[0]),
        'svm_score':    float(engine.oc_svm.score_samples(scaled.reshape(1,-1))[0]),
        'enrolled_at':  datetime.now().isoformat(),
    }
    engine.reset_user_window(req.user_id)
    write_audit('enroll', user_id=req.user_id, layer=3, details='keystroke+mouse')
    add_event(3, 'enrollment', f"User {req.user_id} enrolled (keystroke+mouse)", 'info')
    return {"status":"enrolled","user_id":req.user_id,"mode":"keystroke_and_mouse",
            "total_features":len(combined)}

@app.get("/api/anomaly-users")
async def get_anomaly_users():
    profiles_data = joblib.load("models/user_profiles_v2.pkl")
    results = []
    for uid, data in profiles_data.items():
        f    = np.array(data['features']).reshape(1,-1)
        pred = engine.iso_forest.predict(f)[0]
        sc   = float(engine.iso_forest.score_samples(f)[0])
        if pred == -1:
            results.append({"user_id":uid,"if_score":round(sc,4)})
    results.sort(key=lambda x: x['if_score'])
    return {"anomaly_users":results,"total":len(results)}

@app.get("/api/status")
async def layer3_status():
    engine.clear_expired_alerts()
    return {
        "active_alerts":        list(engine.active_alerts.keys()),
        "n_enrolled_users":     len(engine.user_profiles),
        "context_thresholds":   engine.CONTEXT_THRESHOLDS,
        "alert_reductions":     engine.ALERT_REDUCTIONS,
        "alert_score_boosts":   engine.ALERT_SCORE_BOOST,
        "session_context":      monitor.current_context,
        "n_per_user_models":    len(per_user_models),
    }

# Keep backward compat alert endpoints
class AlertRequest(BaseModel):
    source:   str
    severity: str = 'medium'

@app.post("/api/alert")
async def receive_alert(req: AlertRequest):
    result = engine.receive_alert(req.source, req.severity)
    platform_alerts[req.source] = {
        'timestamp': datetime.now(), 'severity': req.severity,
        'layer': 3, 'message': f"Alert: {req.source}",
    }
    return result

@app.delete("/api/alert/{source}")
async def clear_alert(source: str):
    if source in engine.active_alerts:
        del engine.active_alerts[source]
    if source in platform_alerts:
        del platform_alerts[source]
    return {"status":"cleared","source":source}

@app.post("/api/liveness-check")
async def liveness_check():
    try:
        from face_liveness import detect_liveness
        result = detect_liveness(timeout_seconds=12)
        write_audit('liveness', layer=3,
                    decision='passed' if result['passed'] else 'failed',
                    details=result['reason'])
        add_event(3, 'liveness',
                  f"Liveness {'PASSED' if result['passed'] else 'FAILED'}: {result['reason']}",
                  'low' if result['passed'] else 'high')
        return result
    except ImportError:
        return {"passed":False,"reason":"face_liveness.py not found","duration":0,"blink_count":0}
    except Exception as e:
        return {"passed":False,"reason":str(e),"duration":0,"blink_count":0}


# ══════════════════════════════════════════════════════════
# ROUTES — SIMULATED LAYERS 1, 2, 4
# These give realistic endpoints other team members can replace
# ══════════════════════════════════════════════════════════

class LayerEventRequest(BaseModel):
    event_type: str
    severity:   str  = 'medium'
    details:    str  = ''
    source_ip:  str  = ''
    file_path:  str  = ''

@app.get("/api/layer1/status")
async def layer1_status():
    return {
        "layer": 1, "name": "Network Guardian",
        "status": layer_states[1]['status'],
        "threat_count": layer_states[1]['threat_count'],
        "description": "Zero-day intrusion detection — monitors network traffic",
        "simulated": True,
    }

@app.post("/api/layer1/threat")
async def layer1_threat(req: LayerEventRequest):
    """Simulate Network Guardian detecting a threat."""
    layer_states[1]['status']        = 'alert'
    layer_states[1]['threat_count'] += 1
    msg = f"Network Guardian: {req.event_type}"
    if req.source_ip:
        msg += f" from {req.source_ip}"
    # Notify platform
    platform_alerts['network_guardian'] = {
        'timestamp': datetime.now(), 'severity': req.severity,
        'layer': 1, 'message': msg,
    }
    engine.receive_alert('network_guardian', req.severity)
    add_event(1, req.event_type, msg, req.severity)
    write_audit('layer1_threat', layer=1, details=msg)
    return {"status":"threat_raised","message":msg,"layer3_notified":True}

@app.get("/api/layer2/status")
async def layer2_status():
    return {
        "layer": 2, "name": "Ransomware Killer",
        "status": layer_states[2]['status'],
        "threat_count": layer_states[2]['threat_count'],
        "description": "Endpoint behavioral protection — monitors file system",
        "simulated": True,
    }

@app.post("/api/layer2/threat")
async def layer2_threat(req: LayerEventRequest):
    layer_states[2]['status']        = 'alert'
    layer_states[2]['threat_count'] += 1
    msg = f"Ransomware Killer: {req.event_type}"
    if req.file_path:
        msg += f" — {req.file_path}"
    platform_alerts['ransomware_killer'] = {
        'timestamp': datetime.now(), 'severity': req.severity,
        'layer': 2, 'message': msg,
    }
    engine.receive_alert('ransomware_killer', req.severity)
    add_event(2, req.event_type, msg, req.severity)
    write_audit('layer2_threat', layer=2, details=msg)
    return {"status":"threat_raised","message":msg,"layer3_notified":True}

@app.get("/api/layer4/status")
async def layer4_status():
    return {
        "layer": 4, "name": "Content Threat Detection",
        "status": layer_states[4]['status'],
        "threat_count": layer_states[4]['threat_count'],
        "description": "Deepfake and AI phishing detection",
        "simulated": True,
    }

@app.post("/api/layer4/threat")
async def layer4_threat(req: LayerEventRequest):
    layer_states[4]['status']        = 'alert'
    layer_states[4]['threat_count'] += 1
    msg = f"Content Threat: {req.event_type}"
    platform_alerts['content_threat_detection'] = {
        'timestamp': datetime.now(), 'severity': req.severity,
        'layer': 4, 'message': msg,
    }
    engine.receive_alert('content_threat_detection', req.severity)
    add_event(4, req.event_type, msg, req.severity)
    write_audit('layer4_threat', layer=4, details=msg)
    return {"status":"threat_raised","message":msg,"layer3_notified":True}


# ══════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*55)
    print("  NeuraShield Platform v2.0")
    print("="*55)
    print(f"  Central Dashboard : http://localhost:8000")
    print(f"  Layer 3 Dashboard : http://localhost:8000/layer3")
    print(f"  API Documentation : http://localhost:8000/docs")
    print(f"  Platform Status   : http://localhost:8000/api/platform/status")
    print("="*55 + "\n")
    uvicorn.run("neurashield_platform:app",
                host="0.0.0.0", port=8000, reload=True)

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

# What is new vs main_v2.py:
#   1. Keystroke agent auto-starts as background thread on startup
#   2. Agent score stored and returned in /api/platform/status
#   3. NaN values cleaned before JSON serialisation
#   4. Session monitor auto-detects context from active window
#   5. Per-user models loaded and used alongside global model
#   6. Full audit log written to disk
#   7. Simulated layer endpoints for other team members
#   8. Central dashboard at / — Layer 3 detail at /layer3


from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os, sys, csv, pathlib, threading, time
from datetime import datetime
from collections import deque
from pydantic import BaseModel


sys.path.append(os.path.dirname(__file__))
from risk_engine_v2 import RiskEngine
from session_monitor import SessionMonitor

# ── App ───────────────────────────────────────────────────
app    = FastAPI(title="NeuraShield Platform v2")
engine = RiskEngine()

# ── Session monitor ───────────────────────────────────────
monitor = SessionMonitor()
monitor.start()

# ── Data ──────────────────────────────────────────────────
PROFILES_FILE = "user_behavioral_profiles_combined.csv"
profiles_df   = pd.read_csv(PROFILES_FILE)
FEATURE_COLS  = [c for c in profiles_df.columns if c != 'user']
N_FEATURES    = len(FEATURE_COLS)

WEIGHTS_PATH    = "models/feature_weights_v2.pkl"
feature_weights = (
    joblib.load(WEIGHTS_PATH)
    if os.path.exists(WEIGHTS_PATH)
    else np.ones(N_FEATURES)
)
print(f"[Platform] Features: {N_FEATURES} | Weights: {len(feature_weights)}")

# ── Per-user models ───────────────────────────────────────
USER_MODELS_DIR = os.path.join("models", "user_models")
per_user_models = {}

def load_per_user_models():
    if not os.path.exists(USER_MODELS_DIR):
        print("[Platform] No per-user models — run train_per_user_models.py")
        return
    count = 0
    for fname in os.listdir(USER_MODELS_DIR):
        if fname.endswith("_if.pkl"):
            uid = int(fname.replace("user_","").replace("_if.pkl",""))
            try:
                per_user_models[uid] = {
                    'model':  joblib.load(os.path.join(USER_MODELS_DIR, f"user_{uid}_if.pkl")),
                    'scaler': joblib.load(os.path.join(USER_MODELS_DIR, f"user_{uid}_scaler.pkl")),
                    'stats':  joblib.load(os.path.join(USER_MODELS_DIR, f"user_{uid}_stats.pkl"))
                              if os.path.exists(os.path.join(USER_MODELS_DIR, f"user_{uid}_stats.pkl"))
                              else {},
                }
                count += 1
            except Exception as e:
                print(f"[Platform] Could not load model user {uid}: {e}")
    print(f"[Platform] Loaded {count} per-user models")

load_per_user_models()
# Make per-user models available to RiskEngine without import
RiskEngine.per_user_models = per_user_models
print(f"[Platform] Shared {len(per_user_models)} per-user models with RiskEngine")

# ── Audit log ──────────────────────────────────────────────
AUDIT_LOG  = "audit_log.csv"
AUDIT_HDRS = ['timestamp','event_type','user_id','layer',
              'risk_percent','decision','context','active_alerts','details']
_audit_lk  = threading.Lock()

def write_audit(event_type, user_id=None, layer=None,
                risk_percent=None, decision=None,
                context=None, active_alerts=None, details=None):
    row = {
        'timestamp':     datetime.now().isoformat(),
        'event_type':    event_type or '',
        'user_id':       str(user_id) if user_id is not None else '',
        'layer':         str(layer)   if layer   is not None else '',
        'risk_percent':  str(risk_percent) if risk_percent is not None else '',
        'decision':      decision or '',
        'context':       context  or '',
        'active_alerts': ','.join(active_alerts) if active_alerts else '',
        'details':       details or '',
    }
    exists = os.path.exists(AUDIT_LOG)
    with _audit_lk:
        with open(AUDIT_LOG, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=AUDIT_HDRS)
            if not exists:
                w.writeheader()
            w.writerow(row)

def clean_for_json(obj):
    """
    Recursively replace NaN/Inf/numpy types with JSON-safe values.
    Handles: numpy.float64, numpy.int64, numpy.nan, float nan/inf.
    """
    import math
    # Handle numpy array types
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return clean_for_json(obj.tolist())
        if isinstance(obj, np.bool_):
            return bool(obj)
    except Exception:
        pass

    # Handle standard Python types
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    # Convert any other non-serialisable type to string
    if not isinstance(obj, (str, int, bool, type(None))):
        return str(obj)
    return obj

# ── Platform state ────────────────────────────────────────
platform_alerts = {}
layer_states    = {
    1: {'name':'Network Guardian',    'status':'monitoring','threat_count':0,'color':'teal'},
    2: {'name':'Ransomware Killer',   'status':'monitoring','threat_count':0,'color':'red'},
    3: {'name':'Zero-Trust Auth',     'status':'active',    'threat_count':0,'color':'purple'},
    4: {'name':'Content Threat Det.', 'status':'monitoring','threat_count':0,'color':'amber'},
}
recent_events    = deque(maxlen=50)
_latest_agent_score = {}   # latest score from background agent

def add_event(layer_num, etype, message, severity='info'):
    ev = {
        'timestamp': datetime.now().isoformat(),
        'layer':     layer_num,
        'type':      etype,
        'message':   message,
        'severity':  severity,
    }
    recent_events.appendleft(ev)
    layer_states[layer_num]['last_event'] = datetime.now().isoformat()
    return ev

# ── Helpers ───────────────────────────────────────────────
def pad_features(raw):
    f = list(raw)[:N_FEATURES]
    return f + [0.0]*(N_FEATURES-len(f))

def get_user_features(user_id):
    row = profiles_df[profiles_df['user']==user_id]
    return None if row.empty else row[FEATURE_COLS].values[0].tolist()

def get_per_user_score(user_id, raw_features):
    if user_id not in per_user_models:
        return None
    try:
        m      = per_user_models[user_id]
        feats  = np.array(raw_features[:13]).reshape(1,-1)
        scaled = m['scaler'].transform(feats)
        raw_sc = float(m['model'].score_samples(scaled)[0])
        stats  = m['stats']
        mn,mx  = stats.get('min',-0.7), stats.get('max',-0.3)
        return float(np.clip((mx-raw_sc)/(mx-mn+1e-6),0.0,1.0))
    except Exception:
        return None

def compliance_mode(context):
    return {
        'financial':        {'standard':'PSD2','article':'Article 97 — SCA',
                             'requirement':'Strong Customer Authentication required'},
        'sensitive_access': {'standard':'GDPR + NIST','article':'Art.25 + SP800-207',
                             'requirement':'Privacy by design + Zero Trust verification'},
        'normal_browsing':  {'standard':'NIST SP800-207','article':'Zero Trust Architecture',
                             'requirement':'Continuous verification enforced'},
        'under_attack':     {'standard':'ISO 27001','article':'A.16 — Incident Response',
                             'requirement':'Elevated authentication during active incident'},
    }.get(context, {'standard':'NIST SP800-207','article':'ZTA','requirement':'Continuous verification'})

# ══════════════════════════════════════════════════════════
# AUTO-START KEYSTROKE AGENT
# Starts when the platform starts — no second terminal needed
# ══════════════════════════════════════════════════════════
def _start_agent_background():
    """
    Start keystroke agent in a background thread.
    Waits 5 seconds for the FastAPI server to be fully ready
    before the agent tries to connect to /api/enroll.
    """
    time.sleep(5)
    try:
        from keystroke_agent import KeystrokeAgent
        global _agent
        _agent = KeystrokeAgent(user_id=9001)
        _agent.start()
        print("[Platform] Keystroke agent started automatically (user 9001)")
    except ImportError:
        print("[Platform] keystroke_agent.py not found — agent not started")
    except Exception as e:
        print(f"[Platform] Agent start error: {e}")

_agent = None
_agent_thread = threading.Thread(
    target=_start_agent_background, daemon=True, name="AgentStarter"
)
_agent_thread.start()

# ══════════════════════════════════════════════════════════
# DASHBOARDS
# ══════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def central_dashboard(request: Request):
    p = pathlib.Path(__file__).parent / "templates" / "central_dashboard.html"
    if not p.exists():
        return HTMLResponse("<h2 style='color:red;padding:40px'>central_dashboard.html not found in templates/</h2>")
    return HTMLResponse(p.read_text(encoding="utf-8"))

@app.get("/layer3", response_class=HTMLResponse)
async def layer3_dashboard(request: Request):
    p = pathlib.Path(__file__).parent / "templates" / "index.html"
    if not p.exists():
        return HTMLResponse("<h2 style='color:red;padding:40px'>index.html not found in templates/</h2>")
    return HTMLResponse(p.read_text(encoding="utf-8"))

# ══════════════════════════════════════════════════════════
# PLATFORM STATUS
# ══════════════════════════════════════════════════════════
@app.get("/api/platform/status")
async def platform_status():
    engine.clear_expired_alerts()

    # Expire old platform alerts
    now    = datetime.now()
    active = {
        k: {
            'timestamp': v['timestamp'].isoformat(),
            'severity':  v['severity'],
            'layer':     v.get('layer',0),
            'message':   v.get('message',''),
        }
        for k,v in platform_alerts.items()
        if (now - v['timestamp']).seconds < 600
    }

    n  = len(active)
    tl = 'critical' if n>=3 else 'high' if n>=2 else 'medium' if n>=1 else 'low'

    # Agent status
    agent_status = {}
    if _agent:
        agent_status = _agent.get_status()

    result = {
        'platform_threat_level': tl,
        'active_alerts':         active,
        'n_active_alerts':       n,
        'layer_states':          layer_states,
        'session_monitor':       monitor.get_status(),
        'n_enrolled_users':      len(engine.user_profiles),
        'n_per_user_models':     len(per_user_models),
        'recent_events':         list(recent_events)[:10],
        'latest_agent_score':    _latest_agent_score,
        'agent_status':          agent_status,
        'timestamp':             datetime.now().isoformat(),
    }
    return clean_for_json(result)

@app.get("/api/platform/events")
async def platform_events():
    return clean_for_json({'events': list(recent_events)})

@app.get("/api/platform/audit-log")
async def get_audit_log(limit: int = 50):
    if not os.path.exists(AUDIT_LOG):
        return {'entries':[], 'total':0}
    try:
        df = pd.read_csv(AUDIT_LOG)
        df = df.fillna('')   # replace NaN with empty string
        entries = df.tail(limit).to_dict(orient='records')
        # Extra safety: replace any remaining float nan
        clean = []
        for row in entries:
            clean.append({
                k: ('' if isinstance(v,float) and v!=v else v)
                for k,v in row.items()
            })
        return {'entries': list(reversed(clean)), 'total': len(df)}
    except Exception as e:
        return {'entries':[], 'error': str(e)}

@app.get("/api/session-activity")
async def session_activity():
    return monitor.get_status()

# ── Agent score endpoint ──────────────────────────────────
@app.post("/api/agent-score")
async def receive_agent_score(result: dict):
    """
    Background agent sends its latest score here.
    Dashboard polls /api/platform/status which includes this.
    """
    global _latest_agent_score
    _latest_agent_score = result
    _latest_agent_score['received_at'] = datetime.now().isoformat()

    dec  = result.get('decision','allow')
    risk = result.get('risk_percent',0)

    if dec != 'allow':
        add_event(3, f"agent_{dec}",
                  f"Agent user {result.get('user_id')}: {risk}% — {dec}",
                  'high' if dec=='step_up_auth' else 'medium')
        layer_states[3]['threat_count'] += 1

    write_audit('agent_score',
                user_id=result.get('user_id'), layer=3,
                risk_percent=risk, decision=dec,
                context=result.get('context'),
                active_alerts=result.get('active_alerts',[]))
    return {"status":"received"}

@app.get("/api/agent-score")
async def get_agent_score():
    return clean_for_json(_latest_agent_score or {"status":"no_score_yet"})

# ══════════════════════════════════════════════════════════
# PLATFORM ALERTS
# ══════════════════════════════════════════════════════════
class PlatformAlertRequest(BaseModel):
    source:   str
    severity: str = 'medium'
    layer:    int = 0
    message:  str = ''

@app.post("/api/platform/alert")
async def platform_alert(req: PlatformAlertRequest):
    platform_alerts[req.source] = {
        'timestamp': datetime.now(),
        'severity':  req.severity,
        'layer':     req.layer,
        'message':   req.message or f"Threat: {req.source}",
    }
    engine.receive_alert(req.source, req.severity)
    if req.severity in ('critical','high') or len(platform_alerts)>=2:
        monitor.set_alert_override(True)
        layer_states[3]['status'] = 'elevated'
    if 1<=req.layer<=4:
        layer_states[req.layer]['threat_count'] += 1
        layer_states[req.layer]['status']        = 'alert'
    add_event(req.layer or 3, 'alert', req.message or f"Alert: {req.source}", req.severity)
    write_audit('platform_alert', layer=req.layer, context=req.source,
                active_alerts=list(platform_alerts.keys()),
                details=f"severity={req.severity}")
    return {
        'status':          'alert_received',
        'source':          req.source,
        'platform_alerts': len(platform_alerts),
        'layer3_notified': True,
        'threshold_drop':  engine.ALERT_REDUCTIONS.get(req.source, 0.08),
        'score_boost':     engine.ALERT_SCORE_BOOST.get(req.source, 0.08),
    }

@app.delete("/api/platform/alert/{source}")
async def clear_platform_alert(source: str):
    platform_alerts.pop(source, None)
    engine.active_alerts.pop(source, None)
    if not platform_alerts:
        monitor.set_alert_override(False)
        for ls in layer_states.values():
            if ls['status'] in ('alert','elevated'):
                ls['status'] = 'monitoring'
        layer_states[3]['status'] = 'active'
    add_event(3,'alert_cleared',f"Alert cleared: {source}",'info')
    return {'status':'cleared','source':source}

@app.delete("/api/platform/alerts")
async def clear_all_platform_alerts():
    platform_alerts.clear()
    engine.active_alerts.clear()
    monitor.set_alert_override(False)
    layer_states[3]['status'] = 'active'
    for i in [1,2,4]:
        layer_states[i]['status'] = 'monitoring'
    add_event(3,'all_cleared','All platform alerts cleared','info')
    return {'status':'all_cleared'}

# ══════════════════════════════════════════════════════════
# LAYER 3 — Zero-Trust Auth endpoints
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

class AlertRequest(BaseModel):
    source:   str
    severity: str = 'medium'

@app.get("/api/users")
async def get_users():
    return {"users": sorted(engine.user_profiles.keys()),
            "total": len(engine.user_profiles)}

@app.post("/api/score")
async def score_session(req: ScoreRequest):
    features = pad_features(req.features)
    result   = engine.score_session(features, req.user_id, req.context, use_window=True)

    # Per-user model blend
    pus = get_per_user_score(req.user_id, req.features[:13])
    if pus is not None:
        blended = 0.70*pus + 0.30*result['final_risk_score']
        blended = float(np.clip(blended,0,1))
        result['per_user_score']      = round(pus, 4)
        result['per_user_model_used'] = True
        result['final_risk_score']    = round(blended, 4)
        result['risk_percent']        = int(blended*100)
        result['risk_level']          = engine._risk_level(blended)
        result['decision']            = engine._make_decision(blended, result['adaptive_threshold'])
    else:
        result['per_user_score']      = None
        result['per_user_model_used'] = False

    result['compliance'] = compliance_mode(req.context)

    write_audit('score', user_id=req.user_id, layer=3,
                risk_percent=result['risk_percent'],
                decision=result['decision'], context=req.context,
                active_alerts=result.get('active_alerts',[]))

    if result['decision'] != 'allow':
        add_event(3, result['decision'],
                  f"User {req.user_id}: {result['risk_percent']}% — {result['decision']}",
                  'high' if result['decision']=='step_up_auth' else 'medium')

    return clean_for_json(result)

@app.post("/api/replay")
async def replay_user(req: ReplayRequest):
    features = get_user_features(req.user_id)
    if features is None:
        raise HTTPException(404, f"No profile for user {req.user_id}")
    result = engine.score_session(features, req.user_id, req.context, use_window=False)

    pus = get_per_user_score(req.user_id, features[:13])
    result['per_user_score']      = round(pus,4) if pus else None
    result['per_user_model_used'] = pus is not None
    result['mode']                = 'dataset_replay'
    result['compliance']          = compliance_mode(req.context)

    write_audit('replay', user_id=req.user_id, layer=3,
                risk_percent=result['risk_percent'],
                decision=result['decision'], context=req.context,
                active_alerts=result.get('active_alerts',[]))

    if result['decision'] != 'allow':
        add_event(3, result['decision'],
                  f"Replay {req.user_id}: {result['risk_percent']}% — {result['decision']}",
                  'high' if result['decision']=='step_up_auth' else 'medium')

    return clean_for_json(result)

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
    add_event(3,'enrollment',f"User {req.user_id} enrolled",'info')
    return clean_for_json({
        "status":"enrolled","user_id":req.user_id,"mode":"keystroke_only",
        "if_score":  round(engine.user_profiles[req.user_id]['if_score'],4),
        "svm_score": round(engine.user_profiles[req.user_id]['svm_score'],4),
    })

@app.post("/api/enroll-combined")
async def enroll_combined(req: CombinedEnrollRequest):
    ks  = list(req.ks_features[:15])  + [0.0]*(15-len(req.ks_features[:15]))
    ms  = list(req.mouse_features[:33])+ [0.0]*(33-len(req.mouse_features[:33]))
    combined = ks+ms
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
    add_event(3,'enrollment',f"User {req.user_id} enrolled (ks+mouse)",'info')
    return clean_for_json({
        "status":"enrolled","user_id":req.user_id,
        "mode":"keystroke_and_mouse","total_features":len(combined),
    })

@app.get("/api/anomaly-users")
async def get_anomaly_users():
    data = joblib.load("models/user_profiles_v2.pkl")
    results = []
    for uid,d in data.items():
        f    = np.array(d['features']).reshape(1,-1)
        pred = engine.iso_forest.predict(f)[0]
        sc   = float(engine.iso_forest.score_samples(f)[0])
        if pred==-1:
            results.append({"user_id":uid,"if_score":round(sc,4)})
    results.sort(key=lambda x:x['if_score'])
    return {"anomaly_users":results,"total":len(results)}

@app.get("/api/status")
async def layer3_status():
    engine.clear_expired_alerts()
    return clean_for_json({
        "active_alerts":      list(engine.active_alerts.keys()),
        "n_enrolled_users":   len(engine.user_profiles),
        "context_thresholds": engine.CONTEXT_THRESHOLDS,
        "alert_reductions":   engine.ALERT_REDUCTIONS,
        "alert_score_boosts": engine.ALERT_SCORE_BOOST,
        "session_context":    monitor.current_context,
        "n_per_user_models":  len(per_user_models),
    })

@app.post("/api/alert")
async def receive_alert(req: AlertRequest):
    result = engine.receive_alert(req.source, req.severity)
    platform_alerts[req.source] = {
        'timestamp':datetime.now(),'severity':req.severity,
        'layer':3,'message':f"Alert: {req.source}",
    }
    return result

@app.delete("/api/alert/{source}")
async def clear_alert(source: str):
    engine.active_alerts.pop(source, None)
    platform_alerts.pop(source, None)
    return {"status":"cleared","source":source}

@app.post("/api/liveness-check")
async def liveness_check():
    try:
        from face_liveness import run_from_api
        r = run_from_api(timeout_seconds=25)

        write_audit('liveness', layer=3,
                    decision='passed' if r['passed'] else 'failed',
                    details=r['reason'])
        add_event(3,'liveness',
                  f"Liveness {'PASSED' if r['passed'] else 'FAILED'}: {r['reason']}",
                  'low' if r['passed'] else 'high')
        return r
    except ImportError:
        return {"passed":False,"reason":"face_liveness.py not found","duration":0,"blink_count":0}
    except Exception as e:
        return {"passed":False,"reason":str(e),"duration":0,"blink_count":0}

# ══════════════════════════════════════════════════════════
# SIMULATED LAYERS 1, 2, 4
# ══════════════════════════════════════════════════════════
class LayerEventRequest(BaseModel):
    event_type: str
    severity:   str = 'medium'
    details:    str = ''
    source_ip:  str = ''
    file_path:  str = ''

@app.get("/api/layer1/status")
async def layer1_status():
    return {"layer":1,"name":"Network Guardian",
            "status":layer_states[1]['status'],
            "threat_count":layer_states[1]['threat_count'],"simulated":True}

@app.post("/api/layer1/threat")
async def layer1_threat(req: LayerEventRequest):
    layer_states[1]['status']        = 'alert'
    layer_states[1]['threat_count'] += 1
    msg = f"Network Guardian: {req.event_type}"
    if req.source_ip: msg += f" from {req.source_ip}"
    platform_alerts['network_guardian'] = {
        'timestamp':datetime.now(),'severity':req.severity,'layer':1,'message':msg}
    engine.receive_alert('network_guardian', req.severity)
    add_event(1, req.event_type, msg, req.severity)
    write_audit('layer1_threat', layer=1, details=msg)
    return {"status":"threat_raised","message":msg,"layer3_notified":True}

@app.get("/api/layer2/status")
async def layer2_status():
    return {"layer":2,"name":"Ransomware Killer",
            "status":layer_states[2]['status'],
            "threat_count":layer_states[2]['threat_count'],"simulated":True}

@app.post("/api/layer2/threat")
async def layer2_threat(req: LayerEventRequest):
    layer_states[2]['status']        = 'alert'
    layer_states[2]['threat_count'] += 1
    msg = f"Ransomware Killer: {req.event_type}"
    if req.file_path: msg += f" — {req.file_path}"
    platform_alerts['ransomware_killer'] = {
        'timestamp':datetime.now(),'severity':req.severity,'layer':2,'message':msg}
    engine.receive_alert('ransomware_killer', req.severity)
    add_event(2, req.event_type, msg, req.severity)
    write_audit('layer2_threat', layer=2, details=msg)
    return {"status":"threat_raised","message":msg,"layer3_notified":True}

@app.get("/api/layer4/status")
async def layer4_status():
    return {"layer":4,"name":"Content Threat Detection",
            "status":layer_states[4]['status'],
            "threat_count":layer_states[4]['threat_count'],"simulated":True}

@app.post("/api/layer4/threat")
async def layer4_threat(req: LayerEventRequest):
    layer_states[4]['status']        = 'alert'
    layer_states[4]['threat_count'] += 1
    msg = f"Content Threat: {req.event_type}"
    platform_alerts['content_threat_detection'] = {
        'timestamp':datetime.now(),'severity':req.severity,'layer':4,'message':msg}
    engine.receive_alert('content_threat_detection', req.severity)
    add_event(4, req.event_type, msg, req.severity)
    write_audit('layer4_threat', layer=4, details=msg)
    return {"status":"threat_raised","message":msg,"layer3_notified":True}

class ContextUpdate(BaseModel):
    url: str
    title: str = ""
    timestamp: str = ""

@app.post("/api/context-update")
async def context_update(data: ContextUpdate):
    """
    Receives current tab URL + title from the browser extension
    and updates the session monitor.
    """
    try:
        monitor.update_from_url(data.url, data.title)
        return {
            "status": "received",
            "url": data.url,
            "title": data.title,
            "context": monitor.current_context
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ══════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*55)
    print("  NeuraShield Platform v2.0")
    print("="*55)
    print("  Central Dashboard : http://localhost:8000")
    print("  Layer 3 Dashboard : http://localhost:8000/layer3")
    print("  API Docs          : http://localhost:8000/docs")
    print("  Platform Status   : http://localhost:8000/api/platform/status")
    print("  Keystroke Agent   : starts automatically in 5 seconds")
    print("="*55 + "\n")
    uvicorn.run("neurashield_platform:app",
                host="0.0.0.0", port=8000, reload=False)
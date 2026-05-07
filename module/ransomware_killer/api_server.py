"""
NeuraShield — Ransomware Killer API Server
Flask REST backend that bridges the agent logic with the frontend UI.
"""
import os, math, random, string, time, json, threading, datetime
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── State ────────────────────────────────────────────────────────────────────
DEMO_DIR     = os.path.join(os.path.dirname(__file__), "demo_files")
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
LOG_DIR      = os.path.join(os.path.dirname(__file__), "logs")
ALERT_LOG    = os.path.join(LOG_DIR, "alerts.json")

_state = {
    "status":          "idle",        # idle | running | detected | clean | error
    "iso_score":       None,
    "rf_prob":         None,
    "anomaly_score":   None,
    "entropy_log":     [],
    "files_affected":  [],
    "files_restored":  [],
    "audit_log":       [],
    "alerts":          [],
    "is_ransomware":   False,
    "start_time":      None,
    "end_time":        None,
}
_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _log(msg: str):
    ts  = datetime.datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    with _lock:
        _state["audit_log"].append(entry)
    print(entry)

def _random_text(n=200):
    return "".join(random.choices(string.ascii_letters + " \n", k=n))

def _shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts if c)

def _ensure_dirs():
    for d in [DEMO_DIR, SNAPSHOT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

def _take_snapshot(filepath: str):
    fname = os.path.basename(filepath)
    dest  = os.path.join(SNAPSHOT_DIR, fname + ".bak")
    try:
        import shutil
        shutil.copy2(filepath, dest)
    except Exception as e:
        _log(f"Snapshot failed for {fname}: {e}")

def _rollback_files(files: list):
    import shutil
    restored = []
    for fpath in files:
        fname  = os.path.basename(fpath)
        backup = os.path.join(SNAPSHOT_DIR, fname + ".bak")
        if os.path.exists(backup):
            shutil.copy2(backup, fpath)
            restored.append(fpath)
            _log(f"Rolled back {fname}")
        else:
            _log(f"No snapshot for {fname}")
    return restored

def _dispatch_alert(process_name, pid, score, files_affected, action_taken):
    alert = {
        "timestamp":       datetime.datetime.utcnow().isoformat() + "Z",
        "source_layer":    "RansomwareKiller",
        "severity":        "CRITICAL",
        "event_type":      "RANSOMWARE_DETECTED",
        "process":         {"name": process_name, "pid": pid, "anomaly_score": round(float(score), 4)},
        "files_affected":  files_affected,
        "action_taken":    action_taken,
        "notify_layers":   ["NetworkGuardian", "ZeroTrustAuth", "ContentThreat"],
        "recommended_action": "Suspend all sessions. Escalate to SOC.",
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(ALERT_LOG, "a") as f:
        f.write(json.dumps(alert) + "\n")
    with _lock:
        _state["alerts"].append(alert)
    return alert

# ── Background Demo Runner ────────────────────────────────────────────────────
def _run_demo_background():
    NUM_FILES      = 10
    FAKE_PID       = 99999
    FAKE_PROC_NAME = "cryptolocker_sim.exe"

    with _lock:
        _state["status"]         = "running"
        _state["entropy_log"]    = []
        _state["files_affected"] = []
        _state["files_restored"] = []
        _state["is_ransomware"]  = False
        _state["iso_score"]      = None
        _state["rf_prob"]        = None
        _state["anomaly_score"]  = None
        _state["start_time"]     = datetime.datetime.now().isoformat()
        _state["end_time"]       = None

    _ensure_dirs()

    # Phase 1 – create & snapshot files
    _log("Creating demo files and snapshots…")
    files = []
    for i in range(NUM_FILES):
        fpath = os.path.join(DEMO_DIR, f"document_{i+1:02d}.txt")
        with open(fpath, "w") as f:
            f.write(_random_text())
        _take_snapshot(fpath)
        files.append(fpath)
    _log(f"{NUM_FILES} files created and backed up.")
    time.sleep(0.5)

    # Phase 2 – simulate ransomware encryption
    _log("Simulated ransomware starting rapid file encryption…")
    entropy_log = []
    affected    = []
    for fpath in files:
        with open(fpath, "wb") as f:
            payload = bytes([random.randint(0, 255) for _ in range(512)])
            f.write(payload)
        entropy = _shannon_entropy(open(fpath, "rb").read())
        entry   = {"file": os.path.basename(fpath), "entropy": round(entropy, 3)}
        entropy_log.append(entry)
        affected.append(fpath)
        _log(f"[ENC] {entry['file']} — entropy: {entry['entropy']:.2f} bits/byte")
        time.sleep(0.15)

    with _lock:
        _state["entropy_log"]    = entropy_log
        _state["files_affected"] = affected

    # Phase 3 – detection (simulate models)
    _log("Extracting behavioral features…")
    time.sleep(0.4)

    try:
        import joblib
        MODEL_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ransomware_killer")
        iso  = joblib.load(os.path.join(MODEL_BASE, "isolation_forest.pkl"))
        rf   = joblib.load(os.path.join(MODEL_BASE, "random_forest.pkl"))
        fnames = joblib.load(os.path.join(MODEL_BASE, "feature_names.pkl"))
        vec  = np.random.uniform(2.0, 5.0, size=(1, len(fnames)))
        iso_score = float(iso.decision_function(vec)[0])
        iso_pred  = int(iso.predict(vec)[0])
        rf_prob   = float(rf.predict_proba(vec)[0][1])
        is_ransomware = (iso_pred == -1) or (rf_prob > 0.70)
        _log(f"Isolation Forest score: {iso_score:.4f}  ({'RANSOMWARE' if iso_pred == -1 else 'Benign'})")
        _log(f"Random Forest P(ransom): {rf_prob:.2%}")
    except Exception as e:
        _log(f"Models not found — using simulated detection ({e})")
        iso_score     = -0.42
        rf_prob       = 0.93
        is_ransomware = True

    anomaly_score = abs(iso_score)
    with _lock:
        _state["iso_score"]     = round(iso_score, 4)
        _state["rf_prob"]       = round(rf_prob, 4)
        _state["anomaly_score"] = round(anomaly_score, 4)
        _state["is_ransomware"] = is_ransomware

    if is_ransomware:
        _log("🚨  RANSOMWARE CONFIRMED — triggering automated response…")
        time.sleep(0.3)

        # Kill (simulated — FAKE_PID doesn't exist)
        try:
            import psutil
            psutil.Process(FAKE_PID).kill()
        except Exception:
            _log(f"[KILL] Process '{FAKE_PROC_NAME}' (PID {FAKE_PID}) terminated (simulated).")

        # Rollback
        restored = _rollback_files(affected)
        with _lock:
            _state["files_restored"] = restored

        # Alert
        action = f"Killed PID {FAKE_PID}. Rolled back {len(restored)} files."
        _dispatch_alert(FAKE_PROC_NAME, FAKE_PID, anomaly_score, affected, action)
        _log(f"Alert dispatched. {len(restored)} files restored.")

        with _lock:
            _state["status"] = "detected"
    else:
        _log("No ransomware detected.")
        with _lock:
            _state["status"] = "clean"

    with _lock:
        _state["end_time"] = datetime.datetime.now().isoformat()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def get_status():
    with _lock:
        return jsonify(dict(_state))

@app.route("/api/run", methods=["POST"])
def run_demo():
    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "Demo already running"}), 409
    t = threading.Thread(target=_run_demo_background, daemon=True)
    t.start()
    return jsonify({"message": "Demo started"}), 202

@app.route("/api/reset", methods=["POST"])
def reset():
    with _lock:
        _state.update({
            "status":         "idle",
            "iso_score":      None,
            "rf_prob":        None,
            "anomaly_score":  None,
            "entropy_log":    [],
            "files_affected": [],
            "files_restored": [],
            "audit_log":      [],
            "alerts":         [],
            "is_ransomware":  False,
            "start_time":     None,
            "end_time":       None,
        })
    return jsonify({"message": "State reset"})

@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    alerts = []
    if os.path.exists(ALERT_LOG):
        with open(ALERT_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        alerts.append(json.loads(line))
                    except Exception:
                        pass
    return jsonify(alerts)

if __name__ == "__main__":
    print("NeuraShield — Ransomware Killer API running on http://127.0.0.1:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)

"""
dashboard/app.py
-------------------
Local, READ-MOSTLY dashboard for Layer 2 (Ransomware Killer).

Security model (this runs next to a security tool, so it is strict):
  - Listens on 127.0.0.1 only. To change: NEURASHIELD_DASHBOARD_HOST
    (setting 0.0.0.0 is a conscious decision; a warning is logged).
  - Host header must be a local name (stops DNS-rebinding). Extra names:
    NEURASHIELD_DASHBOARD_ALLOWED_HOSTS="name1,name2".
  - Every POST must carry the header  X-Requested-With: NeuraShield  and,
    when the browser sends an Origin, it must match the Host (CSRF guard).
  - Strict Content-Security-Policy, no inline scripts/styles, no external
    fonts or CDNs. Security headers on every response.
  - Alert ids are validated; rollback approvals may only name files that
    appear in a stored alert (the rollback engine also only restores files
    it has a backup of).

Run:
    python -m src.dashboard.app            (waitress, 127.0.0.1:5050)
"""

import json
import os
import re
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import psutil
from flask import Flask, jsonify, render_template, request

from src.alerting.alert_generator import AlertGenerator
from src.consent import ConsentStore, normalize_folders
from src.dashboard.incident_state import IncidentStateStore
from src.response.command_listener import build_rollback_command
from src.response.incident_store import IncidentStore
from src.utils import paths
from src.utils.logger import get_logger, sanitize_for_log
from src.utils.rotation import RotatingJSONLWriter

logger = get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5050
ALERT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_ROLLBACK_FILES = 200
MAX_PATH_LENGTH = 4096
CSRF_HEADER = "X-Requested-With"
CSRF_VALUE = "NeuraShield"
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")

REQUIRED_CONFIG_SECTIONS = ["live_monitor", "response", "alerting", "security"]


def _allowed_hosts() -> set:
    hosts = {"127.0.0.1", "localhost", "[::1]", "::1"}
    extra = os.environ.get("NEURASHIELD_DASHBOARD_ALLOWED_HOSTS", "")
    hosts.update(h.strip().lower() for h in extra.split(",") if h.strip())
    return hosts


def _host_only(host_header: str) -> str:
    host = (host_header or "").strip().lower()
    if host.startswith("["):                      # [::1]:5050
        return host.split("]")[0] + "]"
    return host.rsplit(":", 1)[0] if ":" in host else host


def estimate_scope(folders, max_files=200000, max_seconds=2.0) -> dict:
    """Counts files/bytes in the folders to be protected, with a time cap, so the
    person can see what they are agreeing to. approximate=True if it hit the cap."""
    started = time.time()
    total_files = total_bytes = 0
    approximate = False
    out = []
    for folder in folders:
        n = size = 0
        exists = os.path.isdir(folder)
        if exists:
            for root, _dirs, names in os.walk(folder):
                for name in names:
                    n += 1
                    try:
                        size += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
                if total_files + n > max_files or time.time() - started > max_seconds:
                    approximate = True
                    break
        total_files += n
        total_bytes += size
        out.append({"path": folder, "exists": exists, "files": n, "mb": round(size / 1048576, 1)})
    return {"folders": out, "files": total_files, "mb": round(total_bytes / 1048576, 1),
            "approximate": approximate}


def create_app(config_path=None, alert_queue_path=None, command_queue_path=None,
               status_path=None, state_path=None, layer_status_path=None, consent_path=None,
               incidents_path=None) -> Flask:
    cfg = paths.load_config(config_path)
    missing = [k for k in REQUIRED_CONFIG_SECTIONS if k not in cfg]
    if missing:
        raise SystemExit(f"[dashboard] FATAL: config.yaml is missing section(s): {missing}")

    app = Flask(
        __name__,
        template_folder=paths.resource(os.path.join("src", "dashboard", "templates")),
        static_folder=paths.resource(os.path.join("src", "dashboard", "static")),
    )
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
    app.config["STATUS_EXTENDERS"] = []     # demo overlay can add layer rows
    app.config["SERVICE"] = None            # set by Layer2 when the dashboard runs inside the service
    app.config["DEMO_MODE"] = False

    alert_queue = alert_queue_path or paths.runtime("data/simulated_events/alert_queue.jsonl")
    command_queue = command_queue_path or paths.runtime("data/simulated_events/command_queue.jsonl")
    service_status = status_path or paths.runtime("data/simulated_events/service_status.json")
    start_time = time.time()
    allowed_hosts = _allowed_hosts()

    layer_status = layer_status_path or paths.runtime("data/simulated_events/layer2_status.json")
    consent_store = ConsentStore(consent_path)
    watch_folders = [os.path.expanduser(p) for p in cfg["live_monitor"]["watch_paths"]]
    max_backup_mb = (cfg.get("response") or {}).get("max_backup_storage_mb", 5120)
    scope_cache = {"at": 0.0, "value": None, "running": False}
    scope_lock = threading.Lock()

    def refresh_scope_async():
        """Counts the files in a background thread so a request never waits for a
        slow disk; the dashboard shows 'counting...' until the first result."""
        with scope_lock:
            if scope_cache["running"]:
                return
            scope_cache["running"] = True

        def work():
            try:
                scope_cache["value"] = estimate_scope(watch_folders)
                scope_cache["at"] = time.time()
            except Exception as e:
                logger.warning(f"[dashboard] file count failed: {sanitize_for_log(e)}")
                scope_cache["at"] = time.time()
            finally:
                scope_cache["running"] = False

        threading.Thread(target=work, daemon=True, name="scope").start()

    verifier = AlertGenerator(config_path=config_path, queue_path=alert_queue)
    incident_state = IncidentStateStore(state_path=state_path)
    incident_store = IncidentStore(incidents_path)
    command_writer = RotatingJSONLWriter(command_queue, max_bytes=5 * 1024 * 1024, backup_count=3)

    app.config["ALERT_QUEUE_PATH"] = alert_queue
    app.config["START_TIME"] = start_time

    # ------------------------------------------------------------ security

    @app.before_request
    def _guard():
        if _host_only(request.host) not in allowed_hosts:
            return jsonify({"status": "error", "reason": "host not allowed"}), 400
        if request.method == "POST":
            if request.headers.get(CSRF_HEADER) != CSRF_VALUE:
                return jsonify({"status": "error", "reason": "missing request header"}), 403
            origin = request.headers.get("Origin")
            if origin and origin.split("://", 1)[-1].lower() != request.host.lower():
                return jsonify({"status": "error", "reason": "cross-origin request refused"}), 403
        return None

    @app.after_request
    def _headers(resp):
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/") or request.path == "/":
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # -------------------------------------------------------------- helpers

    def read_alerts(limit=25, include_acknowledged=True):
        if not os.path.exists(alert_queue):
            return []
        try:
            with open(alert_queue, "r", encoding="utf-8", errors="replace") as f:
                lines = [l for l in f.readlines() if l.strip()]
        except OSError:
            return []
        alerts = []
        for line in lines[-limit:]:
            try:
                a = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(a, dict) and isinstance(a.get("alert_id"), str):
                alerts.append(a)
        alerts.reverse()                       # most recent first
        for a in alerts:
            a["acknowledged"] = incident_state.is_acknowledged(a["alert_id"])
        if not include_acknowledged:
            alerts = [a for a in alerts if not a["acknowledged"]]
        return alerts

    def service_stats():
        try:
            with open(service_status, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if time.time() - float(data.get("timestamp", 0)) > 15:
            return None
        return data

    app.config["READ_ALERTS"] = read_alerts

    def read_layer_status():
        """The service's own status file; None if the service is not running."""
        try:
            with open(layer_status, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if time.time() - float(data.get("timestamp", 0)) > 20:
            return None
        return data

    # --------------------------------------------------------------- routes

    @app.route("/")
    def index():
        return render_template("index.html", demo_mode=app.config["DEMO_MODE"])

    @app.route("/api/health")
    def health():
        checks = {
            "config_loaded": True,
            "alert_queue_writable": os.access(os.path.dirname(alert_queue) or ".", os.W_OK),
            "service_running": service_stats() is not None,
        }
        healthy = checks["config_loaded"] and checks["alert_queue_writable"]
        return jsonify({"status": "healthy" if healthy else "degraded",
                        "uptime_seconds": round(time.time() - start_time, 1),
                        "checks": checks}), (200 if healthy else 503)

    @app.route("/api/status")
    def status():
        alerts = read_alerts(limit=5, include_acknowledged=False)
        now = time.time()
        active_threat = any(isinstance(a.get("timestamp"), (int, float)) and now - a["timestamp"] < 30
                            for a in alerts)
        stats = service_stats()
        if stats:
            cpu, mem, source = stats.get("cpu_percent_avg", 0.0), stats.get("memory_mb_avg", 0.0), "service"
        else:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
            source = "dashboard_process_fallback"
        layers = [{"name": "Ransomware Killer", "role": "Endpoint & ransomware defense",
                   "status": "threat" if active_threat else "active"}]
        for extender in app.config["STATUS_EXTENDERS"]:
            try:
                layers.extend(extender(now))
            except Exception as e:      # a demo extra must never break the status
                logger.warning(f"[dashboard] status extender failed: {sanitize_for_log(e)}")
        snap = read_layer_status()
        return jsonify({"protected": not active_threat, "state": snap["state"] if snap else "offline",
                        "demo_mode": app.config["DEMO_MODE"],
                        "backup_percent": _percent(snap),
                        "uptime_seconds": round(now - start_time, 1),
                        "cpu_percent": cpu, "memory_mb": mem, "vitals_source": source, "layers": layers})

    def _percent(snap):
        if not snap:
            return None
        b = snap.get("backup") or {}
        total = b.get("total") or 0
        return None if not total else min(100, round(100 * (b.get("done") or 0) / total))

    @app.route("/api/protection")
    def protection():
        snap = read_layer_status()
        if time.time() - scope_cache["at"] > 120 or scope_cache["value"] is None:
            refresh_scope_async()
        consent = consent_store.read()
        return jsonify({
            "state": snap["state"] if snap else "offline",
            "service": snap,
            "percent": _percent(snap),
            "scope": scope_cache["value"],
            "max_backup_mb": max_backup_mb,
            "consent": None if consent is None else {
                "granted_at": consent.get("granted_at"), "granted_by": consent.get("granted_by"),
                "covers_current_folders": consent_store.covers(watch_folders)},
        })

    @app.route("/api/protection/consent", methods=["POST"])
    def protection_consent():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or data.get("confirm") is not True:
            return jsonify({"status": "error", "reason": "explicit confirmation required"}), 400
        consent_store.grant(watch_folders, by="dashboard")
        return jsonify({"status": "granted", "folders": normalize_folders(watch_folders)})

    @app.route("/api/protection/revoke", methods=["POST"])
    def protection_revoke():
        consent_store.revoke(by="dashboard")
        return jsonify({"status": "revoked"})

    @app.route("/api/alerts")
    def alerts():
        show = request.args.get("show_acknowledged", "true").lower() == "true"
        recent = read_alerts(limit=25, include_acknowledged=show)
        for a in recent:
            # "acknowledged"/"verified" are display fields, not part of the signed data.
            original = {k: v for k, v in a.items() if k not in ("acknowledged", "verified")}
            a["verified"] = verifier.verify_alert(original)
        return jsonify(recent)

    def _valid_id(alert_id):
        return bool(ALERT_ID_RE.match(alert_id))

    @app.route("/api/alerts/<alert_id>/acknowledge", methods=["POST"])
    def acknowledge_alert(alert_id):
        if not _valid_id(alert_id):
            return jsonify({"status": "error", "reason": "bad alert id"}), 400
        incident_state.acknowledge(alert_id)
        logger.info(f"[dashboard] Alert {alert_id} acknowledged by operator")
        return jsonify({"status": "acknowledged", "alert_id": alert_id})

    @app.route("/api/alerts/<alert_id>/unacknowledge", methods=["POST"])
    def unacknowledge_alert(alert_id):
        if not _valid_id(alert_id):
            return jsonify({"status": "error", "reason": "bad alert id"}), 400
        incident_state.unacknowledge(alert_id)
        return jsonify({"status": "unacknowledged", "alert_id": alert_id})

    @app.route("/api/alerts/clear_all", methods=["POST"])
    def clear_all_alerts():
        ids = [a["alert_id"] for a in read_alerts(limit=1000, include_acknowledged=False)]
        incident_store.reload()
        ids += [i["id"] for i in incident_store.list(limit=100)]
        incident_state.acknowledge_many(ids)
        logger.info(f"[dashboard] {len(ids)} alert(s) cleared by operator")
        return jsonify({"status": "cleared", "count": len(ids)})

    @app.route("/api/approve_rollback", methods=["POST"])
    def approve_rollback():
        data = request.get_json(silent=True)
        files = data.get("files") if isinstance(data, dict) else None
        if (not isinstance(files, list) or not files or len(files) > MAX_ROLLBACK_FILES
                or not all(isinstance(p, str) and 0 < len(p) <= MAX_PATH_LENGTH for p in files)):
            return jsonify({"status": "error", "reason": "files must be a non-empty list of paths"}), 400
        # Only files that appear in a recorded alert can be approved from here.
        known = set()
        for a in read_alerts(limit=1000, include_acknowledged=True):
            known.update(p for p in (a.get("affected_files") or {}).get("paths", []) if isinstance(p, str))
        unknown = [p for p in files if p not in known]
        if unknown:
            return jsonify({"status": "error", "reason": "file is not part of any recorded alert"}), 400
        command = build_rollback_command(target_files=files, secret=verifier._secret,
                                         reason="operator-approved via dashboard")
        command_writer.append(json.dumps(command))
        logger.info(f"[dashboard] Rollback approved by operator for {len(files)} file(s)")
        return jsonify({"status": "sent", "command_id": command["command_id"], "files": len(files)})

    # ---------------------------------------------------- incident reports

    def _service():
        return app.config.get("SERVICE")

    def _no_service():
        return jsonify({"status": "error",
                        "reason": "This action needs the running Layer 2 service "
                                  "(start it with:  python -m main --dashboard)."}), 503

    @app.route("/api/incidents")
    def incidents_list():
        show = request.args.get("show_acknowledged", "false").lower() == "true"
        incident_store.reload()
        rows = incident_store.list(limit=30)
        for r in rows:
            r["acknowledged"] = incident_state.is_acknowledged(r["id"])
        if not show:
            rows = [r for r in rows if not r["acknowledged"]]
        return jsonify(rows)

    @app.route("/api/incidents/<incident_id>")
    def incident_detail(incident_id):
        if not _valid_id(incident_id):
            return jsonify({"status": "error", "reason": "bad incident id"}), 400
        incident_store.reload()
        item = incident_store.get(incident_id)
        if item is None:
            return jsonify({"status": "error", "reason": "unknown incident"}), 404
        from src.response.incident_store import summarize
        item["summary"] = summarize(item)
        item["acknowledged"] = incident_state.is_acknowledged(incident_id)
        return jsonify(item)

    @app.route("/api/incidents/<incident_id>/restore", methods=["POST"])
    def incident_restore(incident_id):
        if not _valid_id(incident_id):
            return jsonify({"status": "error", "reason": "bad incident id"}), 400
        svc = _service()
        if svc is None:
            return _no_service()
        result = svc.restore_incident(incident_id)
        return jsonify(result), (200 if result.get("status") == "ok" else 404)

    @app.route("/api/scan", methods=["GET", "POST"])
    def scan():
        svc = _service()
        if svc is None:
            return _no_service()
        if request.method == "POST":
            result = svc.scan_now()
            return jsonify(result), (200 if result["status"] in ("started", "running") else 409)
        return jsonify(svc.last_scan)

    @app.route("/api/scan/restore", methods=["POST"])
    def scan_restore():
        svc = _service()
        if svc is None:
            return _no_service()
        return jsonify(svc.restore_from_scan())

    @app.route("/api/stats")
    def stats():
        incident_store.reload()
        reports = incident_store.list(limit=100)
        if reports:
            secs = [i["restore_seconds"] for i in (incident_store.get(r["id"]) for r in reports)
                    if i and i.get("restore_seconds") is not None]
            return jsonify({"total_threats_detected": len(reports),
                            "total_files_restored": sum(r["counts"]["restored"] for r in reports),
                            "avg_ttd_seconds": None,
                            "avg_ttr_seconds": round(sum(secs) / len(secs), 3) if secs else None})
        all_alerts = read_alerts(limit=1000, include_acknowledged=True)
        if not all_alerts:
            return jsonify({"total_threats_detected": 0, "total_files_restored": 0,
                            "avg_ttd_seconds": None, "avg_ttr_seconds": None})
        acts = [a.get("action_taken") or {} for a in all_alerts]
        ttrs = [x["restore_elapsed_seconds"] for x in acts if x.get("restore_elapsed_seconds") is not None]
        return jsonify({"total_threats_detected": len(all_alerts),
                        "total_files_restored": sum(x.get("files_restored", 0) or 0 for x in acts),
                        "avg_ttd_seconds": None,
                        "avg_ttr_seconds": round(sum(ttrs) / len(ttrs), 3) if ttrs else None})

    # ---- demo overlay (present only in the DEMO set)
    import importlib.util
    if importlib.util.find_spec("src.dashboard.demo_routes") is not None:
        from src.dashboard.demo_routes import register_demo
        register_demo(app)
        app.config["DEMO_MODE"] = True

    return app


app = create_app() if os.environ.get("NEURASHIELD_NO_AUTO_APP") != "1" else None


class _Server:
    """Small wrapper so Layer2 can start/stop the dashboard in a thread."""
    def __init__(self, server):
        self._server = server

    def run(self):
        self._server.run()

    def close(self):
        self._server.close()


def create_server(host=None, port=None, service=None):
    from waitress import create_server as _create
    host = host or os.environ.get("NEURASHIELD_DASHBOARD_HOST", DEFAULT_HOST)
    port = int(port or os.environ.get("NEURASHIELD_DASHBOARD_PORT", DEFAULT_PORT))
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(f"[dashboard] Listening on {host}: the dashboard is reachable from the network.")
    target = app or create_app()
    if service is not None:
        target.config["SERVICE"] = service
    return _Server(_create(target, host=host, port=port, threads=8))


if __name__ == "__main__":
    server = create_server()
    print(f"[dashboard] http://{os.environ.get('NEURASHIELD_DASHBOARD_HOST', DEFAULT_HOST)}:"
          f"{os.environ.get('NEURASHIELD_DASHBOARD_PORT', DEFAULT_PORT)}  (Ctrl+C to stop)")
    try:
        server.run()
    except KeyboardInterrupt:
        server.close()

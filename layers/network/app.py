# app.py
# NeuraShield — Network Guardian Dashboard
# Run with: venv\Scripts\activate && python app.py
# monitor.py starts automatically when app.py runs

from flask import Flask, jsonify, render_template_string, request
import json, os, sys, subprocess, threading, signal
from datetime import datetime, timezone, timedelta
from collections import defaultdict

app = Flask(__name__)

# Runtime data directory — shared with monitor-2.py (the supervisor sets
# NEURASHIELD_DATA_DIR for both). Both must read the same two files.
_DATA_DIR    = os.environ.get("NEURASHIELD_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(_DATA_DIR, exist_ok=True)
ALERT_FILE   = os.path.join(_DATA_DIR, "alerts.json")
FLOW_LOG     = os.path.join(_DATA_DIR, "flow_log.json")
monitor_proc = None

# ── API-key protection (same contract as the rest of NeuraShield) ──────────
# The dashboard's JSON endpoints require the shared X-API-Key when a key is
# configured (NEURASHIELD_API_KEY, set by the supervisor). This closes the
# previously open local API. If no key is set (standalone dev run), the guard
# is a no-op so the layer still works on its own.
import hmac as _hmac
def _shared_api_key():
    k = (os.environ.get("NEURASHIELD_API_KEY") or "").strip()
    if k:
        return k
    path = os.environ.get("NEURASHIELD_KEY_FILE")
    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError:
            pass
    return ""

_API_KEY = _shared_api_key()


def _require_api_key():
    """Return a Flask (response, status) tuple to abort with, or None if OK."""
    if not _API_KEY:
        return None  # standalone mode: no key configured
    provided = (request.headers.get("X-API-Key") or "").strip()
    if not provided or not _hmac.compare_digest(provided, _API_KEY):
        return jsonify({"detail": "Missing or invalid X-API-Key header."}), 401
    return None

# ── Auto-start monitor.py ──────────────────────────────────────
def _monitor_script():
    """Path of monitor-2.py: next to this file, or (packaged exe) in the bundle."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "monitor-2.py")]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands += [os.path.join(meipass, "layers", "network", "monitor-2.py"),
                  os.path.join(meipass, "monitor-2.py")]
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def start_monitor():
    global monitor_proc
    python = sys.executable
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script = _monitor_script()
    # Packaged exe: sys.executable IS this program, so "python monitor-2.py"
    # would start a second copy of the dashboard (which would start another...).
    # The exe re-enters itself with --monitor instead (handled in __main__).
    monitor_cmd = ([python, "--monitor"] if getattr(sys, "frozen", False)
                   else [python, script])

    if not os.path.exists(script):
        print("  WARNING: monitor-2.py not found")
        return

    if monitor_proc is not None and monitor_proc.poll() is None:
        print(f"  monitor-2.py is already running (PID {monitor_proc.pid})")
        return

    try:
        # Run the capture process headless: its banner uses non-ASCII symbols, so
        # force UTF-8 and send its output to a log file in the data dir rather
        # than opening a console window (and so a redirected stdout cannot crash it).
        child_env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                         PYTHONUNBUFFERED="1")
        log_fh = open(os.path.join(_DATA_DIR, "monitor.log"), "ab", buffering=0)
        monitor_proc = subprocess.Popen(
            monitor_cmd,
            cwd=os.path.dirname(script),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW
                          if sys.platform == "win32" else 0,
        )
        print(f"  monitor-2.py started (PID {monitor_proc.pid}) - log: {log_fh.name}")
        print("  NOTE: Packet capture may require Administrator/root privileges.")
    except Exception as e:
        print(f"  Could not start monitor-2.py: {e}")
        print("  Run monitor-2.py manually as Administrator/root if needed")

def stop_monitor():
    global monitor_proc
    if monitor_proc and monitor_proc.poll() is None:
        monitor_proc.terminate()
        try: monitor_proc.wait(timeout=5)
        except: monitor_proc.kill()
        print("  monitor-2.py stopped")


def categorize_threat(threat_type, message, status):
    t = (threat_type + " " + message).lower()
    if any(x in t for x in ["dos","flood","denial","icmp","syn"]):
        return "Denial of Service"
    if any(x in t for x in ["port scan","probe","sweep","scan","recon"]):
        return "Network Scan"
    if any(x in t for x in ["exfil","large outbound","data transfer","dns tunnel"]):
        return "Data Exfiltration"
    if any(x in t for x in ["c2","command","control","metasploit","irc","tor",
                              "suspicious port","4444","6667","31337","9001"]):
        return "Command & Control"
    if any(x in t for x in ["brute","auth","login","credential","rdp","ssh"]):
        return "Brute Force"
    if any(x in t for x in ["database","db access","sql","mongo","redis"]):
        return "Unauthorized Access"
    if any(x in t for x in ["inbound","remote admin","external"]):
        return "Suspicious Inbound"
    if any(x in t for x in ["ml anomaly","dual model","autoencoder","isolation","anomaly"]):
        return "ML Anomaly"
    if status == "ALERT":
        return "Unknown Threat"
    return "Suspicious Activity"

def risk_score(alert):
    score = 0
    if alert.get("status") == "ALERT":   score += 50
    elif alert.get("status") == "WARNING": score += 25
    conf = float(alert.get("confidence", 0) or 0)
    score += int(conf * 30)
    if alert.get("ml_flagged"): score += 10
    if alert.get("if_anomaly"): score += 5
    if alert.get("ae_anomaly"): score += 5
    return min(score, 100)

def read_json_lines(filepath):
    items = []
    if not os.path.exists(filepath):
        return items
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except:
                        pass
    except:
        pass
    return items

def filter_last_hour(items, time_key="timestamp"):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    out = []
    for item in items:
        try:
            ts_str = item.get(time_key,"")
            if ts_str:
                ts_str = ts_str.replace("Z","+00:00")
                if "+" not in ts_str and ts_str.count("-") < 3:
                    ts_str += "+00:00"
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    out.append(item)
        except:
            out.append(item)
    return out

def deduplicate_alerts(alerts):
    grouped = {}
    for alert in alerts:
        src    = alert.get("details",{}).get("source_ip","?")
        ttype  = alert.get("threat_type","?")
        status = alert.get("status","?")
        key    = f"{src}_{ttype}_{status}"
        if key in grouped:
            grouped[key]["count"] = grouped[key].get("count",1) + 1
            grouped[key]["last_seen"] = alert.get("timestamp")
            if alert.get("status") == "ALERT":
                grouped[key]["status"] = "ALERT"
        else:
            alert["count"]     = 1
            alert["last_seen"] = alert.get("timestamp")
            grouped[key]       = alert
    result = list(grouped.values())
    for a in result:
        if a.get("count",1) > 1:
            a["message"] = f"[×{a['count']}] " + a.get("message","")
    return result

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Network Guardian — NeuraShield</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=Plus+Jakarta+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
/* ── Theme tokens ── */
[data-theme="dark"] {
  --bg:       #0f1117;
  --surface:  #161a24;
  --surface2: #1c2130;
  --surface3: #212738;
  --border:   #252c3d;
  --border2:  #2e3650;
  --text:     #ffffff;
  --text2:    #c8d0e4;
  --text3:    #6a7490;
  --accent:   #3b7eff;
  --red:      #e05252;
  --amber:    #d4870a;
  --green:    #2db87a;
  --shadow:   rgba(0,0,0,.4);
}
[data-theme="light"] {
  --bg:       #f0f2f7;
  --surface:  #ffffff;
  --surface2: #f5f7fb;
  --surface3: #eaecf4;
  --border:   #dde1ee;
  --border2:  #c8cee0;
  --text:     #0f1117;
  --text2:    #3a4260;
  --text3:    #7a85a0;
  --accent:   #2563eb;
  --red:      #dc3545;
  --amber:    #b45309;
  --green:    #16a34a;
  --shadow:   rgba(0,0,0,.08);
}

/* ── Base ── */
*{margin:0;padding:0;box-sizing:border-box;}
html,body{height:100%;}
body{
  background:var(--bg);color:var(--text);
  font-family:'Plus Jakarta Sans',system-ui,sans-serif;
  font-size:13.5px;line-height:1.5;
  transition:background .2s,color .2s;
}
button{font-family:inherit;cursor:pointer;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}

/* ── NAV ── */
nav{
  height:54px;
  display:grid;
  grid-template-columns:1fr auto 1fr;
  align-items:center;
  padding:0 20px;
  background:var(--surface);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:300;
  box-shadow:0 1px 8px var(--shadow);
}
.sidebar-toggle{
  width:34px;height:34px;border-radius:8px;
  border:1px solid var(--border2);background:var(--surface2);
  display:flex;align-items:center;justify-content:center;
  color:var(--text2);font-size:16px;transition:all .15s;
  margin-right:10px;flex-shrink:0;
}
.sidebar-toggle:hover{background:var(--surface3);color:var(--text);}
.brand{display:flex;align-items:center;gap:11px;}
.logo-mark{
  width:34px;height:34px;background:var(--accent);
  border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.logo-mark svg{width:20px;height:20px;fill:none;stroke:#fff;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;}
.brand-name{font-family:'Archivo Black',sans-serif;font-size:15px;color:var(--text);letter-spacing:0;}
.brand-tag{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text3);letter-spacing:1px;}
.nav-tabs{display:flex;gap:1px;background:var(--bg);border:1px solid var(--border);border-radius:7px;padding:3px;justify-self:center;}
.ntab{
  padding:5px 18px;border-radius:5px;border:none;
  background:none;color:var(--text3);
  font-family:'Plus Jakarta Sans',sans-serif;font-size:12.5px;font-weight:500;
  transition:all .15s;
}
.ntab.active{background:var(--surface3);color:var(--text);}
.ntab:hover:not(.active){color:var(--text2);}
.nav-right{display:flex;align-items:center;gap:10px;justify-self:end;}
.clock{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--text3);}
.live-pill{
  display:flex;align-items:center;gap:6px;padding:4px 11px;border-radius:99px;
  background:rgba(45,184,122,.08);border:1px solid rgba(45,184,122,.2);
  font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:500;color:var(--green);
}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:blink 1.8s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.2;}}

/* Theme toggle */
.theme-toggle{
  width:34px;height:34px;border-radius:8px;border:1px solid var(--border2);
  background:var(--surface2);display:flex;align-items:center;justify-content:center;
  color:var(--text2);font-size:15px;transition:all .15s;
}
.theme-toggle:hover{background:var(--surface3);color:var(--text);}

/* ── LAYOUT ── */
.app{display:flex;height:calc(100vh - 54px);overflow:hidden;}

/* ── LEFT SIDEBAR ── */
.sidebar{
  transition:width .25s ease, opacity .25s ease, min-width .25s ease;
}
.sidebar.collapsed{
  width:0 !important;
  min-width:0 !important;
  opacity:0;
  overflow:hidden;
  padding:0;
}
/* ── LEFT SIDEBAR (nav items like teammate's UI) ── */
.sidebar{
  width:200px;
  background:var(--surface);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;
  flex-shrink:0;overflow-y:auto;
  padding:16px 0;
}
.sb-section{
  padding:10px 16px 4px;
  font-family:'Roboto',sans-serif;font-size:10px;font-weight:500;
  letter-spacing:1.2px;text-transform:uppercase;color:var(--text3);
}
.sb-item{
  display:flex;align-items:center;gap:10px;
  padding:8px 16px;margin:1px 8px;border-radius:7px;
  border:none;background:none;width:calc(100% - 16px);text-align:left;
  color:var(--text2);font-family:'Plus Jakarta Sans',sans-serif;
  font-size:13px;font-weight:400;transition:all .12s;
}
.sb-item:hover{background:var(--surface2);color:var(--text);}
.sb-item.active{background:rgba(59,126,255,.12);color:var(--accent);font-weight:500;}
.sb-icon{font-size:14px;flex-shrink:0;width:18px;text-align:center;}
.sb-divider{height:1px;background:var(--border);margin:10px 16px;}

/* Sidebar info blocks */
.sb-info-block{padding:4px 14px 8px;}
.sb-info-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:5px 0;border-bottom:1px solid var(--border);
}
.sb-info-row:last-child{border-bottom:none;}
.sb-info-key{font-family:'Roboto',sans-serif;font-size:11px;color:var(--text3);}
.sb-info-val{
  font-family:'JetBrains Mono',monospace;font-size:11px;
  color:var(--text2);text-align:right;
}
.sb-info-val.ok  {color:var(--green);}
.sb-info-val.bad {color:var(--red);}
.sb-info-val.warn{color:var(--amber);}

/* Sidebar model rows */
.sb-model-row{
  display:flex;align-items:center;gap:10px;
  padding:8px 0;border-bottom:1px solid var(--border);
}
.sb-model-row:last-child{border-bottom:none;}
.sb-model-dot{
  width:8px;height:8px;border-radius:50%;flex-shrink:0;
  background:var(--text3);
}
.sb-model-dot.on {background:var(--green);}
.sb-model-dot.off{background:var(--red);}
.sb-model-info{}
.sb-model-name{
  font-family:'Plus Jakarta Sans',sans-serif;font-size:12px;
  font-weight:500;color:var(--text);
}
.sb-model-sub{
  font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text3);
  margin-top:1px;
}

/* Sidebar stat rows */
.sb-stat-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:5px 0;border-bottom:1px solid var(--border);
}
.sb-stat-row:last-child{border-bottom:none;}
.sb-stat-label{font-family:'Roboto',sans-serif;font-size:11px;color:var(--text3);}
.sb-stat-val{
  font-family:'JetBrains Mono',monospace;font-size:12px;
  font-weight:500;color:var(--text);
}
.sb-stat-val.bad {color:var(--red);}
.sb-stat-val.warn{color:var(--amber);}
.sb-stat-val.ok  {color:var(--green);}

/* ── MAIN ── */
.main{flex:1;overflow-y:auto;}
.tab-page{display:none;}
.tab-page.active{display:block;}
.page-pad{padding:20px 22px 48px;}

/* KPI row */
.kpi-row{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px;}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:9px;padding:15px 16px;}
.kpi-label{
  font-family:'Roboto',sans-serif;font-size:11px;font-weight:500;
  color:var(--text3);letter-spacing:.3px;margin-bottom:9px;
}
.kpi-val{
  font-family:'JetBrains Mono',monospace;
  font-size:26px;font-weight:500;color:var(--text);line-height:1;
}
.kpi.red   .kpi-val{color:var(--red);}
.kpi.amber .kpi-val{color:var(--amber);}
.kpi-hint{font-family:'Roboto',sans-serif;font-size:11px;color:var(--text3);margin-top:5px;}

/* Model bar */
.model-bar{
  background:var(--surface);border:1px solid var(--border);
  border-radius:9px;padding:10px 16px;margin-bottom:16px;
  display:flex;align-items:center;gap:10px;flex-wrap:wrap;
}
.model-lbl{font-family:'Roboto',sans-serif;font-size:10px;font-weight:500;
            letter-spacing:1px;text-transform:uppercase;color:var(--text3);margin-right:6px;}
.m-chip{
  display:flex;align-items:center;gap:6px;padding:4px 11px;
  border-radius:5px;border:1px solid;
  font-family:'JetBrains Mono',monospace;font-size:11px;
}
.m-chip.on {background:rgba(45,184,122,.07);border-color:rgba(45,184,122,.2);color:var(--green);}
.m-chip.off{background:rgba(224,82,82,.07);border-color:rgba(224,82,82,.2);color:var(--red);}
.m-dot{width:5px;height:5px;border-radius:50%;background:currentColor;}
.m-info{margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text3);}

/* Chart row */
.chart-row{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-bottom:16px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:9px;padding:16px;}
.card-title{
  font-family:'Roboto',sans-serif;font-size:11px;font-weight:500;
  letter-spacing:1px;text-transform:uppercase;color:var(--text3);margin-bottom:14px;
}

/* Bot grid */
.bot-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:9px;overflow:hidden;}
.panel-head{
  padding:11px 14px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  background:var(--surface2);
}
/* WHITE BOLD panel title */
.panel-title{
  font-family:'Plus Jakarta Sans',sans-serif;font-size:13px;
  font-weight:700;color:var(--text);letter-spacing:0;
}
.panel-cnt{
  font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--text3);
  background:var(--bg);border:1px solid var(--border);padding:2px 8px;border-radius:5px;
}

/* Alert feed */
.alert-feed{padding:8px;max-height:310px;overflow-y:auto;}
.af{
  display:flex;align-items:flex-start;gap:10px;padding:10px 11px;
  border-radius:7px;margin-bottom:4px;
  background:var(--bg);border:1px solid var(--border);
  cursor:pointer;transition:border-color .12s;
}
.af:hover{border-color:var(--border2);}
.af.a{border-left:2px solid var(--red);}
.af.w{border-left:2px solid var(--amber);}
.af-body{flex:1;min-width:0;}
.af-top{display:flex;align-items:center;gap:6px;margin-bottom:3px;flex-wrap:wrap;}
/* WHITE BOLD category name */
.af-cat{font-family:'Plus Jakarta Sans',sans-serif;font-size:13px;font-weight:700;color:var(--text);}
/* WHITE BOLD IPs */
.af-ips{
  font-family:'JetBrains Mono',monospace;font-size:11px;
  color:#ffffff;font-weight:500;margin-bottom:2px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
[data-theme="light"] .af-ips{color:var(--accent);}
.af-msg{font-family:'Plus Jakarta Sans',sans-serif;font-size:11.5px;color:var(--text3);
         white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.af-right{text-align:right;flex-shrink:0;}
.af-time{font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--text3);}
.risk-bar{width:44px;height:3px;background:var(--border);border-radius:2px;margin-top:5px;margin-left:auto;}
.risk-fill{height:100%;border-radius:2px;}

/* Badges */
.badge{
  font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:500;
  padding:1px 7px;border-radius:4px;border:1px solid;
}
.badge.ALERT  {color:var(--red);  border-color:rgba(224,82,82,.3); background:rgba(224,82,82,.07);}
.badge.WARNING{color:var(--amber);border-color:rgba(212,135,10,.3);background:rgba(212,135,10,.07);}
.badge.NORMAL {color:var(--green);border-color:rgba(45,184,122,.3);background:rgba(45,184,122,.07);}
.ml-tag{
  font-family:'JetBrains Mono',monospace;font-size:10px;
  color:var(--text3);background:var(--surface2);
  border:1px solid var(--border2);padding:1px 6px;border-radius:4px;
}

/* IP list */
.ip-list{padding:8px 14px;max-height:310px;overflow-y:auto;}
.ip-row{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--border);}
.ip-row:last-child{border-bottom:none;}
/* WHITE BOLD IP address */
.ip-addr{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:500;color:#ffffff;flex:1;}
[data-theme="light"] .ip-addr{color:var(--accent);}
.ip-bar-bg{flex:2;background:var(--bg);border-radius:2px;height:4px;}
.ip-bar-fill{height:4px;border-radius:2px;background:var(--red);opacity:.5;}
.ip-cnt{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text3);min-width:24px;text-align:right;}

/* empty */
.empty{padding:40px 20px;text-align:center;color:var(--text3);
        font-family:'Roboto',sans-serif;font-size:12px;}

/* ── INCIDENTS TAB ── */
.inc-wrap{padding:20px 22px;}
.inc-toolbar{display:flex;align-items:center;gap:8px;margin-bottom:14px;}
.inc-search{
  flex:1;max-width:300px;
  background:var(--surface);border:1px solid var(--border2);
  border-radius:6px;padding:7px 12px;color:var(--text);outline:none;
  font-family:'JetBrains Mono',monospace;font-size:12px;
}
.inc-search:focus{border-color:var(--accent);}
.inc-search::placeholder{color:var(--text3);}
.sort-btn{
  padding:6px 12px;background:var(--surface);border:1px solid var(--border2);
  border-radius:6px;color:var(--text2);font-family:'Plus Jakarta Sans',sans-serif;
  font-size:12px;transition:all .12s;
}
.sort-btn:hover,.sort-btn.on{background:var(--surface3);color:var(--text);}
.export-btn{
  margin-left:auto;padding:6px 14px;
  background:var(--surface);border:1px solid var(--border2);
  border-radius:6px;color:var(--text2);
  font-family:'Plus Jakarta Sans',sans-serif;font-size:12px;transition:all .12s;
}
.export-btn:hover{color:var(--text);}
.table-wrap{
  background:var(--surface);border:1px solid var(--border);
  border-radius:9px;overflow:hidden;
}
table{width:100%;border-collapse:collapse;}
th{
  padding:10px 14px;text-align:left;
  font-family:'Roboto',sans-serif;font-size:10px;font-weight:500;
  letter-spacing:1.2px;text-transform:uppercase;color:var(--text3);
  background:var(--surface2);border-bottom:1px solid var(--border);
  cursor:pointer;user-select:none;white-space:nowrap;
}
th:hover{color:var(--text);}
td{padding:10px 14px;border-bottom:1px solid var(--border);font-size:12.5px;}
tr:last-child td{border-bottom:none;}
tr:hover td{background:var(--surface2);cursor:pointer;}
/* WHITE BOLD table IPs */
.td-ip{font-family:'JetBrains Mono',monospace;font-size:11.5px;font-weight:500;color:#ffffff;}
[data-theme="light"] .td-ip{color:var(--accent);}
.td-mono{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--text2);}
.td-dim{font-family:'Plus Jakarta Sans',sans-serif;color:var(--text3);
         max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.risk-pill{
  display:inline-block;min-width:34px;padding:2px 7px;border-radius:4px;
  font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:500;text-align:center;
}
.rh{color:var(--red);  background:rgba(224,82,82,.08); border:1px solid rgba(224,82,82,.2);}
.rm{color:var(--amber);background:rgba(212,135,10,.08);border:1px solid rgba(212,135,10,.2);}
.rl{color:var(--green);background:rgba(45,184,122,.08);border:1px solid rgba(45,184,122,.2);}
.pager{
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 16px;border-top:1px solid var(--border);background:var(--surface2);
}
.pg-info{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text3);}
.pg-btns{display:flex;gap:3px;}
.pg-btn{
  padding:4px 10px;background:var(--surface);border:1px solid var(--border);
  border-radius:5px;color:var(--text2);font-family:'JetBrains Mono',monospace;font-size:11px;
  transition:all .12s;
}
.pg-btn:hover,.pg-btn.on{background:rgba(59,126,255,.1);color:var(--accent);border-color:rgba(59,126,255,.3);}
.pg-btn:disabled{opacity:.3;cursor:not-allowed;}

/* ── ANALYTICS TAB ── */
.a-wrap{padding:20px 22px 48px;}
/* Search in analytics */
.analytics-search-wrap{margin-bottom:16px;}
.analytics-search{
  width:100%;max-width:380px;
  background:var(--surface);border:1px solid var(--border2);
  border-radius:6px;padding:8px 14px;color:var(--text);outline:none;
  font-family:'JetBrains Mono',monospace;font-size:12px;
}
.analytics-search:focus{border-color:var(--accent);}
.analytics-search::placeholder{color:var(--text3);}
.a-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.a-grid.full{grid-template-columns:1fr;}
.heatmap-row{display:grid;grid-template-columns:repeat(24,1fr);gap:2px;}
.hm{height:22px;border-radius:3px;background:var(--border);cursor:pointer;position:relative;}
.hm:hover::after{
  content:attr(data-t);
  position:absolute;bottom:calc(100% + 4px);left:50%;transform:translateX(-50%);
  background:var(--surface3);border:1px solid var(--border2);border-radius:4px;
  padding:3px 7px;font-size:10px;font-family:'JetBrains Mono',monospace;
  color:var(--text);white-space:nowrap;z-index:10;
}
.hm-lbl-row{display:grid;grid-template-columns:repeat(24,1fr);gap:2px;margin-top:4px;}
.hm-lbl{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--text3);text-align:center;}
.m-tbl{width:100%;border-collapse:collapse;}
.m-tbl td{padding:8px 10px;border-bottom:1px solid var(--border);font-size:12.5px;}
.m-tbl tr:last-child td{border-bottom:none;}
.m-key{color:var(--text3);font-family:'Roboto',sans-serif;font-size:11.5px;width:60%;}
/* WHITE BOLD metric values */
.m-val{font-family:'JetBrains Mono',monospace;font-size:12.5px;font-weight:500;color:#ffffff;text-align:right;}
[data-theme="light"] .m-val{color:var(--text);}
.m-val.ok {color:var(--green);}
.m-val.w  {color:var(--amber);}
.m-val.bad{color:var(--red);}
/* Filtered results in analytics */
.analytics-results{margin-top:14px;}
.filtered-table-wrap{
  background:var(--surface);border:1px solid var(--border);
  border-radius:9px;overflow:hidden;margin-top:10px;
}

/* ── MODAL ── */
.ov{
  position:fixed;inset:0;background:rgba(0,0,0,.65);backdrop-filter:blur(4px);
  z-index:500;display:flex;align-items:center;justify-content:center;padding:20px;
  opacity:0;pointer-events:none;transition:opacity .18s;
}
.ov.open{opacity:1;pointer-events:all;}
.modal{
  background:var(--surface);border:1px solid var(--border2);border-radius:12px;
  width:100%;max-width:700px;max-height:88vh;overflow:hidden;
  display:flex;flex-direction:column;
  transform:translateY(10px);transition:transform .18s;
}
.ov.open .modal{transform:translateY(0);}
.modal-head{
  padding:16px 20px;border-bottom:1px solid var(--border);
  display:flex;align-items:flex-start;justify-content:space-between;
  background:var(--surface2);
}
/* WHITE BOLD modal title */
.modal-title{
  font-family:'Plus Jakarta Sans',sans-serif;font-size:16px;
  font-weight:700;color:var(--text);
}
.modal-sub{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text3);margin-top:3px;}
.modal-x{
  background:none;border:none;color:var(--text3);font-size:18px;
  padding:2px 6px;border-radius:4px;transition:all .12s;flex-shrink:0;margin-left:12px;
}
.modal-x:hover{color:var(--text);background:var(--surface3);}
.modal-body{overflow-y:auto;padding:20px;}
.sec-title{
  font-family:'Roboto',sans-serif;font-size:10px;font-weight:500;
  letter-spacing:1.5px;text-transform:uppercase;color:var(--text3);
  padding-bottom:8px;border-bottom:1px solid var(--border);margin-bottom:12px;
}
.attr-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.attr{background:var(--bg);border:1px solid var(--border);border-radius:7px;padding:10px 13px;}
.attr-k{
  font-family:'Roboto',sans-serif;font-size:10px;font-weight:500;
  letter-spacing:.5px;text-transform:uppercase;color:var(--text3);margin-bottom:5px;
}
.attr-v{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--text);word-break:break-all;}
.attr-v.red   {color:var(--red);}
.attr-v.amber {color:var(--amber);}
.attr-v.green {color:var(--green);}
.attr-v.accent{color:var(--accent);}
/* WHITE BOLD IP in modal */
.attr-v.ip{color:#ffffff;font-weight:500;}
[data-theme="light"] .attr-v.ip{color:var(--accent);}
.m-sec{margin-bottom:20px;}
.gauge{display:flex;align-items:center;gap:12px;margin-top:8px;}
.gauge-track{flex:1;height:6px;background:var(--bg);border-radius:3px;overflow:hidden;}
.gauge-fill{height:100%;border-radius:3px;transition:width .4s;}
.gauge-lbl{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:500;min-width:38px;}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div style="display:flex;align-items:center;">
    <button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()" title="Toggle sidebar">
      <span id="toggleIcon">☰</span>
    </button>
    <div class="brand">
      <div class="logo-mark">
        <svg viewBox="0 0 24 24">
          <path d="M12 2L4 6v6c0 5.25 3.5 10.15 8 11.35C16.5 22.15 20 17.25 20 12V6l-8-4z"/>
          <polyline points="9 12 11 14 15 10"/>
        </svg>
      </div>
      <div>
        <div class="brand-name">Network Guardian</div>
        <div class="brand-tag">NEURASHIELD</div>
      </div>
    </div>
  </div>

  <div class="nav-tabs">
    <button class="ntab active" onclick="switchTab('dashboard',this)">Dashboard</button>
    <button class="ntab" onclick="switchTab('incidents',this)">Incidents</button>
    <button class="ntab" onclick="switchTab('analytics',this)">Analytics</button>
  </div>

  <div class="nav-right">
    <div class="clock" id="clock">--:--:-- UTC</div>
    <button class="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark">
      <span id="themeIcon">☀</span>
    </button>
    <div class="live-pill"><span class="live-dot"></span>Live</div>
  </div>
</nav>

<div class="app">

<!-- LEFT SIDEBAR -->
<div class="sidebar">

  <div class="sb-section">ML Models</div>
  <div class="sb-info-block">
    <div class="sb-model-row" id="sbIfRow">
      <span class="sb-model-dot" id="sbIfDot"></span>
      <div class="sb-model-info">
        <div class="sb-model-name">Isolation Forest</div>
        <div class="sb-model-sub">Ensemble Detector</div>
      </div>
    </div>
    <div class="sb-model-row" id="sbAeRow">
      <span class="sb-model-dot" id="sbAeDot"></span>
      <div class="sb-model-info">
        <div class="sb-model-name">Autoencoder</div>
        <div class="sb-model-sub">Neural Anomaly Detector</div>
      </div>
    </div>
  </div>

  <div class="sb-divider"></div>
  <div class="sb-section">Session Stats</div>
  <div class="sb-info-block" id="sbStats">
    <div class="sb-stat-row">
      <span class="sb-stat-label">Alerts</span>
      <span class="sb-stat-val bad" id="sbAlert">0</span>
    </div>
    <div class="sb-stat-row">
      <span class="sb-stat-label">Warnings</span>
      <span class="sb-stat-val warn" id="sbWarn">0</span>
    </div>
    <div class="sb-stat-row">
      <span class="sb-stat-label">Normal</span>
      <span class="sb-stat-val ok" id="sbNormal">0</span>
    </div>
    <div class="sb-stat-row">
      <span class="sb-stat-label">ML Flags</span>
      <span class="sb-stat-val" id="sbML">0</span>
    </div>
  </div>

  <div class="sb-divider"></div>
  <div class="sb-section">System</div>
  <div class="sb-info-block">
    <div class="sb-info-row">
      <span class="sb-info-key">Features</span>
      <span class="sb-info-val">40</span>
    </div>
    <div class="sb-info-row">
      <span class="sb-info-key">Privacy</span>
      <span class="sb-info-val ok">No payload inspect</span>
    </div>
    <div class="sb-info-row">
      <span class="sb-info-key">Zero-day</span>
      <span class="sb-info-val ok">Enabled</span>
    </div>
  </div>

</div>

<!-- MAIN CONTENT -->
<div class="main">

<!-- DASHBOARD -->
<div class="tab-page active" id="tab-dashboard">
<div class="page-pad">

  <div class="kpi-row">
    <div class="kpi">
      <div class="kpi-label">Total Flows</div>
      <div class="kpi-val" id="kTotal">0</div>
      <div class="kpi-hint">Captured</div>
    </div>
    <div class="kpi red">
      <div class="kpi-label">Alerts</div>
      <div class="kpi-val" id="kAlert">0</div>
      <div class="kpi-hint">High severity</div>
    </div>
    <div class="kpi amber">
      <div class="kpi-label">Warnings</div>
      <div class="kpi-val" id="kWarn">0</div>
      <div class="kpi-hint">Needs review</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Normal</div>
      <div class="kpi-val" id="kNormal">0</div>
      <div class="kpi-hint">Safe</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">ML Detections</div>
      <div class="kpi-val" id="kML">0</div>
      <div class="kpi-hint">Model flags</div>
    </div>
  </div>

  <div class="model-bar">
    <span class="model-lbl">Detection Models</span>
    <div class="m-chip" id="ifChip"><span class="m-dot"></span>Isolation Forest Ensemble</div>
    <div class="m-chip" id="aeChip"><span class="m-dot"></span>Autoencoder</div>
    <span class="m-info" id="mInfo">Checking…</span>
  </div>

  <div class="chart-row">
    <div class="card">
      <div class="card-title">Event Timeline — last 60 min</div>
      <div style="height:170px;position:relative;"><canvas id="timeline"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Threat Distribution</div>
      <div style="height:170px;position:relative;"><canvas id="donut"></canvas></div>
    </div>
  </div>

  <div class="bot-grid">
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">Recent Detections</span>
        <span class="panel-cnt" id="feedCnt">0</span>
      </div>
      <div class="alert-feed" id="feed">
        <div class="empty">No detections yet</div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">Top Source IPs</span>
        <span class="panel-cnt" id="ipCnt">0</span>
      </div>
      <div class="ip-list" id="ipList">
        <div class="empty">No data yet</div>
      </div>
    </div>
  </div>

</div>
</div>

<!-- INCIDENTS -->
<div class="tab-page" id="tab-incidents">
<div class="inc-wrap">
  <div class="inc-toolbar">
    <input class="inc-search" id="incQ" placeholder="Search incidents…" oninput="renderTable()">
    <button class="sort-btn" id="sTime" onclick="setSort('time')">Time ↓</button>
    <button class="sort-btn" id="sRisk" onclick="setSort('risk')">Risk</button>
    <button class="sort-btn" id="sSev"  onclick="setSort('sev')">Severity</button>
    <button class="export-btn" onclick="exportCSV()">Export CSV</button>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th onclick="setSort('time')">Time</th>
        <th onclick="setSort('sev')">Severity</th>
        <th onclick="setSort('risk')">Risk</th>
        <th>Category</th>
        <th>Source IP</th>
        <th>Destination</th>
        <th>Protocol</th>
        <th>ML</th>
        <th>Message</th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table>
    <div class="pager">
      <span class="pg-info" id="pgInfo">0 incidents</span>
      <div class="pg-btns" id="pgBtns"></div>
    </div>
  </div>
</div>
</div>

<!-- ANALYTICS -->
<div class="tab-page" id="tab-analytics">
<div class="a-wrap">

  <!-- Search bar in Analytics tab -->
  <div class="analytics-search-wrap">
    <input class="analytics-search" id="analyticsQ"
           placeholder="Search by IP, category, or threat type…"
           oninput="renderAnalyticsSearch()">
  </div>

  <div class="a-grid full" style="margin-bottom:14px;">
    <div class="card">
      <div class="card-title">Alert Frequency by Hour (UTC)</div>
      <div class="heatmap-row" id="heatmap"></div>
      <div class="hm-lbl-row" id="hmLabels"></div>
    </div>
  </div>
  <div class="a-grid">
    <div class="card">
      <div class="card-title">Detection Metrics</div>
      <table class="m-tbl">
        <tr><td class="m-key">Total flows analysed</td><td class="m-val" id="mTotal">—</td></tr>
        <tr><td class="m-key">Alert rate</td><td class="m-val" id="mAR">—</td></tr>
        <tr><td class="m-key">ML detection rate</td><td class="m-val" id="mMR">—</td></tr>
        <tr><td class="m-key">Dual model flags</td><td class="m-val" id="mDual">—</td></tr>
        <tr><td class="m-key">Top threat category</td><td class="m-val" id="mTopC">—</td></tr>
        <tr><td class="m-key">Most flagged source</td><td class="m-val" id="mTopI">—</td></tr>
      </table>
    </div>
    <div class="card">
      <div class="card-title">Threats by Category</div>
      <div style="height:220px;position:relative;"><canvas id="barChart"></canvas></div>
    </div>
  </div>

  <!-- Analytics search results -->
  <div class="analytics-results" id="analyticsResults" style="display:none;">
    <div class="card">
      <div class="card-title">Search Results</div>
      <div class="filtered-table-wrap">
        <table>
          <thead><tr>
            <th>Time</th><th>Severity</th><th>Category</th>
            <th>Source IP</th><th>Destination</th><th>Message</th>
          </tr></thead>
          <tbody id="analyticsTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

</div>
</div>

</div><!-- /main -->


</div><!-- /app -->

<!-- MODAL -->
<div class="ov" id="ov" onclick="closeOv(event)">
  <div class="modal">
    <div class="modal-head">
      <div>
        <div class="modal-title" id="mTitle">Incident</div>
        <div class="modal-sub"   id="mSub">—</div>
      </div>
      <button class="modal-x" onclick="closeOvDirect()">✕</button>
    </div>
    <div class="modal-body" id="mBody"></div>
  </div>
</div>

<script>
// ── NeuraShield shared API key (injected server-side) ──────────
const NEURASHIELD_API_KEY = "__NEURASHIELD_API_KEY__";
const AUTH_HEADERS = NEURASHIELD_API_KEY ? { "X-API-Key": NEURASHIELD_API_KEY } : {};
// ── State ──────────────────────────────────────────────────────
let allAlerts  = [];
let activeCat  = 'All';
let activeSev  = 'All';
let sortF      = 'time';
let page       = 1;
const PG       = 20;
let tChart = null, dChart = null, bChart = null;
let currentTab = 'dashboard';

const CAT_COLOR = {
  'All':                 '#3b7eff',
  'Denial of Service':   '#e05252',
  'Network Scan':        '#d4870a',
  'Data Exfiltration':   '#e05252',
  'Command & Control':   '#e05252',
  'Brute Force':         '#d4870a',
  'Unauthorized Access': '#d4870a',
  'Suspicious Inbound':  '#6a7490',
  'ML Anomaly':          '#2db87a',
  'Suspicious Activity': '#6a7490',
  'Unknown Threat':      '#6a7490',
};

function riskColor(s){return s>=70?'#e05252':s>=40?'#d4870a':'#2db87a';}
function riskClass(s){return s>=70?'rh':s>=40?'rm':'rl';}
function riskOf(a){
  let s=0;
  if(a.status==='ALERT')s+=50;else if(a.status==='WARNING')s+=25;
  s+=Math.round((a.confidence||0)*30);
  if(a.ml_flagged)s+=10;if(a.if_anomaly)s+=5;if(a.ae_anomaly)s+=5;
  return Math.min(s,100);
}

// ── Theme ──────────────────────────────────────────────────────
function toggleTheme(){
  const html=document.documentElement;
  const isDark=html.getAttribute('data-theme')==='dark';
  html.setAttribute('data-theme', isDark?'light':'dark');
  document.getElementById('themeIcon').textContent = isDark ? '☾' : '☀';
  // Redraw charts for new theme
  renderTimeline(lastTimeline);
  renderDonut();
  if(currentTab==='analytics') renderAnalytics();
}
let lastTimeline = null;

// ── Clock ──────────────────────────────────────────────────────
function tick(){
  const n=new Date(),p=v=>String(v).padStart(2,'0');
  document.getElementById('clock').textContent=
    `${p(n.getUTCHours())}:${p(n.getUTCMinutes())}:${p(n.getUTCSeconds())} UTC`;
}
setInterval(tick,1000);tick();

// ── Tab ────────────────────────────────────────────────────────
function switchTab(name, btn){
  currentTab = name;
  document.querySelectorAll('.tab-page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.ntab').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if(btn) btn.classList.add('active');
  
  if(name==='analytics') renderAnalytics();
}

function setSbActive(btn){
  // no-op — sidebar is now info-only not navigation
}

function toggleSidebar(){
  const sb   = document.querySelector('.sidebar');
  const icon = document.getElementById('toggleIcon');
  const collapsed = sb.classList.toggle('collapsed');
  icon.textContent = collapsed ? '→' : '☰';
}

// ── Load ───────────────────────────────────────────────────────
async function load(){
  try{
    const d=await(await fetch('/api/data',{headers:AUTH_HEADERS})).json();
    allAlerts=d.alerts||[];
    allAlerts.forEach(a=>a.risk_score=riskOf(a));
    renderKPIs(d.stats);
    renderModels(d.models);
    renderTimeline(d.timeline);
    renderDonut();
    renderFeed(filtered());
    document.getElementById('feedCnt').textContent=filtered().length;
    renderIPs();
    
    if(currentTab==='analytics') renderAnalytics();
    if(currentTab==='incidents') renderTable();
  }catch(e){}
}

function renderKPIs(s){
  document.getElementById('kTotal').textContent =s.total.toLocaleString();
  document.getElementById('kAlert').textContent =s.alert;
  document.getElementById('kWarn').textContent  =s.warning;
  document.getElementById('kNormal').textContent=s.normal.toLocaleString();
  document.getElementById('kML').textContent    =s.ml_flags;
  // Update right panel session stats
  const rpt=document.getElementById('rpTotal');if(rpt)rpt.textContent=s.total.toLocaleString();
  const rpa=document.getElementById('rpAlert'); if(rpa)rpa.textContent=s.alert;
  const rpw=document.getElementById('rpWarn');  if(rpw)rpw.textContent=s.warning;
  const rpn=document.getElementById('rpNormal');if(rpn)rpn.textContent=s.normal.toLocaleString();
  const rpm=document.getElementById('rpML');    if(rpm)rpm.textContent=s.ml_flags;
  // Update left sidebar stats
  const sbt=document.getElementById('sbTotal'); if(sbt)sbt.textContent=s.total.toLocaleString();
  const sba=document.getElementById('sbAlert'); if(sba)sba.textContent=s.alert;
  const sbw=document.getElementById('sbWarn');  if(sbw)sbw.textContent=s.warning;
  const sbn=document.getElementById('sbNormal');if(sbn)sbn.textContent=s.normal.toLocaleString();
  const sbm=document.getElementById('sbML');    if(sbm)sbm.textContent=s.ml_flags;
}

function renderModels(m){
  document.getElementById('ifChip').className='m-chip '+(m.if_enabled?'on':'off');
  document.getElementById('aeChip').className='m-chip '+(m.ae_enabled?'on':'off');
  document.getElementById('mInfo').textContent=
    (m.if_enabled&&m.ae_enabled)?'Dual-model ensemble — zero-day detection enabled':
    m.if_enabled?'Isolation Forest only':m.ae_enabled?'Autoencoder only':'No models active';
  // Update right panel
  const rpif=document.getElementById('rpIF');
  const rpae=document.getElementById('rpAE');
  if(rpif){rpif.textContent=m.if_enabled?'Active':'Offline';
            rpif.className='sys-val '+(m.if_enabled?'ok':'bad');}
  if(rpae){rpae.textContent=m.ae_enabled?'Active':'Offline';
            rpae.className='sys-val '+(m.ae_enabled?'ok':'bad');}
  // sidebar model dots
  const sdif=document.getElementById('sbIfDot');
  const sdae=document.getElementById('sbAeDot');
  if(sdif)sdif.className='sb-model-dot '+(m.if_enabled?'on':'off');
  if(sdae)sdae.className='sb-model-dot '+(m.ae_enabled?'on':'off');
}

function getTextColor(){
  return document.documentElement.getAttribute('data-theme')==='dark'?'#6a7490':'#7a85a0';
}
function getGridColor(){
  return document.documentElement.getAttribute('data-theme')==='dark'?
    'rgba(255,255,255,.03)':'rgba(0,0,0,.05)';
}

function renderTimeline(tl){
  lastTimeline=tl;
  const ctx=document.getElementById('timeline').getContext('2d');
  if(tChart)tChart.destroy();
  const tc=getTextColor(),gc=getGridColor();
  tChart=new Chart(ctx,{
    type:'line',
    data:{
      labels:tl?tl.labels:[],
      datasets:[
        {label:'Alerts',data:tl?tl.alerts:[],
         borderColor:'#e05252',backgroundColor:'rgba(224,82,82,.08)',
         fill:true,tension:.4,pointRadius:1,borderWidth:1.5},
        {label:'Warnings',data:tl?tl.warnings:[],
         borderColor:'#d4870a',backgroundColor:'rgba(212,135,10,.05)',
         fill:true,tension:.4,pointRadius:1,borderWidth:1.5}
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:tc,font:{size:11,family:'Plus Jakarta Sans'}}}},
      scales:{
        x:{ticks:{color:tc,font:{size:10,family:'JetBrains Mono'}},grid:{color:gc}},
        y:{ticks:{color:tc,font:{size:10},stepSize:1},grid:{color:gc},beginAtZero:true}
      }
    }
  });
}

function renderDonut(){
  const cats={};
  allAlerts.forEach(a=>{const c=a.category||'Unknown';cats[c]=(cats[c]||0)+1;});
  const labels=Object.keys(cats);
  const vals=Object.values(cats);
  const cols=labels.map(l=>CAT_COLOR[l]||'#6a7490');
  const ctx=document.getElementById('donut').getContext('2d');
  if(dChart)dChart.destroy();
  if(!labels.length)return;
  const tc=getTextColor();
  dChart=new Chart(ctx,{
    type:'doughnut',
    data:{labels,datasets:[{data:vals,backgroundColor:cols,borderWidth:0,hoverOffset:3}]},
    options:{
      responsive:true,maintainAspectRatio:false,cutout:'68%',
      plugins:{
        legend:{position:'right',labels:{color:tc,font:{size:11,family:'Plus Jakarta Sans'},boxWidth:10,padding:8}},
        tooltip:{callbacks:{label:c=>` ${c.label}: ${c.parsed}`}}
      }
    }
  });
}

function filtered(){
  const q=(document.getElementById('incQ')?.value||'').toLowerCase();
  return allAlerts.filter(a=>{
    const co=activeCat==='All'||a.category===activeCat;
    const so=activeSev==='All'||a.status===activeSev;
    const qo=!q||
      (a.details?.source_ip||'').includes(q)||
      (a.details?.destination_ip||'').includes(q)||
      String(a.details?.dest_port||'').includes(q)||
      (a.category||'').toLowerCase().includes(q)||
      (a.message||'').toLowerCase().includes(q);
    return co&&so&&qo;
  });
}

function renderFeed(list){
  const el=document.getElementById('feed');
  if(!list.length){el.innerHTML='<div class="empty">No detections match filters</div>';return;}
  el.innerHTML=[...list].reverse().slice(0,40).map(a=>{
    const cls=a.status==='ALERT'?'a':a.status==='WARNING'?'w':'';
    const icon=a.status==='ALERT'?'▲':'◆';
    const s=a.risk_score||0;
    const rc=riskColor(s);
    const idx=allAlerts.indexOf(a);
    return `<div class="af ${cls}" onclick="openModal(${idx})">
      <div style="font-size:11px;color:${a.status==='ALERT'?'var(--red)':'var(--amber)'};flex-shrink:0;margin-top:2px">${icon}</div>
      <div class="af-body">
        <div class="af-top">
          <span class="af-cat">${a.category||'Unknown'}</span>
          <span class="badge ${a.status}">${a.status}</span>
          ${a.ml_flagged?'<span class="ml-tag">ML</span>':''}
          ${(a.count||1)>1?`<span style="font-size:10px;color:var(--amber);font-family:'JetBrains Mono',monospace">×${a.count}</span>`:''}
        </div>
        <div class="af-ips">${a.details?.source_ip||'—'} → ${a.details?.destination_ip||'—'}:${a.details?.dest_port||'—'}</div>
        <div class="af-msg">${a.message||''}</div>
      </div>
      <div class="af-right">
        <div class="af-time">${(a.timestamp||'').substring(11,19)}</div>
        <div class="risk-bar"><div class="risk-fill" style="width:${s}%;background:${rc}"></div></div>
      </div>
    </div>`;
  }).join('');
}

function renderIPs(){
  const counts={};
  allAlerts.forEach(a=>{const ip=a.details?.source_ip;if(ip)counts[ip]=(counts[ip]||0)+1;});
  const sorted=Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,8);
  document.getElementById('ipCnt').textContent=sorted.length;
  const el=document.getElementById('ipList');
  if(!sorted.length){el.innerHTML='<div class="empty">No data yet</div>';return;}
  const mx=sorted[0][1];
  el.innerHTML=sorted.map(([ip,cnt])=>`
    <div class="ip-row">
      <span class="ip-addr">${ip}</span>
      <div class="ip-bar-bg"><div class="ip-bar-fill" style="width:${Math.round(cnt/mx*100)}%"></div></div>
      <span class="ip-cnt">${cnt}</span>
    </div>`).join('');
}

function setCat(cat){
  activeCat=cat;
  renderTable();
  
}

function setSev(btn){
  activeSev=btn.dataset.s;
  document.querySelectorAll('.rpsev').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  renderTable();
}

// Incidents table
function setSort(f){
  sortF=f;
  document.querySelectorAll('.sort-btn').forEach(b=>b.classList.remove('on'));
  const m={time:'sTime',risk:'sRisk',sev:'sSev'};
  if(m[f])document.getElementById(m[f]).classList.add('on');
  page=1;renderTable();
}

function renderTable(){
  const q=(document.getElementById('incQ')?.value||'').toLowerCase();
  let list=allAlerts.filter(a=>{
    const co=activeCat==='All'||a.category===activeCat;
    const so=activeSev==='All'||a.status===activeSev;
    const qo=!q||
      (a.details?.source_ip||'').includes(q)||
      (a.details?.destination_ip||'').includes(q)||
      (a.category||'').toLowerCase().includes(q)||
      (a.message||'').toLowerCase().includes(q);
    return co&&so&&qo;
  });
  list=[...list];
  if(sortF==='risk')list.sort((a,b)=>(b.risk_score||0)-(a.risk_score||0));
  else if(sortF==='sev'){const o={ALERT:0,WARNING:1,NORMAL:2};list.sort((a,b)=>(o[a.status]||2)-(o[b.status]||2));}
  else list.reverse();
  const total=list.length;
  const pages=Math.ceil(total/PG)||1;
  page=Math.min(page,pages);
  const rows=list.slice((page-1)*PG,(page-1)*PG+PG);
  const tbody=document.getElementById('tbody');
  tbody.innerHTML=!rows.length?
    `<tr><td colspan="9" style="text-align:center;padding:28px;color:var(--text3);font-family:'Roboto',sans-serif;">No incidents</td></tr>`:
    rows.map(a=>{
      const s=a.risk_score||0,rc=riskClass(s);
      const idx=allAlerts.indexOf(a);
      const ts=(a.timestamp||'').replace('T',' ').substring(0,19);
      return `<tr onclick="openModal(${idx})">
        <td class="td-mono" style="font-size:11px;color:var(--text3)">${ts}</td>
        <td><span class="badge ${a.status}">${a.status}</span></td>
        <td><span class="risk-pill ${rc}">${s}</span></td>
        <td style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:500">${a.category||'—'}</td>
        <td class="td-ip">${a.details?.source_ip||'—'}</td>
        <td class="td-ip">${a.details?.destination_ip||'—'}:${a.details?.dest_port||'?'}</td>
        <td class="td-mono">${a.details?.protocol||'—'}</td>
        <td>${a.ml_flagged?'<span class="ml-tag">ML</span>':'<span style="color:var(--text3)">—</span>'}</td>
        <td class="td-dim">${a.message||'—'}</td>
      </tr>`;
    }).join('');
  document.getElementById('pgInfo').textContent=`${total} incidents — page ${page} of ${pages}`;
  let pb=`<button class="pg-btn" onclick="changePg(${page-1})" ${page<=1?'disabled':''}>‹</button>`;
  for(let p2=1;p2<=pages;p2++){
    if(pages<=6||Math.abs(p2-page)<=2||p2===1||p2===pages)
      pb+=`<button class="pg-btn ${p2===page?'on':''}" onclick="changePg(${p2})">${p2}</button>`;
    else if(Math.abs(p2-page)===3)pb+=`<span style="padding:4px;color:var(--text3)">…</span>`;
  }
  pb+=`<button class="pg-btn" onclick="changePg(${page+1})" ${page>=pages?'disabled':''}>›</button>`;
  document.getElementById('pgBtns').innerHTML=pb;
}
function changePg(p){page=p;renderTable();}

function exportCSV(){
  const h=['Timestamp','Status','Risk','Category','Source IP','Dest IP','Port','Protocol','ML','Message'];
  const rows=allAlerts.map(a=>[
    a.timestamp||'',a.status||'',a.risk_score||0,a.category||'',
    a.details?.source_ip||'',a.details?.destination_ip||'',a.details?.dest_port||'',
    a.details?.protocol||'',a.ml_flagged?'Yes':'No',(a.message||'').replace(/,/g,';')
  ]);
  const a2=document.createElement('a');
  a2.href=URL.createObjectURL(new Blob([[h,...rows].map(r=>r.join(',')).join('\n')],{type:'text/csv'}));
  a2.download='ng_incidents.csv';a2.click();
}

// Analytics search
function renderAnalyticsSearch(){
  const q=(document.getElementById('analyticsQ')?.value||'').trim().toLowerCase();
  const resultsEl=document.getElementById('analyticsResults');
  if(!q){resultsEl.style.display='none';return;}
  const list=allAlerts.filter(a=>
    (a.details?.source_ip||'').includes(q)||
    (a.details?.destination_ip||'').includes(q)||
    (a.category||'').toLowerCase().includes(q)||
    (a.threat_type||'').toLowerCase().includes(q)||
    (a.message||'').toLowerCase().includes(q)
  );
  resultsEl.style.display='block';
  document.getElementById('analyticsTableBody').innerHTML=
    !list.length?`<tr><td colspan="6" style="text-align:center;padding:20px;color:var(--text3)">No results for "${q}"</td></tr>`:
    [...list].reverse().slice(0,50).map(a=>{
      const idx=allAlerts.indexOf(a);
      return `<tr onclick="openModal(${idx})" style="cursor:pointer">
        <td class="td-mono" style="font-size:11px;color:var(--text3)">${(a.timestamp||'').replace('T',' ').substring(0,19)}</td>
        <td><span class="badge ${a.status}">${a.status}</span></td>
        <td style="font-family:'Plus Jakarta Sans',sans-serif">${a.category||'—'}</td>
        <td class="td-ip">${a.details?.source_ip||'—'}</td>
        <td class="td-ip">${a.details?.destination_ip||'—'}:${a.details?.dest_port||'?'}</td>
        <td class="td-dim">${a.message||'—'}</td>
      </tr>`;
    }).join('');
}

// Analytics
function renderAnalytics(){
  const hours=new Array(24).fill(0);
  allAlerts.forEach(a=>{
    try{const h=parseInt((a.timestamp||'').substring(11,13));if(!isNaN(h))hours[h]++;}catch(e){}
  });
  const mx=Math.max(...hours,1);
  document.getElementById('heatmap').innerHTML=hours.map((v,i)=>{
    const alpha=v===0?0:0.1+(v/mx)*0.85;
    const col=v===0?'var(--border)':`rgba(224,82,82,${alpha.toFixed(2)})`;
    return `<div class="hm" style="background:${col}" data-t="${String(i).padStart(2,'0')}:00 — ${v} alerts"></div>`;
  }).join('');
  document.getElementById('hmLabels').innerHTML=hours.map((_,i)=>
    `<div class="hm-lbl">${i%4===0?String(i).padStart(2,'0'):''}</div>`
  ).join('');
  const total=allAlerts.length;
  const alertC=allAlerts.filter(a=>a.status==='ALERT').length;
  const mlC=allAlerts.filter(a=>a.ml_flagged).length;
  const dualC=allAlerts.filter(a=>a.if_anomaly&&a.ae_anomaly).length;
  const cats={},ips={};
  allAlerts.forEach(a=>{
    const c=a.category||'?';cats[c]=(cats[c]||0)+1;
    const ip=a.details?.source_ip;if(ip)ips[ip]=(ips[ip]||0)+1;
  });
  const topC=Object.entries(cats).sort((a,b)=>b[1]-a[1])[0];
  const topI=Object.entries(ips).sort((a,b)=>b[1]-a[1])[0];
  const ar=total?(alertC/total*100).toFixed(1):0;
  const mr=total?(mlC/total*100).toFixed(1):0;
  document.getElementById('mTotal').textContent=total.toLocaleString();
  const arEl=document.getElementById('mAR');
  arEl.textContent=ar+'%';arEl.className='m-val '+(parseFloat(ar)>10?'bad':parseFloat(ar)>5?'w':'ok');
  document.getElementById('mMR').textContent=mr+'%';
  document.getElementById('mDual').textContent=dualC;
  document.getElementById('mTopC').textContent=topC?topC[0]:'—';
  document.getElementById('mTopI').textContent=topI?topI[0]:'—';
  const labs=Object.keys(cats);
  const vals=Object.values(cats);
  const cols=labs.map(l=>CAT_COLOR[l]||'#6a7490');
  const ctx=document.getElementById('barChart').getContext('2d');
  if(bChart)bChart.destroy();
  const tc=getTextColor(),gc=getGridColor();
  bChart=new Chart(ctx,{
    type:'bar',
    data:{labels:labs,datasets:[{data:vals,backgroundColor:cols,borderWidth:0,borderRadius:4}]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:tc,font:{size:9,family:'JetBrains Mono'},maxRotation:30},grid:{display:false}},
        y:{ticks:{color:tc,font:{size:10},stepSize:1},grid:{color:gc},beginAtZero:true}
      }
    }
  });
}

// Modal
function openModal(idx){
  const a=allAlerts[idx];if(!a)return;
  const d=a.details||{};
  const s=a.risk_score||0,sc=riskColor(s);
  const stc=a.status==='ALERT'?'red':a.status==='WARNING'?'amber':'green';
  document.getElementById('mTitle').textContent=a.category||'Incident';
  document.getElementById('mSub').textContent=
    (a.timestamp||'').replace('T',' ').substring(0,19)+' UTC  ·  '+
    (d.source_ip||'—')+' → '+(d.destination_ip||'—');
  document.getElementById('mBody').innerHTML=`
    <div class="m-sec">
      <div class="sec-title">Summary</div>
      <div class="gauge" style="margin-bottom:14px;">
        <span style="font-size:11px;font-family:'Roboto',sans-serif;color:var(--text3);min-width:70px;">Risk Score</span>
        <div class="gauge-track"><div class="gauge-fill" style="width:${s}%;background:${sc}"></div></div>
        <span class="gauge-lbl" style="color:${sc}">${s}</span>
      </div>
      <div class="attr-grid">
        <div class="attr"><div class="attr-k">Severity</div><div class="attr-v ${stc}">${a.status||'—'}</div></div>
        <div class="attr"><div class="attr-k">Category</div><div class="attr-v">${a.category||'—'}</div></div>
        <div class="attr"><div class="attr-k">Threat Type</div><div class="attr-v">${(a.threat_type||'—').replace(/_/g,' ')}</div></div>
        <div class="attr"><div class="attr-k">Timestamp</div><div class="attr-v">${(a.timestamp||'—').replace('T',' ').substring(0,19)}</div></div>
        <div class="attr" style="grid-column:span 2">
          <div class="attr-k">Description</div>
          <div class="attr-v" style="font-size:12px;color:var(--text2);line-height:1.6">${a.message||'—'}</div>
        </div>
      </div>
    </div>
    <div class="m-sec">
      <div class="sec-title">Network Attributes</div>
      <div class="attr-grid">
        <div class="attr"><div class="attr-k">Source IP</div><div class="attr-v ip">${d.source_ip||'—'}</div></div>
        <div class="attr"><div class="attr-k">Destination IP</div><div class="attr-v ip">${d.destination_ip||'—'}</div></div>
        <div class="attr"><div class="attr-k">Src Port</div><div class="attr-v">${d.source_port??'—'}</div></div>
        <div class="attr"><div class="attr-k">Dst Port</div><div class="attr-v">${d.dest_port??'—'}</div></div>
        <div class="attr"><div class="attr-k">Protocol</div><div class="attr-v">${d.protocol||'—'}</div></div>
        <div class="attr"><div class="attr-k">Bytes</div><div class="attr-v">${d.bytes!=null?d.bytes.toLocaleString()+' B':'—'}</div></div>
        <div class="attr"><div class="attr-k">Packets</div><div class="attr-v">${d.packets??'—'}</div></div>
        <div class="attr"><div class="attr-k">Duration</div><div class="attr-v">${d.duration!=null?d.duration+'s':'—'}</div></div>
        <div class="attr"><div class="attr-k">Action</div><div class="attr-v ${stc}">${a.action||'—'}</div></div>
      </div>
    </div>
    <div class="m-sec">
      <div class="sec-title">ML Analysis</div>
      <div class="attr-grid">
        <div class="attr"><div class="attr-k">Anomaly Score</div>
  	  <div class="attr-v ${a.if_anomaly?'red':'green'}">${a.if_anomaly?'Anomalous':'Within normal range'}</div></div>
	<div class="attr"><div class="attr-k">Behaviour Analysis</div>
  	  <div class="attr-v ${a.ae_anomaly?'red':'green'}">${a.ae_anomaly?'Deviation detected':'Normal pattern'}</div></div>
        <div class="attr"><div class="attr-k">Confidence</div>
          <div class="attr-v">${a.confidence!=null?(a.confidence*100).toFixed(1)+'%':'—'}</div></div>
        <div class="attr"><div class="attr-k">ML Overall</div>
          <div class="attr-v ${a.ml_flagged?'red':'green'}">${a.ml_flagged?'Flagged anomalous':'Within normal range'}</div></div>
      </div>
    </div>`;
  document.getElementById('ov').classList.add('open');
}
function closeOv(e){if(e.target===document.getElementById('ov'))closeOvDirect();}
function closeOvDirect(){document.getElementById('ov').classList.remove('open');}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeOvDirect();});

// Initial right panel




load();
setInterval(load,3000);
</script>
</body>
</html>
"""

@app.route("/health")
def health():
    # Liveness probe for the NeuraShield supervisor (no data, so no key needed).
    return jsonify({"status": "ok", "layer": "network_guardian",
                    "monitor_running": bool(monitor_proc and monitor_proc.poll() is None)})

@app.route("/")
def dashboard():
    # INTEGRATION SECURITY FIX: the page used to be served to any local caller
    # with the real X-API-Key embedded in it. It now needs the key itself and no
    # longer embeds it - the dashboard is reached through the NeuraShield
    # console, whose proxy adds X-API-Key to every request server-side.
    _deny = _require_api_key()
    if _deny:
        # Opened directly in a browser (no key): go through the console instead.
        if "text/html" in (request.headers.get("Accept") or ""):
            from flask import redirect
            base = os.environ.get("NEURASHIELD_CONSOLE_URL", "http://127.0.0.1:9000").rstrip("/")
            return redirect(base + "/embed/network", code=303)
        return _deny
    html = DASHBOARD_HTML.replace("__NEURASHIELD_API_KEY__", "")
    return render_template_string(html)

@app.route("/api/data")
def api_data():
    _deny = _require_api_key()
    if _deny:
        return _deny
    all_alerts = read_json_lines(ALERT_FILE)
    flows      = read_json_lines(FLOW_LOG)
    for a in all_alerts:
        a["category"]   = categorize_threat(
            a.get("threat_type",""), a.get("message",""), a.get("status",""))
        a["risk_score"] = risk_score(a)
    hour_alerts = filter_last_hour(all_alerts)
    hour_alerts = deduplicate_alerts(hour_alerts)
    total    = len(flows)
    alert_c  = sum(1 for f in flows if f.get("Status")=="ALERT")
    warn_c   = sum(1 for f in flows if f.get("Status")=="WARNING")
    normal_c = sum(1 for f in flows if f.get("Status")=="NORMAL")
    ml_c     = sum(1 for f in flows if f.get("ML Flagged"))
    now = datetime.now(timezone.utc)
    ba,bw = defaultdict(int),defaultdict(int)
    for a in all_alerts:
        try:
            ts=a.get("timestamp","").replace("Z","+00:00")
            if "+" not in ts and ts.count("-")<3: ts+="+00:00"
            t=datetime.fromisoformat(ts)
            if t.tzinfo is None: t=t.replace(tzinfo=timezone.utc)
            diff=int((now-t).total_seconds()/60)
            if 0<=diff<60:
                b=59-diff
                if a.get("status")=="ALERT": ba[b]+=1
                else: bw[b]+=1
        except: pass
    tl_labels   = [f"-{59-i}m" if (59-i)%10==0 else "" for i in range(60)]
    tl_alerts   = [ba.get(i,0) for i in range(60)]
    tl_warnings = [bw.get(i,0) for i in range(60)]
    recent    = flows[-20:] if flows else []
    if_active = any(f.get("IF Score") is not None for f in recent)
    ae_active = any(f.get("AE Error") is not None for f in recent)
    return jsonify({
        "stats":    {"total":total,"alert":alert_c,"warning":warn_c,"normal":normal_c,"ml_flags":ml_c},
        "models":   {"if_enabled":if_active,"ae_enabled":ae_active},
        "alerts":   hour_alerts,
        "timeline": {"labels":tl_labels,"alerts":tl_alerts,"warnings":tl_warnings},
    })

@app.route("/api/alerts")
def api_alerts():
    _deny = _require_api_key()
    if _deny:
        return _deny
    items = read_json_lines(ALERT_FILE)
    for a in items:
        a["category"]   = categorize_threat(a.get("threat_type",""),a.get("message",""),a.get("status",""))
        a["risk_score"] = risk_score(a)
    return jsonify(items)



@app.route("/api/flows")
def api_flows():
    _deny = _require_api_key()
    if _deny:
        return _deny
    return jsonify(read_json_lines(FLOW_LOG))

import atexit
atexit.register(lambda: stop_monitor())

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        # Packaged build: run the capture process in this (second) instance.
        import runpy
        _ms = _monitor_script()
        os.chdir(os.path.dirname(_ms))
        runpy.run_path(_ms, run_name="__main__")
        sys.exit(0)
    print("="*50)
    print("  Network Guardian — NeuraShield")
    print("="*50)
    print("  Starting monitor-2.py automatically...")
    start_monitor()
    print("  Dashboard : http://127.0.0.1:5000")
    print("  Stop      : Ctrl+C")
    print("="*50)
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        stop_monitor()
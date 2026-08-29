# app.py
# NeuraShield — Network Guardian Dashboard
# Run with: source venv/bin/activate && python3 app.py
 
from flask import Flask, jsonify, render_template_string
import json, os
from datetime import datetime, timezone, timedelta
 
app = Flask(__name__)
ALERT_FILE = "alerts.json"
FLOW_LOG   = "flow_log.json"
 
# ── Attack category mapping ────────────────────────────────────
def categorize_threat(threat_type, message, status):
    t = (threat_type + " " + message).lower()
    if any(x in t for x in ["dos", "flood", "denial", "icmp flood", "dns flood", "syn"]):
        return "Denial of Service"
    if any(x in t for x in ["port scan", "probe", "sweep", "scan"]):
        return "Network Scan"
    if any(x in t for x in ["exfil", "large outbound", "data transfer", "dns tunnel"]):
        return "Data Exfiltration"
    if any(x in t for x in ["c2", "command", "control", "metasploit", "irc", "tor",
                              "suspicious port", "4444", "6666", "6667", "31337"]):
        return "Command & Control"
    if any(x in t for x in ["brute", "auth", "login", "credential", "rdp", "ssh"]):
        return "Brute Force"
    if any(x in t for x in ["database", "db access", "sql", "mongo", "redis"]):
        return "Unauthorized Access"
    if any(x in t for x in ["inbound", "remote admin", "external"]):
        return "Suspicious Inbound"
    if any(x in t for x in ["ml anomaly", "dual model", "autoencoder", "isolation"]):
        return "ML Anomaly"
    if status == "ALERT":
        return "Unknown Threat"
    return "Suspicious Activity"
 
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
            ts_str = item.get(time_key, "")
            if ts_str:
                ts_str = ts_str.replace("Z", "+00:00")
                if "+" not in ts_str and ts_str.count("-") < 3:
                    ts_str += "+00:00"
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    out.append(item)
        except:
            out.append(item)  # keep if parse fails
    return out
 
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeuraShield</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:          #07090f;
  --surface:     #0d1117;
  --surface2:    #111820;
  --surface3:    #151e2c;
  --border:      #1a2332;
  --border2:     #243040;
  --text:        #d0dcea;
  --text-dim:    #4d6480;
  --text-soft:   #8aa0b8;
  --blue:        #4f8ef7;
  --blue-bg:     #0c1f3a;
  --red:         #f06060;
  --red-bg:      #2a0f0f;
  --amber:       #f0a840;
  --amber-bg:    #2a1c08;
  --green:       #3dd68c;
  --green-bg:    #082a18;
  --purple:      #9d7fea;
  --purple-bg:   #1a1030;
  --sidebar-w:   240px;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:'Sora',sans-serif;font-size:14px;display:flex;flex-direction:column;min-height:100vh;}
 
/* ── NAV ── */
nav{height:60px;display:flex;align-items:center;justify-content:space-between;
    padding:0 24px;background:var(--surface);border-bottom:1px solid var(--border);
    position:sticky;top:0;z-index:200;flex-shrink:0;}
.brand-name{font-size:20px;font-weight:700;color:#fff;letter-spacing:-0.3px;line-height:1.1;}
.brand-sub{font-size:10px;color:var(--text-dim);font-family:'JetBrains Mono',monospace;
           letter-spacing:1.5px;text-transform:uppercase;margin-top:2px;}
.live-pill{display:flex;align-items:center;gap:8px;padding:6px 14px;border-radius:99px;
           background:var(--green-bg);border:1px solid rgba(61,214,140,.2);
           font-size:11px;font-weight:600;color:var(--green);
           font-family:'JetBrains Mono',monospace;}
.live-dot{width:7px;height:7px;background:var(--green);border-radius:50%;
          animation:blink 1.6s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.2;}}
 
/* ── SHELL ── */
.shell{display:flex;flex:1;overflow:hidden;}
 
/* ── SIDEBAR ── */
aside{width:var(--sidebar-w);background:var(--surface);border-right:1px solid var(--border);
      display:flex;flex-direction:column;flex-shrink:0;overflow-y:auto;}
.aside-section{padding:18px 14px 8px;font-family:'JetBrains Mono',monospace;
               font-size:10px;letter-spacing:1.8px;text-transform:uppercase;
               color:var(--text-dim);}
.aside-section:first-of-type{padding-top:20px;}
 
/* search box */
.search-wrap{padding:0 12px 12px;}
.search-box{width:100%;background:var(--bg);border:1px solid var(--border2);
            border-radius:8px;padding:8px 12px;color:var(--text);
            font-family:'JetBrains Mono',monospace;font-size:12px;outline:none;
            transition:border-color .15s;}
.search-box::placeholder{color:var(--text-dim);}
.search-box:focus{border-color:var(--blue);}
 
/* filter pills */
.filter-list{padding:0 10px 16px;}
.filter-btn{width:100%;text-align:left;background:none;border:none;
            padding:8px 10px;border-radius:7px;cursor:pointer;
            display:flex;align-items:center;justify-content:space-between;
            color:var(--text-soft);font-size:12px;font-family:'Sora',sans-serif;
            transition:background .12s,color .12s;margin-bottom:2px;}
.filter-btn:hover{background:var(--surface2);color:var(--text);}
.filter-btn.active{background:var(--blue-bg);color:var(--blue);}
.filter-btn .f-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-right:8px;}
.filter-btn .f-count{font-family:'JetBrains Mono',monospace;font-size:10px;
                     color:var(--text-dim);background:var(--surface2);
                     padding:1px 7px;border-radius:8px;}
.filter-btn.active .f-count{background:var(--blue-bg);color:var(--blue);}
 
/* severity filters */
.sev-list{padding:0 10px 20px;}
.sev-btn{width:100%;text-align:left;background:none;border:none;
         padding:7px 10px;border-radius:7px;cursor:pointer;
         display:flex;align-items:center;gap:8px;
         font-size:12px;font-family:'Sora',sans-serif;
         color:var(--text-soft);transition:background .12s;margin-bottom:2px;}
.sev-btn:hover{background:var(--surface2);}
.sev-btn.active{background:var(--surface2);}
 
/* ── MAIN ── */
.main{flex:1;overflow-y:auto;padding:26px 28px 48px;}
.main::-webkit-scrollbar{width:3px;}
.main::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
 
.section-label{font-size:10px;font-family:'JetBrains Mono',monospace;
               letter-spacing:2px;text-transform:uppercase;color:var(--text-dim);
               margin-bottom:11px;}
 
/* ── STAT ROW ── */
.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;margin-bottom:28px;}
.stat-card{background:var(--surface);border:1px solid var(--border);
           border-radius:11px;padding:18px 18px 16px;position:relative;overflow:hidden;}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:11px 11px 0 0;}
.stat-card.blue::before  {background:var(--blue);}
.stat-card.red::before   {background:var(--red);}
.stat-card.amber::before {background:var(--amber);}
.stat-card.green::before {background:var(--green);}
.stat-num{font-family:'JetBrains Mono',monospace;font-size:34px;font-weight:500;line-height:1;margin-bottom:7px;}
.stat-card.blue  .stat-num{color:var(--blue);}
.stat-card.red   .stat-num{color:var(--red);}
.stat-card.amber .stat-num{color:var(--amber);}
.stat-card.green .stat-num{color:var(--green);}
.stat-name{font-size:12px;color:var(--text-soft);}
.stat-hint{font-size:10px;color:var(--text-dim);font-family:'JetBrains Mono',monospace;margin-top:3px;}
 
/* ── MODEL BAR ── */
.model-bar{background:var(--surface);border:1px solid var(--border);border-radius:10px;
           padding:12px 18px;margin-bottom:28px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.model-bar-label{font-size:10px;color:var(--text-dim);font-family:'JetBrains Mono',monospace;
                 letter-spacing:1px;text-transform:uppercase;margin-right:6px;flex-shrink:0;}
.model-chip{display:flex;align-items:center;gap:7px;padding:5px 13px;border-radius:6px;
            border:1px solid;font-size:12px;font-family:'JetBrains Mono',monospace;}
.model-chip.on{background:var(--green-bg);border-color:rgba(61,214,140,.2);color:var(--green);}
.model-chip.off{background:var(--red-bg);border-color:rgba(240,96,96,.2);color:var(--red);}
.chip-dot{width:6px;height:6px;background:currentColor;border-radius:50%;}
.ml-info{margin-left:auto;font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--text-dim);}
 
/* ── ALERT CARDS ── */
.alerts-panel{background:var(--surface);border:1px solid var(--border);
              border-radius:11px;overflow:hidden;margin-bottom:28px;}
.panel-header{padding:13px 18px;border-bottom:1px solid var(--border);
              display:flex;align-items:center;justify-content:space-between;
              background:var(--surface2);}
.panel-header-title{font-size:13px;font-weight:600;color:var(--text);}
.count-chip{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text-dim);
            background:var(--bg);border:1px solid var(--border);
            border-radius:8px;padding:3px 10px;}
 
.alert-list{padding:10px;max-height:400px;overflow-y:auto;}
.alert-list::-webkit-scrollbar{width:3px;}
.alert-list::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
 
.alert-item{display:flex;align-items:flex-start;gap:13px;padding:13px 14px;
            border-radius:9px;margin-bottom:6px;background:var(--bg);
            border:1px solid var(--border);cursor:pointer;
            transition:border-color .15s,background .15s;}
.alert-item:last-child{margin-bottom:0;}
.alert-item:hover{border-color:var(--border2);background:var(--surface3);}
.alert-item.is-alert {border-left:3px solid var(--red);}
.alert-item.is-warn  {border-left:3px solid var(--amber);}
.alert-item.is-normal{border-left:3px solid var(--green);}
 
.alert-icon{font-size:17px;flex-shrink:0;margin-top:1px;}
.alert-body{flex:1;min-width:0;}
.alert-top{display:flex;align-items:center;gap:7px;margin-bottom:4px;flex-wrap:wrap;}
.alert-type{font-size:13px;font-weight:600;color:var(--text);}
.alert-category{font-size:10px;color:var(--text-dim);font-family:'JetBrains Mono',monospace;
                background:var(--surface2);border:1px solid var(--border2);
                padding:2px 8px;border-radius:4px;}
.alert-ips{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--blue);
           margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.alert-msg{font-size:11px;color:var(--text-dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.alert-meta{text-align:right;flex-shrink:0;}
.alert-time{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text-dim);display:block;margin-bottom:5px;}
.score-block{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-dim);line-height:1.7;text-align:right;}
.s-good{color:var(--green);}
.s-bad {color:var(--red);}
.view-btn{font-size:10px;color:var(--blue);font-family:'JetBrains Mono',monospace;
          background:var(--blue-bg);border:1px solid rgba(79,142,247,.2);
          padding:3px 9px;border-radius:5px;margin-top:5px;display:inline-block;
          text-align:center;}
 
/* badges */
.badge{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:500;
       padding:2px 8px;border-radius:4px;border:1px solid;letter-spacing:.3px;}
.badge.ALERT  {color:var(--red);  background:var(--red-bg);  border-color:rgba(240,96,96,.3);}
.badge.WARNING{color:var(--amber);background:var(--amber-bg);border-color:rgba(240,168,64,.3);}
.badge.NORMAL {color:var(--green);background:var(--green-bg);border-color:rgba(61,214,140,.3);}
.ml-badge{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--purple);
          background:var(--purple-bg);border:1px solid rgba(157,127,234,.25);
          padding:2px 8px;border-radius:4px;}
 
/* empty */
.empty{padding:48px 20px;text-align:center;color:var(--text-dim);
       font-family:'JetBrains Mono',monospace;font-size:12px;}
.empty .e-icon{font-size:28px;margin-bottom:11px;opacity:.4;}
 
/* footer */
.footer{margin-top:32px;text-align:center;font-family:'JetBrains Mono',monospace;
        font-size:11px;color:var(--text-dim);}
 
/* ── MODAL OVERLAY ── */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);
               backdrop-filter:blur(6px);z-index:500;
               display:flex;align-items:center;justify-content:center;
               padding:20px;opacity:0;pointer-events:none;transition:opacity .2s;}
.modal-overlay.open{opacity:1;pointer-events:all;}
 
.modal{background:var(--surface);border:1px solid var(--border2);border-radius:14px;
       width:100%;max-width:720px;max-height:88vh;overflow:hidden;
       display:flex;flex-direction:column;
       transform:translateY(16px);transition:transform .2s;}
.modal-overlay.open .modal{transform:translateY(0);}
 
.modal-head{padding:18px 22px;border-bottom:1px solid var(--border);
            display:flex;align-items:flex-start;justify-content:space-between;
            background:var(--surface2);}
.modal-title{font-size:15px;font-weight:700;color:var(--text);line-height:1.3;}
.modal-sub{font-size:11px;color:var(--text-dim);font-family:'JetBrains Mono',monospace;margin-top:3px;}
.modal-close{background:none;border:none;color:var(--text-dim);font-size:20px;
             cursor:pointer;padding:2px 6px;border-radius:5px;
             transition:color .15s,background .15s;flex-shrink:0;margin-left:12px;}
.modal-close:hover{color:var(--text);background:var(--surface3);}
 
.modal-body{overflow-y:auto;padding:22px;}
.modal-body::-webkit-scrollbar{width:3px;}
.modal-body::-webkit-scrollbar-thumb{background:var(--border2);}
 
/* incident sections */
.inc-section{margin-bottom:24px;}
.inc-section-title{font-size:10px;font-family:'JetBrains Mono',monospace;
                   letter-spacing:2px;text-transform:uppercase;color:var(--text-dim);
                   margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border);}
.attr-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.attr-item{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:11px 14px;}
.attr-key{font-size:10px;font-family:'JetBrains Mono',monospace;letter-spacing:.8px;
          text-transform:uppercase;color:var(--text-dim);margin-bottom:5px;}
.attr-val{font-size:13px;font-family:'JetBrains Mono',monospace;color:var(--text);
          word-break:break-all;}
.attr-val.blue  {color:var(--blue);}
.attr-val.green {color:var(--green);}
.attr-val.red   {color:var(--red);}
.attr-val.amber {color:var(--amber);}
.attr-val.purple{color:var(--purple);}
 
/* packet box */
.packet-box{background:var(--bg);border:1px solid var(--border);border-radius:8px;
            padding:14px 16px;font-family:'JetBrains Mono',monospace;font-size:12px;
            line-height:1.9;color:var(--text-soft);}
.packet-box .pk-row{display:flex;gap:12px;border-bottom:1px solid var(--border);
                    padding-bottom:7px;margin-bottom:7px;}
.packet-box .pk-row:last-child{border-bottom:none;padding-bottom:0;margin-bottom:0;}
.pk-field{color:var(--text-dim);flex-shrink:0;min-width:120px;}
.pk-value{color:var(--text);}
.pk-value.hi{color:var(--blue);}
.pk-value.red{color:var(--red);}
.pk-value.green{color:var(--green);}
.pk-value.amber{color:var(--amber);}
 
/* ── CATEGORY COLORS ── */
.cat-dos  {color:#f06060;}
.cat-scan {color:#f0a840;}
.cat-exfil{color:#9d7fea;}
.cat-c2   {color:#f06060;}
.cat-brute{color:#f0a840;}
.cat-unauth{color:#9d7fea;}
.cat-inbound{color:#8aa0b8;}
.cat-ml   {color:#3dd68c;}
.cat-other{color:#4d6480;}
</style>
</head>
<body>
 
<!-- NAV -->
<nav>
  <div>
    <div class="brand-name">NeuraShield</div>
    <div class="brand-sub">Network Protection Layer</div>
  </div>
  <div class="live-pill"><span class="live-dot"></span>Live</div>
</nav>
 
<!-- SHELL: sidebar + main -->
<div class="shell">
 
  <!-- SIDEBAR -->
  <aside>
    <div class="aside-section">Search</div>
    <div class="search-wrap">
      <input class="search-box" id="searchBox" placeholder="IP, port, threat..." oninput="applyFilters()">
    </div>
 
    <div class="aside-section">Category</div>
    <div class="filter-list" id="catList"></div>
 
    <div class="aside-section">Severity</div>
    <div class="sev-list">
      <button class="sev-btn active" data-sev="All" onclick="setSev(this)">
        <span>🔘</span> All Levels
      </button>
      <button class="sev-btn" data-sev="ALERT" onclick="setSev(this)">
        <span>🚨</span> Alert
      </button>
      <button class="sev-btn" data-sev="WARNING" onclick="setSev(this)">
        <span>⚠️</span> Warning
      </button>
      <button class="sev-btn" data-sev="NORMAL" onclick="setSev(this)">
        <span>✅</span> Normal
      </button>
    </div>
  </aside>
 
  <!-- MAIN -->
  <div class="main">
 
    <!-- STAT CARDS -->
    <div class="section-label">Overview</div>
    <div class="stat-row">
      <div class="stat-card blue">
        <div class="stat-num" id="sTotal">0</div>
        <div class="stat-name">Total Flows</div>
        <div class="stat-hint">All traffic captured</div>
      </div>
      <div class="stat-card red">
        <div class="stat-num" id="sAlert">0</div>
        <div class="stat-name">Alerts</div>
        <div class="stat-hint">High risk detections</div>
      </div>
      <div class="stat-card amber">
        <div class="stat-num" id="sWarn">0</div>
        <div class="stat-name">Warnings</div>
        <div class="stat-hint">Needs review</div>
      </div>
      <div class="stat-card green">
        <div class="stat-num" id="sNormal">0</div>
        <div class="stat-name">Normal</div>
        <div class="stat-hint">Safe traffic</div>
      </div>
    </div>
 
    <!-- MODEL STATUS -->
    <div class="section-label">Detection Models</div>
    <div class="model-bar">
      <span class="model-bar-label">Status</span>
      <div class="model-chip" id="ifChip">
        <span class="chip-dot"></span><span>Isolation Forest</span>
      </div>
      <div class="model-chip" id="aeChip">
        <span class="chip-dot"></span><span>Autoencoder</span>
      </div>
      <span class="ml-info" id="mlInfo">Checking...</span>
    </div>
 
    <!-- ALERTS FEED -->
    <div class="section-label">
      Alerts &amp; Warnings
      <span id="hourLabel" style="margin-left:8px;color:var(--text-dim)">(last hour)</span>
    </div>
    <div class="alerts-panel">
      <div class="panel-header">
        <span class="panel-header-title">Recent Detections</span>
        <span class="count-chip" id="alertCount">0</span>
      </div>
      <div class="alert-list" id="alertList">
        <div class="empty"><div class="e-icon">🔍</div><div>No alerts yet</div></div>
      </div>
    </div>
 
    <div class="footer">NeuraShield · Network Guardian · Refreshes every 3s · UTC</div>
  </div>
</div>
 
<!-- INCIDENT MODAL -->
<div class="modal-overlay" id="modalOverlay" onclick="closeModal(event)">
  <div class="modal" id="modalBox">
    <div class="modal-head">
      <div>
        <div class="modal-title" id="modalTitle">Incident Detail</div>
        <div class="modal-sub"  id="modalSub">—</div>
      </div>
      <button class="modal-close" onclick="closeModalDirect()">✕</button>
    </div>
    <div class="modal-body" id="modalBody"></div>
  </div>
</div>
 
<script>
// ── State ──────────────────────────────────────────────────────
let allAlerts  = [];
let activeCat  = 'All';
let activeSev  = 'All';
let searchTerm = '';
 
const CAT_COLORS = {
  'All':                 '#4f8ef7',
  'Denial of Service':   '#f06060',
  'Network Scan':        '#f0a840',
  'Data Exfiltration':   '#9d7fea',
  'Command & Control':   '#f06060',
  'Brute Force':         '#f0a840',
  'Unauthorized Access': '#9d7fea',
  'Suspicious Inbound':  '#8aa0b8',
  'ML Anomaly':          '#3dd68c',
  'Suspicious Activity': '#4d6480',
  'Unknown Threat':      '#4d6480',
};
 
// ── Fetch ──────────────────────────────────────────────────────
async function load() {
  try {
    const r    = await fetch('/api/data');
    const data = await r.json();
    allAlerts  = data.alerts || [];
    renderStats(data.stats);
    renderModels(data.models);
    buildCategoryList();
    applyFilters();
  } catch(e) {}
}
 
// ── Stats ──────────────────────────────────────────────────────
function renderStats(s) {
  document.getElementById('sTotal').textContent  = s.total;
  document.getElementById('sAlert').textContent  = s.alert;
  document.getElementById('sWarn').textContent   = s.warning;
  document.getElementById('sNormal').textContent = s.normal;
}
 
// ── Models ─────────────────────────────────────────────────────
function renderModels(m) {
  const ifChip = document.getElementById('ifChip');
  ifChip.className = 'model-chip ' + (m.if_enabled ? 'on' : 'off');
  const aeChip = document.getElementById('aeChip');
  aeChip.className = 'model-chip ' + (m.ae_enabled ? 'on' : 'off');
  document.getElementById('mlInfo').textContent =
    (m.if_enabled && m.ae_enabled) ? 'Dual-model detection active' :
    m.if_enabled ? 'Isolation Forest only' :
    m.ae_enabled ? 'Autoencoder only' : 'No ML models loaded';
}
 
// ── Category sidebar ───────────────────────────────────────────
function buildCategoryList() {
  const counts = { 'All': allAlerts.length };
  allAlerts.forEach(a => {
    const c = a.category || 'Unknown Threat';
    counts[c] = (counts[c] || 0) + 1;
  });
 
  const order = ['All','Denial of Service','Network Scan','Data Exfiltration',
                 'Command & Control','Brute Force','Unauthorized Access',
                 'Suspicious Inbound','ML Anomaly','Suspicious Activity','Unknown Threat'];
 
  const catList = document.getElementById('catList');
  catList.innerHTML = order
    .filter(c => c === 'All' || counts[c])
    .map(c => {
      const col   = CAT_COLORS[c] || '#4d6480';
      const count = counts[c] || 0;
      return `
      <button class="filter-btn ${activeCat===c?'active':''}"
              onclick="setCat(this,'${c}')">
        <span style="display:flex;align-items:center;gap:7px;min-width:0">
          <span class="f-dot" style="background:${col};flex-shrink:0"></span>
          <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
            ${c}
          </span>
        </span>
        <span class="f-count">${count}</span>
      </button>`;
    }).join('');
}
 
function setCat(btn, cat) {
  activeCat = cat;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}
 
function setSev(btn) {
  activeSev = btn.dataset.sev;
  document.querySelectorAll('.sev-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}
 
// ── Filter + render alert list ─────────────────────────────────
function applyFilters() {
  searchTerm = document.getElementById('searchBox').value.toLowerCase();
 
  let filtered = allAlerts.filter(a => {
    const catOk  = activeCat === 'All' || a.category === activeCat;
    const sevOk  = activeSev === 'All' || a.status   === activeSev;
    const q      = searchTerm;
    const searchOk = !q ||
      (a.details?.source_ip      || '').includes(q) ||
      (a.details?.destination_ip || '').includes(q) ||
      String(a.details?.dest_port|| '').includes(q) ||
      (a.category                || '').toLowerCase().includes(q) ||
      (a.threat_type             || '').toLowerCase().includes(q) ||
      (a.message                 || '').toLowerCase().includes(q);
    return catOk && sevOk && searchOk;
  });
 
  renderAlertList(filtered);
  document.getElementById('alertCount').textContent = filtered.length;
}
 
function renderAlertList(list) {
  const el = document.getElementById('alertList');
  if (!list.length) {
    el.innerHTML = `<div class="empty"><div class="e-icon">🔍</div>
                    <div>No alerts match your filters</div></div>`;
    return;
  }
 
  el.innerHTML = list.slice().reverse().slice(0, 60).map((a, idx) => {
    const cls  = a.status === 'ALERT'   ? 'is-alert'
               : a.status === 'WARNING' ? 'is-warn' : 'is-normal';
    const icon = a.status === 'ALERT'   ? '🚨'
               : a.status === 'WARNING' ? '⚠️' : '✅';
    const cat  = a.category || 'Unknown';
    const col  = CAT_COLORS[cat] || '#4d6480';
 
    const ifScore = (a.if_score != null) ? a.if_score.toFixed(3) : null;
    const aeErr   = (a.ae_error != null) ? a.ae_error.toFixed(4) : null;
    const ifCls   = ifScore != null ? (parseFloat(ifScore) < 0 ? 's-bad' : 's-good') : '';
    const aeCls   = a.ae_anomaly ? 's-bad' : 's-good';
 
    // real index in allAlerts for modal
    const realIdx = allAlerts.length - 1 - allAlerts.slice().reverse().indexOf(a);
 
    return `
    <div class="alert-item ${cls}" onclick="openModal(${realIdx})">
      <div class="alert-icon">${icon}</div>
      <div class="alert-body">
        <div class="alert-top">
          <span class="alert-type">${cat}</span>
          <span class="badge ${a.status}">${a.status}</span>
          ${a.ml_flagged ? '<span class="ml-badge">🧠 ML</span>' : ''}
        </div>
        <div class="alert-ips">
          ${a.details?.source_ip || '—'} → ${a.details?.destination_ip || '—'}:${a.details?.dest_port || '—'}
          &nbsp;·&nbsp; ${a.details?.protocol || '—'}
        </div>
        <div class="alert-msg">${a.message || ''}</div>
      </div>
      <div class="alert-meta">
        <span class="alert-time">${(a.timestamp||'').substring(11,19)}</span>
        <div class="score-block">
          ${ifScore != null ? `<div>IF <span class="${ifCls}">${ifScore}</span></div>` : ''}
          ${aeErr   != null ? `<div>AE <span class="${aeCls}">${aeErr}</span></div>`   : ''}
        </div>
        <div class="view-btn">View →</div>
      </div>
    </div>`;
  }).join('');
}
 
// ── Modal ──────────────────────────────────────────────────────
function openModal(idx) {
  const a = allAlerts[idx];
  if (!a) return;
 
  const d = a.details || {};
  const cat = a.category || 'Unknown';
 
  // header
  document.getElementById('modalTitle').textContent = cat;
  document.getElementById('modalSub').textContent   =
    (a.timestamp || '').replace('T',' ').substring(0,19) + ' UTC  ·  ' +
    (d.source_ip || '—') + ' → ' + (d.destination_ip || '—');
 
  // severity color for attrs
  const statusColor = a.status === 'ALERT' ? 'red'
                    : a.status === 'WARNING' ? 'amber' : 'green';
 
  // IF score display
  const ifScore = a.if_score != null ? a.if_score.toFixed(4) : '—';
  const ifCol   = a.if_score != null
                  ? (a.if_score < 0 ? 'red' : 'green') : '';
 
  // AE error display
  const aeErr = a.ae_error != null ? a.ae_error.toFixed(6) : '—';
  const aeCol = a.ae_anomaly ? 'red' : 'green';
 
  // bytes friendly
  const bytes = d.bytes != null ? d.bytes.toLocaleString() + ' B' : '—';
 
  document.getElementById('modalBody').innerHTML = `
 
    <!-- Section 1: Incident summary -->
    <div class="inc-section">
      <div class="inc-section-title">Incident Summary</div>
      <div class="attr-grid">
        <div class="attr-item">
          <div class="attr-key">Severity</div>
          <div class="attr-val ${statusColor}">${a.status || '—'}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Attack Category</div>
          <div class="attr-val">${cat}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Threat Type</div>
          <div class="attr-val">${(a.threat_type||'—').replace(/_/g,' ')}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Timestamp (UTC)</div>
          <div class="attr-val">${(a.timestamp||'—').replace('T',' ').substring(0,19)}</div>
        </div>
        <div class="attr-item" style="grid-column:span 2">
          <div class="attr-key">Description</div>
          <div class="attr-val" style="font-size:12px;line-height:1.6;color:var(--text-soft)">
            ${a.message || '—'}
          </div>
        </div>
      </div>
    </div>
 
    <!-- Section 2: Network packet attributes -->
    <div class="inc-section">
      <div class="inc-section-title">Network Packet Attributes</div>
      <div class="attr-grid">
        <div class="attr-item">
          <div class="attr-key">Source IP</div>
          <div class="attr-val blue">${d.source_ip || '—'}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Destination IP</div>
          <div class="attr-val blue">${d.destination_ip || '—'}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Source Port</div>
          <div class="attr-val">${d.source_port != null ? d.source_port : '—'}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Destination Port</div>
          <div class="attr-val">${d.dest_port != null ? d.dest_port : '—'}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Protocol</div>
          <div class="attr-val">${d.protocol || '—'}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Total Bytes</div>
          <div class="attr-val">${bytes}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Packet Count</div>
          <div class="attr-val">${d.packets != null ? d.packets : '—'}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Avg Packet Size</div>
          <div class="attr-val">${d.avg_pkt_size != null ? d.avg_pkt_size + ' B' : '—'}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Duration</div>
          <div class="attr-val">${d.duration != null ? d.duration + 's' : '—'}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Recommended Action</div>
          <div class="attr-val ${statusColor}">${a.action || '—'}</div>
        </div>
      </div>
    </div>
 
    <!-- Section 3: Network packet (raw view) -->
    <div class="inc-section">
      <div class="inc-section-title">Network Packet</div>
      <div class="packet-box">
        <div class="pk-row">
          <span class="pk-field">[ FLOW ID ]</span>
          <span class="pk-value hi">${d.source_ip || '—'}:${d.source_port != null ? d.source_port : '?'}
            &nbsp;→&nbsp;
            ${d.destination_ip || '—'}:${d.dest_port != null ? d.dest_port : '?'}</span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ PROTOCOL ]</span>
          <span class="pk-value">${d.protocol || '—'}</span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ DIRECTION ]</span>
          <span class="pk-value">
            ${isPrivate(d.source_ip) ? 'Internal' : 'External'}
            &nbsp;→&nbsp;
            ${isPrivate(d.destination_ip) ? 'Internal' : 'External'}
          </span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ PACKETS ]</span>
          <span class="pk-value">${d.packets != null ? d.packets : '—'}</span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ BYTES ]</span>
          <span class="pk-value">${bytes}</span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ AVG SIZE ]</span>
          <span class="pk-value">${d.avg_pkt_size != null ? d.avg_pkt_size + ' bytes' : '—'}</span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ DURATION ]</span>
          <span class="pk-value">${d.duration != null ? d.duration + ' seconds' : '—'}</span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ TIMESTAMP ]</span>
          <span class="pk-value">${(a.timestamp||'—').replace('T',' ')}</span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ IF SCORE ]</span>
          <span class="pk-value ${a.if_score != null && a.if_score < 0 ? 'red' : 'green'}">${ifScore}</span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ AE ERROR ]</span>
          <span class="pk-value ${a.ae_anomaly ? 'red' : 'green'}">${aeErr}</span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ ML FLAGGED ]</span>
          <span class="pk-value ${a.ml_flagged ? 'red' : 'green'}">${a.ml_flagged ? 'YES — anomaly detected' : 'NO — within normal range'}</span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ VERDICT ]</span>
          <span class="pk-value ${statusColor}">${a.status || '—'}</span>
        </div>
        <div class="pk-row">
          <span class="pk-field">[ ACTION ]</span>
          <span class="pk-value amber">${a.action || '—'}</span>
        </div>
      </div>
    </div>
 
    <!-- Section 4: ML analysis -->
    <div class="inc-section">
      <div class="inc-section-title">Machine Learning Analysis</div>
      <div class="attr-grid">
        <div class="attr-item">
          <div class="attr-key">Isolation Forest Score</div>
          <div class="attr-val ${ifCol}">${ifScore}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">IF Anomaly Flag</div>
          <div class="attr-val ${a.if_anomaly ? 'red' : 'green'}">
            ${a.if_anomaly ? 'ANOMALY' : 'Normal'}
          </div>
        </div>
        <div class="attr-item">
          <div class="attr-key">Autoencoder Error</div>
          <div class="attr-val ${aeCol}">${aeErr}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">AE Anomaly Flag</div>
          <div class="attr-val ${a.ae_anomaly ? 'red' : 'green'}">
            ${a.ae_anomaly ? 'ANOMALY' : 'Normal'}
          </div>
        </div>
        <div class="attr-item">
          <div class="attr-key">ML Confidence</div>
          <div class="attr-val">${a.confidence != null ? (a.confidence * 100).toFixed(1) + '%' : '—'}</div>
        </div>
        <div class="attr-item">
          <div class="attr-key">ML Overall Flag</div>
          <div class="attr-val ${a.ml_flagged ? 'red' : 'green'}">
            ${a.ml_flagged ? '🧠 Flagged as anomalous' : 'Within normal range'}
          </div>
        </div>
      </div>
    </div>
  `;
 
  document.getElementById('modalOverlay').classList.add('open');
}
 
function isPrivate(ip) {
  if (!ip) return false;
  return ip.startsWith('10.') || ip.startsWith('192.168.') ||
         ip.startsWith('172.') || ip.startsWith('127.');
}
 
function closeModal(e) {
  if (e.target === document.getElementById('modalOverlay')) closeModalDirect();
}
 
function closeModalDirect() {
  document.getElementById('modalOverlay').classList.remove('open');
}
 
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModalDirect();
});
 
load();
setInterval(load, 3000);
</script>
</body>
</html>
"""
 
# ── Routes ─────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)
 
@app.route("/api/data")
def api_data():
    all_alerts = read_json_lines(ALERT_FILE)
    flows      = read_json_lines(FLOW_LOG)
 
    # Enrich alerts with category and filter to last hour
    for a in all_alerts:
        a["category"] = categorize_threat(
            a.get("threat_type",""),
            a.get("message",""),
            a.get("status","")
        )
 
    hour_alerts = filter_last_hour(all_alerts)
 
    total    = len(flows)
    alert_c  = sum(1 for f in flows if f.get("Status") == "ALERT")
    warn_c   = sum(1 for f in flows if f.get("Status") == "WARNING")
    normal_c = sum(1 for f in flows if f.get("Status") == "NORMAL")
    ml_c     = sum(1 for f in flows if f.get("ML Flagged"))
 
    recent    = flows[-20:] if flows else []
    if_active = any(f.get("IF Score") is not None for f in recent)
    ae_active = any(f.get("AE Error") is not None for f in recent)
 
    return jsonify({
        "stats":  {"total":total,"alert":alert_c,"warning":warn_c,"normal":normal_c,"ml_flags":ml_c},
        "models": {"if_enabled":if_active,"ae_enabled":ae_active},
        "alerts": hour_alerts,
        "flows":  flows[-50:],
    })
 
@app.route("/api/alerts")
def api_alerts():
    items = read_json_lines(ALERT_FILE)
    for a in items:
        a["category"] = categorize_threat(
            a.get("threat_type",""), a.get("message",""), a.get("status",""))
    return jsonify(items)
 
@app.route("/api/flows")
def api_flows():
    return jsonify(read_json_lines(FLOW_LOG))
 
if __name__ == "__main__":
    print("=" * 48)
    print("  NeuraShield — Network Guardian Dashboard")
    print("=" * 48)
    print(f"  URL  : http://127.0.0.1:5000")
    print(f"  Stop : Ctrl+C")
    print("=" * 48)
    app.run(host="127.0.0.1", port=5000, debug=False)

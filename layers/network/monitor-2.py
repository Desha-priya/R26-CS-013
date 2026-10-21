# monitor.py
# Network Guardian — NeuraShield
# Run as Administrator: venv\Scripts\activate && python monitor.py

from scapy.all import sniff, IP, TCP, UDP, ICMP
from datetime import datetime, timezone
import time, json, threading, pickle, os
import numpy as np

# The saved scaler/forest pickles were written with numpy 2 (which names its core
# package "numpy._core"). The platform runs every layer on one environment whose
# numpy is 1.26 (TensorFlow requires < 2), where the same package is "numpy.core".
# Alias it so those pickles load; a no-op on numpy 2.
import sys as _sys
if int(np.__version__.split(".")[0]) < 2:
    # (numpy 1.26 ships a stub "numpy._core" package, so test the version, not the name)
    import numpy.core as _npcore
    _sys.modules["numpy._core"] = _npcore
    for _sub in ("numeric", "multiarray", "_multiarray_umath", "umath", "numerictypes",
                 "fromnumeric", "arrayprint", "defchararray", "records", "shape_base",
                 "_internal", "_dtype", "_methods"):
        try:
            _sys.modules["numpy._core." + _sub] = __import__("numpy.core." + _sub, fromlist=["x"])
        except Exception:
            pass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
# INTERFACE can be overridden from the environment (set by the NeuraShield
# supervisor / installer) so the capture NIC is not hard-coded to one machine.
# Empty string => Scapy sniffs the default interface.
# Empty / unset => None => Scapy picks the default capture interface. (The
# previous built-in default was one specific machine's NPF adapter GUID, which
# makes sniff() fail on every other computer.)
INTERFACE    = os.environ.get("NEURASHIELD_NET_INTERFACE", "").strip() or None
FLOW_TIMEOUT = 30
PRINT_FLOWS  = True

# Runtime data directory. The supervisor points every layer at its own data
# folder via NEURASHIELD_DATA_DIR; the dashboard (app.py) reads the same two
# files from the same place, so both must resolve them identically.
_DATA_DIR    = os.environ.get("NEURASHIELD_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(_DATA_DIR, exist_ok=True)
ALERT_FILE   = os.path.join(_DATA_DIR, "alerts.json")
FLOW_LOG     = os.path.join(_DATA_DIR, "flow_log.json")

# ─────────────────────────────────────────
# PLATFORM INTEGRATION (NeuraShield ZTA platform — Layer 3)
# ─────────────────────────────────────────
# The Network Guardian forwards every WARNING/ALERT flow to the central
# platform as POST /api/layer1/threat with the shared X-API-Key header, using
# exactly the same API-key contract as the other layers. No key is built in:
# it must come from NEURASHIELD_API_KEY (set by the supervisor). Without a key,
# alerts are still written locally for the dashboard but are not sent.
import urllib.request
import urllib.error

_PLATFORM_ORIGIN = (os.environ.get("NEURASHIELD_PLATFORM_URL", "http://127.0.0.1:8000")
                    .strip().rstrip("/"))
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

_PLATFORM_API_KEY = _shared_api_key()
_PLATFORM_ALERT_URL = _PLATFORM_ORIGIN + "/api/layer1/threat"
# Only these severities are forwarded; low-severity flows stay local.
_PLATFORM_SEND_SEVERITIES = {"medium", "high", "critical"}
_platform_warned_no_key = False


def send_to_platform(platform_alert):
    """Send one network alert to the ZTA platform's Layer 1 endpoint.

    Never raises: a platform outage must never interrupt packet capture.
    `platform_alert` is the dict returned by save_alert():
    {source, severity, layer, message}. The platform's LayerEventRequest
    expects {event_type, severity}, so we map message->event_type.
    """
    global _platform_warned_no_key
    severity = str(platform_alert.get("severity", "medium")).lower()
    if severity not in _PLATFORM_SEND_SEVERITIES:
        return
    if not _PLATFORM_API_KEY:
        if not _platform_warned_no_key:
            print("  [Network to Platform] No NEURASHIELD_API_KEY set — "
                  "alerts are NOT sent to the platform (written locally only).")
            _platform_warned_no_key = True
        return
    body = json.dumps({
        "event_type": str(platform_alert.get("message", "network_anomaly"))[:300],
        "severity": severity,
    }).encode("utf-8")
    req = urllib.request.Request(
        _PLATFORM_ALERT_URL, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "X-API-Key": _PLATFORM_API_KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                print(f"  [Network to Platform] alert sent ({severity})")
            else:
                print(f"  [Network to Platform] alert rejected HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        print(f"  [Network to Platform] platform rejected request (HTTP {e.code})")
    except urllib.error.URLError as e:
        print(f"  [Network to Platform] could not reach platform at "
              f"{_PLATFORM_ALERT_URL}: {e.reason}")
    except Exception as e:
        print(f"  [Network to Platform] unexpected error sending alert: {e}")

IF_THRESHOLD          = -0.0100
AE_THRESHOLD_MULTIPLIER = 3.0

SAFE_PORTS = {
    53, 80, 123, 443, 5353, 67, 68,
    546, 547, 1900, 5355, 137, 138, 139, 8080, 8443,
}

SAFE_IPS = {
    "8.8.8.8","8.8.4.4","1.1.1.1","1.0.0.1",
    "162.159.200.1","162.159.200.123",
    "216.239.35.0","216.239.35.4",
    "17.253.84.125","91.189.91.157","91.189.89.199",
    "192.168.64.1",
}

SAFE_IP_PREFIXES = (
    "224.","239.","255.","169.254.",
    "142.250.","142.251.","172.217.","172.253.",
    "216.58.","74.125.","34.107.","34.120.",
    "146.75.","151.101.",
)

print("=" * 60)
print("  Network Guardian — NeuraShield")
print("  Autonomous Cyber Defense Platform")
print("=" * 60)

# ─────────────────────────────────────────
# LOAD ISOLATION FOREST
# ─────────────────────────────────────────
ML_ENABLED  = False
ml_model    = None
ml_scaler   = None
ml_features = None

try:
    with open("isolation_forest.pkl", "rb") as f:
        ml_model = pickle.load(f)
    if isinstance(ml_model, dict) and "scaler" in ml_model:
        ml_scaler = ml_model["scaler"]
    else:
        with open("scaler.pkl", "rb") as f:
            ml_scaler = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        ml_features = pickle.load(f)
    ML_ENABLED = True
    if os.path.exists("model_config.json"):
        with open("model_config.json") as f:
            cfg = json.load(f)
        stored = cfg.get("if_threshold", IF_THRESHOLD)
        if stored < 0:
            IF_THRESHOLD = stored
        is_ensemble = cfg.get("ensemble", False)
        model_count = cfg.get("model_count", 1)
        if is_ensemble:
            print(f"  ✅ Isolation Forest : Ensemble ({model_count} models)")
            print(f"     Weights          : {cfg.get('weights',[1.0])}")
        else:
            print(f"  ✅ Isolation Forest : Single model")
    else:
        print(f"  ✅ Isolation Forest : loaded")
    print(f"     IF Threshold     : {IF_THRESHOLD}")
except Exception as e:
    print(f"  ❌ Isolation Forest : {e}")

# ─────────────────────────────────────────
# AUTOENCODER
# ─────────────────────────────────────────
if TORCH_AVAILABLE:
    class NetworkAutoencoder(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(),
                nn.Linear(32, 16),        nn.ReLU(),
                nn.Linear(16, 8),         nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 16),         nn.ReLU(),
                nn.Linear(16, 32),        nn.ReLU(),
                nn.Linear(32, input_dim),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

AE_ENABLED   = False
ae_model     = None
ae_scaler    = None
ae_config    = None
AE_THRESHOLD = 0.10

if TORCH_AVAILABLE:
    try:
        with open("autoencoder_config.pkl", "rb") as f:
            ae_config = pickle.load(f)
        with open("ae_scaler.pkl", "rb") as f:
            ae_scaler = pickle.load(f)
        ae_model = NetworkAutoencoder(ae_config["input_dim"])
        ae_model.load_state_dict(
            torch.load("autoencoder.pth", map_location=torch.device("cpu"))
        )
        ae_model.eval()
        AE_THRESHOLD = ae_config["threshold"] * AE_THRESHOLD_MULTIPLIER
        AE_ENABLED   = True
        print(f"  ✅ Autoencoder      : loaded")
        print(f"     AE Threshold     : {AE_THRESHOLD:.4f}")
    except Exception as e:
        print(f"  ❌ Autoencoder      : {e}")

print(f"  📡 Interface        : {INTERFACE or '(system default)'}")
print(f"  📁 Alert file       : {ALERT_FILE}")
print(f"  📁 Flow log         : {FLOW_LOG}")
print(f"  🕐 Started          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Press Ctrl+C to stop")
print("=" * 60 + "\n")

# ─────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────
active_flows    = {}
completed_flows = []
stats = {"total":0,"normal":0,"warning":0,"alert":0,"ml_flags":0}

WELL_KNOWN_PORTS = {
    20:"FTP Data",21:"FTP",22:"SSH",23:"Telnet",25:"SMTP",
    53:"DNS",80:"HTTP",110:"POP3",143:"IMAP",443:"HTTPS",
    445:"SMB",993:"IMAPS",995:"POP3S",1433:"MSSQL",3306:"MySQL",
    3389:"RDP",4444:"Metasploit",5432:"PostgreSQL",5900:"VNC",
    6379:"Redis",6666:"IRC/C2",6667:"IRC/C2",8080:"HTTP Proxy",
    8443:"HTTPS Alt",9001:"Tor",9050:"Tor SOCKS",27017:"MongoDB",
}

SUSPICIOUS_PORTS   = {4444,1337,31337,6666,6667,6668,9001,9050,12345,54321}
DATABASE_PORTS     = {1433,3306,5432,6379,27017}
REMOTE_ADMIN_PORTS = {23,3389,5900}

PRIVATE_PREFIXES = (
    "10.","192.168.","127.",
    "172.16.","172.17.","172.18.","172.19.",
    "172.20.","172.21.","172.22.","172.23.",
    "172.24.","172.25.","172.26.","172.27.",
    "172.28.","172.29.","172.30.","172.31.",
)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_proto_name(n):
    return {6:"TCP",17:"UDP",1:"ICMP",2:"IGMP"}.get(n,f"PROTO({n})")

def is_private(ip):
    return ip.startswith(PRIVATE_PREFIXES)

def is_whitelisted(ip, port):
    if port in SAFE_PORTS: return True
    if ip in SAFE_IPS: return True
    if any(ip.startswith(p) for p in SAFE_IP_PREFIXES): return True
    return False

def port_label(p):
    return WELL_KNOWN_PORTS.get(p, f"Port {p}")

def get_flow_key(pkt):
    if IP not in pkt: return None
    src   = pkt[IP].src
    dst   = pkt[IP].dst
    proto = pkt[IP].proto
    sp    = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
    dp    = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
    return (src, dst, sp, dp, proto)

# ─────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────
def build_features(flow, feature_list):
    dur_ms   = flow["Duration (seconds)"] * 1000
    dur_sec  = max(flow["Duration (seconds)"], 0.001)
    total_b  = flow["Total Bytes"]
    pkts     = flow["Packet Count"]
    avg_ps   = flow["Average Packet Size"]
    proto    = flow["Protocol"]

    src_pkts  = pkts * 0.6
    src_bytes = total_b * 0.6
    dst_pkts  = pkts * 0.4
    dst_bytes = total_b * 0.4
    bps       = total_b / dur_sec
    pps       = pkts / dur_sec
    fpps      = src_pkts / dur_sec
    bpps      = dst_pkts / dur_sec
    mean_piat = dur_ms / max(pkts, 1)
    std_piat  = mean_piat * 0.3

    fmap = {
        "bidirectional_duration_ms":    dur_ms,
        "src2dst_packets":              src_pkts,
        "src2dst_bytes":                src_bytes,
        "dst2src_packets":              dst_pkts,
        "dst2src_bytes":                dst_bytes,
        "fwd_packet_length_max":        avg_ps * 1.5,
        "fwd_packet_length_min":        avg_ps * 0.3,
        "fwd_packet_length_mean":       avg_ps * 0.9,
        "fwd_packet_length_std":        avg_ps * 0.2,
        "bwd_packet_length_max":        avg_ps * 1.8,
        "bwd_packet_length_min":        avg_ps * 0.2,
        "bwd_packet_length_mean":       avg_ps * 1.1,
        "bwd_packet_length_std":        avg_ps * 0.3,
        "flow_bytes_per_second":        bps,
        "flow_packets_per_second":      pps,
        "fwd_packets_per_second":       fpps,
        "bwd_packets_per_second":       bpps,
        "bidirectional_mean_piat_ms":   mean_piat,
        "bidirectional_stddev_piat_ms": std_piat,
        "bidirectional_max_piat_ms":    dur_ms,
        "bidirectional_min_piat_ms":    0.1,
        "fwd_iat_mean":                 mean_piat * 1.1,
        "fwd_iat_std":                  std_piat,
        "fwd_iat_max":                  dur_ms * 0.8,
        "fwd_iat_min":                  0.1,
        "bwd_iat_mean":                 mean_piat * 0.9,
        "bwd_iat_std":                  std_piat * 0.8,
        "bwd_iat_max":                  dur_ms * 0.6,
        "bwd_iat_min":                  0.1,
        "fwd_header_length":            src_pkts * 32,
        "bwd_header_length":            dst_pkts * 32,
        "bidirectional_min_ps":         avg_ps * 0.5,
        "bidirectional_max_ps":         avg_ps * 1.5,
        "bidirectional_mean_ps":        avg_ps,
        "bidirectional_stddev_ps":      avg_ps * 0.2,
        "bidirectional_variance_ps":    (avg_ps * 0.2) ** 2,
        "avg_packet_size":              avg_ps,
        "bidirectional_fin_packets":    1 if proto == "TCP" else 0,
        "bidirectional_psh_packets":    max(pkts // 2, 0),
        "bidirectional_ack_packets":    max(pkts - 1, 0),
    }

    import pandas as pd
    return pd.DataFrame(
        [[fmap.get(f, 0.0) for f in feature_list]],
        columns=feature_list
    )

# ─────────────────────────────────────────
# ML SCORING
# ─────────────────────────────────────────
def score_if(flow):
    if not ML_ENABLED:
        return None, False, 0.0
    try:
        feat   = build_features(flow, ml_features)
        scaled = ml_scaler.transform(feat)
        if isinstance(ml_model, dict) and "model_A" in ml_model:
            weights = ml_model["weights"]
            score_A = float(ml_model["model_A"].decision_function(scaled)[0])
            score_B = float(ml_model["model_B"].decision_function(scaled)[0])
            score_C = float(ml_model["model_C"].decision_function(scaled)[0])
            score   = (weights[0]*score_A + weights[1]*score_B + weights[2]*score_C)
        else:
            score = float(ml_model.decision_function(scaled)[0])
        is_an = score < IF_THRESHOLD
        conf  = min(abs(score - IF_THRESHOLD) * 300, 100.0) if is_an else 0.0
        return round(score, 4), is_an, round(conf, 1)
    except:
        return None, False, 0.0

def score_ae(flow):
    if not AE_ENABLED:
        return None, False, 0.0
    try:
        feat   = build_features(flow, ae_config["feature_cols"])
        scaled = ae_scaler.transform(feat)
        tensor = torch.FloatTensor(scaled)
        with torch.no_grad():
            recon = ae_model(tensor)
            error = float(torch.mean((tensor - recon)**2).item())
        is_an = error > AE_THRESHOLD
        conf  = 0.0
        if is_an:
            excess = error - AE_THRESHOLD
            conf   = min(100.0 * (1 - 1/(1+(excess/AE_THRESHOLD)*3)), 99.0)
        return round(error, 6), is_an, round(conf, 1)
    except:
        return None, False, 0.0

# ─────────────────────────────────────────
# RULE-BASED ANALYSIS
# ─────────────────────────────────────────
def analyze_flow(flow):
    src_ip   = flow["Source IP"]
    dst_ip   = flow["Destination IP"]
    src_port = flow["Source Port"]
    dst_port = flow["Destination Port"]
    protocol = flow["Protocol"]
    packets  = flow["Packet Count"]
    total_b  = flow["Total Bytes"]
    avg_size = flow["Average Packet Size"]
    duration = flow["Duration (seconds)"]

    if (is_whitelisted(src_ip, src_port) or
            is_whitelisted(dst_ip, dst_port)):
        return ("NORMAL","Safe Traffic",
                f"Whitelisted {src_ip} → {dst_ip} ({protocol} port {dst_port}).")

    src_priv = is_private(src_ip)
    dst_priv = is_private(dst_ip)
    pps      = packets / max(duration, 0.001)
    bps      = total_b / max(duration, 0.001)

    if pps > 1000 and duration < 60:
        return ("ALERT","DoS Attack Pattern",
                f"Extremely high packet rate {pps:.0f} pps — DoS flood.")

    if pps > 500 and avg_size < 100 and packets > 100:
        return ("ALERT","DDoS Pattern",
                f"High rate {pps:.0f} pps with small packets ({avg_size:.0f}B avg).")

    if duration < 0.01 and total_b < 100 and packets <= 2:
        if dst_port not in SAFE_PORTS:
            return ("WARNING","Port Scan Probe",
                    f"Ultra-short flow to port {dst_port} — port scan.")

    if total_b > 10_000_000 and src_priv and not dst_priv and bps > 1_000_000:
        return ("ALERT","Data Exfiltration Pattern",
                f"Large high-speed outbound {total_b/1_000_000:.1f}MB at {bps/1_000_000:.1f}MB/s.")

    if protocol == "ICMP":
        if packets > 500 and duration < 10:
            return ("ALERT","ICMP Flood",
                    f"High ICMP rate — {packets} pkts in {duration:.1f}s.")
        if packets > 100 and avg_size < 30:
            return ("WARNING","ICMP Sweep",
                    f"{src_ip} sent {packets} tiny ICMP packets.")
        return ("NORMAL","ICMP Ping", f"{src_ip} → {dst_ip}.")

    if dst_port == 53 or src_port == 53:
        return ("NORMAL","DNS Query", f"{src_ip} queried DNS at {dst_ip}.")

    if dst_port in SUSPICIOUS_PORTS:
        return ("ALERT","Suspicious Port",
                f"Connection to known attack port {dst_port} ({port_label(dst_port)}) at {dst_ip}.")

    if pps > 2000:
        return ("ALERT","Possible DoS Attack",
                f"Extremely high packet rate {pps:.0f} pps from {src_ip}.")

    if (protocol == "TCP" and packets == 1 and total_b < 80 and
            dst_port not in SAFE_PORTS and dst_port > 1024 and not dst_priv):
        return ("WARNING","Port Probe",
                f"{src_ip} sent single packet to external {dst_ip}:{dst_port}.")

    if dst_port in DATABASE_PORTS and not src_priv:
        return ("ALERT","External Database Access",
                f"External {src_ip} accessing database port {dst_port}.")

    if dst_port in REMOTE_ADMIN_PORTS and not src_priv:
        return ("WARNING","External Admin Access",
                f"External {src_ip} connecting to admin port {dst_port}.")

    if total_b > 100_000_000 and src_priv and not dst_priv:
        return ("WARNING","Large Data Transfer",
                f"{src_ip} sent {total_b/1_000_000:.1f} MB to external {dst_ip}.")

    if src_priv and dst_priv:
        return ("NORMAL","Internal Traffic",
                f"{src_ip} → {dst_ip}:{dst_port} ({protocol}).")

    if src_priv and not dst_priv:
        return ("NORMAL","Outbound Connection",
                f"{src_ip} → {dst_ip}:{dst_port} ({protocol}).")

    if not src_priv and dst_priv:
        if packets == 1 and dst_port not in SAFE_PORTS:
            return ("WARNING","Single Packet Inbound",
                    f"External {src_ip} sent one packet to {dst_ip}:{dst_port}.")
        return ("NORMAL","Inbound Connection",
                f"External {src_ip} → {dst_ip}:{dst_port} ({packets} pkts).")

    return ("NORMAL","General Traffic",
            f"{src_ip}:{src_port} → {dst_ip}:{dst_port} ({protocol}) {packets} pkts.")

# ─────────────────────────────────────────
# SAVE ALERT — NeuraShield platform format
# AND extended format for dashboard
# ─────────────────────────────────────────
def save_alert(flow):
    status     = flow.get("Status", "WARNING")
    label      = flow.get("Label", "Unknown")
    src_ip     = flow.get("Source IP", "")
    dst_ip     = flow.get("Destination IP", "")
    dst_port   = flow.get("Destination Port", "")
    src_port   = flow.get("Source Port", "")
    protocol   = flow.get("Protocol", "")
    packets    = flow.get("Packet Count", 0)
    total_b    = flow.get("Total Bytes", 0)
    duration   = flow.get("Duration (seconds)", 0)
    avg_ps     = flow.get("Average Packet Size", 0)
    if_score   = flow.get("IF Score")
    ae_error   = flow.get("AE Error")
    if_anom    = flow.get("IF Anomaly", False)
    ae_anom    = flow.get("AE Anomaly", False)
    if_conf    = flow.get("IF Confidence", 0)
    ae_conf    = flow.get("AE Confidence", 0)
    ml_flagged = flow.get("ML Flagged", False)
    timestamp  = flow.get("Timestamp", datetime.now(timezone.utc).isoformat())

    # ── Severity mapping ──────────────────────────────────────
    severity_map = {"ALERT":"critical","WARNING":"medium","NORMAL":"low"}
    severity     = severity_map.get(status, "medium")
    high_risk    = ["suspicious port","c2","command","control",
                    "dos attack","ddos","exfiltration","database access"]
    if any(r in label.lower() for r in high_risk) and severity == "medium":
        severity = "high"

    # ── Human readable message ────────────────────────────────
    msg = (f"{label} detected. "
           f"{src_ip} → {dst_ip}:{dst_port} ({protocol}). "
           f"Packets: {packets}, Bytes: {total_b}, Duration: {duration}s.")
    if ml_flagged:
        msg += f" ML: IF={if_score:.4f}" if if_score is not None else " ML: IF=N/A"
        if ae_error is not None:
            msg += f" AE={ae_error:.4f}"
        msg += "."

    # ── NeuraShield platform format (for Autonomous Brain) ────
    platform_alert = {
        "source":   "network_guardian",
        "severity": severity,
        "layer":    1,
        "message":  msg,
    }

    # ── Extended format (for dashboard) ──────────────────────
    # Dashboard needs more fields to display properly
    dashboard_alert = {
        # Platform fields
        "source":   "network_guardian",
        "severity": severity,
        "layer":    1,
        "message":  msg,
        # Dashboard display fields
        "timestamp":   timestamp,
        "status":      status,
        "threat_type": label.lower().replace(" ", "_"),
        "action":      "Investigate" if status == "ALERT" else "Monitor",
        "confidence":  round((if_conf + ae_conf) / 2 / 100, 2) if ml_flagged else 0,
        "ml_flagged":  ml_flagged,
        "if_anomaly":  if_anom,
        "ae_anomaly":  ae_anom,
        "if_score":    if_score,
        "ae_error":    ae_error,
        "details": {
            "source_ip":        src_ip,
            "destination_ip":   dst_ip,
            "source_port":      src_port,
            "dest_port":        dst_port,
            "protocol":         protocol,
            "packets":          packets,
            "bytes":            total_b,
            "duration":         duration,
            "avg_pkt_size":     round(avg_ps, 2),
        }
    }

    with open(ALERT_FILE, "a") as f:
        f.write(json.dumps(dashboard_alert) + "\n")

    return platform_alert

def save_flow_log(flow):
    with open(FLOW_LOG, "a") as f:
        f.write(json.dumps(flow) + "\n")

# ─────────────────────────────────────────
# TERMINAL DISPLAY
# ─────────────────────────────────────────
COLORS = {"NORMAL":"\033[92m","WARNING":"\033[93m","ALERT":"\033[91m"}
ICONS  = {"NORMAL":"✅","WARNING":"⚠️ ","ALERT":"🚨"}
RESET  = "\033[0m"

def print_flow(flow):
    status = flow["Status"]
    color  = COLORS.get(status,"")
    icon   = ICONS.get(status,"  ")
    parts  = []
    if flow.get("IF Score") is not None:
        parts.append(f"IF:{flow['IF Score']:+.3f}")
    if flow.get("AE Error") is not None:
        parts.append(f"AE:{flow['AE Error']:.4f}")
    conf = flow.get("IF Confidence") or flow.get("AE Confidence") or 0
    if conf > 0:
        parts.append(f"Conf:{conf:.0f}%")
    ml_tag = (" | " + " | ".join(parts)) if parts else ""
    if flow.get("ML Flagged"):
        ml_tag += " 🧠"
    print(
        f"\n{icon} {color}[{status:7s}]{RESET} "
        f"[{flow['Timestamp'][:19]}]{ml_tag}\n"
        f"   {flow['Protocol']:5s} "
        f"{flow['Source IP']:15s}:{str(flow['Source Port']):<6} → "
        f"{flow['Destination IP']:15s}:{str(flow['Destination Port']):<6}\n"
        f"   Pkts:{flow['Packet Count']:5d} | "
        f"Bytes:{flow['Total Bytes']:9,d} | "
        f"Dur:{flow['Duration (seconds)']:6.2f}s\n"
        f"   {color}[{flow['Label']}]{RESET} {flow['Message']}"
    )

# ─────────────────────────────────────────
# PACKET HANDLER
# ─────────────────────────────────────────
def process_packet(pkt):
    if IP not in pkt:
        return
    key = get_flow_key(pkt)
    if not key:
        return
    now = time.time()
    if key not in active_flows:
        active_flows[key] = {
            "Source IP":        key[0],
            "Destination IP":   key[1],
            "Source Port":      key[2],
            "Destination Port": key[3],
            "Protocol":         get_proto_name(key[4]),
            "Start Time":       now,
            "Last Seen":        now,
            "Packet Count":     0,
            "Total Bytes":      0,
        }
    fl = active_flows[key]
    fl["Last Seen"]     = now
    fl["Packet Count"] += 1
    fl["Total Bytes"]  += len(pkt)
    if TCP in pkt:
        flags = pkt[TCP].flags
        if flags & 0x01 or flags & 0x04:
            finish_flow(key)

# ─────────────────────────────────────────
# FLOW FINALIZATION
# ─────────────────────────────────────────
def finish_flow(key):
    if key not in active_flows:
        return
    raw      = active_flows.pop(key)
    duration = max(raw["Last Seen"] - raw["Start Time"], 0.001)
    avg_ps   = raw["Total Bytes"] / max(raw["Packet Count"], 1)

    flow = {
        "Timestamp":          datetime.now(timezone.utc).isoformat(),
        "Protocol":           raw["Protocol"],
        "Source IP":          raw["Source IP"],
        "Destination IP":     raw["Destination IP"],
        "Source Port":        raw["Source Port"],
        "Destination Port":   raw["Destination Port"],
        "Packet Count":       raw["Packet Count"],
        "Total Bytes":        raw["Total Bytes"],
        "Average Packet Size":round(avg_ps, 2),
        "Duration (seconds)": round(duration, 3),
    }

    status, label, message = analyze_flow(flow)

    if_score, if_anom, if_conf = score_if(flow)
    flow["IF Score"]      = if_score
    flow["IF Anomaly"]    = if_anom
    flow["IF Confidence"] = if_conf

    ae_error, ae_anom, ae_conf = score_ae(flow)
    flow["AE Error"]      = ae_error
    flow["AE Anomaly"]    = ae_anom
    flow["AE Confidence"] = ae_conf

    both_flagged       = if_anom and ae_anom
    flow["ML Flagged"] = both_flagged

    if both_flagged:
        avg_conf = round((if_conf + ae_conf) / 2, 1)
        if status == "NORMAL":
            status  = "WARNING"
            label   = "Dual Model Anomaly"
            message = (f"Both ML models flagged. "
                       f"IF={if_score:.3f} AE={ae_error:.4f} conf={avg_conf:.0f}%.")
        elif status == "WARNING":
            status  = "ALERT"
            label   = f"ML + Rule: {label}"
            message = (f"Both ML models AND rule engine flagged. "
                       f"IF={if_score:.3f} AE={ae_error:.4f} conf={avg_conf:.0f}%. {message}")

    flow["Status"]  = status
    flow["Label"]   = label
    flow["Message"] = message

    stats["total"] += 1
    stats[status.lower()] = stats.get(status.lower(), 0) + 1
    if both_flagged:
        stats["ml_flags"] += 1

    save_flow_log(flow)
    if status in ("WARNING","ALERT"):
        platform_alert = save_alert(flow)
        # Forward to the central ZTA platform (never blocks/raises).
        send_to_platform(platform_alert)

    completed_flows.append(flow)
    if PRINT_FLOWS:
        print_flow(flow)

# ─────────────────────────────────────────
# CLEANUP + STATS
# ─────────────────────────────────────────
def cleanup_old_flows():
    now     = time.time()
    expired = [k for k,v in active_flows.items()
               if now - v["Last Seen"] > FLOW_TIMEOUT]
    for k in expired:
        finish_flow(k)

def stats_reporter():
    while True:
        time.sleep(30)
        cleanup_old_flows()
        total  = stats["total"]
        alerts = stats.get("alert", 0)
        warns  = stats.get("warning", 0)
        rate   = (alerts+warns)/total*100 if total > 0 else 0
        print(f"\n{'─'*60}")
        print(f"  📊 STATS — {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Active flows    : {len(active_flows)}")
        print(f"  Total completed : {total}")
        print(f"  🚨 Alerts       : {alerts}")
        print(f"  ⚠️  Warnings     : {warns}")
        print(f"  ✅ Normal       : {stats.get('normal',0)}")
        print(f"  🧠 ML flags     : {stats['ml_flags']}")
        print(f"  Alert rate      : {rate:.1f}%")
        print(f"{'─'*60}\n")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    open(ALERT_FILE, "w").close()
    open(FLOW_LOG,   "w").close()
    print(f"  Log files cleared\n")
    t = threading.Thread(target=stats_reporter, daemon=True)
    t.start()
    print(f"  🔍 Capturing live traffic...\n")
    try:
        sniff(
            iface=INTERFACE,
            prn=process_packet,
            store=False,
            filter="ip"
        )
    except KeyboardInterrupt:
        print(f"\n\n⛔ Stopping Network Guardian...")
        cleanup_old_flows()
        total  = stats["total"]
        alerts = stats.get("alert",0)
        warns  = stats.get("warning",0)
        rate   = (alerts+warns)/total*100 if total > 0 else 0
        print(f"\n  Final stats:")
        print(f"  Total:{total} | Alerts:{alerts} | Warnings:{warns} | Rate:{rate:.1f}%")
        print(f"\n  Goodbye!\n")
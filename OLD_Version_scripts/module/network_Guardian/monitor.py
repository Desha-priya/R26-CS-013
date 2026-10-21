# monitor.py
# Network Guardian — NeuraShield
# Run with: sudo venv/bin/python3 monitor.py
 
from scapy.all import sniff, IP, TCP, UDP, ICMP
from datetime import datetime
import time, json, threading, pickle, os
import numpy as np
 
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
 
# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
INTERFACE    = None
FLOW_TIMEOUT = 30
PRINT_FLOWS  = True
ALERT_FILE   = "alerts.json"
FLOW_LOG     = "flow_log.json"
 
# ML thresholds
IF_THRESHOLD             = -0.04
AE_THRESHOLD_MULTIPLIER  = 3.0
 
# ── Whitelist ports — never flagged ───────────────────────────
SAFE_PORTS = {
    53,    # DNS  — moved here to stop DNS false positives
    80,    # HTTP
    123,   # NTP
    443,   # HTTPS — moved here to stop HTTPS false positives
    5353,  # mDNS
    67, 68,# DHCP
    546, 547,
    1900,
    5355,
    137, 138, 139,
    8080, 8443,
}
 
# ── Whitelist IPs — never flagged ─────────────────────────────
SAFE_IPS = {
    "8.8.8.8", "8.8.4.4",           # Google DNS
    "1.1.1.1", "1.0.0.1",           # Cloudflare DNS
    "162.159.200.1","162.159.200.123",
    "216.239.35.0","216.239.35.4",
    "17.253.84.125",
    "91.189.91.157","91.189.89.199",
    "192.168.64.1",                  # UTM gateway
}
 
# ── Whitelist IP prefixes — never flagged ─────────────────────
SAFE_IP_PREFIXES = (
    "224.","239.","255.",
    "169.254.",
    "142.250.",   # Google services
    "142.251.",   # Google services
    "172.217.",   # Google services
    "172.253.",   # Google services
    "216.58.",    # Google services
    "74.125.",    # Google services
    "34.107.",    # Google Cloud
    "34.120.",    # Google Cloud
    "146.75.",    # Fastly CDN
    "151.101.",   # Fastly CDN
)
 
# ─────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────
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
    print(f"  ✅ Isolation Forest : loaded ({len(ml_features)} features)")
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
                nn.Linear(input_dim, 16), nn.ReLU(),
                nn.Linear(16, 8),         nn.ReLU(),
                nn.Linear(8, 4),          nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(4, 8),          nn.ReLU(),
                nn.Linear(8, 16),         nn.ReLU(),
                nn.Linear(16, input_dim),
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
            torch.load("autoencoder.pth",
                       map_location=torch.device("cpu"))
        )
        ae_model.eval()
        AE_THRESHOLD = ae_config["threshold"] * AE_THRESHOLD_MULTIPLIER
        AE_ENABLED   = True
        print(f"  ✅ Autoencoder      : loaded")
        print(f"     AE Threshold     : {AE_THRESHOLD:.4f}")
    except Exception as e:
        print(f"  ❌ Autoencoder      : {e}")
 
print(f"  📡 Interface        : {INTERFACE or 'all interfaces'}")
print(f"  📁 Alert file       : {ALERT_FILE}")
print(f"  📁 Flow log         : {FLOW_LOG}")
print(f"  🕐 Started          : "
      f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Press Ctrl+C to stop")
print("=" * 60 + "\n")
 
# ─────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────
active_flows    = {}
completed_flows = []
stats = {
    "total":0,"normal":0,
    "warning":0,"alert":0,"ml_flags":0,
}
 
# ─────────────────────────────────────────
# PORT DEFINITIONS
# ─────────────────────────────────────────
WELL_KNOWN_PORTS = {
    20:"FTP Data",    21:"FTP",         22:"SSH",
    23:"Telnet",      25:"SMTP",        53:"DNS",
    80:"HTTP",        110:"POP3",       143:"IMAP",
    443:"HTTPS",      445:"SMB",        993:"IMAPS",
    995:"POP3S",      1433:"MSSQL",     3306:"MySQL",
    3389:"RDP",       4444:"Metasploit",5432:"PostgreSQL",
    5900:"VNC",       6379:"Redis",     6666:"IRC/C2",
    6667:"IRC/C2",    8080:"HTTP Proxy",8443:"HTTPS Alt",
    9001:"Tor",       9050:"Tor SOCKS", 27017:"MongoDB",
}
 
# Real suspicious ports — known malware/attack ports only
SUSPICIOUS_PORTS   = {4444,1337,31337,6666,6667,6668,
                      9001,9050,12345,54321}
DATABASE_PORTS     = {1433,3306,5432,6379,27017}
REMOTE_ADMIN_PORTS = {23,3389,5900}   # removed 22 SSH — too many false positives
 
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
    """Return True if this IP or port should never be flagged"""
    if port in SAFE_PORTS:
        return True
    if ip in SAFE_IPS:
        return True
    if any(ip.startswith(p) for p in SAFE_IP_PREFIXES):
        return True
    return False
 
def port_label(p):
    return WELL_KNOWN_PORTS.get(p, f"Port {p}")
 
def get_flow_key(pkt):
    if IP not in pkt:
        return None
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
    dur_ms  = flow["Duration (seconds)"] * 1000
    total_b = flow["Total Bytes"]
    pkts    = flow["Packet Count"]
    avg_ps  = flow["Average Packet Size"]
    proto   = flow["Protocol"]
 
    fmap = {
        "bidirectional_duration_ms":    dur_ms,
        "bidirectional_packets":        pkts,
        "bidirectional_bytes":          total_b,
        "src2dst_packets":              pkts * 0.6,
        "src2dst_bytes":                total_b * 0.6,
        "dst2src_packets":              pkts * 0.4,
        "dst2src_bytes":                total_b * 0.4,
        "bidirectional_min_ps":         avg_ps * 0.5,
        "bidirectional_mean_ps":        avg_ps,
        "bidirectional_stddev_ps":      avg_ps * 0.2,
        "bidirectional_max_ps":         avg_ps * 1.5,
        "bidirectional_mean_piat_ms":   dur_ms / max(pkts,1),
        "bidirectional_stddev_piat_ms": dur_ms / max(pkts,1) * 0.3,
        "bidirectional_min_piat_ms":    0.1,
        "bidirectional_max_piat_ms":    dur_ms,
        "bidirectional_syn_packets":    1 if proto == "TCP" else 0,
        "bidirectional_ack_packets":    max(pkts-1, 0),
        "bidirectional_rst_packets":    0,
        "bidirectional_fin_packets":    1 if proto == "TCP" else 0,
        "bidirectional_psh_packets":    max(pkts//2, 0),
    }
    return np.array([[fmap.get(f,0.0) for f in feature_list]])
 
# ─────────────────────────────────────────
# ML SCORING
# ─────────────────────────────────────────
def score_if(flow):
    if not ML_ENABLED:
        return None, False, 0.0
    try:
        feat   = build_features(flow, ml_features)
        scaled = ml_scaler.transform(feat)
        score  = float(ml_model.decision_function(scaled)[0])
        is_an  = score < IF_THRESHOLD
        conf   = min(abs(score - IF_THRESHOLD)*300, 100.0) if is_an else 0.0
        return round(score,4), is_an, round(conf,1)
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
            error = float(torch.mean((tensor-recon)**2).item())
        is_an = error > AE_THRESHOLD
        conf  = min((error-AE_THRESHOLD)/AE_THRESHOLD*100,100.0) if is_an else 0.0
        return round(error,6), is_an, round(conf,1)
    except:
        return None, False, 0.0
 
# ─────────────────────────────────────────
# RULE-BASED ANALYSIS
# The key fixes are here:
# 1. DNS and HTTPS are whitelisted before any rules run
# 2. Port scan rule is much stricter
# 3. Inbound rule only fires on truly unsolicited traffic
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
 
    # ── Whitelist FIRST — before any other rule ────────────────
    # Check both source and destination IP and port
    if (is_whitelisted(src_ip, src_port) or
        is_whitelisted(dst_ip, dst_port)):
        return ("NORMAL","Safe Traffic",
                f"Whitelisted traffic {src_ip} → {dst_ip} "
                f"({protocol} port {dst_port}).")
 
    src_priv = is_private(src_ip)
    dst_priv = is_private(dst_ip)
    pps      = packets / max(duration, 0.001)
 
    # ── ICMP ──────────────────────────────────────────────────
    if protocol == "ICMP":
        if packets > 500 and duration < 10:
            return ("ALERT","ICMP Flood",
                    f"High ICMP rate — {packets} pkts in "
                    f"{duration:.1f}s. Possible DoS attack.")
        if packets > 100 and avg_size < 30:
            return ("WARNING","ICMP Sweep",
                    f"{src_ip} sent {packets} tiny ICMP packets.")
        return ("NORMAL","ICMP Ping",
                f"{src_ip} → {dst_ip}.")
 
    # ── DNS (non-standard ports only — port 53 is whitelisted) ─
    if dst_port == 53 or src_port == 53:
        return ("NORMAL","DNS Query",
                f"{src_ip} queried DNS at {dst_ip}.")
 
    # ── Known attack ports ────────────────────────────────────
    if dst_port in SUSPICIOUS_PORTS:
        return ("ALERT","Suspicious Port",
                f"Connection to known attack port "
                f"{dst_port} ({port_label(dst_port)}) "
                f"at {dst_ip} from {src_ip}.")
 
    # ── DoS pattern ───────────────────────────────────────────
    if pps > 2000:
        return ("ALERT","Possible DoS Attack",
                f"Extremely high packet rate {pps:.0f} pps "
                f"from {src_ip} to {dst_ip}.")
 
    # ── Port scan — very strict ────────────────────────────────
    # Must be TCP, 1 packet only, tiny size, AND to a non-standard port
    # This avoids flagging normal connection attempts
    if (protocol == "TCP" and
        packets == 1 and
        total_b < 80 and
        dst_port not in SAFE_PORTS and
        dst_port > 1024 and      # ignore standard service ports
        not dst_priv):           # only flag external targets
        return ("WARNING","Port Probe",
                f"{src_ip} sent single packet to "
                f"external {dst_ip}:{dst_port}.")
 
    # ── External DB access ────────────────────────────────────
    if dst_port in DATABASE_PORTS and not src_priv:
        return ("ALERT","External Database Access",
                f"External {src_ip} accessing "
                f"database port {dst_port} on {dst_ip}.")
 
    # ── External remote admin ─────────────────────────────────
    if dst_port in REMOTE_ADMIN_PORTS and not src_priv:
        return ("WARNING","External Admin Access",
                f"External {src_ip} connecting to "
                f"admin port {dst_port} ({port_label(dst_port)}) "
                f"on {dst_ip}.")
 
    # ── Very large outbound transfer ──────────────────────────
    # Only flag if more than 100MB — not regular web traffic
    if total_b > 100_000_000 and src_priv and not dst_priv:
        return ("WARNING","Large Data Transfer",
                f"{src_ip} sent {total_b/1_000_000:.1f} MB "
                f"to external {dst_ip}.")
 
    # ── Internal traffic ──────────────────────────────────────
    if src_priv and dst_priv:
        return ("NORMAL","Internal Traffic",
                f"{src_ip} → {dst_ip}:{dst_port} ({protocol}).")
 
    # ── Outbound ──────────────────────────────────────────────
    if src_priv and not dst_priv:
        return ("NORMAL","Outbound Connection",
                f"{src_ip} → {dst_ip}:{dst_port} ({protocol}).")
 
    # ── Inbound ───────────────────────────────────────────────
    # Only flag if single packet AND to a non-whitelisted port
    # Multi-packet inbound = established connection = normal
    if not src_priv and dst_priv:
        if packets == 1 and dst_port not in SAFE_PORTS:
            return ("WARNING","Single Packet Inbound",
                    f"External {src_ip} sent one packet "
                    f"to {dst_ip}:{dst_port}.")
        return ("NORMAL","Inbound Connection",
                f"External {src_ip} → {dst_ip}:{dst_port} "
                f"({packets} pkts).")
 
    return ("NORMAL","General Traffic",
            f"{src_ip}:{src_port} → {dst_ip}:{dst_port} "
            f"({protocol}) {packets} pkts.")
 
# ─────────────────────────────────────────
# SAVE ALERT
# ─────────────────────────────────────────
def save_alert(flow):
    if_conf = flow.get("IF Confidence", 0.0) or 0.0
    ae_conf = flow.get("AE Confidence", 0.0) or 0.0
 
    if flow.get("IF Anomaly") and flow.get("AE Anomaly"):
        confidence = round((if_conf + ae_conf) / 2, 1)
    elif flow.get("IF Anomaly"):
        confidence = round(if_conf, 1)
    elif flow.get("AE Anomaly"):
        confidence = round(ae_conf, 1)
    else:
        confidence = 75.0
 
    alert = {
        "component":   "NetworkGuardian",
        "timestamp":   flow["Timestamp"],
        "status":      flow["Status"],
        "threat_type": flow["Label"].lower().replace(" ","_"),
        "confidence":  confidence / 100.0,
        "if_score":    flow.get("IF Score"),
        "if_anomaly":  flow.get("IF Anomaly", False),
        "ae_error":    flow.get("AE Error"),
        "ae_anomaly":  flow.get("AE Anomaly", False),
        "ml_flagged":  flow.get("ML Flagged", False),
        "details": {
            "source_ip":      flow["Source IP"],
            "destination_ip": flow["Destination IP"],
            "source_port":    flow["Source Port"],
            "dest_port":      flow["Destination Port"],
            "protocol":       flow["Protocol"],
            "bytes":          flow["Total Bytes"],
            "packets":        flow["Packet Count"],
            "duration":       flow["Duration (seconds)"],
            "avg_pkt_size":   flow["Average Packet Size"],
        },
        "message": flow["Message"],
        "action":  "investigate" if flow["Status"] == "ALERT"
                   else "monitor",
    }
    with open(ALERT_FILE, "a") as f:
        f.write(json.dumps(alert) + "\n")
 
def save_flow_log(flow):
    with open(FLOW_LOG, "a") as f:
        f.write(json.dumps(flow) + "\n")
 
# ─────────────────────────────────────────
# TERMINAL DISPLAY
# ─────────────────────────────────────────
COLORS = {
    "NORMAL":"\033[92m",
    "WARNING":"\033[93m",
    "ALERT":"\033[91m",
}
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
        f"{flow['Destination IP']:15s}:"
        f"{str(flow['Destination Port']):<6}\n"
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
        "Timestamp":           datetime.utcnow().isoformat(),
        "Protocol":            raw["Protocol"],
        "Source IP":           raw["Source IP"],
        "Destination IP":      raw["Destination IP"],
        "Source Port":         raw["Source Port"],
        "Destination Port":    raw["Destination Port"],
        "Packet Count":        raw["Packet Count"],
        "Total Bytes":         raw["Total Bytes"],
        "Average Packet Size": round(avg_ps, 2),
        "Duration (seconds)":  round(duration, 3),
    }
 
    # Step 1: Rules
    status, label, message = analyze_flow(flow)
 
    # Step 2: IF scoring
    if_score, if_anom, if_conf = score_if(flow)
    flow["IF Score"]      = if_score
    flow["IF Anomaly"]    = if_anom
    flow["IF Confidence"] = if_conf
 
    # Step 3: AE scoring
    ae_error, ae_anom, ae_conf = score_ae(flow)
    flow["AE Error"]      = ae_error
    flow["AE Anomaly"]    = ae_anom
    flow["AE Confidence"] = ae_conf
 
    # Step 4: STRICT ML — only escalate when BOTH agree
    both_flagged       = if_anom and ae_anom
    flow["ML Flagged"] = both_flagged
 
    if both_flagged:
        avg_conf = round((if_conf + ae_conf) / 2, 1)
        if status == "NORMAL":
            status  = "WARNING"
            label   = "Dual Model Anomaly"
            message = (
                f"Both ML models flagged this flow. "
                f"IF={if_score:.3f} AE={ae_error:.4f} "
                f"conf={avg_conf:.0f}%."
            )
        elif status == "WARNING":
            status  = "ALERT"
            label   = f"ML + Rule: {label}"
            message = (
                f"Both ML models AND rule engine flagged this. "
                f"IF={if_score:.3f} AE={ae_error:.4f} "
                f"conf={avg_conf:.0f}%. {message}"
            )
 
    flow["Status"]  = status
    flow["Label"]   = label
    flow["Message"] = message
 
    stats["total"] += 1
    stats[status.lower()] = stats.get(status.lower(), 0) + 1
    if both_flagged:
        stats["ml_flags"] += 1
 
    save_flow_log(flow)
    if status in ("WARNING", "ALERT"):
        save_alert(flow)
 
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
        print(f"  Total : {total} | Alerts : {alerts} | "
              f"Warnings : {warns} | Rate : {rate:.1f}%")
        print(f"\n  Goodbye!\n")

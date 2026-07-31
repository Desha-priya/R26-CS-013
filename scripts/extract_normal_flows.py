# extract_normal_flows.py
# Extracts clean normal flows from flow_log.json
# Run with: python3 extract_normal_flows.py
 
import json
import pandas as pd
import numpy as np
import os
 
print("="*55)
print(" NeuraShield — Extract Normal Traffic Baseline")
print("="*55)
 
if not os.path.exists("flow_log.json"):
    print("ERROR: flow_log.json not found.")
    print("Run monitor.py for 15+ minutes first then try again.")
    exit(1)
 
print("\n[1/3] Reading flow log...")
flows = []
with open("flow_log.json") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                flows.append(json.loads(line))
            except:
                pass
 
print(f"      Total flows read : {len(flows):,}")
 
# Keep only flows the rule engine marked NORMAL
# AND that have reasonable values
normal = []
for f in flows:
    if f.get("Status") != "NORMAL":
        continue
    pkts = f.get("Packet Count", 0)
    byt  = f.get("Total Bytes", 0)
    dur  = f.get("Duration (seconds)", 0)
    # Filter out zero or garbage flows
    if pkts < 1 or byt < 1 or dur < 0:
        continue
    normal.append(f)
 
print(f"      Clean normal flows : {len(normal):,}")
 
if len(normal) < 100:
    print(f"\n⚠️  Only {len(normal)} normal flows found.")
    print("   Run monitor.py for longer (15+ mins) and try again.")
    exit(1)
 
print("\n[2/3] Building feature vectors...")
 
FEATURE_COLS = [
    "bidirectional_duration_ms",
    "bidirectional_packets",
    "bidirectional_bytes",
    "src2dst_packets",
    "src2dst_bytes",
    "dst2src_packets",
    "dst2src_bytes",
    "bidirectional_min_ps",
    "bidirectional_mean_ps",
    "bidirectional_stddev_ps",
    "bidirectional_max_ps",
    "bidirectional_mean_piat_ms",
    "bidirectional_stddev_piat_ms",
    "bidirectional_min_piat_ms",
    "bidirectional_max_piat_ms",
    "bidirectional_syn_packets",
    "bidirectional_ack_packets",
    "bidirectional_rst_packets",
    "bidirectional_fin_packets",
    "bidirectional_psh_packets",
]
 
rows = []
for f in normal:
    dur_ms  = f.get("Duration (seconds)", 0) * 1000
    packets = f.get("Packet Count", 1)
    total_b = f.get("Total Bytes", 0)
    avg_ps  = f.get("Average Packet Size", 0)
    proto   = f.get("Protocol", "")
 
    rows.append({
        "bidirectional_duration_ms":    dur_ms,
        "bidirectional_packets":        packets,
        "bidirectional_bytes":          total_b,
        "src2dst_packets":              packets * 0.6,
        "src2dst_bytes":                total_b * 0.6,
        "dst2src_packets":              packets * 0.4,
        "dst2src_bytes":                total_b * 0.4,
        "bidirectional_min_ps":         avg_ps * 0.5,
        "bidirectional_mean_ps":        avg_ps,
        "bidirectional_stddev_ps":      avg_ps * 0.2,
        "bidirectional_max_ps":         avg_ps * 1.5,
        "bidirectional_mean_piat_ms":   dur_ms / max(packets, 1),
        "bidirectional_stddev_piat_ms": dur_ms / max(packets, 1) * 0.3,
        "bidirectional_min_piat_ms":    0.1,
        "bidirectional_max_piat_ms":    dur_ms,
        "bidirectional_syn_packets":    1 if proto == "TCP" else 0,
        "bidirectional_ack_packets":    max(packets - 1, 0),
        "bidirectional_rst_packets":    0,
        "bidirectional_fin_packets":    1 if proto == "TCP" else 0,
        "bidirectional_psh_packets":    max(packets // 2, 0),
    })
 
df = pd.DataFrame(rows, columns=FEATURE_COLS)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
 
print("\n[3/3] Saving...")
df.to_csv("my_normal_traffic.csv", index=False)
 
print(f"\n{'='*55}")
print(f"✅ Saved {len(df):,} normal flows to my_normal_traffic.csv")
print(f"   Protocol breakdown:")
proto_counts = {}
for f in normal:
    p = f.get("Protocol","?")
    proto_counts[p] = proto_counts.get(p,0) + 1
for p,c in sorted(proto_counts.items(), key=lambda x:-x[1]):
    print(f"   {p:<8} : {c:,}")
print(f"{'='*55}")
print("Next: run retrain_model.py")
 

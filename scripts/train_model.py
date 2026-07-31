# train_model.py
# Network Guardian — ML Model Training
# Uses nfstream CSV output from Wednesday-workingHours.pcap
# Run with: python3 train_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("="*60)
print(" Network Guardian — Model Training")
print("="*60)

# ── Step 1: Load the CSV ───────────────────────────────────────
CSV_PATH = "cicflow_output/Wednesday-workingHours.pcap_ISCX.csv"

if not os.path.exists(CSV_PATH):
    print(f"ERROR: Cannot find {CSV_PATH}")
    exit(1)

print(f"\n[1/6] Loading dataset...")
df = pd.read_csv(CSV_PATH, low_memory=False)
df.columns = df.columns.str.strip()
print(f"      Rows loaded  : {len(df):,}")
print(f"      Columns      : {len(df.columns)}")

# ── Step 2: Select features ────────────────────────────────────
# These are confirmed column names from YOUR nfstream CSV output
# Chosen because they match what monitor.py extracts from live traffic
print("\n[2/6] Selecting features...")

FEATURE_COLS = [
    # Flow duration and volume
    "bidirectional_duration_ms",      # how long the flow lasted
    "bidirectional_packets",          # total packets both directions
    "bidirectional_bytes",            # total bytes both directions

    # Sent vs received (ratio reveals exfiltration, C2, etc.)
    "src2dst_packets",                # packets sent by source
    "src2dst_bytes",                  # bytes sent by source
    "dst2src_packets",                # packets received by source
    "dst2src_bytes",                  # bytes received by source

    # Packet size statistics
    "bidirectional_min_ps",           # smallest packet size
    "bidirectional_mean_ps",          # average packet size
    "bidirectional_stddev_ps",        # variation in packet size
    "bidirectional_max_ps",           # largest packet size

    # Timing between packets (reveals DoS, scanning patterns)
    "bidirectional_mean_piat_ms",     # avg time between packets
    "bidirectional_stddev_piat_ms",   # variation in timing
    "bidirectional_min_piat_ms",      # shortest gap between packets
    "bidirectional_max_piat_ms",      # longest gap between packets

    # TCP flag counts (reveal scanning, SYN floods, etc.)
    "bidirectional_syn_packets",      # SYN flags (connection attempts)
    "bidirectional_ack_packets",      # ACK flags
    "bidirectional_rst_packets",      # RST flags (rejected connections)
    "bidirectional_fin_packets",      # FIN flags (closed connections)
    "bidirectional_psh_packets",      # PSH flags (data being pushed)
]

# Verify all columns exist (safety check)
missing = [f for f in FEATURE_COLS if f not in df.columns]
if missing:
    print(f"      WARNING: Missing columns: {missing}")
    FEATURE_COLS = [f for f in FEATURE_COLS if f in df.columns]

print(f"      Using {len(FEATURE_COLS)} features")
for f in FEATURE_COLS:
    print(f"        ✓ {f}")

# ── Step 3: Clean the data ─────────────────────────────────────
print("\n[3/6] Cleaning data...")

df_clean = df[FEATURE_COLS].copy()
rows_before = len(df_clean)

# Remove infinite values
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows with any missing values
df_clean.dropna(inplace=True)

# Remove flows with negative duration (corrupted records)
df_clean = df_clean[df_clean["bidirectional_duration_ms"] >= 0]

# Remove flows with zero packets (empty flows)
df_clean = df_clean[df_clean["bidirectional_packets"] > 0]

rows_after = len(df_clean)
print(f"      Rows before cleaning : {rows_before:,}")
print(f"      Rows after cleaning  : {rows_after:,}")
print(f"      Rows removed         : {rows_before - rows_after:,}")

# Show a quick summary of the cleaned data
print(f"\n      Feature summary (cleaned data):")
print(f"      {'Feature':<35} {'Min':>12} {'Mean':>12} {'Max':>12}")
print(f"      {'-'*75}")
for col in FEATURE_COLS[:5]:  # show first 5 as sample
    print(f"      {col:<35} "
          f"{df_clean[col].min():>12.2f} "
          f"{df_clean[col].mean():>12.2f} "
          f"{df_clean[col].max():>12.2f}")
print(f"      ... and {len(FEATURE_COLS)-5} more features")

# ── Step 4: Scale the features ─────────────────────────────────
# StandardScaler normalizes each feature to mean=0, stddev=1
# This is critical — without it, large byte counts would
# dominate over packet counts and break the model
print("\n[4/6] Scaling features...")

X = df_clean[FEATURE_COLS].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"      Scaled {X_scaled.shape[0]:,} flows × {X_scaled.shape[1]} features")

# ── Step 5: Train Isolation Forest ────────────────────────────
# Isolation Forest works by randomly partitioning data
# Normal points need many cuts to isolate (they cluster together)
# Anomalous points need very few cuts (they are far from the cluster)
print("\n[5/6] Training Isolation Forest...")
print(f"      This may take 1-3 minutes for {len(X_scaled):,} flows...")

model = IsolationForest(
    n_estimators=300,     # 300 isolation trees
    contamination=0.05,   # expect ~5% anomalies in the dataset
    max_samples="auto",   # auto-select sample size per tree
    random_state=42,      # fixed seed for reproducibility
    n_jobs=-1             # use all CPU cores to speed up training
)
model.fit(X_scaled)
print("      ✓ Training complete!")

# ── Step 6: Evaluate, score and save ──────────────────────────
print("\n[6/6] Scoring all flows...")

scores = model.decision_function(X_scaled)
preds  = model.predict(X_scaled)

normal_count  = (preds ==  1).sum()
anomaly_count = (preds == -1).sum()
anomaly_rate  = anomaly_count / len(preds) * 100

print(f"\n      Scoring results:")
print(f"      Total flows scored      : {len(preds):,}")
print(f"      Flagged as NORMAL       : {normal_count:,}")
print(f"      Flagged as ANOMALY      : {anomaly_count:,}")
print(f"      Anomaly rate            : {anomaly_rate:.1f}%")
print(f"\n      Anomaly score range:")
print(f"      Most anomalous score    : {scores.min():.4f}")
print(f"      Most normal score       : {scores.max():.4f}")
print(f"      Average score           : {scores.mean():.4f}")
print(f"\n      NOTE: Scores below 0.0 = anomalous")
print(f"            Scores above 0.0 = normal")
print(f"            More negative = more suspicious")

# ── Save everything ────────────────────────────────────────────
print("\n      Saving model files...")

with open("isolation_forest.pkl", "wb") as f:
    pickle.dump(model, f)
print("      ✓ isolation_forest.pkl saved")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("      ✓ scaler.pkl saved")

with open("feature_cols.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)
print("      ✓ feature_cols.pkl saved")

print("\n" + "="*60)
print("✅ Phase 4 complete!")
print()
print("   Files created:")
print("   isolation_forest.pkl  — the trained model")
print("   scaler.pkl            — the feature scaler")
print("   feature_cols.pkl      — the 20 feature names")
print()
print("   Next: Phase 5 — integrate model into monitor.py")
print("="*60)

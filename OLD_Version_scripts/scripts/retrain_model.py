# retrain_model.py
# Retrains Isolation Forest using ONLY your personal traffic
# This gives the model a proper baseline for your VM environment
# Run with: python3 retrain_model.py
 
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os
 
print("="*55)
print(" NeuraShield — Isolation Forest Retraining")
print("="*55)
 
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
 
# ── Load your personal normal traffic ─────────────────────────
print("\n[1/5] Loading your normal traffic...")
if not os.path.exists("my_normal_traffic.csv"):
    print("ERROR: my_normal_traffic.csv not found.")
    print("Run extract_normal_flows.py first.")
    exit(1)
 
df = pd.read_csv("my_normal_traffic.csv")
df = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
 
# Add any missing columns as zeros
for c in FEATURE_COLS:
    if c not in df.columns:
        df[c] = 0.0
df = df[FEATURE_COLS].astype(float)
 
print(f"      Normal flows loaded : {len(df):,}")
 
if len(df) < 50:
    print("ERROR: Not enough data. Run monitor.py longer and re-extract.")
    exit(1)
 
# ── Scale ──────────────────────────────────────────────────────
print("\n[2/5] Scaling features...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(f"      Scaled {X_scaled.shape[0]:,} x {X_scaled.shape[1]} features")
 
# ── Train ──────────────────────────────────────────────────────
# contamination=0.01 means we expect only 1% of
# YOUR normal traffic to be odd — very strict
print("\n[3/5] Training Isolation Forest on YOUR traffic only...")
model = IsolationForest(
    n_estimators=400,
    contamination=0.01,
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled)
print("      Training complete!")
 
# ── Find the right threshold ───────────────────────────────────
print("\n[4/5] Calculating optimal threshold...")
scores = model.decision_function(X_scaled)
 
print(f"      Score distribution on YOUR normal traffic:")
print(f"      Min    : {scores.min():.4f}")
print(f"      1st %  : {np.percentile(scores, 1):.4f}")
print(f"      5th %  : {np.percentile(scores, 5):.4f}")
print(f"      Mean   : {scores.mean():.4f}")
print(f"      Median : {np.median(scores):.4f}")
 
# Use 1st percentile so only the absolute bottom
# 1% of YOUR normal traffic gets flagged
# Everything above this is considered normal
threshold = float(np.percentile(scores, 1))
print(f"\n      ✅ Recommended threshold : {threshold:.4f}")
print(f"         (1st percentile of your normal traffic)")
print(f"         This means only the most extreme outliers")
print(f"         will be flagged as anomalies.")
 
# ── Save ───────────────────────────────────────────────────────
print("\n[5/5] Saving model files...")
with open("isolation_forest.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("feature_cols.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)
 
# Save threshold to a config file so monitor.py can read it
config = {
    "if_threshold": threshold,
    "trained_on":   len(df),
    "features":     FEATURE_COLS,
}
with open("model_config.json", "w") as f:
    import json
    json.dump(config, f, indent=2)
 
print(f"\n{'='*55}")
print(f"✅ Isolation Forest retrained!")
print(f"")
print(f"   NOW UPDATE monitor.py:")
print(f"   Find:    IF_THRESHOLD = ...")
print(f"   Change to: IF_THRESHOLD = {threshold:.4f}")
print(f"{'='*55}")
 

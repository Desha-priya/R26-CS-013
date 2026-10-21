# retrain_autoencoder.py
# Retrains autoencoder using ONLY your personal normal traffic
# Run with: python3 retrain_autoencoder.py
 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import os
 
print("="*55)
print(" NeuraShield — Autoencoder Retraining")
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
 
EPOCHS     = 100
BATCH_SIZE = 64
LR         = 0.0005
 
# ── Load your normal traffic ───────────────────────────────────
print("\n[1/6] Loading your normal traffic...")
if not os.path.exists("my_normal_traffic.csv"):
    print("ERROR: Run extract_normal_flows.py first.")
    exit(1)
 
df = pd.read_csv("my_normal_traffic.csv")
df = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
for c in FEATURE_COLS:
    if c not in df.columns:
        df[c] = 0.0
df = df[FEATURE_COLS].astype(float)
print(f"      Normal flows : {len(df):,}")
 
if len(df) < 50:
    print("ERROR: Not enough data. Run monitor.py longer.")
    exit(1)
 
# ── Scale ──────────────────────────────────────────────────────
print("\n[2/6] Scaling features...")
ae_scaler = StandardScaler()
X_scaled  = ae_scaler.fit_transform(df)
X_tensor  = torch.FloatTensor(X_scaled)
loader    = DataLoader(
    TensorDataset(X_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)
print(f"      Scaled {X_scaled.shape[0]:,} x {X_scaled.shape[1]} features")
 
# ── Build autoencoder ──────────────────────────────────────────
print("\n[3/6] Building autoencoder...")
INPUT_DIM = len(FEATURE_COLS)
 
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
 
model     = NetworkAutoencoder(INPUT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
 
total_params = sum(p.numel() for p in model.parameters())
print(f"      Architecture : {INPUT_DIM}→16→8→4→8→16→{INPUT_DIM}")
print(f"      Parameters   : {total_params:,}")
 
# ── Train ──────────────────────────────────────────────────────
print(f"\n[4/6] Training for {EPOCHS} epochs...")
print(f"      {'Epoch':<8} {'Loss':<14} {'Status'}")
print(f"      {'-'*36}")
 
best_loss = float("inf")
model.train()
 
for epoch in range(1, EPOCHS + 1):
    total_loss  = 0.0
    batch_count = 0
    for batch in loader:
        x    = batch[0]
        out  = model(x)
        loss = criterion(out, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss  += loss.item()
        batch_count += 1
 
    avg = total_loss / batch_count
 
    improved = ""
    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), "autoencoder_best.pth")
        improved = "← best"
 
    if epoch % 10 == 0 or epoch == 1:
        print(f"      Epoch {epoch:<4} Loss: {avg:<12.6f} {improved}")
 
# Load best checkpoint
model.load_state_dict(torch.load("autoencoder_best.pth"))
model.eval()
print(f"\n      Best loss achieved : {best_loss:.6f}")
 
# ── Calculate threshold from YOUR traffic only ─────────────────
print(f"\n[5/6] Calculating anomaly threshold...")
print(f"      Scoring all your normal flows...")
 
errors = []
with torch.no_grad():
    for i in range(0, len(X_tensor), 256):
        batch = X_tensor[i:i+256]
        recon = model(batch)
        err   = torch.mean((batch - recon) ** 2, dim=1)
        errors.extend(err.numpy().tolist())
 
errors = np.array(errors)
 
# Use 99th percentile — only top 1% of YOUR normal traffic
# reconstruction errors will be considered anomalous
threshold = float(np.percentile(errors, 99))
 
print(f"\n      Reconstruction error on YOUR normal traffic:")
print(f"      Min     : {errors.min():.6f}")
print(f"      Mean    : {errors.mean():.6f}")
print(f"      95th %  : {np.percentile(errors,95):.6f}")
print(f"      99th %  : {threshold:.6f}  ← new threshold")
print(f"      Max     : {errors.max():.6f}")
 
# How many of YOUR flows would be flagged with this threshold
flagged = (errors > threshold).sum()
print(f"\n      Flows flagged as anomaly : {flagged} "
      f"({flagged/len(errors)*100:.1f}% of your normal traffic)")
print(f"      This should be around 1% — if much higher,")
print(f"      collect more normal traffic and retrain.")
 
# ── Save ───────────────────────────────────────────────────────
print(f"\n[6/6] Saving model files...")
torch.save(model.state_dict(), "autoencoder.pth")
 
ae_config = {
    "input_dim":    INPUT_DIM,
    "threshold":    threshold,
    "feature_cols": FEATURE_COLS,
    "epochs":       EPOCHS,
    "best_loss":    best_loss,
}
with open("autoencoder_config.pkl", "wb") as f:
    pickle.dump(ae_config, f)
with open("ae_scaler.pkl", "wb") as f:
    pickle.dump(ae_scaler, f)
 
print(f"\n{'='*55}")
print(f"✅ Autoencoder retrained successfully!")
print(f"")
print(f"   New threshold : {threshold:.6f}")
print(f"   Best loss     : {best_loss:.6f}")
print(f"   Trained on    : {len(df):,} of YOUR normal flows")
print(f"")
print(f"   monitor.py will load this automatically.")
print(f"   Restart monitor.py to apply the new model.")
print(f"{'='*55}")
 

# train_autoencoder.py
# Network Guardian — Autoencoder Anomaly Detection Model
# Second ML model — works alongside Isolation Forest
# Run with: python3 train_autoencoder.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("="*60)
print(" Network Guardian — Autoencoder Training")
print("="*60)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CSV_PATH      = "cicflow_output/Wednesday-workingHours.pcap_ISCX.csv"
EPOCHS        = 50          # number of training passes through the data
BATCH_SIZE    = 256         # flows processed together in each step
LEARNING_RATE = 0.001       # how fast the model learns
THRESHOLD_PCT = 95          # flag flows above this reconstruction error percentile

# ─────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────
print(f"\n[1/7] Loading dataset...")

if not os.path.exists(CSV_PATH):
    print(f"ERROR: Cannot find {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH, low_memory=False)
df.columns = df.columns.str.strip()
print(f"      Rows loaded  : {len(df):,}")
print(f"      Columns      : {len(df.columns)}")

# ─────────────────────────────────────────
# STEP 2 — SELECT FEATURES
# Same 20 features used by Isolation Forest
# so both models are comparable
# ─────────────────────────────────────────
print(f"\n[2/7] Selecting features...")

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

# Only keep columns that exist in the CSV
FEATURE_COLS = [f for f in FEATURE_COLS if f in df.columns]
print(f"      Using {len(FEATURE_COLS)} features")

# ─────────────────────────────────────────
# STEP 3 — CLEAN DATA
# ─────────────────────────────────────────
print(f"\n[3/7] Cleaning data...")

df_clean = df[FEATURE_COLS].copy()
rows_before = len(df_clean)

df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean.dropna(inplace=True)
df_clean = df_clean[df_clean["bidirectional_duration_ms"] >= 0]
df_clean = df_clean[df_clean["bidirectional_packets"] > 0]

rows_after = len(df_clean)
print(f"      Rows before : {rows_before:,}")
print(f"      Rows after  : {rows_after:,}")
print(f"      Removed     : {rows_before - rows_after:,}")

# ─────────────────────────────────────────
# STEP 4 — SCALE FEATURES
# ─────────────────────────────────────────
print(f"\n[4/7] Scaling features...")

X = df_clean[FEATURE_COLS].astype(float).values

# Use a NEW scaler specifically for the autoencoder
ae_scaler = StandardScaler()
X_scaled  = ae_scaler.fit_transform(X)

print(f"      Scaled {X_scaled.shape[0]:,} flows x {X_scaled.shape[1]} features")

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_scaled)
dataset  = TensorDataset(X_tensor)
loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"      Batches per epoch : {len(loader)}")

# ─────────────────────────────────────────
# STEP 5 — BUILD THE AUTOENCODER
# Architecture:
#   Input(20) → Encode → 10 → 5 → Decode → 10 → Output(20)
#
# The bottleneck (5 neurons) forces the model to learn
# a compressed representation of normal traffic.
# Attack traffic cannot be compressed and reconstructed
# accurately → high reconstruction error → flagged
# ─────────────────────────────────────────
print(f"\n[5/7] Building autoencoder architecture...")

INPUT_DIM = len(FEATURE_COLS)   # 20

class NetworkAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(NetworkAutoencoder, self).__init__()

        # Encoder — compresses input down to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),   # 20 → 16
            nn.ReLU(),
            nn.Linear(16, 8),           # 16 → 8
            nn.ReLU(),
            nn.Linear(8, 4),            # 8  → 4  (bottleneck)
            nn.ReLU(),
        )

        # Decoder — reconstructs from bottleneck back to original
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),            # 4  → 8
            nn.ReLU(),
            nn.Linear(8, 16),           # 8  → 16
            nn.ReLU(),
            nn.Linear(16, input_dim),   # 16 → 20
        )

    def forward(self, x):
        encoded   = self.encoder(x)
        decoded   = self.decoder(encoded)
        return decoded

    def reconstruction_error(self, x):
        """Calculate per-sample mean squared reconstruction error"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            errors = torch.mean((x - reconstructed) ** 2, dim=1)
        return errors

model     = NetworkAutoencoder(INPUT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"      Input dimensions  : {INPUT_DIM}")
print(f"      Bottleneck size   : 4 neurons")
print(f"      Total parameters  : {total_params}")
print(f"      Architecture      : {INPUT_DIM}→16→8→4→8→16→{INPUT_DIM}")

# ─────────────────────────────────────────
# STEP 6 — TRAIN THE AUTOENCODER
# ─────────────────────────────────────────
print(f"\n[6/7] Training autoencoder for {EPOCHS} epochs...")
print(f"      Batch size   : {BATCH_SIZE}")
print(f"      Learning rate: {LEARNING_RATE}")
print(f"      {'Epoch':<8} {'Loss':<12} {'Status'}")
print(f"      {'-'*35}")

model.train()
best_loss    = float("inf")
loss_history = []

for epoch in range(1, EPOCHS + 1):
    epoch_loss   = 0.0
    batch_count  = 0

    for batch in loader:
        inputs = batch[0]

        # Forward pass
        outputs = model(inputs)
        loss    = criterion(outputs, inputs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss  += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count
    loss_history.append(avg_loss)

    # Track best model
    improved = ""
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "autoencoder_best.pth")
        improved = "← best"

    # Print every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        print(f"      Epoch {epoch:<4} Loss: {avg_loss:.6f}   {improved}")

print(f"\n      Training complete!")
print(f"      Best loss achieved : {best_loss:.6f}")

# Load best weights
model.load_state_dict(torch.load("autoencoder_best.pth"))
model.eval()

# ─────────────────────────────────────────
# STEP 7 — CALCULATE THRESHOLD AND SAVE
# The threshold is the reconstruction error
# value above which we consider a flow anomalous.
# We set it at the 95th percentile of training errors
# meaning only the top 5% most unusual flows
# are flagged as anomalies.
# ─────────────────────────────────────────
print(f"\n[7/7] Calculating anomaly threshold and saving...")

# Calculate reconstruction errors on all training data
print(f"      Calculating reconstruction errors on all flows...")
all_errors = []

with torch.no_grad():
    for batch in DataLoader(TensorDataset(X_tensor),
                            batch_size=1024, shuffle=False):
        errors = model.reconstruction_error(batch[0])
        all_errors.extend(errors.numpy().tolist())

all_errors = np.array(all_errors)

# Set threshold at 95th percentile
threshold = float(np.percentile(all_errors, THRESHOLD_PCT))

print(f"\n      Reconstruction error stats:")
print(f"      Min error    : {all_errors.min():.6f}")
print(f"      Mean error   : {all_errors.mean():.6f}")
print(f"      Median error : {np.median(all_errors):.6f}")
print(f"      95th pct     : {threshold:.6f}  ← anomaly threshold")
print(f"      Max error    : {all_errors.max():.6f}")

flagged = (all_errors > threshold).sum()
print(f"\n      Flows above threshold : {flagged:,} ({flagged/len(all_errors)*100:.1f}%)")

# Save everything
torch.save(model.state_dict(), "autoencoder.pth")

ae_config = {
    "input_dim":     INPUT_DIM,
    "threshold":     threshold,
    "feature_cols":  FEATURE_COLS,
    "epochs":        EPOCHS,
    "best_loss":     best_loss,
    "loss_history":  loss_history,
}
with open("autoencoder_config.pkl", "wb") as f:
    pickle.dump(ae_config, f)

with open("ae_scaler.pkl", "wb") as f:
    pickle.dump(ae_scaler, f)

print(f"\n      Files saved:")
print(f"      autoencoder.pth         — trained model weights")
print(f"      autoencoder_best.pth    — best checkpoint")
print(f"      autoencoder_config.pkl  — threshold + config")
print(f"      ae_scaler.pkl           — feature scaler")

print("\n" + "="*60)
print("✅ Phase 6 complete!")
print()
print("   Model summary:")
print(f"   Architecture  : {INPUT_DIM}→16→8→4→8→16→{INPUT_DIM}")
print(f"   Features used : {len(FEATURE_COLS)}")
print(f"   Best loss     : {best_loss:.6f}")
print(f"   Threshold     : {threshold:.6f}")
print()
print("   Next: integrate autoencoder into monitor.py")
print("="*60)

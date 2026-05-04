# train_models.py
# Trains Isolation Forest + One-Class SVM on combined behavioral profiles
# Saves: models/isolation_forest.pkl, models/oneclass_svm.pkl, models/scaler.pkl
#        models/user_profiles.pkl  (per-user feature vectors for live comparison)

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from pathlib import Path

# ** Config ************************************************
DATA_FILE   =  Path(__file__).resolve().parent.parent / "user_behavioral_profiles_combined.csv"
MODELS_DIR  = Path(__file__).resolve().parent.parent.parent.parent / "models" / "zero_trust_auth"
os.makedirs(MODELS_DIR, exist_ok=True)

# Features to use - all 48 columns except 'user'
EXCLUDE_COLS = ['user']

# ** Load data ********************************************─
print("Loading combined behavioral profiles...")
df = pd.read_csv(DATA_FILE)
print(f"Shape: {df.shape}")

feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
X = df[feature_cols].fillna(0).values   
users = df['user'].values

print(f"Features: {len(feature_cols)}")
print(f"Users: {len(users)}")

# ** Step 1: Scale features ********************************
# StandardScaler: transforms each feature to mean=0, std=1
# This is critical for One-Class SVM - it's distance-based so scale matters
# Isolation Forest doesn't strictly need it but it helps consistency
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("Saved: scaler.pkl")

# ** Step 2: Train Isolation Forest ************************
# contamination=0.05 means we expect ~5% of sessions to be anomalous
# n_estimators=200 means 200 trees - more = more accurate, slower to train
# random_state=42 makes results reproducible
print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    max_samples='auto',
    random_state=42,
    n_jobs=-1        # use all CPU cores
)
iso_forest.fit(X_scaled)

# Get anomaly scores for all users (-1=anomaly, +1=normal)
# score_samples gives raw scores - more negative = more anomalous
if_scores = iso_forest.score_samples(X_scaled)
if_predictions = iso_forest.predict(X_scaled)

print(f"  Isolation Forest trained")
print(f"  Users flagged as anomaly in training data: {np.sum(if_predictions == -1)}")
print(f"  Score range: {if_scores.min():.3f} to {if_scores.max():.3f}")
print(f"  Score mean:  {if_scores.mean():.3f} (more negative = more unusual)")

joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
print("Saved: isolation_forest.pkl")

# ** Step 3: Train One-Class SVM **************************─
# nu=0.05 - similar to contamination, expect 5% outliers
# kernel='rbf' - radial basis function, handles non-linear boundaries
# gamma='scale' - auto-calculates best gamma from data
# This model learns the "shape" of normal behaviour in high-dimensional space
print("\nTraining One-Class SVM...")
oc_svm = OneClassSVM(
    kernel='rbf',
    nu=0.05,
    gamma='scale'
)
oc_svm.fit(X_scaled)

svm_scores      = oc_svm.score_samples(X_scaled)
svm_predictions = oc_svm.predict(X_scaled)

print(f"  One-Class SVM trained")
print(f"  Users flagged as anomaly: {np.sum(svm_predictions == -1)}")
print(f"  Score range: {svm_scores.min():.3f} to {svm_scores.max():.3f}")

joblib.dump(oc_svm, os.path.join(MODELS_DIR, "oneclass_svm.pkl"))
print("Saved: oneclass_svm.pkl")

# ** Step 4: Save per-user profiles **********************─
# This is used during live sessions - compare incoming features
# against the specific enrolled user's profile vector
print("\nSaving per-user profiles...")
user_profiles = {}
for i, uid in enumerate(users):
    user_profiles[int(uid)] = {
        'features': X_scaled[i].tolist(),        # scaled feature vector
        'raw_features': X[i].tolist(),            # original unscaled
        'feature_names': feature_cols,
        'if_score': float(if_scores[i]),          # their normal score
        'svm_score': float(svm_scores[i]),
    }

joblib.dump(user_profiles, os.path.join(MODELS_DIR, "user_profiles.pkl"))
print(f"Saved: user_profiles.pkl ({len(user_profiles)} users)")

# ** Step 5: Model summary ******************
print("\n" + "="*55)
print("MODEL TRAINING COMPLETE - SUMMARY")
print("="*55)
print(f"Training data    : {len(users)} users × {len(feature_cols)} features")
print(f"Isolation Forest : {iso_forest.n_estimators} trees, contamination={iso_forest.contamination}")
print(f"One-Class SVM    : kernel=rbf, nu={oc_svm.nu}, gamma=scale")
print(f"\nIF  - normal users (score > threshold): {np.sum(if_predictions == 1)}/{len(users)}")
print(f"SVM - normal users (score > threshold): {np.sum(svm_predictions == 1)}/{len(users)}")
print(f"\nFiles saved to ./{MODELS_DIR}/")
print("  isolation_forest.pkl")
print("  oneclass_svm.pkl")
print("  scaler.pkl")
print("  user_profiles.pkl")
print("\nNext: run risk_engine.py then main.py")
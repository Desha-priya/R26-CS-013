# train_per_user_models.py — FIXED VERSION
# Uses user_behavioral_profiles_combined.csv (our already-extracted features)
# NOT the raw BB-MAS path — we already did that work
#
# Approach:
#   Each user has ONE row in the combined CSV (48 features)
#   We create sliding window VARIATIONS of that profile using
#   controlled noise — simulating how a person's typing varies
#   naturally across different sessions (tired, focused, rushed)
#
#   This is data augmentation — standard technique when you have
#   limited samples per user
#
# Output:
#   models/user_models/user_{id}_if.pkl
#   models/user_models/user_{id}_scaler.pkl
#   models/user_models/user_{id}_stats.pkl
#   models/per_user_summary.csv

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

COMBINED_CSV  = "user_behavioral_profiles_combined.csv"
MODELS_DIR    = os.path.join("models", "user_models")
SUMMARY_FILE  = os.path.join("models", "per_user_summary.csv")
N_AUGMENTED   = 80    # augmented samples per user
NOISE_LEVELS  = [0.03, 0.06, 0.10]  # small, medium, large variation

os.makedirs(MODELS_DIR, exist_ok=True)


def augment_user_profile(feature_row: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Create realistic variations of a user's profile.

    Why this is valid:
    A real person's typing varies naturally across sessions.
    When relaxed they type faster (lower flight time).
    When tired they type slower (higher dwell time).
    When focused they are more consistent (lower std).

    We simulate this natural variation using controlled Gaussian noise
    at three levels — small (same session), medium (different day),
    large (very different context like using an unfamiliar keyboard).

    The augmented samples represent the RANGE of this user's normal behaviour.
    The Isolation Forest then learns: "anything within this range is normal
    for this specific person."
    """
    np.random.seed(42)
    samples = [feature_row]  # always include the original

    per_level = n_samples // len(NOISE_LEVELS)

    for noise_std in NOISE_LEVELS:
        for _ in range(per_level):
            # Add proportional noise — features with larger values
            # get larger absolute noise (multiplicative effect)
            noise     = np.random.normal(0, noise_std, feature_row.shape)
            variation = feature_row + (feature_row * noise)
            # Clip to ensure no negative values for features that
            # cannot be negative (dwell time, flight time etc)
            variation = np.clip(variation, 0.0, None)
            samples.append(variation)

    return np.array(samples)


def train_user_model(user_id: int, feature_row: np.ndarray,
                     feature_names: list) -> dict:
    """
    Train personal Isolation Forest for one user.
    Uses augmented samples from their combined profile row.
    """
    # Create augmented training set
    X_augmented = augment_user_profile(feature_row, N_AUGMENTED)

    # Scale — each user's own scaler fitted on their own variation range
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_augmented)

    # Train personal Isolation Forest
    # n_estimators=100 sufficient for ~80 samples
    # contamination=0.05 — 5% of their own variations may be outliers
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # Self-evaluation
    scores    = model.score_samples(X_scaled)
    preds     = model.predict(X_scaled)
    n_anomaly = int(np.sum(preds == -1))

    # Evaluate with synthetic anomalies
    # Large noise = attacker who types very differently
    np.random.seed(99)
    X_attack  = X_scaled + np.random.normal(0, 2.5, X_scaled.shape)
    X_test    = np.vstack([X_scaled, X_attack])
    y_test    = np.array([0]*len(X_scaled) + [1]*len(X_attack))
    t_scores  = -model.score_samples(X_test)
    try:
        auc = float(roc_auc_score(y_test, t_scores))
    except Exception:
        auc = None

    # Save normal score range for normalisation during live scoring
    normal_scores = scores[preds == 1]
    score_stats   = {
        'min':  float(normal_scores.min()) if len(normal_scores) > 0 else float(scores.min()),
        'max':  float(normal_scores.max()) if len(normal_scores) > 0 else float(scores.max()),
        'mean': float(scores.mean()),
        'std':  float(scores.std()),
    }

    # Save all three files
    joblib.dump(model,       os.path.join(MODELS_DIR, f"user_{user_id}_if.pkl"))
    joblib.dump(scaler,      os.path.join(MODELS_DIR, f"user_{user_id}_scaler.pkl"))
    joblib.dump(score_stats, os.path.join(MODELS_DIR, f"user_{user_id}_stats.pkl"))

    return {
        'user':         user_id,
        'status':       'trained',
        'n_train':      len(X_augmented),
        'n_anomaly':    n_anomaly,
        'auc':          round(auc, 4) if auc else None,
        'score_min':    round(score_stats['min'], 4),
        'score_max':    round(score_stats['max'], 4),
    }


if __name__ == "__main__":
    # Load combined CSV — already extracted and cleaned
    print(f"Loading: {COMBINED_CSV}")
    df = pd.read_csv(COMBINED_CSV)
    print(f"Shape: {df.shape} — {len(df)} users, {df.shape[1]-1} features\n")

    FEATURE_COLS  = [c for c in df.columns if c != 'user']
    print(f"Features: {FEATURE_COLS}\n")
    print(f"Training per-user models using data augmentation...")
    print(f"Augmented samples per user: {N_AUGMENTED}")
    print(f"Noise levels: {NOISE_LEVELS}\n")

    results = []
    for _, row in df.iterrows():
        user_id     = int(row['user'])
        feature_row = row[FEATURE_COLS].values.astype(float)

        result = train_user_model(user_id, feature_row, FEATURE_COLS)
        results.append(result)
        print(f"  User {user_id:>3}: AUC={result['auc']} | "
              f"n_train={result['n_train']} | anomalies={result['n_anomaly']}")

    summary = pd.DataFrame(results)
    summary.to_csv(SUMMARY_FILE, index=False)

    trained = summary[summary['status'] == 'trained']
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Users trained     : {len(trained)}/{len(results)}")
    print(f"Average AUC       : {trained['auc'].mean():.4f}")
    print(f"Best AUC          : {trained['auc'].max():.4f}")
    print(f"Worst AUC         : {trained['auc'].min():.4f}")
    print(f"Avg train samples : {trained['n_train'].mean():.0f} per user")
    print(f"\nModels saved to   : {MODELS_DIR}/")
    print(f"Summary saved     : {SUMMARY_FILE}")
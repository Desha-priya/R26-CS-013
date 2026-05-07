# =============================================================
# NeuraShield — Multimodal Content Threat Detection Layer
# Script  : src/image/train_deepfake_model.py
# Purpose : Train a Random Forest classifier on the extracted
#           video features to detect deepfake videos.
#
# Input   : data/video/processed/video_features.csv
# Output  : models/image/deepfake_video_model_v5.pkl
#           outputs/results/video_confusion_matrix_v5.png
#
# Why Random Forest for video features?
#   Our features are tabular (one row per video, numeric columns).
#   Random Forest handles tabular data very well, is robust to
#   outliers, and gives us feature importance scores — useful
#   when explaining to the supervisor which visual cues matter most.
#
# Run from project root:
#   python src/image/train_deepfake_model.py
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

print("=" * 60)
print("  NeuraShield — Deepfake Video Classifier Training")
print("=" * 60)

os.makedirs("models/image",     exist_ok=True)
os.makedirs("outputs/results",  exist_ok=True)

# ── Load extracted features ──────────────────────────────────
DATA_PATH = "data/video/processed/improved_visual_features_v3.csv"
print(f"\nLoading features from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Total samples : {len(df)}")
print(f"Class balance :")
print(df['label'].value_counts().to_string())

# ── Prepare feature matrix and label vector ──────────────────
# Drop non-numeric / identifier columns — the model only needs numbers
# 'file' is a filename, 'label' is what we're predicting, 'modality' is
# always "video" in this dataset so it carries no information and must
# be dropped — sklearn cannot process string columns
drop_cols = ['file', 'label', 'modality']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df['label']   # "real" or "fake"

print(f"\nFeatures used    : {feature_cols}")
print(f"Feature matrix   : {X.shape}")

# ── Train / Test Split ───────────────────────────────────────
# 25% held out for testing — a bit more than text because we have
# fewer video samples so we want a meaningful test set size
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y       # keep real/fake ratio equal in both splits
)

print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── Train the Random Forest ───────────────────────────────────
# n_estimators=400 → 400 decision trees are built independently
# and their predictions are majority-voted for final classification.
# More trees = lower variance (more stable) but slower to train.
# class_weight='balanced' compensates for the fact that we have
# more fake videos than real ones in FaceForensics++.
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1          # use all available CPU cores
)
model.fit(X_train, y_train)
print("Training complete.")

# ── Evaluate the Model ───────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)   # confidence scores for demo
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("  TRAINING RESULTS")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# ── Feature Importance ───────────────────────────────────────
# Random Forest tells us which features contributed most to its decisions.
# This is very useful to explain to the supervisor WHY the model works.
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)
print("\nFeature importances (which visual cues matter most):")
for feat, imp in importances.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:15s} {imp:.4f}  {bar}")

# ── Confusion Matrix Plot ────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=['fake', 'real'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
ax.set_title('Deepfake Video Classifier — Confusion Matrix')
ax.set_ylabel('Actual Label')
ax.set_xlabel('Predicted Label')
plt.tight_layout()
PLOT_PATH = "outputs/results/video_confusion_matrix.png"
plt.savefig(PLOT_PATH, dpi=150)
plt.close()
print(f"\nConfusion matrix saved to: {PLOT_PATH}")

# ── Save the Trained Model ───────────────────────────────────
MODEL_PATH = "models/image/deepfake_video_model_v5.pkl"
joblib.dump(model, MODEL_PATH)
print(f"Model saved      : {MODEL_PATH}")

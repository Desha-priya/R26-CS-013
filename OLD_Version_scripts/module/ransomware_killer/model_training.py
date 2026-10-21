import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib, os, matplotlib.pyplot as plt, seaborn as sns

FEAT_PATH = r"src\ransomware_killer\data\processed\malmem_features.csv"

def train():
    data = pd.read_csv(FEAT_PATH)
    print(f"[*] Training data shape: {data.shape}")

    X = data.drop(columns=["label"], errors="ignore")
    y = data["label"] if "label" in data.columns else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Isolation Forest (anomaly detection, unsupervised) ---
    print("\n[*] Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.1,   # ~10% expected ransomware
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)
    joblib.dump(iso, r"models\ransomware_killer/isolation_forest.pkl")
    print("[+] Isolation Forest saved.")

    # Evaluate IF (convert: -1 = anomaly = ransomware = 1, 1 = normal = 0)
    iso_preds = iso.predict(X_test)
    iso_labels = np.where(iso_preds == -1, 1, 0)
    print("\nIsolation Forest Report:")
    print(classification_report(y_test, iso_labels, target_names=["Benign","Ransomware"]))

    # --- Random Forest (supervised baseline) ---
    print("\n[*] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, r"models\ransomware_killer/random_forest.pkl")
    print("[+] Random Forest saved.")

    rf_preds = rf.predict(X_test)
    print("\nRandom Forest Report:")
    print(classification_report(y_test, rf_preds, target_names=["Benign","Ransomware"]))

    # Confusion matrix plot (show to supervisor)
    cm = confusion_matrix(y_test, rf_preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Benign","Ransomware"],
                yticklabels=["Benign","Ransomware"])
    plt.title("Random Forest — Confusion Matrix")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(r"models\ransomware_killer/confusion_matrix.png", dpi=150)
    print("[+] Confusion matrix saved to models/confusion_matrix.png")

    # Feature importance plot
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(8,5))
    plt.barh([X.columns[i] for i in top_idx][::-1],
             importances[top_idx][::-1], color="crimson")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()

    plt.savefig(r"models\ransomware_killer/feature_importance.png", dpi=150)
    print("[+] Feature importance plot saved to models/feature_importance.png")

if __name__ == "__main__":
    train()
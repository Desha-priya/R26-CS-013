import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib, os

IN_PATH  = r"src\ransomware_killer\data\processed\malmem_cleaned.csv"
OUT_PATH = r"src\ransomware_killer\data\processed\malmem_features.csv"

# Key behavioral features in CIC-MalMem-2022
# (file I/O patterns, process memory, VAD info)
FEATURE_KEYWORDS = [
    "pslist", "dlllist", "handles", "ldrmodules",
    "malfind", "svcscan", "callbacks", "label"
]

def engineer_features():
    data = pd.read_csv(IN_PATH, low_memory=False)
    print(f"[*] Loaded cleaned data: {data.shape}")

    # Select columns matching behavioral keywords
    selected_cols = []
    for col in data.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in FEATURE_KEYWORDS):
            selected_cols.append(col)

    # Always include label
    if "label" not in selected_cols and "label" in data.columns:
        selected_cols.append("label")

    print(f"[*] Selected {len(selected_cols)} behavioral features")

    df = data[selected_cols].copy()

    # Separate features and label
    X = df.drop(columns=["label"], errors="ignore")
    y = df["label"] if "label" in df.columns else None

    # Keep only numeric
    X = X.select_dtypes(include=[np.number])

    # Fill any remaining NaN
    X.fillna(X.median(), inplace=True)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Save scaler for use in live monitor
   
    joblib.dump(scaler, r"models\ransomware_killer/scaler.pkl")
    joblib.dump(list(X.columns), r"models\ransomware_killer/feature_names.pkl")

    # Combine and save
    if y is not None:
        X_scaled_df["label"] = y.values
    X_scaled_df.to_csv(OUT_PATH, index=False)
    print(f"[+] Saved engineered features to {OUT_PATH}")
    print(f"[+] Feature columns: {list(X.columns)[:8]} ...")

if __name__ == "__main__":
    engineer_features()
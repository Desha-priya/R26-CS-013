import pandas as pd
import numpy as np
import os

RAW_PATH = r"src\ransomware_killer\data\raw"
OUT_PATH = r"src\ransomware_killer\data\processed\malmem_cleaned.csv"

def load_and_clean():
    files = [f for f in os.listdir(RAW_PATH) if f.endswith(".csv")]
    if not files:
        print("[!] No CSV files found in data/raw/. Download CIC-MalMem-2022 first.")
        return

    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(RAW_PATH, f), low_memory=False)
        print(f"[+] Loaded {f}: {df.shape}")
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"\n[*] Combined shape: {data.shape}")

    # Strip whitespace from column names
    data.columns = data.columns.str.strip()

    # Drop rows with all NaN
    data.dropna(how="all", inplace=True)

    # Drop duplicate rows
    data.drop_duplicates(inplace=True)

    # Drop columns with >50% missing
    thresh = len(data) * 0.5
    data.dropna(axis=1, thresh=thresh, inplace=True)

    # Fill remaining NaN with column median (numeric only)
    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())

    # Encode label column (adjust column name if needed)
    label_col = "Class"  # CIC-MalMem-2022 uses 'Class'
    if label_col in data.columns:
        data[label_col] = data[label_col].str.strip()
        data["label"] = (data[label_col] != "Benign").astype(int)
        print(f"\n[*] Label distribution:\n{data['label'].value_counts()}")


    data.to_csv(OUT_PATH, index=False)
    print(f"\n[+] Saved cleaned data to {OUT_PATH}")

if __name__ == "__main__":
    load_and_clean()
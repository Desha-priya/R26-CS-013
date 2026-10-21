# digraph_features.py
# Extracts key-pair (digraph/bigram) transition timing features
# from raw keystroke data.
#
# Why this matters:
# keystroke data is SEQUENTIAL not structured.
# dwell_mean alone loses WHO typed what specific key pairs.
# Digraph features capture: "how long does THIS user take to type TH,
# HE, IN, ER, AN specifically?" — these are highly personal muscle memory
# patterns that are much harder to spoof than average typing speed.
#
# These features are used by:
#   - Per-user model training (train_per_user_models.py)
#   - Live session scoring (compared against enrolled digraph profile)
#
# Output: user_digraph_profiles.csv — per-user digraph timing stats

import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict

# Most common English letter pairs — these are the most discriminative
# because they appear frequently enough to get reliable statistics
TOP_DIGRAPHS = [
    'th', 'he', 'in', 'er', 'an', 're', 'on', 'en', 'at', 'es',
    'ed', 'or', 'ti', 'hi', 'as', 'to', 'ou', 'ea', 'ng', 'al',
    'de', 'se', 'le', 'sa', 'si', 'ar', 'it', 'is', 'ha', 'et',
    'nt', 'ne', 'te', 'no', 'st', 'me', 'ec', 'io', 'li', 'di', 'so'
]

# Feature prefix for each digraph
# Each digraph generates 3 features: mean, std, count
# Total: 40 digraphs × 3 = 120 features + 5 global = 125 features


def extract_digraph_features_from_df(df: pd.DataFrame) -> dict:
    """
    Extract digraph timing features from a raw keystroke dataframe.

    Input df must have columns: key, flight_time
    The key column should be single characters (ignores special keys).

    Returns dict of features — mean/std flight time per key pair.
    """
    # Filter to only regular printable characters
    df = df.copy()
    df['key_clean'] = df['key'].astype(str).str.lower()

    # Keep only single letter/number keys — ignore BACKSPACE, SHIFT etc
    df = df[df['key_clean'].str.len() == 1]
    df = df[df['key_clean'].str.match(r'[a-z0-9 .,]')]
    df['flight_time'] = pd.to_numeric(df['flight_time'], errors='coerce')
    df = df.dropna(subset=['flight_time'])
    df = df[df['flight_time'] >= 0]
    df = df[df['flight_time'] < 5.0]   # cap at 5 seconds

    if len(df) < 10:
        return {}

    # Build list of consecutive key pairs with their flight times
    keys    = df['key_clean'].values
    flights = df['flight_time'].values

    # Group flight times by digraph
    digraph_flights = defaultdict(list)
    for i in range(len(keys) - 1):
        pair = keys[i] + keys[i+1]
        digraph_flights[pair].append(flights[i+1])   # flight to the NEXT key

    features = {}

    # Extract features for top digraphs
    for dg in TOP_DIGRAPHS:
        timings = digraph_flights.get(dg, [])
        col     = 'dg_' + dg   # e.g. dg_th, dg_he

        if len(timings) >= 3:
            arr = np.array(timings)
            features[col + '_mean']  = float(np.mean(arr))
            features[col + '_std']   = float(np.std(arr))
            features[col + '_count'] = int(len(arr))
        else:
            # Not enough samples for this pair — use NaN (filled later)
            features[col + '_mean']  = np.nan
            features[col + '_std']   = np.nan
            features[col + '_count'] = 0

    # Global digraph statistics
    all_timings = [t for ts in digraph_flights.values() for t in ts]
    if all_timings:
        arr = np.array(all_timings)
        features['dg_global_mean']    = float(np.mean(arr))
        features['dg_global_std']     = float(np.std(arr))
        features['dg_global_min']     = float(np.min(arr))
        features['dg_global_max']     = float(np.max(arr))
        features['dg_unique_pairs']   = int(len(digraph_flights))
    else:
        features['dg_global_mean']  = np.nan
        features['dg_global_std']   = np.nan
        features['dg_global_min']   = np.nan
        features['dg_global_max']   = np.nan
        features['dg_unique_pairs'] = 0

    return features


def extract_all_users(base_path: str, output_file: str = "user_digraph_profiles.csv"):
    """
    Extract digraph features for all users in BB-MAS dataset.
    Reads Desktop keyboard files only (most data per user).
    """
    user_folders = sorted(
        [f for f in os.listdir(base_path) if f.isdigit()],
        key=lambda x: int(x)
    )
    print(f"Found {len(user_folders)} users. Extracting digraph features...\n")

    results = []
    for uid in user_folders:
        folder    = os.path.join(base_path, uid)
        ks_files  = glob.glob(os.path.join(folder, "*Desktop*Keyboard*.csv"))

        if not ks_files:
            print(f"  User {uid}: no desktop keyboard file found — skipping")
            continue

        try:
            df       = pd.read_csv(ks_files[0])
            features = extract_digraph_features_from_df(df)
            if features:
                features['user'] = int(uid)
                results.append(features)
                print(f"  User {uid}: {features.get('dg_unique_pairs', 0)} unique digraphs found")
            else:
                print(f"  User {uid}: insufficient data")
        except Exception as e:
            print(f"  User {uid}: ERROR — {e}")

    if not results:
        print("No results — check base_path")
        return None

    df_out = pd.DataFrame(results)

    # Move user column to front
    cols   = ['user'] + [c for c in df_out.columns if c != 'user']
    df_out = df_out[cols]

    # Fill NaN with column median — standard for missing digraph data
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[numeric_cols] = df_out[numeric_cols].fillna(
        df_out[numeric_cols].median()
    )

    df_out.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    print(f"Shape: {df_out.shape}")
    print(f"Features per user: {df_out.shape[1] - 1}")

    # Show most discriminative digraphs
    mean_cols = [c for c in df_out.columns if c.endswith('_mean') and c.startswith('dg_')]
    stds      = df_out[mean_cols].std()
    top5      = stds.nlargest(5)
    print(f"\nMost discriminative digraphs (highest std across users):")
    for col, val in top5.items():
        print(f"  {col:<20}: std = {val:.4f}")

    return df_out


if __name__ == "__main__":
    BASE_PATH = r"BB-MAS_Dataset"
    extract_all_users(BASE_PATH)

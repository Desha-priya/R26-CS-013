import pandas as pd
import numpy as np
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT_DIR.parent / "BB-MAS_Dataset"
root_folder = DEFAULT_DATASET if DEFAULT_DATASET.exists() else Path("BB-MAS_Dataset")
PROCESSED_DIR = ROOT_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "all_keystroke_features(FinalCleaned).csv"

def extract_keystroke_features(df, user_id):

    # ---------- TIME PROCESSING ----------
    df['time_dt'] = pd.to_datetime(df['time'])
    df = df.dropna(subset=['time_dt']).reset_index(drop=True)
    df = df.sort_values('time_dt').reset_index(drop=True)

    # convert to relative seconds
    start_time = df['time_dt'].iloc[0]
    df['time_sec'] = (df['time_dt'] - start_time).dt.total_seconds()

    # ---------- PRESS-RELEASE MATCHING ----------
    active_keys = {}
    records = []

    for _, row in df.iterrows():

        key = row['key']
        direction = row['direction']
        t_sec = row['time_sec']
        t_raw = row['time_dt']

        # (optional) ignore CAPSLOCK if noisy
        # if key == "CAPSLOCK":
        #     continue

        if direction == 0:  # PRESS
            active_keys.setdefault(key, []).append((t_sec, t_raw))

        elif direction == 1:  # RELEASE

            if key in active_keys and len(active_keys[key]) > 0:

                press_sec, press_raw = active_keys[key].pop(0)

                # prevent invalid pairing
                if t_sec <= press_sec:
                    continue

                dwell = t_sec - press_sec

                # remove only extreme corrupted matches
                if dwell > 10:
                    continue

                records.append({
                    "user": user_id,
                    "key": key,
                    "press_time": press_raw,
                    "release_time": t_raw,
                    "press_sec": press_sec,
                    "release_sec": t_sec,
                    "dwell_time": round(dwell, 4)
                })

    ks = pd.DataFrame(records)

    if ks.empty:
        return ks

    # ---------- FLIGHT TIME (CORRECT DEFINITION) ----------
    # Flight1 = next_press - current_release

    ks = ks.sort_values("press_sec").reset_index(drop=True)

    ks["next_press"] = ks["press_sec"].shift(-1)

    ks["flight_time"] = ks["next_press"] - ks["release_sec"]

    # ---------- HANDLE OVERLAP ----------
    ks["overlap"] = ks["flight_time"] < 0

    # keep overlap but clip for stability
    ks["flight_time"] = ks["flight_time"].apply(
        lambda x: 0 if pd.isna(x) else round(max(x, 0), 4)
    )

    # ---------- SESSION BREAK FLAG ----------
    ks["session_break"] = ks["flight_time"] > 5

    # cleanup
    ks = ks.drop(columns=["press_sec", "release_sec", "next_press"])

    return ks

# =============== PROCESS ALL USERS ===============
all_features = []

for user_folder in os.listdir(root_folder):
    user_path = os.path.join(root_folder, user_folder)
    if not os.path.isdir(user_path):
        continue

    # Find the Desktop Keyboard file
    keyboard_file = None
    for f in os.listdir(user_path):
        if f.endswith("_Desktop_Keyboard.csv"):
            keyboard_file = os.path.join(user_path, f)
            break

    if keyboard_file and os.path.exists(keyboard_file):
        print(f"Processing user {user_folder}...")
        df = pd.read_csv(keyboard_file)
        features = extract_keystroke_features(df, user_folder)
        
        if not features.empty:
            all_features.append(features)
            print(f"   → Extracted {len(features)} features from user {user_folder}")
        else:
            print(f"   → No features extracted from user {user_folder} (check data format)")

# Combine all
if all_features:
    final_df = pd.concat(all_features, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n SUCCESS! Total features extracted from all users: {len(final_df)}")
    print("Saved to: all_keystroke_features.csv")
else:
    print(" No keyboard files found. Check the root_folder path.")

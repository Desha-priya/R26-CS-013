# aggregate_keystroke_profiles.py
# Converts raw keystroke rows → one summary row per user
# This is your proper "keystroke behavioral profile"

from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
RAW_KEYSTROKE_FILE = ROOT_DIR / "raw" / "all_keystroke_features(FinalCleaned).csv"
if not RAW_KEYSTROKE_FILE.exists():
    RAW_KEYSTROKE_FILE = ROOT_DIR / "processed" / "all_keystroke_features(FinalCleaned).csv"

OUTPUT_FILE = ROOT_DIR / "processed" / "user_keystroke_profiles.csv"

print("Loading raw keystroke data...")
df = pd.read_csv(RAW_KEYSTROKE_FILE)
print(f"Raw shape: {df.shape}")

# Convert dwell_time and flight_time to numeric (in case of any string values)
df['dwell_time']  = pd.to_numeric(df['dwell_time'],  errors='coerce')
df['flight_time'] = pd.to_numeric(df['flight_time'], errors='coerce')
df['overlap']     = df['overlap'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
df['session_break'] = df['session_break'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})

print("Aggregating per user...")

agg = df.groupby('user').agg(
    # Dwell time features (how long keys are held)
    dwell_mean        = ('dwell_time',  'mean'),
    dwell_std         = ('dwell_time',  'std'),
    dwell_min         = ('dwell_time',  'min'),
    dwell_max         = ('dwell_time',  'max'),
    dwell_median      = ('dwell_time',  'median'),

    # Flight time features (gap between keys)
    flight_mean       = ('flight_time', 'mean'),
    flight_std        = ('flight_time', 'std'),
    flight_min        = ('flight_time', 'min'),
    flight_max        = ('flight_time', 'max'),
    flight_median     = ('flight_time', 'median'),

    # Typing rhythm features
    overlap_rate      = ('overlap',       'mean'),   # how often keys overlap (fast typist = higher)
    session_break_rate= ('session_break', 'mean'),   # how often user pauses mid-session
    total_keystrokes  = ('dwell_time',    'count'),  # typing volume

).reset_index()

# Typing speed proxy: lower average flight time = faster typist
# We also add coefficient of variation (std/mean) — measures typing consistency
agg['dwell_cv']  = agg['dwell_std']  / agg['dwell_mean']   # consistency of key hold
agg['flight_cv'] = agg['flight_std'] / agg['flight_mean']  # consistency of key rhythm

# Fill any NaN (users with very few keystrokes)
agg = agg.fillna(agg.median(numeric_only=True))

agg.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE}")
print(f"Shape: {agg.shape}  ← should be ~116 rows, 15 columns")
print(f"\nFeature columns: {agg.columns.tolist()}")
print(f"\nSample:\n{agg.head(3).to_string()}")
import pandas as pd

df = pd.read_csv("all_keystroke_features(FinalCleaned).csv")

# ---------- BASIC PREVIEW ----------
print("HEAD:")
print(df.head())

print("\nDESCRIBE:")
print(df.describe())

# ---------- SANITY CHECK 1: Missing values ----------
print("\nMissing values:")
print(df.isnull().sum())

# ---------- SANITY CHECK 2: Negative or invalid values ----------
print("\nInvalid values check:")
print("Negative dwell:", (df['dwell_time'] < 0).sum())
print("Negative flight:", (df['flight_time'] < 0).sum())

# ---------- SANITY CHECK 3: Extreme values ----------
print("\nExtreme values:")
print("Dwell > 10 sec:", (df['dwell_time'] > 10).sum())
print("Flight > 5 sec:", (df['flight_time'] > 5).sum())

# ---------- SANITY CHECK 4: Overlap ratio ----------
overlap_ratio = df['overlap'].mean()
print(f"\nOverlap ratio: {overlap_ratio:.4f}")

# ---------- SANITY CHECK 5: Session breaks ----------
session_ratio = df['session_break'].mean()
print(f"Session break ratio: {session_ratio:.4f}")

# ---------- SANITY CHECK 6: Key distribution ----------
print("\nTop 10 keys:")
print(df['key'].value_counts().head(10))

# ---------- SANITY CHECK 7: Basic dataset health summary ----------
print("\nDataset Health Summary:")
print(f"Total rows: {len(df)}")
print(f"Unique users: {df['user'].nunique()}")
print(f"Unique keys: {df['key'].nunique()}")
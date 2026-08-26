import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# AALTO FEATURE VALIDATION — SINGLE PARTICIPANT
# ============================================================

FILE = Path(r"processed/features/5_features.csv")

# Our 16 candidate behavioral features
FEATURES = [
    "DWELL_TIME",
    "PRESS_INTERVAL",
    "RELEASE_INTERVAL",
    "RELEASE_PRESS_LATENCY",
    "OVERLAP_DURATION",
    "OVERLAP_INDICATOR",
    "DWELL_DIFFERENCE",
    "DWELL_RATIO",
    "TRIGRAPH_PRESS_INTERVAL",
    "MEAN_DWELL",
    "STD_DWELL",
    "MEDIAN_DWELL",
    "MEAN_IKI",
    "STD_IKI",
    "IKI_CV",
    "LOCAL_PAUSE_FREQUENCY"
]

print("=" * 70)
print("AALTO FEATURE VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------

df = pd.read_csv(FILE)

print("\nFILE:")
print(FILE)

print("\nSHAPE:")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

print("\nALL COLUMNS:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2}. {col}")

# ------------------------------------------------------------
# 2. Check required features
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE AVAILABILITY")
print("=" * 70)

missing_features = [f for f in FEATURES if f not in df.columns]

if missing_features:
    print("\nMISSING FEATURES:")
    for f in missing_features:
        print(" -", f)
else:
    print("\nAll 16 candidate features are present.")

# ------------------------------------------------------------
# 3. Missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df[FEATURES].isna().sum()

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": (missing / len(df)) * 100
})

print(missing_report.to_string())

# ------------------------------------------------------------
# 4. Infinite values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INFINITE VALUES")
print("=" * 70)

numeric = df[FEATURES].apply(pd.to_numeric, errors="coerce")

infinite = np.isinf(numeric).sum()

for feature, count in infinite.items():
    if count > 0:
        print(f"{feature}: {count}")

if infinite.sum() == 0:
    print("No infinite values found.")

# ------------------------------------------------------------
# 5. Statistical summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

stats = numeric.describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T

print(stats.to_string())

# ------------------------------------------------------------
# 6. Negative values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NEGATIVE VALUES")
print("=" * 70)

for feature in FEATURES:
    values = pd.to_numeric(df[feature], errors="coerce")

    count = (values < 0).sum()

    if count > 0:
        print(f"{feature}: {count:,} negative values")

# ------------------------------------------------------------
# 7. Section distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SECTION DISTRIBUTION")
print("=" * 70)

section_counts = df.groupby("TEST_SECTION_ID").size()

print(section_counts.to_string())

print(f"\nNumber of sections: {section_counts.shape[0]}")
print(f"Minimum rows/section: {section_counts.min()}")
print(f"Maximum rows/section: {section_counts.max()}")

# ------------------------------------------------------------
# 8. Check section boundaries
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SECTION BOUNDARY CHECK")
print("=" * 70)

# Show first and last rows of every section
boundary = (
    df.groupby("TEST_SECTION_ID")
      .agg(
          first_keystroke=("KEYSTROKE_ID", "first"),
          last_keystroke=("KEYSTROKE_ID", "last"),
          rows=("KEYSTROKE_ID", "count")
      )
)

print(boundary.to_string())

# ------------------------------------------------------------
# 9. Check chronological ordering
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CHRONOLOGICAL ORDER CHECK")
print("=" * 70)

bad_sections = []

for section_id, group in df.groupby("TEST_SECTION_ID"):

    press = pd.to_numeric(group["PRESS_TIME"], errors="coerce")

    if not press.is_monotonic_increasing:
        bad_sections.append(section_id)

if bad_sections:
    print("Sections NOT chronologically ordered:")
    print(bad_sections)
else:
    print("All sections are chronologically ordered.")

# ------------------------------------------------------------
# 10. Correlation matrix
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE CORRELATION")
print("=" * 70)

corr = numeric.corr()

# Find highly correlated pairs
high_corr = []

for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):

        value = corr.iloc[i, j]

        if abs(value) >= 0.90:
            high_corr.append(
                (
                    corr.columns[i],
                    corr.columns[j],
                    value
                )
            )

if high_corr:

    print("\nHighly correlated pairs (|r| >= 0.90):")

    for a, b, value in sorted(
        high_corr,
        key=lambda x: abs(x[2]),
        reverse=True
    ):
        print(f"{a:30} <-> {b:30} : {value:.4f}")

else:
    print("\nNo correlations above |r| >= 0.90.")

# ------------------------------------------------------------
# 11. Example rows
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 FEATURE ROWS")
print("=" * 70)

print(df[
    ["TEST_SECTION_ID", "KEYSTROKE_ID"] + FEATURES
].head(10).to_string(index=False))

# ------------------------------------------------------------
# DONE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
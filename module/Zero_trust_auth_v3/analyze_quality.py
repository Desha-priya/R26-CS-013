from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

REPORT_FILE = Path("reports/preprocessing_report.csv")
OUTPUT_DIR = Path("reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD REPORT
# ============================================================

df = pd.read_csv(REPORT_FILE)

print("=" * 70)
print("AALTO HIGH-QUALITY SUBSET ANALYSIS")
print("=" * 70)

print(f"\nTotal participant files: {len(df):,}")


# ============================================================
# KEEP SUCCESSFULLY PROCESSED PARTICIPANTS
# ============================================================

clean = df[df["status"] == "OK"].copy()

print(
    f"Successfully processed: {len(clean):,}"
)

# ------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BASIC STATISTICS")
print("=" * 70)

for column in [
    "rows_final",
    "sections_valid",
    "duplicates_removed",
    "invalid_timestamp_rows",
]:

    print(f"\n{column}")

    print(
        clean[column].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )


# ============================================================
# QUALITY THRESHOLD EXPERIMENT
# ============================================================

print("\n" + "=" * 70)
print("QUALITY THRESHOLD ANALYSIS")
print("=" * 70)

# We test several reasonable minimum-keystroke thresholds.
# We are NOT selecting one yet.

keystroke_thresholds = [
    500,
    550,
    600,
    650,
    700,
    750,
    800,
    900,
    1000,
]

section_thresholds = [
    10,
    12,
    15,
]


results = []

for min_sections in section_thresholds:

    for min_keystrokes in keystroke_thresholds:

        subset = clean[
            (clean["sections_valid"] >= min_sections)
            &
            (clean["rows_final"] >= min_keystrokes)
        ]

        results.append({
            "min_sections": min_sections,
            "min_keystrokes": min_keystrokes,
            "participants": len(subset),
            "percentage_of_clean": (
                len(subset) /
                len(clean) *
                100
            ),
        })


threshold_df = pd.DataFrame(results)

print(
    threshold_df.to_string(
        index=False,
        formatters={
            "percentage_of_clean": "{:.2f}".format
        }
    )
)


# ============================================================
# RECOMMENDED CANDIDATE
# ============================================================

# For the first candidate we want:
#
#   - all 15 sections
#   - at least 600 keystroke events
#
# This is ONLY a candidate.
# We will inspect the resulting size before finalizing.

candidate = clean[
    (clean["sections_valid"] >= 15)
    &
    (clean["rows_final"] >= 600)
].copy()


print("\n" + "=" * 70)
print("CANDIDATE HIGH-QUALITY SUBSET")
print("=" * 70)

print(
    f"Participants: {len(candidate):,}"
)

print(
    f"Percentage of clean participants: "
    f"{len(candidate) / len(clean) * 100:.2f}%"
)

print(
    f"Total processed keystrokes: "
    f"{candidate['rows_final'].sum():,}"
)

print(
    f"Average keystrokes/participant: "
    f"{candidate['rows_final'].mean():.2f}"
)

print(
    f"Median keystrokes/participant: "
    f"{candidate['rows_final'].median():.2f}"
)


# ============================================================
# PARTICIPANT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SECTION DISTRIBUTION")
print("=" * 70)

print(
    candidate["sections_valid"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# SAVE RESULTS
# ============================================================

threshold_file = (
    OUTPUT_DIR /
    "quality_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False
)


candidate_file = (
    OUTPUT_DIR /
    "candidate_high_quality_participants.csv"
)

candidate[
    [
        "file",
        "filename_participant_id",
        "rows_final",
        "sections_valid",
        "duplicates_removed",
        "invalid_timestamp_rows",
    ]
].to_csv(
    candidate_file,
    index=False
)


print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    f"\nThreshold analysis:\n{threshold_file}"
)

print(
    f"\nCandidate participant list:\n{candidate_file}"
)

print("\nDONE.")
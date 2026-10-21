from pathlib import Path
import pandas as pd
import numpy as np
import json
import random
import time

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = Path("processed/features")
OUTPUT_DIR = Path("modeling")

RANDOM_SEED = 42

# Development cohort
TARGET_PARTICIPANTS = 5000

# Quality requirements
MIN_SECTIONS = 15
MIN_KEYSTROKES = 600

# Behavioral features selected for the experiment
FEATURE_COLUMNS = [
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
    "LOCAL_PAUSE_FREQUENCY",
]

REQUIRED_COLUMNS = [
    "PARTICIPANT_ID",
    "TEST_SECTION_ID",
    "KEYSTROKE_ID",
] + FEATURE_COLUMNS


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_DIR = OUTPUT_DIR / "sequences"
SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

start_time = time.time()


# ============================================================
# FIND FEATURE FILES
# ============================================================

files = sorted(FEATURE_DIR.glob("*_features.csv"))

print("=" * 70)
print("AALTO SEQUENCE DATASET CONSTRUCTION")
print("=" * 70)

print(f"\nFeature directory : {FEATURE_DIR}")
print(f"Files discovered  : {len(files)}")
print(f"Target participants: {TARGET_PARTICIPANTS}")
print(f"Minimum sections  : {MIN_SECTIONS}")
print(f"Minimum keystrokes: {MIN_KEYSTROKES}")
print(f"Random seed       : {RANDOM_SEED}")


# ============================================================
# PASS 1 — QUALITY SCREENING
# ============================================================

eligible = []
failed = []

print("\n" + "=" * 70)
print("PASS 1 — QUALITY SCREENING")
print("=" * 70)

for i, file in enumerate(files, start=1):

    try:
        df = pd.read_csv(file)

        # ----------------------------------------------------
        # Basic structural validation
        # ----------------------------------------------------

        missing_columns = [
            c for c in REQUIRED_COLUMNS
            if c not in df.columns
        ]

        if missing_columns:
            failed.append({
                "file": file.name,
                "reason": "missing_required_columns",
                "details": ",".join(missing_columns)
            })
            continue

        if df.empty:
            failed.append({
                "file": file.name,
                "reason": "empty_file",
                "details": ""
            })
            continue

        # ----------------------------------------------------
        # Participant ID consistency
        # ----------------------------------------------------

        filename_id = file.name.replace("_features.csv", "")

        try:
            filename_id = int(filename_id)
        except ValueError:
            failed.append({
                "file": file.name,
                "reason": "invalid_filename_id",
                "details": ""
            })
            continue

        participant_ids = df["PARTICIPANT_ID"].dropna().unique()

        if len(participant_ids) != 1:
            failed.append({
                "file": file.name,
                "reason": "multiple_participant_ids",
                "details": str(participant_ids[:10])
            })
            continue

        participant_id = int(participant_ids[0])

        if participant_id != filename_id:
            failed.append({
                "file": file.name,
                "reason": "filename_id_mismatch",
                "details": f"{filename_id} != {participant_id}"
            })
            continue

        # ----------------------------------------------------
        # Section quality
        # ----------------------------------------------------

        sections = (
            df["TEST_SECTION_ID"]
            .dropna()
            .unique()
        )

        section_count = len(sections)

        if section_count < MIN_SECTIONS:
            failed.append({
                "file": file.name,
                "reason": "insufficient_sections",
                "details": str(section_count)
            })
            continue

        # ----------------------------------------------------
        # Keystroke quality
        # ----------------------------------------------------

        keystroke_count = len(df)

        if keystroke_count < MIN_KEYSTROKES:
            failed.append({
                "file": file.name,
                "reason": "insufficient_keystrokes",
                "details": str(keystroke_count)
            })
            continue

        # ----------------------------------------------------
        # Required feature existence
        # ----------------------------------------------------

        feature_available = (
            df[FEATURE_COLUMNS]
            .notna()
            .any(axis=1)
            .sum()
        )

        if feature_available < MIN_KEYSTROKES * 0.90:
            failed.append({
                "file": file.name,
                "reason": "too_many_missing_features",
                "details": str(feature_available)
            })
            continue

        # ----------------------------------------------------
        # Temporal ordering check
        # ----------------------------------------------------

        temporal_ok = True

        for section_id, section in df.groupby(
            "TEST_SECTION_ID",
            sort=False
        ):
            key_ids = section["KEYSTROKE_ID"].to_numpy()

            # Keystroke IDs should generally progress.
            # We don't require strict +1 because AALTO can
            # contain missing IDs.
            if len(key_ids) > 1:
                if np.any(np.diff(key_ids) < 0):
                    temporal_ok = False
                    break

        if not temporal_ok:
            failed.append({
                "file": file.name,
                "reason": "temporal_order_problem",
                "details": ""
            })
            continue

        # ----------------------------------------------------
        # Eligible
        # ----------------------------------------------------

        eligible.append({
            "participant_id": participant_id,
            "file": str(file),
            "sections": section_count,
            "keystrokes": keystroke_count
        })

    except Exception as e:

        failed.append({
            "file": file.name,
            "reason": "read_error",
            "details": str(e)
        })

    if i % 5000 == 0:
        print(
            f"Processed {i:,}/{len(files):,} | "
            f"eligible={len(eligible):,} | "
            f"failed={len(failed):,}"
        )


# ============================================================
# ELIGIBLE PARTICIPANTS
# ============================================================

eligible_df = pd.DataFrame(eligible)

failed_df = pd.DataFrame(failed)

print("\nEligible participants:", len(eligible_df))
print("Failed participants  :", len(failed_df))


if len(eligible_df) < TARGET_PARTICIPANTS:
    raise RuntimeError(
        f"Only {len(eligible_df)} participants passed "
        f"quality screening. Cannot select {TARGET_PARTICIPANTS}."
    )


# ============================================================
# FIXED RANDOM SAMPLING
# ============================================================

selected_df = (
    eligible_df
    .sample(
        n=TARGET_PARTICIPANTS,
        random_state=RANDOM_SEED
    )
    .sort_values("participant_id")
    .reset_index(drop=True)
)

selected_df.to_csv(
    REPORT_DIR / "selected_participants.csv",
    index=False
)

eligible_df.to_csv(
    REPORT_DIR / "eligible_participants.csv",
    index=False
)

failed_df.to_csv(
    REPORT_DIR / "failed_participants.csv",
    index=False
)

print("\nSelected participants:", len(selected_df))


# ============================================================
# PASS 2 — BUILD SEQUENCES
# ============================================================

print("\n" + "=" * 70)
print("PASS 2 — BUILDING SENTENCE-LEVEL SEQUENCES")
print("=" * 70)

sequence_records = []
sequence_summary = []

timing_issues = []

for idx, row in selected_df.iterrows():

    participant_id = int(row["participant_id"])
    file = Path(row["file"])

    df = pd.read_csv(file)

    # --------------------------------------------------------
    # Preserve temporal order
    # --------------------------------------------------------

    df = df.sort_values(
        ["TEST_SECTION_ID", "KEYSTROKE_ID"],
        kind="stable"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Process each sentence / test section
    # --------------------------------------------------------

    for section_id, section in df.groupby(
        "TEST_SECTION_ID",
        sort=False
    ):

        section = section.copy()

        section = section.sort_values(
            "KEYSTROKE_ID",
            kind="stable"
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Validate sequence length
        # ----------------------------------------------------

        if len(section) < 10:
            continue

        # ----------------------------------------------------
        # Detect suspicious timing values
        # ----------------------------------------------------

        for feature in [
            "DWELL_TIME",
            "PRESS_INTERVAL",
            "RELEASE_INTERVAL",
            "RELEASE_PRESS_LATENCY",
            "OVERLAP_DURATION",
            "DWELL_DIFFERENCE",
            "DWELL_RATIO",
            "TRIGRAPH_PRESS_INTERVAL",
            "MEAN_DWELL",
            "STD_DWELL",
            "MEDIAN_DWELL",
            "MEAN_IKI",
            "STD_IKI",
            "IKI_CV",
            "LOCAL_PAUSE_FREQUENCY",
        ]:

            values = pd.to_numeric(
                section[feature],
                errors="coerce"
            )

            finite_values = values[np.isfinite(values)]

            if len(finite_values) == 0:
                continue

            # Record extreme values rather than deleting them
            # at this stage.
            if feature != "OVERLAP_INDICATOR":
                extreme_count = (
                    (finite_values.abs() > 5000)
                    .sum()
                )

                if extreme_count > 0:
                    timing_issues.append({
                        "participant_id": participant_id,
                        "section_id": section_id,
                        "feature": feature,
                        "extreme_values": int(extreme_count)
                    })

        # ----------------------------------------------------
        # Extract modeling representation
        # ----------------------------------------------------

        feature_data = section[FEATURE_COLUMNS].copy()

        # Force numerical representation
        for feature in FEATURE_COLUMNS:
            feature_data[feature] = pd.to_numeric(
                feature_data[feature],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Store sequence as compressed NPZ
        # ----------------------------------------------------

        sequence_id = (
            f"{participant_id}_section_{int(section_id)}"
        )

        sequence_path = (
            SEQUENCE_DIR /
            f"{sequence_id}.npz"
        )

        np.savez_compressed(
            sequence_path,
            X=feature_data.to_numpy(dtype=np.float32),
            participant_id=np.int64(participant_id),
            section_id=np.int64(section_id),
            length=np.int64(len(feature_data))
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        sequence_summary.append({
            "sequence_id": sequence_id,
            "participant_id": participant_id,
            "section_id": int(section_id),
            "sequence_length": len(section),
            "missing_values": int(
                feature_data.isna().sum().sum()
            ),
            "missing_percentage": (
                feature_data.isna().sum().sum()
                / feature_data.size
                * 100
            ),
            "path": str(sequence_path)
        })

    if (idx + 1) % 250 == 0:
        print(
            f"Built sequences for "
            f"{idx + 1:,}/{len(selected_df):,} participants"
        )


# ============================================================
# SAVE METADATA
# ============================================================

sequence_df = pd.DataFrame(sequence_summary)

sequence_df.to_csv(
    OUTPUT_DIR / "sequence_metadata.csv",
    index=False
)

pd.DataFrame(timing_issues).to_csv(
    REPORT_DIR / "timing_quality_report.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

elapsed = time.time() - start_time

summary = {
    "random_seed": RANDOM_SEED,
    "files_discovered": len(files),
    "eligible_participants": len(eligible_df),
    "selected_participants": len(selected_df),
    "failed_participants": len(failed_df),
    "sequences_created": len(sequence_df),
    "features_per_keystroke": len(FEATURE_COLUMNS),
    "minimum_sections": MIN_SECTIONS,
    "minimum_keystrokes": MIN_KEYSTROKES,
    "sequence_length_min": (
        int(sequence_df["sequence_length"].min())
        if len(sequence_df) else None
    ),
    "sequence_length_median": (
        float(sequence_df["sequence_length"].median())
        if len(sequence_df) else None
    ),
    "sequence_length_max": (
        int(sequence_df["sequence_length"].max())
        if len(sequence_df) else None
    ),
    "total_keystrokes": (
        int(sequence_df["sequence_length"].sum())
        if len(sequence_df) else 0
    ),
    "elapsed_seconds": elapsed
}

with open(
    REPORT_DIR / "sequence_dataset_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4)


print("\n" + "=" * 70)
print("SEQUENCE DATASET CONSTRUCTION COMPLETE")
print("=" * 70)

for key, value in summary.items():
    print(f"{key:30}: {value}")

print("\nOutput:")
print(f"Sequences : {SEQUENCE_DIR}")
print(f"Metadata  : {OUTPUT_DIR / 'sequence_metadata.csv'}")
print(f"Reports   : {REPORT_DIR}")
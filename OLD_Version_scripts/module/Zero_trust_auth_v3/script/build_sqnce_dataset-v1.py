from pathlib import Path
import pandas as pd
import numpy as np
import random
import json
import time

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = Path("processed/features")
CANDIDATE_FILE = Path(
    "reports/candidate_high_quality_participants.csv"
)

OUTPUT_DIR = Path("processed/sequence_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_PARTICIPANTS = 5000

MIN_SECTIONS = 15
MIN_KEYSTROKES = 600

SEQUENCE_LENGTH = 50
STRIDE = 25

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

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
    "LOCAL_PAUSE_FREQUENCY",
]

META_COLUMNS = [
    "PARTICIPANT_ID",
    "TEST_SECTION_ID",
    "KEYSTROKE_ID",
]

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

start_time = time.time()

print("=" * 70)
print("AALTO SEQUENCE DATASET CONSTRUCTION")
print("=" * 70)

print(f"Feature directory       : {FEATURE_DIR}")
print(f"Candidate file          : {CANDIDATE_FILE}")
print(f"Target participants     : {TARGET_PARTICIPANTS}")
print(f"Minimum sections        : {MIN_SECTIONS}")
print(f"Minimum keystrokes     : {MIN_KEYSTROKES}")
print(f"Sequence length        : {SEQUENCE_LENGTH}")
print(f"Sequence stride        : {STRIDE}")
print(f"Random seed             : {SEED}")
print("=" * 70)


# ============================================================
# PHASE 1
# LOAD EXISTING HIGH-QUALITY PARTICIPANT LIST
# ============================================================
# ============================================================
# PHASE 1
# LOAD EXISTING HIGH-QUALITY PARTICIPANT LIST
# ============================================================

print("\nPHASE 1 — LOADING HIGH-QUALITY PARTICIPANTS")

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(
        f"Candidate file not found: {CANDIDATE_FILE}"
    )

candidate_df = pd.read_csv(CANDIDATE_FILE)

print(f"Candidate columns: {list(candidate_df.columns)}")
print(f"Candidate rows   : {len(candidate_df):,}")

# ------------------------------------------------------------
# The high-quality report stores the participant ID in:
# filename_participant_id
# ------------------------------------------------------------

if "filename_participant_id" not in candidate_df.columns:
    raise ValueError(
        "Expected 'filename_participant_id' column "
        "in the high-quality participant report."
    )

candidate_ids = (
    pd.to_numeric(
        candidate_df["filename_participant_id"],
        errors="coerce"
    )
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

print(
    f"Unique high-quality participants: "
    f"{len(candidate_ids):,}"
)

# ============================================================
# PHASE 2
# SELECT 5,000 PARTICIPANTS
# ============================================================

print("\nPHASE 2 — SELECTING PARTICIPANTS")

if len(candidate_ids) < TARGET_PARTICIPANTS:
    raise ValueError(
        f"Only {len(candidate_ids)} candidates available; "
        f"{TARGET_PARTICIPANTS} required."
    )

rng = np.random.default_rng(SEED)

selected_ids = rng.choice(
    candidate_ids,
    size=TARGET_PARTICIPANTS,
    replace=False
)

selected_ids = sorted(
    [int(x) for x in selected_ids]
)

print(f"Selected participants: {len(selected_ids):,}")

# Save selected participant list
selected_df = pd.DataFrame({
    "PARTICIPANT_ID": selected_ids
})

selected_df.to_csv(
    OUTPUT_DIR / "selected_5000_participants.csv",
    index=False
)


# ============================================================
# PHASE 3
# PARTICIPANT-LEVEL TRAIN/VAL/TEST SPLIT
# ============================================================

print("\nPHASE 3 — PARTICIPANT-LEVEL SPLIT")

shuffled_ids = selected_ids.copy()
rng.shuffle(shuffled_ids)

n = len(shuffled_ids)

train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_ids = sorted(shuffled_ids[:train_end])
val_ids = sorted(shuffled_ids[train_end:val_end])
test_ids = sorted(shuffled_ids[val_end:])

print(f"Train participants      : {len(train_ids):,}")
print(f"Validation participants : {len(val_ids):,}")
print(f"Test participants       : {len(test_ids):,}")

assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)

split_df = pd.DataFrame({
    "PARTICIPANT_ID": (
        train_ids +
        val_ids +
        test_ids
    ),
    "SPLIT": (
        ["train"] * len(train_ids) +
        ["validation"] * len(val_ids) +
        ["test"] * len(test_ids)
    )
})

split_df.to_csv(
    OUTPUT_DIR / "participant_splits.csv",
    index=False
)

split_lookup = dict(
    zip(
        split_df["PARTICIPANT_ID"],
        split_df["SPLIT"]
    )
)


# ============================================================
# PHASE 4
# BUILD TEMPORAL SEQUENCES
# ============================================================

print("\nPHASE 4 — BUILDING TEMPORAL SEQUENCES")
print("-" * 70)

sequence_files = {
    "train": OUTPUT_DIR / "train_sequences.npz",
    "validation": OUTPUT_DIR / "validation_sequences.npz",
    "test": OUTPUT_DIR / "test_sequences.npz",
}

# Store sequences separately by split
X_data = {
    "train": [],
    "validation": [],
    "test": [],
}

y_data = {
    "train": [],
    "validation": [],
    "test": [],
}

section_data = {
    "train": [],
    "validation": [],
    "test": [],
}

successful = 0
failed = 0
total_sequences = {
    "train": 0,
    "validation": 0,
    "test": 0,
}

for index, participant_id in enumerate(selected_ids, start=1):

    feature_file = (
        FEATURE_DIR /
        str(participant_id) /
        f"{participant_id}_features.csv"
    )

    if not feature_file.exists():
        # Also support files directly inside processed/features
        alternative = (
            FEATURE_DIR /
            f"{participant_id}_features.csv"
        )

        if alternative.exists():
            feature_file = alternative
        else:
            failed += 1
            continue

    try:

        df = pd.read_csv(feature_file)

        # ----------------------------------------------------
        # Basic structural validation
        # ----------------------------------------------------

        missing_columns = [
            col for col in FEATURES
            if col not in df.columns
        ]

        if missing_columns:
            failed += 1
            continue

        required_meta = [
            col for col in META_COLUMNS
            if col not in df.columns
        ]

        if required_meta:
            failed += 1
            continue

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        for col in FEATURES:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Remove infinite values
        # ----------------------------------------------------

        df[FEATURES] = df[FEATURES].replace(
            [np.inf, -np.inf],
            np.nan
        )

        # ----------------------------------------------------
        # Verify sections
        # ----------------------------------------------------

        sections = sorted(
            df["TEST_SECTION_ID"]
            .dropna()
            .unique()
        )

        if len(sections) < MIN_SECTIONS:
            failed += 1
            continue

        if len(df) < MIN_KEYSTROKES:
            failed += 1
            continue

        split = split_lookup[participant_id]

        participant_sequence_count = 0

        # ----------------------------------------------------
        # Build sequences SECTION BY SECTION
        # ----------------------------------------------------

        for section_id in sections:

            section = df[
                df["TEST_SECTION_ID"] == section_id
            ].copy()

            # Preserve chronological order
            section = section.sort_values(
                "KEYSTROKE_ID"
            )

            # Need enough rows for at least one sequence
            if len(section) < SEQUENCE_LENGTH:
                continue

            # ------------------------------------------------
            # Impute missing values using section median.
            # Fall back to participant median.
            # ------------------------------------------------

            for col in FEATURES:

                if section[col].isna().any():

                    section_median = section[col].median()

                    if pd.isna(section_median):
                        section_median = df[col].median()

                    if pd.isna(section_median):
                        section_median = 0.0

                    section[col] = section[col].fillna(
                        section_median
                    )

            values = section[FEATURES].to_numpy(
                dtype=np.float32
            )

            # ------------------------------------------------
            # Sliding-window sequence generation
            # ------------------------------------------------

            for start in range(
                0,
                len(values) - SEQUENCE_LENGTH + 1,
                STRIDE
            ):

                end = start + SEQUENCE_LENGTH

                sequence = values[start:end]

                if sequence.shape != (
                    SEQUENCE_LENGTH,
                    len(FEATURES)
                ):
                    continue

                X_data[split].append(sequence)

                # Genuine sequence belongs to this participant
                y_data[split].append(participant_id)

                section_data[split].append(section_id)

                participant_sequence_count += 1
                total_sequences[split] += 1

        if participant_sequence_count > 0:
            successful += 1
        else:
            failed += 1

    except Exception as e:
        failed += 1

    # Progress
    if index % 100 == 0 or index == len(selected_ids):
        elapsed = time.time() - start_time

        print(
            f"[{index:5d}/{len(selected_ids):5d}] "
            f"successful={successful:,} "
            f"failed={failed:,} "
            f"sequences="
            f"{sum(total_sequences.values()):,} "
            f"time={elapsed/60:.1f} min"
        )


# ============================================================
# PHASE 5
# CONVERT TO NUMPY ARRAYS
# ============================================================

print("\nPHASE 5 — SAVING SEQUENCE DATA")

for split in ["train", "validation", "test"]:

    if len(X_data[split]) == 0:
        raise RuntimeError(
            f"No sequences generated for {split}."
        )

    X = np.stack(X_data[split]).astype(
        np.float32
    )

    participant_labels = np.asarray(
        y_data[split],
        dtype=np.int32
    )

    section_labels = np.asarray(
        section_data[split],
        dtype=np.int32
    )

    output_file = sequence_files[split]

    np.savez_compressed(
        output_file,
        X=X,
        participant_id=participant_labels,
        section_id=section_labels,
    )

    print(
        f"{split:12s}: "
        f"X={X.shape} "
        f"saved={output_file}"
    )


# ============================================================
# PHASE 6
# DATASET SUMMARY
# ============================================================

elapsed = time.time() - start_time

summary = {
    "dataset": "AALTO",
    "candidate_participants": len(candidate_ids),
    "selected_participants": len(selected_ids),
    "train_participants": len(train_ids),
    "validation_participants": len(val_ids),
    "test_participants": len(test_ids),
    "sequence_length": SEQUENCE_LENGTH,
    "stride": STRIDE,
    "num_features": len(FEATURES),
    "features": FEATURES,
    "train_sequences": total_sequences["train"],
    "validation_sequences": total_sequences["validation"],
    "test_sequences": total_sequences["test"],
    "successful_participants": successful,
    "failed_participants": failed,
    "random_seed": SEED,
    "elapsed_minutes": elapsed / 60,
}

with open(
    OUTPUT_DIR / "dataset_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=4
    )

print("\n" + "=" * 70)
print("SEQUENCE DATASET CONSTRUCTION COMPLETE")
print("=" * 70)

print(f"Candidate pool        : {len(candidate_ids):,}")
print(f"Selected participants : {len(selected_ids):,}")
print(f"Successful            : {successful:,}")
print(f"Failed                : {failed:,}")

print(
    f"Train sequences       : "
    f"{total_sequences['train']:,}"
)

print(
    f"Validation sequences  : "
    f"{total_sequences['validation']:,}"
)

print(
    f"Test sequences        : "
    f"{total_sequences['test']:,}"
)

print(f"Features              : {len(FEATURES)}")
print(f"Sequence length       : {SEQUENCE_LENGTH}")
print(f"Stride                : {STRIDE}")
print(f"Time                  : {elapsed/60:.2f} minutes")

print("\nOutput:")
print(OUTPUT_DIR)
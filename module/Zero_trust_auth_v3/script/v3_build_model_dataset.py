from pathlib import Path
import pandas as pd
import numpy as np
import json
import random
import time

# ============================================================
# CONFIGURATION
# ============================================================

QUALITY_FILE = Path(
    "reports/deep_quality_v3/final_top_5000_participants_v3.csv"
)

FEATURE_DIR = Path("processed/features")
OUTPUT_DIR = Path("processed/siamese_bilstm_v3")

RANDOM_SEED = 42

TOTAL_PARTICIPANTS = 5000

TRAIN_PARTICIPANTS = 3500
VAL_PARTICIPANTS = 750
TEST_PARTICIPANTS = 750

SEQUENCE_LENGTH = 50

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
    "PRESS_TIME",
    "RELEASE_TIME",
] + FEATURE_COLUMNS


# ============================================================
# SETUP
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

start_time = time.time()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "validation"
TEST_DIR = OUTPUT_DIR / "test"

TRAIN_DIR.mkdir(exist_ok=True)
VAL_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("AALTO V3 MODEL DATASET CONSTRUCTION")
print("=" * 70)

print(f"\nQuality cohort : {QUALITY_FILE}")
print(f"Feature dir    : {FEATURE_DIR}")
print(f"Output dir     : {OUTPUT_DIR}")
print(f"Random seed    : {RANDOM_SEED}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Features       : {len(FEATURE_COLUMNS)}")


# ============================================================
# LOAD TOP-5000 COHORT
# ============================================================

print("\n" + "=" * 70)
print("LOADING LOCKED TOP-5000 COHORT")
print("=" * 70)

if not QUALITY_FILE.exists():
    raise FileNotFoundError(
        f"V3 cohort file not found:\n{QUALITY_FILE}"
    )

cohort = pd.read_csv(QUALITY_FILE)

if "participant_id" not in cohort.columns:
    raise RuntimeError(
        "participant_id column missing from V3 cohort file."
    )

cohort["participant_id"] = pd.to_numeric(
    cohort["participant_id"],
    errors="coerce"
)

cohort = cohort.dropna(
    subset=["participant_id"]
).copy()

cohort["participant_id"] = cohort[
    "participant_id"
].astype(int)

cohort = cohort.drop_duplicates(
    subset=["participant_id"]
).reset_index(drop=True)

print(f"Participants loaded : {len(cohort):,}")

if len(cohort) != TOTAL_PARTICIPANTS:
    raise RuntimeError(
        f"Expected exactly {TOTAL_PARTICIPANTS} participants, "
        f"but found {len(cohort)}."
    )


# ============================================================
# VERIFY FILES BEFORE SPLIT
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING PARTICIPANT FEATURE FILES")
print("=" * 70)

missing_files = []

for i, row in cohort.iterrows():

    participant_id = int(row["participant_id"])

    file_path = FEATURE_DIR / (
        f"{participant_id}_features.csv"
    )

    if not file_path.exists():
        missing_files.append({
            "participant_id": participant_id,
            "file": str(file_path)
        })

    if (i + 1) % 1000 == 0:
        print(
            f"Checked {i + 1:,}/{len(cohort):,}"
        )

if missing_files:

    pd.DataFrame(missing_files).to_csv(
        REPORT_DIR / "missing_feature_files.csv",
        index=False
    )

    raise RuntimeError(
        f"{len(missing_files)} selected participants "
        f"have missing feature files."
    )

print("All selected participant files exist.")
print("File integrity : PASS")


# ============================================================
# PARTICIPANT-LEVEL RANDOM SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING PARTICIPANT-LEVEL TRAIN / VAL / TEST SPLIT")
print("=" * 70)

shuffled = cohort.sample(
    frac=1.0,
    random_state=RANDOM_SEED
).reset_index(drop=True)

train_ids = shuffled.iloc[
    :TRAIN_PARTICIPANTS
]["participant_id"].tolist()

val_ids = shuffled.iloc[
    TRAIN_PARTICIPANTS:
    TRAIN_PARTICIPANTS + VAL_PARTICIPANTS
]["participant_id"].tolist()

test_ids = shuffled.iloc[
    TRAIN_PARTICIPANTS + VAL_PARTICIPANTS:
]["participant_id"].tolist()

train_ids = set(train_ids)
val_ids = set(val_ids)
test_ids = set(test_ids)

print(f"Train participants : {len(train_ids):,}")
print(f"Validation         : {len(val_ids):,}")
print(f"Test               : {len(test_ids):,}")


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("PARTICIPANT LEAKAGE CHECK")
print("=" * 70)

train_val = train_ids & val_ids
train_test = train_ids & test_ids
val_test = val_ids & test_ids

print(f"Train ∩ Validation : {len(train_val)}")
print(f"Train ∩ Test       : {len(train_test)}")
print(f"Validation ∩ Test  : {len(val_test)}")

if train_val or train_test or val_test:
    raise RuntimeError(
        "PARTICIPANT LEAKAGE DETECTED."
    )

print("Participant separation : PASS")


# ============================================================
# SAVE PARTICIPANT SPLIT
# ============================================================

split_records = []

for pid in sorted(train_ids):
    split_records.append({
        "participant_id": pid,
        "split": "train"
    })

for pid in sorted(val_ids):
    split_records.append({
        "participant_id": pid,
        "split": "validation"
    })

for pid in sorted(test_ids):
    split_records.append({
        "participant_id": pid,
        "split": "test"
    })

split_df = pd.DataFrame(split_records)

split_df.to_csv(
    REPORT_DIR / "participant_split_v3.csv",
    index=False
)


# ============================================================
# SEQUENCE BUILDER
# ============================================================

def build_sequences(
    participant_ids,
    output_dir,
    split_name
):

    sequence_records = []

    total_sequences = 0
    total_keystrokes = 0

    nan_values = 0
    inf_values = 0

    participant_sequence_counts = []

    print("\n" + "-" * 70)
    print(f"BUILDING {split_name.upper()} SEQUENCES")
    print("-" * 70)

    for idx, participant_id in enumerate(
        sorted(participant_ids),
        start=1
    ):

        file_path = FEATURE_DIR / (
            f"{participant_id}_features.csv"
        )

        df = pd.read_csv(file_path)

        # ----------------------------------------------------
        # Structural validation
        # ----------------------------------------------------

        missing_columns = [
            c for c in REQUIRED_COLUMNS
            if c not in df.columns
        ]

        if missing_columns:
            raise RuntimeError(
                f"{participant_id}: missing columns: "
                f"{missing_columns}"
            )

        # ----------------------------------------------------
        # Participant consistency
        # ----------------------------------------------------

        participant_values = (
            df["PARTICIPANT_ID"]
            .dropna()
            .unique()
        )

        if len(participant_values) != 1:
            raise RuntimeError(
                f"{participant_id}: multiple participant IDs."
            )

        if int(participant_values[0]) != participant_id:
            raise RuntimeError(
                f"{participant_id}: participant ID mismatch."
            )

        # ----------------------------------------------------
        # Numerical conversion
        # ----------------------------------------------------

        for feature in FEATURE_COLUMNS:

            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # PRESS_TIME is the temporal reference.
        #
        # We DO NOT use KEYSTROKE_ID for temporal sorting.
        # ----------------------------------------------------

        df["PRESS_TIME"] = pd.to_numeric(
            df["PRESS_TIME"],
            errors="coerce"
        )

        df["RELEASE_TIME"] = pd.to_numeric(
            df["RELEASE_TIME"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Sort by section + PRESS_TIME
        # ----------------------------------------------------

        df = df.sort_values(
            [
                "TEST_SECTION_ID",
                "PRESS_TIME"
            ],
            kind="stable"
        ).reset_index(drop=True)

        participant_sequence_count = 0

        # ----------------------------------------------------
        # Process sections
        # ----------------------------------------------------

        for section_id, section in df.groupby(
            "TEST_SECTION_ID",
            sort=False
        ):

            section = section.copy()

            section = section.sort_values(
                "PRESS_TIME",
                kind="stable"
            ).reset_index(drop=True)

            if len(section) < SEQUENCE_LENGTH:
                continue

            feature_data = section[
                FEATURE_COLUMNS
            ].copy()

            X = feature_data.to_numpy(
                dtype=np.float32
            )

            nan_count = int(
                np.isnan(X).sum()
            )

            inf_count = int(
                np.isinf(X).sum()
            )

            nan_values += nan_count
            inf_values += inf_count

            # ------------------------------------------------
            # Replace invalid values ONLY for sequence
            # construction.
            #
            # This is not participant selection.
            # ------------------------------------------------

            X = np.nan_to_num(
                X,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # ------------------------------------------------
            # Sliding windows of 50 keystrokes
            # ------------------------------------------------

            for start in range(
                0,
                len(X) - SEQUENCE_LENGTH + 1,
                SEQUENCE_LENGTH
            ):

                sequence = X[
                    start:
                    start + SEQUENCE_LENGTH
                ]

                sequence_id = (
                    f"{participant_id}_"
                    f"section_{int(section_id)}_"
                    f"seq_{participant_sequence_count}"
                )

                output_file = (
                    output_dir /
                    f"{sequence_id}.npz"
                )

                np.savez_compressed(
                    output_file,
                    X=sequence.astype(
                        np.float32
                    ),
                    participant_id=np.int64(
                        participant_id
                    ),
                    section_id=np.int64(
                        section_id
                    ),
                    sequence_id=sequence_id
                )

                sequence_records.append({
                    "sequence_id": sequence_id,
                    "participant_id": participant_id,
                    "section_id": int(section_id),
                    "sequence_length": SEQUENCE_LENGTH,
                    "start_index": start,
                    "split": split_name,
                    "path": str(output_file)
                })

                participant_sequence_count += 1
                total_sequences += 1
                total_keystrokes += SEQUENCE_LENGTH

        participant_sequence_counts.append({
            "participant_id": participant_id,
            "split": split_name,
            "sequences": participant_sequence_count
        })

        if idx % 250 == 0 or idx == len(participant_ids):

            print(
                f"Processed "
                f"{idx:,}/{len(participant_ids):,} | "
                f"sequences={total_sequences:,}"
            )

    return (
        sequence_records,
        participant_sequence_counts,
        total_sequences,
        total_keystrokes,
        nan_values,
        inf_values
    )


# ============================================================
# BUILD TRAIN
# ============================================================

(
    train_sequences,
    train_counts,
    train_total,
    train_keystrokes,
    train_nan,
    train_inf
) = build_sequences(
    train_ids,
    TRAIN_DIR,
    "train"
)


# ============================================================
# BUILD VALIDATION
# ============================================================

(
    val_sequences,
    val_counts,
    val_total,
    val_keystrokes,
    val_nan,
    val_inf
) = build_sequences(
    val_ids,
    VAL_DIR,
    "validation"
)


# ============================================================
# BUILD TEST
# ============================================================

(
    test_sequences,
    test_counts,
    test_total,
    test_keystrokes,
    test_nan,
    test_inf
) = build_sequences(
    test_ids,
    TEST_DIR,
    "test"
)


# ============================================================
# COMBINE METADATA
# ============================================================

all_sequences = (
    train_sequences +
    val_sequences +
    test_sequences
)

sequence_df = pd.DataFrame(
    all_sequences
)

sequence_df.to_csv(
    REPORT_DIR / "sequence_metadata_v3.csv",
    index=False
)

participant_counts = pd.DataFrame(
    train_counts +
    val_counts +
    test_counts
)

participant_counts.to_csv(
    REPORT_DIR /
    "participant_sequence_counts_v3.csv",
    index=False
)


# ============================================================
# SEQUENCE INTEGRITY
# ============================================================

print("\n" + "=" * 70)
print("SEQUENCE INTEGRITY AUDIT")
print("=" * 70)

if len(sequence_df) == 0:
    raise RuntimeError(
        "No sequences were generated."
    )

invalid_lengths = (
    sequence_df["sequence_length"] !=
    SEQUENCE_LENGTH
).sum()

invalid_participants = (
    ~sequence_df["participant_id"].isin(
        train_ids |
        val_ids |
        test_ids
    )
).sum()

print(
    f"Total sequences       : "
    f"{len(sequence_df):,}"
)

print(
    f"Invalid sequence len  : "
    f"{invalid_lengths}"
)

print(
    f"Unknown participants  : "
    f"{invalid_participants}"
)

if invalid_lengths != 0:
    raise RuntimeError(
        "Invalid sequence lengths detected."
    )

if invalid_participants != 0:
    raise RuntimeError(
        "Unknown participant IDs detected."
    )

print("Sequence integrity : PASS")


# ============================================================
# PARTICIPANT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PARTICIPANT / SEQUENCE DISTRIBUTION")
print("=" * 70)

for split_name in [
    "train",
    "validation",
    "test"
]:

    subset = participant_counts[
        participant_counts["split"] ==
        split_name
    ]

    print(f"\n{split_name.upper()}")

    print(
        f"Participants : {len(subset):,}"
    )

    print(
        f"Mean seq/user: "
        f"{subset['sequences'].mean():.2f}"
    )

    print(
        f"Median       : "
        f"{subset['sequences'].median():.2f}"
    )

    print(
        f"Minimum      : "
        f"{subset['sequences'].min()}"
    )

    print(
        f"Maximum      : "
        f"{subset['sequences'].max()}"
    )

    print(
        f"Total seq    : "
        f"{subset['sequences'].sum():,}"
    )


# ============================================================
# SAVE DATASET SUMMARY
# ============================================================

elapsed = time.time() - start_time

summary = {

    "random_seed": RANDOM_SEED,

    "source_cohort": str(QUALITY_FILE),

    "participants": {
        "total": TOTAL_PARTICIPANTS,
        "train": len(train_ids),
        "validation": len(val_ids),
        "test": len(test_ids)
    },

    "features": {
        "count": len(FEATURE_COLUMNS),
        "sequence_length": SEQUENCE_LENGTH
    },

    "sequences": {
        "train": train_total,
        "validation": val_total,
        "test": test_total,
        "total": (
            train_total +
            val_total +
            test_total
        )
    },

    "keystrokes_represented": {
        "train": train_keystrokes,
        "validation": val_keystrokes,
        "test": test_keystrokes
    },

    "invalid_values": {
        "train_nan": train_nan,
        "train_inf": train_inf,
        "validation_nan": val_nan,
        "validation_inf": val_inf,
        "test_nan": test_nan,
        "test_inf": test_inf
    },

    "temporal_reference": "PRESS_TIME",

    "participant_leakage": {
        "train_validation": len(train_val),
        "train_test": len(train_test),
        "validation_test": len(val_test)
    },

    "processing_seconds": elapsed,

    "status": "PASS"
}

with open(
    REPORT_DIR /
    "dataset_construction_summary_v3.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("AALTO V3 MODEL DATASET CONSTRUCTION COMPLETE")
print("=" * 70)

print(
    f"\nTrain participants      : "
    f"{len(train_ids):,}"
)

print(
    f"Validation participants : "
    f"{len(val_ids):,}"
)

print(
    f"Test participants       : "
    f"{len(test_ids):,}"
)

print(
    f"\nTrain sequences         : "
    f"{train_total:,}"
)

print(
    f"Validation sequences    : "
    f"{val_total:,}"
)

print(
    f"Test sequences          : "
    f"{test_total:,}"
)

print(
    f"Total sequences         : "
    f"{len(sequence_df):,}"
)

print(
    f"\nNaN values              : "
    f"{train_nan + val_nan + test_nan:,}"
)

print(
    f"Inf values              : "
    f"{train_inf + val_inf + test_inf:,}"
)

print(
    f"Processing time         : "
    f"{elapsed / 60:.2f} minutes"
)

print("\nParticipant leakage : PASS")
print("Temporal reference  : PRESS_TIME")
print("Dataset status      : PASS")

print("\nOutputs:")
print(f"Train       : {TRAIN_DIR}")
print(f"Validation  : {VAL_DIR}")
print(f"Test        : {TEST_DIR}")
print(f"Reports     : {REPORT_DIR}")
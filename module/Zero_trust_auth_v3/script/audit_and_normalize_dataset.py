from pathlib import Path
import numpy as np
import pandas as pd
import json
import time
import joblib

from sklearn.preprocessing import RobustScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("processed/sequence_dataset")
OUTPUT_DIR = DATA_DIR / "normalized"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
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

N_FEATURES = len(FEATURE_NAMES)
SEQUENCE_LENGTH = 50

TRAIN_FILE = DATA_DIR / "train_sequences.npz"
VAL_FILE = DATA_DIR / "validation_sequences.npz"
TEST_FILE = DATA_DIR / "test_sequences.npz"

SPLIT_FILE = DATA_DIR / "participant_splits.csv"

SEED = 42

start_time = time.time()


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AALTO DATASET — LEAKAGE AUDIT + NORMALIZATION")
print("=" * 70)

print(f"Dataset directory : {DATA_DIR}")
print(f"Features           : {N_FEATURES}")
print(f"Sequence length    : {SEQUENCE_LENGTH}")
print("Scaler             : RobustScaler")
print("Scaler fit         : TRAIN ONLY")
print(f"Random seed        : {SEED}")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nPHASE 1 — LOADING DATA")

train_data = np.load(TRAIN_FILE)
val_data = np.load(VAL_FILE)
test_data = np.load(TEST_FILE)

X_train = train_data["X"].astype(np.float32)
y_train = train_data["participant_id"]

X_val = val_data["X"].astype(np.float32)
y_val = val_data["participant_id"]

X_test = test_data["X"].astype(np.float32)
y_test = test_data["participant_id"]

print(f"Train X : {X_train.shape}")
print(f"Val X   : {X_val.shape}")
print(f"Test X  : {X_test.shape}")


# ============================================================
# BASIC STRUCTURE AUDIT
# ============================================================

print("\nPHASE 2 — STRUCTURAL AUDIT")

expected_train = (50, N_FEATURES)
expected_val = (50, N_FEATURES)
expected_test = (50, N_FEATURES)

assert X_train.shape[1:] == expected_train
assert X_val.shape[1:] == expected_val
assert X_test.shape[1:] == expected_test

print("✓ Sequence dimensions valid")
print("✓ Feature count valid")
print("✓ Sequence length valid")


# ============================================================
# PARTICIPANT LEAKAGE AUDIT
# ============================================================

print("\nPHASE 3 — PARTICIPANT LEAKAGE AUDIT")

train_users = set(y_train.tolist())
val_users = set(y_val.tolist())
test_users = set(y_test.tolist())

train_val_overlap = train_users & val_users
train_test_overlap = train_users & test_users
val_test_overlap = val_users & test_users

print(f"Unique train participants : {len(train_users):,}")
print(f"Unique val participants   : {len(val_users):,}")
print(f"Unique test participants  : {len(test_users):,}")

print(
    f"Train ↔ Val overlap       : "
    f"{len(train_val_overlap)}"
)

print(
    f"Train ↔ Test overlap      : "
    f"{len(train_test_overlap)}"
)

print(
    f"Val ↔ Test overlap        : "
    f"{len(val_test_overlap)}"
)

if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):
    raise RuntimeError(
        "CRITICAL: Participant leakage detected!"
    )

print("✓ NO PARTICIPANT LEAKAGE")


# ============================================================
# SEQUENCE DUPLICATE AUDIT
# ============================================================

print("\nPHASE 4 — SEQUENCE DUPLICATE AUDIT")

# Exact duplicate detection is performed using hashes.
# This avoids storing giant tuple objects.

def sequence_hashes(X):
    hashes = set()

    for seq in X:
        hashes.add(hash(seq.tobytes()))

    return hashes


train_hashes = sequence_hashes(X_train)

val_hashes = sequence_hashes(X_val)
test_hashes = sequence_hashes(X_test)

train_val_duplicate = len(
    train_hashes & val_hashes
)

train_test_duplicate = len(
    train_hashes & test_hashes
)

val_test_duplicate = len(
    val_hashes & test_hashes
)

print(
    f"Exact train-val duplicates : "
    f"{train_val_duplicate}"
)

print(
    f"Exact train-test duplicates : "
    f"{train_test_duplicate}"
)

print(
    f"Exact val-test duplicates : "
    f"{val_test_duplicate}"
)

if (
    train_val_duplicate
    or train_test_duplicate
    or val_test_duplicate
):
    print(
        "WARNING: Exact sequence duplicates detected."
    )
else:
    print("✓ NO EXACT CROSS-SPLIT SEQUENCE DUPLICATES")


# ============================================================
# NaN / INF AUDIT
# ============================================================

print("\nPHASE 5 — NUMERICAL AUDIT")

for name, X in [
    ("TRAIN", X_train),
    ("VALIDATION", X_val),
    ("TEST", X_test),
]:

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    print(
        f"{name:12s} "
        f"NaN={nan_count:,} "
        f"INF={inf_count:,}"
    )

    if nan_count > 0 or inf_count > 0:
        raise RuntimeError(
            f"Invalid numerical values detected in {name}"
        )

print("✓ Numerical values valid")


# ============================================================
# FIT ROBUST SCALER
# ============================================================

print("\nPHASE 6 — FITTING TRAIN-ONLY ROBUST SCALER")

# Flatten only the temporal dimension.
#
# Shape:
#   (samples, 50, 16)
#
# becomes:
#   (samples * 50, 16)

train_flat = X_train.reshape(
    -1,
    N_FEATURES
)

print(
    f"Scaler fitting matrix: "
    f"{train_flat.shape}"
)

scaler = RobustScaler(
    quantile_range=(25.0, 75.0)
)

scaler.fit(train_flat)

print("✓ RobustScaler fitted using TRAIN ONLY")


# ============================================================
# TRANSFORM ALL SPLITS
# ============================================================

print("\nPHASE 7 — NORMALIZATION")

def transform_sequences(X, scaler):

    original_shape = X.shape

    flat = X.reshape(
        -1,
        N_FEATURES
    )

    transformed = scaler.transform(flat)

    transformed = transformed.reshape(
        original_shape
    )

    return transformed.astype(
        np.float32
    )


X_train_norm = transform_sequences(
    X_train,
    scaler
)

X_val_norm = transform_sequences(
    X_val,
    scaler
)

X_test_norm = transform_sequences(
    X_test,
    scaler
)

print(
    f"Normalized train : {X_train_norm.shape}"
)

print(
    f"Normalized val   : {X_val_norm.shape}"
)

print(
    f"Normalized test  : {X_test_norm.shape}"
)


# ============================================================
# SAVE NORMALIZED DATA
# ============================================================

print("\nPHASE 8 — SAVING")

np.savez_compressed(
    OUTPUT_DIR / "train_normalized.npz",
    X=X_train_norm,
    participant_id=y_train,
)

np.savez_compressed(
    OUTPUT_DIR / "validation_normalized.npz",
    X=X_val_norm,
    participant_id=y_val,
)

np.savez_compressed(
    OUTPUT_DIR / "test_normalized.npz",
    X=X_test_norm,
    participant_id=y_test,
)

joblib.dump(
    scaler,
    OUTPUT_DIR / "robust_scaler.joblib"
)


# ============================================================
# NORMALIZATION CHECK
# ============================================================

print("\nPHASE 9 — NORMALIZATION CHECK")

train_norm_flat = X_train_norm.reshape(
    -1,
    N_FEATURES
)

val_norm_flat = X_val_norm.reshape(
    -1,
    N_FEATURES
)

test_norm_flat = X_test_norm.reshape(
    -1,
    N_FEATURES
)

train_medians = np.median(
    train_norm_flat,
    axis=0
)

train_iqrs = (
    np.percentile(
        train_norm_flat,
        75,
        axis=0
    )
    -
    np.percentile(
        train_norm_flat,
        25,
        axis=0
    )
)

normalization_report = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "train_median": train_medians,
    "train_IQR": train_iqrs,
})

print(normalization_report.to_string(index=False))


# ============================================================
# SAVE AUDIT REPORT
# ============================================================

summary = {
    "train_sequences": int(len(X_train)),
    "validation_sequences": int(len(X_val)),
    "test_sequences": int(len(X_test)),

    "train_participants": int(len(train_users)),
    "validation_participants": int(len(val_users)),
    "test_participants": int(len(test_users)),

    "train_validation_participant_overlap": int(
        len(train_val_overlap)
    ),

    "train_test_participant_overlap": int(
        len(train_test_overlap)
    ),

    "validation_test_participant_overlap": int(
        len(val_test_overlap)
    ),

    "train_validation_exact_duplicates": int(
        train_val_duplicate
    ),

    "train_test_exact_duplicates": int(
        train_test_duplicate
    ),

    "validation_test_exact_duplicates": int(
        val_test_duplicate
    ),

    "num_features": N_FEATURES,
    "sequence_length": SEQUENCE_LENGTH,
    "scaler": "RobustScaler",
    "scaler_fit_split": "train_only",
    "random_seed": SEED,
}

with open(
    OUTPUT_DIR / "normalization_audit.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

normalization_report.to_csv(
    OUTPUT_DIR / "normalization_statistics.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

elapsed = time.time() - start_time

print("\n" + "=" * 70)
print("LEAKAGE AUDIT + NORMALIZATION COMPLETE")
print("=" * 70)

print(
    f"Train participants : {len(train_users):,}"
)

print(
    f"Val participants   : {len(val_users):,}"
)

print(
    f"Test participants  : {len(test_users):,}"
)

print("Participant leakage : 0")
print("Scaler leakage      : 0")
print(
    f"Processing time     : "
    f"{elapsed / 60:.2f} minutes"
)

print("\nOutput directory:")
print(OUTPUT_DIR)
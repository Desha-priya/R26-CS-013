import numpy as np
from pathlib import Path

DATA_DIR = Path(r"processed\sequence_dataset\normalized")

files = {
    "TRAIN": "train_normalized.npz",
    "VALIDATION": "validation_normalized.npz",
    "TEST": "test_normalized.npz"
}

datasets = {}

print("=" * 75)
print("FINAL PARTICIPANT + SEQUENCE INTEGRITY AUDIT")
print("=" * 75)

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

for split, filename in files.items():

    data = np.load(DATA_DIR / filename)

    X = data["X"]
    participant_id = data["participant_id"]

    datasets[split] = participant_id

    unique_ids, counts = np.unique(
        participant_id,
        return_counts=True
    )

    print(f"\n{split}")
    print("-" * 75)

    print(f"Sequences           : {len(participant_id):,}")
    print(f"Unique participants : {len(unique_ids):,}")

    print(f"Min sequences/user  : {counts.min()}")
    print(f"Max sequences/user  : {counts.max()}")
    print(f"Mean sequences/user : {counts.mean():.2f}")
    print(f"Median sequences/user: {np.median(counts):.2f}")

    print(f"X shape             : {X.shape}")

# ---------------------------------------------------------
# CROSS-SPLIT LEAKAGE
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("CROSS-SPLIT PARTICIPANT LEAKAGE")
print("=" * 75)

train_ids = set(datasets["TRAIN"])
val_ids   = set(datasets["VALIDATION"])
test_ids  = set(datasets["TEST"])

train_val = train_ids & val_ids
train_test = train_ids & test_ids
val_test = val_ids & test_ids

print(f"Train ∩ Validation : {len(train_val)}")
print(f"Train ∩ Test       : {len(train_test)}")
print(f"Validation ∩ Test  : {len(val_test)}")

# ---------------------------------------------------------
# EXPECTED PARTICIPANT COUNTS
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("EXPECTED PARTICIPANT COUNTS")
print("=" * 75)

print(f"Train expected      : 3500")
print(f"Validation expected : 750")
print(f"Test expected       : 750")

print(f"\nTrain actual        : {len(train_ids)}")
print(f"Validation actual   : {len(val_ids)}")
print(f"Test actual         : {len(test_ids)}")

# ---------------------------------------------------------
# OVERALL
# ---------------------------------------------------------

if (
    len(train_val) == 0
    and len(train_test) == 0
    and len(val_test) == 0
    and len(train_ids) == 3500
    and len(val_ids) == 750
    and len(test_ids) == 750
):
    print("\n" + "=" * 75)
    print("✓ DATASET INTEGRITY CHECK PASSED")
    print("=" * 75)
else:
    print("\n" + "=" * 75)
    print("⚠ DATASET INTEGRITY ISSUE DETECTED")
    print("=" * 75)
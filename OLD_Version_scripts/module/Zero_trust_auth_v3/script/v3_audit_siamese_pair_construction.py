import os
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("processed/siamese_bilstm_v3")

TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "validation"
TEST_DIR = BASE_DIR / "test"

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "pair_construction_audit"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SEQUENCE_LENGTH = 50
EXPECTED_FEATURES = 16

RANDOM_SEED = 42


# ============================================================
# NPZ DISCOVERY
# ============================================================

def discover_npz(directory):

    files = sorted(directory.glob("*.npz"))

    if not files:
        raise RuntimeError(
            f"No NPZ files found in:\n{directory}"
        )

    return files


# ============================================================
# NPZ INSPECTION
# ============================================================

def inspect_npz(path):

    data = np.load(path, allow_pickle=False)

    result = {
        "path": str(path),
        "keys": list(data.keys()),
        "arrays": {}
    }

    for key in data.keys():

        arr = data[key]

        result["arrays"][key] = {
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "ndim": arr.ndim,
            "size": int(arr.size)
        }

    return result


# ============================================================
# PARTICIPANT EXTRACTION
# ============================================================

def participant_from_filename(path):

    """
    Expected individual sequence format:

        100032_section_1091453_seq_0.npz

    Participant ID = first component.
    """

    name = path.stem

    parts = name.split("_")

    if len(parts) >= 1:

        try:
            return int(parts[0])
        except ValueError:
            pass

    return None


# ============================================================
# SEQUENCE IDENTITY HASH
# ============================================================

def sequence_hash(path):

    data = np.load(path, allow_pickle=False)

    arrays = []

    for key in sorted(data.keys()):

        arr = np.asarray(data[key])

        if arr.dtype.kind in "fiu":
            arrays.append(
                np.ascontiguousarray(arr).tobytes()
            )

    if not arrays:
        return None

    digest = hashlib.sha256()

    for raw in arrays:
        digest.update(raw)

    return digest.hexdigest()


# ============================================================
# INDIVIDUAL DATASET AUDIT
# ============================================================

def audit_split(directory, split_name):

    files = discover_npz(directory)

    participants = set()
    participant_sequences = {}

    invalid_shape = 0
    invalid_values = 0

    hashes = {}
    duplicate_sequences = 0

    print()
    print("=" * 70)
    print(f"{split_name.upper()} SEQUENCE AUDIT")
    print("=" * 70)

    print(f"NPZ files : {len(files):,}")

    for i, path in enumerate(files, 1):

        participant_id = participant_from_filename(path)

        if participant_id is None:
            print(
                f"WARNING: Could not determine participant "
                f"from {path.name}"
            )
            continue

        participants.add(participant_id)

        participant_sequences.setdefault(
            participant_id,
            []
        ).append(path.name)

        data = np.load(path, allow_pickle=False)

        numeric_arrays = []

        for key in data.keys():

            arr = np.asarray(data[key])

            if arr.ndim == 2 and arr.shape == (
                EXPECTED_SEQUENCE_LENGTH,
                EXPECTED_FEATURES
            ):
                numeric_arrays.append(arr)

        if not numeric_arrays:

            invalid_shape += 1

        else:

            X = numeric_arrays[0]

            if not np.all(np.isfinite(X)):
                invalid_values += 1

            # Duplicate detection
            digest = hashlib.sha256(
                np.ascontiguousarray(X).tobytes()
            ).hexdigest()

            if digest in hashes:
                duplicate_sequences += 1
            else:
                hashes[digest] = path.name

        if i % 5000 == 0 or i == len(files):

            print(
                f"Processed "
                f"{i:,}/{len(files):,}",
                flush=True
            )

    sequence_counts = [
        len(v)
        for v in participant_sequences.values()
    ]

    print()
    print("PARTICIPANTS")
    print("-" * 70)

    print(
        f"Participants : "
        f"{len(participants):,}"
    )

    print(
        f"Mean seq/user: "
        f"{np.mean(sequence_counts):.2f}"
    )

    print(
        f"Median       : "
        f"{np.median(sequence_counts):.2f}"
    )

    print(
        f"Minimum      : "
        f"{np.min(sequence_counts)}"
    )

    print(
        f"Maximum      : "
        f"{np.max(sequence_counts)}"
    )

    print()
    print("INTEGRITY")
    print("-" * 70)

    print(
        f"Invalid shape        : "
        f"{invalid_shape:,}"
    )

    print(
        f"Invalid values       : "
        f"{invalid_values:,}"
    )

    print(
        f"Duplicate sequences  : "
        f"{duplicate_sequences:,}"
    )

    return {
        "split": split_name,
        "files": len(files),
        "participants": participants,
        "participant_sequences": participant_sequences,
        "invalid_shape": invalid_shape,
        "invalid_values": invalid_values,
        "duplicate_sequences": duplicate_sequences
    }


# ============================================================
# PARTICIPANT LEAKAGE
# ============================================================

def participant_leakage_check(train, validation, test):

    train_ids = train["participants"]
    val_ids = validation["participants"]
    test_ids = test["participants"]

    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids

    print()
    print("=" * 70)
    print("PARTICIPANT LEAKAGE AUDIT")
    print("=" * 70)

    print(
        f"Train ∩ Validation : "
        f"{len(train_val):,}"
    )

    print(
        f"Train ∩ Test       : "
        f"{len(train_test):,}"
    )

    print(
        f"Validation ∩ Test  : "
        f"{len(val_test):,}"
    )

    passed = (
        len(train_val) == 0
        and len(train_test) == 0
        and len(val_test) == 0
    )

    print()

    if passed:
        print("Participant leakage : PASS")
    else:
        print("Participant leakage : FAIL")

    return {
        "train_validation": len(train_val),
        "train_test": len(train_test),
        "validation_test": len(val_test),
        "pass": passed
    }


# ============================================================
# PAIR FILE DISCOVERY
# ============================================================

def discover_pair_files():

    """
    Look for likely pair-generation outputs.

    We intentionally DO NOT assume that they exist.
    """

    candidate_dirs = [

        BASE_DIR / "pairs",
        BASE_DIR / "pair_data",
        BASE_DIR / "siamese_pairs",
        BASE_DIR / "training_pairs",
        BASE_DIR / "validation_pairs",
        BASE_DIR / "test_pairs",

        Path("processed/siamese_bilstm_v3/pairs"),
        Path("processed/siamese_pairs_v3"),

        Path("reports/siamese_pairs_v3"),
    ]

    found = []

    for directory in candidate_dirs:

        if directory.exists():

            for pattern in (
                "*.npz",
                "*.csv",
                "*.json"
            ):

                found.extend(
                    directory.glob(pattern)
                )

    return sorted(set(found))


# ============================================================
# PAIR FILE STRUCTURE INSPECTION
# ============================================================

def inspect_pair_candidates(files):

    print()
    print("=" * 70)
    print("PAIR OUTPUT DISCOVERY")
    print("=" * 70)

    if not files:

        print(
            "No existing pair-generation output was found."
        )

        print()
        print(
            "This is EXPECTED if pairs have not been generated yet."
        )

        return {
            "pair_files_found": 0,
            "pair_files": []
        }

    print(
        f"Candidate pair files : "
        f"{len(files):,}"
    )

    reports = []

    for path in files[:100]:

        print()
        print(f"FILE: {path}")

        if path.suffix.lower() == ".npz":

            try:

                result = inspect_npz(path)

                for key, info in result["arrays"].items():

                    print(
                        f"  {key}: "
                        f"shape={info['shape']} "
                        f"dtype={info['dtype']}"
                    )

                reports.append(result)

            except Exception as e:

                print(
                    f"  ERROR: {e}"
                )

        elif path.suffix.lower() == ".csv":

            try:

                df = pd.read_csv(path, nrows=5)

                print(
                    f"  Columns: "
                    f"{list(df.columns)}"
                )

                print(
                    f"  Sample rows: "
                    f"{len(df)}"
                )

            except Exception as e:

                print(
                    f"  ERROR: {e}"
                )

        elif path.suffix.lower() == ".json":

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    obj = json.load(f)

                if isinstance(obj, dict):

                    print(
                        f"  JSON keys: "
                        f"{list(obj.keys())}"
                    )

                elif isinstance(obj, list):

                    print(
                        f"  JSON list length: "
                        f"{len(obj)}"
                    )

            except Exception as e:

                print(
                    f"  ERROR: {e}"
                )

    return {
        "pair_files_found": len(files),
        "pair_files": [
            str(x) for x in files
        ],
        "sample_inspections": reports
    }


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print("=" * 70)
    print("SIAMESE PAIR-CONSTRUCTION AUDIT V3")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This script DOES NOT modify the dataset."
    )

    print(
        "It audits the existing sequence dataset "
        "and searches for existing pair outputs."
    )

    # --------------------------------------------------------
    # AUDIT TRAIN
    # --------------------------------------------------------

    train = audit_split(
        TRAIN_DIR,
        "train"
    )

    # --------------------------------------------------------
    # AUDIT VALIDATION
    # --------------------------------------------------------

    validation = audit_split(
        VAL_DIR,
        "validation"
    )

    # --------------------------------------------------------
    # AUDIT TEST
    # --------------------------------------------------------

    test = audit_split(
        TEST_DIR,
        "test"
    )

    # --------------------------------------------------------
    # LEAKAGE
    # --------------------------------------------------------

    leakage = participant_leakage_check(
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # PAIR DISCOVERY
    # --------------------------------------------------------

    pair_files = discover_pair_files()

    pair_report = inspect_pair_candidates(
        pair_files
    )

    # --------------------------------------------------------
    # DATASET TOTALS
    # --------------------------------------------------------

    total_sequences = (
        train["files"]
        + validation["files"]
        + test["files"]
    )

    total_invalid_shape = (
        train["invalid_shape"]
        + validation["invalid_shape"]
        + test["invalid_shape"]
    )

    total_invalid_values = (
        train["invalid_values"]
        + validation["invalid_values"]
        + test["invalid_values"]
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    sequence_integrity_pass = (
        total_invalid_shape == 0
        and total_invalid_values == 0
    )

    overall_pass = (
        sequence_integrity_pass
        and leakage["pass"]
    )

    print()
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(
        f"Total sequences : "
        f"{total_sequences:,}"
    )

    print(
        f"Invalid shape   : "
        f"{total_invalid_shape:,}"
    )

    print(
        f"Invalid values  : "
        f"{total_invalid_values:,}"
    )

    print()

    print(
        "Sequence integrity : "
        + (
            "PASS"
            if sequence_integrity_pass
            else "FAIL"
        )
    )

    print(
        "Participant leakage : "
        + (
            "PASS"
            if leakage["pass"]
            else "FAIL"
        )
    )

    print()

    if pair_report["pair_files_found"] == 0:

        print(
            "Pair construction status:"
        )

        print(
            "NOT YET GENERATED / NOT FOUND"
        )

        print()
        print(
            "This is NOT a failure."
        )

        print(
            "The next step will be to generate "
            "the Siamese pairs under a controlled protocol."
        )

    else:

        print(
            "Pair output files found:"
        )

        print(
            f"{pair_report['pair_files_found']:,}"
        )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    summary = {

        "dataset": "AALTO_V3",

        "expected_sequence_length":
            EXPECTED_SEQUENCE_LENGTH,

        "expected_features":
            EXPECTED_FEATURES,

        "sequence_counts": {
            "train": train["files"],
            "validation": validation["files"],
            "test": test["files"],
            "total": total_sequences
        },

        "sequence_integrity": {
            "invalid_shape":
                total_invalid_shape,
            "invalid_values":
                total_invalid_values,
            "pass":
                sequence_integrity_pass
        },

        "participant_leakage": leakage,

        "pair_output": {
            "files_found":
                pair_report["pair_files_found"],
            "files":
                pair_report["pair_files"]
        },

        "overall_audit_pass":
            overall_pass,

        "processing_seconds":
            time.time() - start
    }

    summary_path = (
        REPORT_DIR
        / "pair_construction_audit_summary_v3.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # PARTICIPANT DISTRIBUTION CSV
    # --------------------------------------------------------

    rows = []

    for split_data in (
        train,
        validation,
        test
    ):

        split_name = split_data["split"]

        for participant_id, sequences in (
            split_data["participant_sequences"].items()
        ):

            rows.append({
                "split": split_name,
                "participant_id": participant_id,
                "sequence_count": len(sequences)
            })

    distribution_path = (
        REPORT_DIR
        / "participant_sequence_distribution_v3.csv"
    )

    pd.DataFrame(rows).to_csv(
        distribution_path,
        index=False
    )

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"Summary      : {summary_path}"
    )

    print(
        f"Distribution : {distribution_path}"
    )

    print()
    print(
        f"Processing time : "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
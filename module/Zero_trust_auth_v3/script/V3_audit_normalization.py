import os
import json
import glob
import time
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("processed/siamese_bilstm_v3")

TRAIN_DIR = BASE_DIR / "train"
VAL_DIR   = BASE_DIR / "validation"
TEST_DIR  = BASE_DIR / "test"

REPORT_DIR = BASE_DIR / "reports" / "normalization_audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SEQ_LEN = 50
EXPECTED_FEATURES = 16

EPS = 1e-8

# ============================================================
# HELPERS
# ============================================================

def find_npz_files(directory):
    return sorted(directory.glob("*.npz"))


def load_npz(path):
    data = np.load(path)

    # Detect the sequence array without assuming an exact key.
    if "X" in data:
        X = data["X"]
    elif "features" in data:
        X = data["features"]
    elif "sequence" in data:
        X = data["sequence"]
    else:
        keys = list(data.keys())

        arrays = []
        for key in keys:
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                arrays.append((key, arr))

        if not arrays:
            raise ValueError(f"No suitable array found in {path}")

        # Prefer 3D sequence arrays.
        three_d = [(k, a) for k, a in arrays if a.ndim == 3]

        if three_d:
            X = three_d[0][1]
        else:
            X = arrays[0][1]

    return np.asarray(X, dtype=np.float64)


def inspect_shape(X, path):
    if X.ndim != 2:
        raise ValueError(
            f"Invalid dimensionality in {path}: "
            f"expected 2D (50, 16), got shape {X.shape}"
        )

    if X.shape[0] != EXPECTED_SEQ_LEN:
        raise ValueError(
            f"Invalid sequence length in {path}: "
            f"expected {EXPECTED_SEQ_LEN}, got {X.shape[0]}"
        )

    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Invalid feature count in {path}: "
            f"expected {EXPECTED_FEATURES}, got {X.shape[1]}"
        )


def collect_training_statistics(files):
    """
    Calculate mean/std using TRAIN ONLY.
    Statistics are calculated feature-wise across
    participant x sequence x timestep.
    """

    feature_sum = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
    feature_sq_sum = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
    feature_count = np.zeros(EXPECTED_FEATURES, dtype=np.int64)

    total_sequences = 0

    for i, path in enumerate(files, 1):

        X = load_npz(path)
        inspect_shape(X, path)

        if not np.all(np.isfinite(X)):
            raise ValueError(
                f"Non-finite value found in training file: {path}"
            )

        flat = X

        feature_sum += np.sum(flat, axis=0)
        feature_sq_sum += np.sum(flat ** 2, axis=0)
        feature_count += flat.shape[0]

        total_sequences += 1

        if i % 1000 == 0 or i == len(files):
            print(
                f"TRAIN statistics: "
                f"{i:,}/{len(files):,}",
                flush=True
            )

    mean = feature_sum / feature_count

    variance = (
        feature_sq_sum / feature_count
        - mean ** 2
    )

    variance = np.maximum(variance, 0.0)

    std = np.sqrt(variance)

    constant_features = std < EPS

    return {
        "mean": mean,
        "std": std,
        "count": feature_count,
        "constant_features": constant_features,
        "total_sequences": total_sequences
    }


def audit_split(files, mean, std, split_name):

    total_sequences = 0
    total_values = 0

    raw_min = np.full(EXPECTED_FEATURES, np.inf)
    raw_max = np.full(EXPECTED_FEATURES, -np.inf)

    normalized_min = np.full(EXPECTED_FEATURES, np.inf)
    normalized_max = np.full(EXPECTED_FEATURES, -np.inf)

    normalized_sum = np.zeros(EXPECTED_FEATURES)
    normalized_sq_sum = np.zeros(EXPECTED_FEATURES)

    nan_values = 0
    inf_values = 0
    extreme_values = 0

    # Values with |z| > 10
    extreme_threshold = 10.0

    for i, path in enumerate(files, 1):

        X = load_npz(path)
        inspect_shape(X, path)

        total_sequences += 1
        total_values += X.size

        nan_values += int(np.isnan(X).sum())
        inf_values += int(np.isinf(X).sum())

        if not np.all(np.isfinite(X)):
            raise ValueError(
                f"Non-finite raw value found in {split_name}: {path}"
            )

        flat = X

        raw_min = np.minimum(
            raw_min,
            np.min(flat, axis=0)
        )

        raw_max = np.maximum(
            raw_max,
            np.max(flat, axis=0)
        )

        normalized = (flat - mean) / np.maximum(std, EPS)

        normalized_min = np.minimum(
            normalized_min,
            np.min(normalized, axis=0)
        )

        normalized_max = np.maximum(
            normalized_max,
            np.max(normalized, axis=0)
        )

        normalized_sum += np.sum(normalized, axis=0)
        normalized_sq_sum += np.sum(normalized ** 2, axis=0)

        extreme_values += int(
            np.sum(np.abs(normalized) > extreme_threshold)
        )

        if i % 1000 == 0 or i == len(files):
            print(
                f"{split_name.upper()}: "
                f"{i:,}/{len(files):,}",
                flush=True
            )

    normalized_mean = normalized_sum / (
        total_sequences * EXPECTED_SEQ_LEN
    )

    normalized_variance = (
        normalized_sq_sum /
        (total_sequences * EXPECTED_SEQ_LEN)
        - normalized_mean ** 2
    )

    normalized_variance = np.maximum(
        normalized_variance,
        0.0
    )

    normalized_std = np.sqrt(normalized_variance)

    return {
        "split": split_name,
        "files": len(files),
        "sequences": total_sequences,
        "values": total_values,
        "nan_values": nan_values,
        "inf_values": inf_values,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "normalized_min": normalized_min,
        "normalized_max": normalized_max,
        "normalized_mean": normalized_mean,
        "normalized_std": normalized_std,
        "extreme_values_abs_z_gt_10": extreme_values
    }


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print("=" * 70)
    print("NORMALIZATION / SCALING AUDIT V3")
    print("=" * 70)

    train_files = find_npz_files(TRAIN_DIR)
    val_files   = find_npz_files(VAL_DIR)
    test_files  = find_npz_files(TEST_DIR)

    print()
    print("NPZ FILE COUNTS")
    print("-" * 70)
    print(f"Train      : {len(train_files):,}")
    print(f"Validation : {len(val_files):,}")
    print(f"Test       : {len(test_files):,}")

    if not train_files:
        raise RuntimeError("No training NPZ files found.")

    if not val_files:
        raise RuntimeError("No validation NPZ files found.")

    if not test_files:
        raise RuntimeError("No test NPZ files found.")

    # --------------------------------------------------------
    # TRAIN-ONLY STATISTICS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 1: TRAIN-ONLY NORMALIZATION PARAMETERS")
    print("=" * 70)

    stats = collect_training_statistics(train_files)

    mean = stats["mean"]
    std = stats["std"]
    counts = stats["count"]
    constant = stats["constant_features"]

    print()
    print("TRAINING FEATURE PARAMETERS")
    print("-" * 70)

    for i in range(EXPECTED_FEATURES):
        print(
            f"Feature {i+1:02d}: "
            f"mean={mean[i]:.8f} | "
            f"std={std[i]:.8f} | "
            f"count={counts[i]:,} | "
            f"constant={bool(constant[i])}"
        )

    if np.any(constant):
        print()
        print("WARNING:")
        print(
            "One or more features have near-zero training variance."
        )
    else:
        print()
        print("All features have non-zero training variance.")

    # --------------------------------------------------------
    # AUDIT ALL SPLITS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 2: NORMALIZATION AUDIT")
    print("=" * 70)

    train_audit = audit_split(
        train_files,
        mean,
        std,
        "train"
    )

    val_audit = audit_split(
        val_files,
        mean,
        std,
        "validation"
    )

    test_audit = audit_split(
        test_files,
        mean,
        std,
        "test"
    )

    audits = [
        train_audit,
        val_audit,
        test_audit
    ]

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NORMALIZATION RESULTS")
    print("=" * 70)

    for result in audits:

        print()
        print(result["split"].upper())

        print(
            f"Sequences        : "
            f"{result['sequences']:,}"
        )

        print(
            f"NaN values       : "
            f"{result['nan_values']:,}"
        )

        print(
            f"Inf values       : "
            f"{result['inf_values']:,}"
        )

        print(
            f"|Z| > 10 values  : "
            f"{result['extreme_values_abs_z_gt_10']:,}"
        )

        print(
            f"Normalized mean range : "
            f"{np.min(result['normalized_mean']):.6f} "
            f"to "
            f"{np.max(result['normalized_mean']):.6f}"
        )

        print(
            f"Normalized std range  : "
            f"{np.min(result['normalized_std']):.6f} "
            f"to "
            f"{np.max(result['normalized_std']):.6f}"
        )

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NORMALIZATION LEAKAGE CHECK")
    print("=" * 70)

    print()
    print("Normalization parameters source:")
    print("    TRAIN ONLY")

    print()
    print("Validation used to calculate mean/std : NO")
    print("Test used to calculate mean/std        : NO")

    # --------------------------------------------------------
    # PASS / FAIL
    # --------------------------------------------------------

    no_invalid = all(
        r["nan_values"] == 0 and
        r["inf_values"] == 0
        for r in audits
    )

    no_constant = not np.any(constant)

    finite_normalized = all(
        np.all(np.isfinite(r["normalized_mean"])) and
        np.all(np.isfinite(r["normalized_std"]))
        for r in audits
    )

    passed = (
        no_invalid and
        no_constant and
        finite_normalized
    )

    print()
    print("=" * 70)

    if passed:
        print("NORMALIZATION AUDIT: PASS")
    else:
        print("NORMALIZATION AUDIT: REVIEW REQUIRED")

    print("=" * 70)

    # --------------------------------------------------------
    # SAVE SCALER
    # --------------------------------------------------------

    scaler_path = REPORT_DIR / "train_only_scaler_v3.npz"

    np.savez(
        scaler_path,
        mean=mean,
        std=std,
        feature_count=EXPECTED_FEATURES,
        sequence_length=EXPECTED_SEQ_LEN
    )

    # --------------------------------------------------------
    # SAVE JSON SUMMARY
    # --------------------------------------------------------

    summary = {
        "dataset": "AALTO_V3",

        "sequence_length": EXPECTED_SEQ_LEN,
        "features": EXPECTED_FEATURES,

        "normalization": {
            "method": "standardization",
            "formula": "(x - train_mean) / train_std",
            "statistics_source": "TRAIN_ONLY"
        },

        "leakage_control": {
            "validation_used_for_scaler": False,
            "test_used_for_scaler": False
        },

        "training_parameters": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "sample_counts": counts.tolist(),
            "constant_features": np.where(constant)[0].tolist()
        },

        "splits": []
    }

    for result in audits:

        summary["splits"].append({
            "split": result["split"],
            "files": result["files"],
            "sequences": result["sequences"],
            "values": result["values"],
            "nan_values": result["nan_values"],
            "inf_values": result["inf_values"],
            "extreme_values_abs_z_gt_10":
                result["extreme_values_abs_z_gt_10"],
            "normalized_mean":
                result["normalized_mean"].tolist(),
            "normalized_std":
                result["normalized_std"].tolist(),
            "normalized_min":
                result["normalized_min"].tolist(),
            "normalized_max":
                result["normalized_max"].tolist()
        })

    summary["audit_pass"] = bool(passed)

    summary["processing_seconds"] = time.time() - start

    summary_path = REPORT_DIR / "normalization_audit_summary_v3.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # FEATURE REPORT CSV
    # --------------------------------------------------------

    csv_path = REPORT_DIR / "feature_normalization_report_v3.csv"

    with open(csv_path, "w", encoding="utf-8") as f:

        f.write(
            "feature,"
            "train_mean,"
            "train_std,"
            "train_constant,"
            "train_norm_mean,"
            "train_norm_std,"
            "validation_norm_mean,"
            "validation_norm_std,"
            "test_norm_mean,"
            "test_norm_std\n"
        )

        for i in range(EXPECTED_FEATURES):

            f.write(
                f"{i+1},"
                f"{mean[i]},"
                f"{std[i]},"
                f"{bool(constant[i])},"
                f"{train_audit['normalized_mean'][i]},"
                f"{train_audit['normalized_std'][i]},"
                f"{val_audit['normalized_mean'][i]},"
                f"{val_audit['normalized_std'][i]},"
                f"{test_audit['normalized_mean'][i]},"
                f"{test_audit['normalized_std'][i]}\n"
            )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"Scaler                  : "
        f"{scaler_path}"
    )

    print(
        f"Summary                 : "
        f"{summary_path}"
    )

    print(
        f"Feature report          : "
        f"{csv_path}"
    )

    print()
    print(
        f"Processing time         : "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("=" * 70)

    if passed:
        print("READY FOR MODEL PREPROCESSING")
    else:
        print("DO NOT TRAIN YET — REVIEW AUDIT")

    print("=" * 70)


if __name__ == "__main__":
    main()
#1,340,675 NaN values after V3 sequcne pair
#check those 1.34M NaNs first


from pathlib import Path
import numpy as np
import pandas as pd
import json
import time

# ============================================================
# AALTO V3 — NaN / INVALID VALUE AUDIT
# ============================================================

DATASET_DIR = Path("processed/siamese_bilstm_v3")
REPORT_DIR = DATASET_DIR / "reports" / "nan_audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]

EXPECTED_FEATURES = 16
EXPECTED_SEQUENCE_LENGTH = 50

start_time = time.time()

print("=" * 70)
print("AALTO V3 NaN / INVALID VALUE AUDIT")
print("=" * 70)

print(f"Dataset directory : {DATASET_DIR}")
print(f"Expected features : {EXPECTED_FEATURES}")
print(f"Sequence length   : {EXPECTED_SEQUENCE_LENGTH}")

# ============================================================
# HELPERS
# ============================================================

def inspect_npz(file_path):

    result = {
        "file": str(file_path),
        "valid": True,
        "shape": None,
        "nan_count": 0,
        "inf_count": 0,
        "finite_count": 0,
        "total_values": 0,
        "nan_percentage": 0.0,
        "zero_count": 0,
        "constant_features": 0,
        "error": ""
    }

    try:

        data = np.load(file_path, allow_pickle=False)

        if "X" not in data:
            result["valid"] = False
            result["error"] = "missing_X_array"
            return result

        X = data["X"]

        result["shape"] = list(X.shape)

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        if X.ndim != 2:
            result["valid"] = False
            result["error"] = f"invalid_dimensions_{X.ndim}"

        elif X.shape[0] != EXPECTED_SEQUENCE_LENGTH:
            result["valid"] = False
            result["error"] = (
                f"invalid_sequence_length_{X.shape[0]}"
            )

        elif X.shape[1] != EXPECTED_FEATURES:
            result["valid"] = False
            result["error"] = (
                f"invalid_feature_count_{X.shape[1]}"
            )

        # ----------------------------------------------------
        # Numeric validity
        # ----------------------------------------------------

        X_float = X.astype(np.float64, copy=False)

        nan_mask = np.isnan(X_float)
        inf_mask = np.isinf(X_float)
        finite_mask = np.isfinite(X_float)

        nan_count = int(nan_mask.sum())
        inf_count = int(inf_mask.sum())
        finite_count = int(finite_mask.sum())
        total_values = int(X_float.size)

        result["nan_count"] = nan_count
        result["inf_count"] = inf_count
        result["finite_count"] = finite_count
        result["total_values"] = total_values

        if total_values > 0:
            result["nan_percentage"] = (
                nan_count / total_values * 100
            )

        # ----------------------------------------------------
        # Zero count
        # ----------------------------------------------------

        result["zero_count"] = int(
            np.sum(
                np.isfinite(X_float) &
                (X_float == 0)
            )
        )

        # ----------------------------------------------------
        # Constant feature count
        # ----------------------------------------------------

        constant_count = 0

        for feature_idx in range(X_float.shape[1]):

            column = X_float[:, feature_idx]

            finite_values = column[np.isfinite(column)]

            if len(finite_values) == 0:
                continue

            if np.nanmin(finite_values) == np.nanmax(finite_values):
                constant_count += 1

        result["constant_features"] = constant_count

        # ----------------------------------------------------
        # Any Inf = invalid
        # ----------------------------------------------------

        if inf_count > 0:
            result["valid"] = False
            result["error"] = "contains_inf"

        data.close()

    except Exception as e:

        result["valid"] = False
        result["error"] = f"read_error: {str(e)}"

    return result


# ============================================================
# PROCESS SPLITS
# ============================================================

all_records = []
split_summary = {}

for split in SPLITS:

    split_dir = DATASET_DIR / split

    files = sorted(split_dir.glob("*.npz"))

    print("\n" + "=" * 70)
    print(f"AUDITING {split.upper()}")
    print("=" * 70)

    print(f"NPZ files: {len(files):,}")

    split_records = []

    for i, file_path in enumerate(files, start=1):

        result = inspect_npz(file_path)

        result["split"] = split
        result["sequence_id"] = file_path.stem

        split_records.append(result)
        all_records.append(result)

        if i % 5000 == 0 or i == len(files):

            elapsed = time.time() - start_time

            print(
                f"Processed {i:,}/{len(files):,} | "
                f"elapsed={elapsed / 60:.2f} min"
            )

    # --------------------------------------------------------
    # Split totals
    # --------------------------------------------------------

    split_df = pd.DataFrame(split_records)

    total_values = int(
        split_df["total_values"].sum()
    )

    total_nan = int(
        split_df["nan_count"].sum()
    )

    total_inf = int(
        split_df["inf_count"].sum()
    )

    total_finite = int(
        split_df["finite_count"].sum()
    )

    invalid_sequences = int(
        (~split_df["valid"]).sum()
    )

    split_summary[split] = {
        "sequences": len(split_df),
        "total_values": total_values,
        "nan_values": total_nan,
        "inf_values": total_inf,
        "finite_values": total_finite,
        "nan_percentage": (
            total_nan / total_values * 100
            if total_values else 0
        ),
        "invalid_sequences": invalid_sequences
    }

    # Save split-level detailed report

    split_df.to_csv(
        REPORT_DIR /
        f"{split}_nan_sequence_report.csv",
        index=False
    )

    print(f"\n{split.upper()} SUMMARY")

    print(
        f"Sequences             : {len(split_df):,}"
    )

    print(
        f"Total values          : {total_values:,}"
    )

    print(
        f"NaN values            : {total_nan:,}"
    )

    print(
        f"NaN percentage        : "
        f"{total_nan / total_values * 100:.4f}%"
        if total_values
        else "NaN percentage        : 0%"
    )

    print(
        f"Inf values            : {total_inf:,}"
    )

    print(
        f"Finite values         : {total_finite:,}"
    )

    print(
        f"Invalid sequences     : {invalid_sequences:,}"
    )


# ============================================================
# COMBINE REPORT
# ============================================================

all_df = pd.DataFrame(all_records)

all_df.to_csv(
    REPORT_DIR / "all_nan_sequence_report.csv",
    index=False
)

# ============================================================
# GLOBAL TOTALS
# ============================================================

total_values = int(
    all_df["total_values"].sum()
)

total_nan = int(
    all_df["nan_count"].sum()
)

total_inf = int(
    all_df["inf_count"].sum()
)

total_finite = int(
    all_df["finite_count"].sum()
)

invalid_sequences = int(
    (~all_df["valid"]).sum()
)

total_sequences = len(all_df)

# ============================================================
# MOST AFFECTED SEQUENCES
# ============================================================

worst_sequences = (
    all_df
    .sort_values(
        ["nan_count", "nan_percentage"],
        ascending=False
    )
    .head(1000)
)

worst_sequences.to_csv(
    REPORT_DIR / "worst_1000_nan_sequences.csv",
    index=False
)

# ============================================================
# NAN DISTRIBUTION
# ============================================================

nan_distribution = (
    all_df["nan_count"]
    .value_counts()
    .sort_index()
    .reset_index()

)

nan_distribution.columns = [
    "nan_count",
    "sequences"
]

nan_distribution.to_csv(
    REPORT_DIR / "nan_count_distribution.csv",
    index=False
)

# ============================================================
# SEQUENCES WITH NO NaNs
# ============================================================

no_nan_sequences = all_df[
    all_df["nan_count"] == 0
].copy()

no_nan_sequences.to_csv(
    REPORT_DIR / "sequences_with_zero_nan.csv",
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

elapsed = time.time() - start_time

summary = {

    "dataset": "AALTO_V3",

    "expected_sequence_length":
        EXPECTED_SEQUENCE_LENGTH,

    "expected_features":
        EXPECTED_FEATURES,

    "total_sequences":
        total_sequences,

    "total_values":
        total_values,

    "nan_values":
        total_nan,

    "nan_percentage":
        (
            total_nan / total_values * 100
            if total_values
            else 0
        ),

    "inf_values":
        total_inf,

    "finite_values":
        total_finite,

    "invalid_sequences":
        invalid_sequences,

    "sequences_with_zero_nan":
        int(len(no_nan_sequences)),

    "sequences_with_nan":
        int(
            (all_df["nan_count"] > 0).sum()
        ),

    "maximum_nan_in_sequence":
        int(all_df["nan_count"].max())
        if len(all_df)
        else 0,

    "median_nan_per_sequence":
        float(all_df["nan_count"].median())
        if len(all_df)
        else 0,

    "mean_nan_per_sequence":
        float(all_df["nan_count"].mean())
        if len(all_df)
        else 0,

    "split_summary":
        split_summary,

    "processing_seconds":
        elapsed
}

with open(
    REPORT_DIR /
    "nan_audit_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("NAN / INVALID VALUE AUDIT COMPLETE")
print("=" * 70)

print(
    f"Total sequences          : {total_sequences:,}"
)

print(
    f"Total values             : {total_values:,}"
)

print(
    f"NaN values               : {total_nan:,}"
)

print(
    f"NaN percentage           : "
    f"{total_nan / total_values * 100:.4f}%"
    if total_values
    else "NaN percentage           : 0%"
)

print(
    f"Inf values               : {total_inf:,}"
)

print(
    f"Finite values            : {total_finite:,}"
)

print(
    f"Sequences with NaN       : "
    f"{(all_df['nan_count'] > 0).sum():,}"
)

print(
    f"Sequences without NaN    : "
    f"{len(no_nan_sequences):,}"
)

print(
    f"Invalid sequences        : {invalid_sequences:,}"
)

print(
    f"Maximum NaN/sequence     : "
    f"{all_df['nan_count'].max():,}"
)

print(
    f"Median NaN/sequence      : "
    f"{all_df['nan_count'].median():.1f}"
)

print(
    f"Mean NaN/sequence        : "
    f"{all_df['nan_count'].mean():.2f}"
)

print(
    f"Processing time          : "
    f"{elapsed / 60:.2f} minutes"
)

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"Summary                  : "
    f"{REPORT_DIR / 'nan_audit_summary.json'}"
)

print(
    f"All sequences            : "
    f"{REPORT_DIR / 'all_nan_sequence_report.csv'}"
)

print(
    f"Worst 1000               : "
    f"{REPORT_DIR / 'worst_1000_nan_sequences.csv'}"
)

print(
    f"Zero-NaN sequences       : "
    f"{REPORT_DIR / 'sequences_with_zero_nan.csv'}"
)

print(
    f"Distribution             : "
    f"{REPORT_DIR / 'nan_count_distribution.csv'}"
)

print("\nIMPORTANT:")
print(
    "This audit DOES NOT modify any dataset files."
)
print(
    "Do not perform imputation yet."
)
print(
    "Use the audit results to choose the preprocessing policy."
)

print("=" * 70)
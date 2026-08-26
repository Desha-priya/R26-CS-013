from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
import math

# ============================================================
# AALTO DEEP QUALITY SCREENING V2
# ============================================================
#
# PURPOSE:
#   Select the highest-quality 5,000 AALTO participants using
#   ONLY dataset-quality characteristics.
#
# IMPORTANT:
#   - No model performance is used.
#   - No validation/test metrics are used.
#   - No existing feature files are modified.
#   - Participant IDs are preserved.
#   - Selection is participant-level.
#
# INPUT:
#   processed/features/*_features.csv
#
# OUTPUT:
#   reports/deep_quality_v2/
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = Path("processed/features")

OUTPUT_DIR = Path("reports/deep_quality_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

TARGET_PARTICIPANTS = 5000

# Minimum participant-level requirements
MIN_SECTIONS = 15
MIN_KEYSTROKES = 600

# Minimum percentage of rows with usable feature values
MIN_FINITE_RATIO = 0.90

# Minimum feature variability threshold
MIN_FEATURE_VARIABILITY = 1e-8

# Maximum fraction of constant features allowed
MAX_CONSTANT_FEATURE_RATIO = 0.50

# Extreme timing threshold used ONLY for diagnostics.
# Values are not automatically deleted.
EXTREME_VALUE_THRESHOLD = 5000.0


# ============================================================
# FEATURES
# ============================================================

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

np.random.seed(RANDOM_SEED)

start_time = time.time()

print("=" * 78)
print("AALTO DEEP QUALITY SCREENING V2")
print("=" * 78)

print(f"Feature directory       : {FEATURE_DIR}")
print(f"Target participants     : {TARGET_PARTICIPANTS:,}")
print(f"Minimum sections        : {MIN_SECTIONS}")
print(f"Minimum keystrokes      : {MIN_KEYSTROKES}")
print(f"Minimum finite ratio     : {MIN_FINITE_RATIO:.2f}")
print(f"Random seed              : {RANDOM_SEED}")
print("=" * 78)


# ============================================================
# FIND ACTUAL FEATURE FILES
# ============================================================

files = sorted(FEATURE_DIR.glob("*_features.csv"))

print("\nFiles discovered:", f"{len(files):,}")

if len(files) == 0:
    raise FileNotFoundError(
        f"\nNo feature files found in:\n"
        f"  {FEATURE_DIR.resolve()}\n\n"
        f"Expected files such as:\n"
        f"  123456_features.csv\n"
        f"  123457_features.csv\n"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return default
    except Exception:
        return default


def extract_filename_participant_id(file):
    """
    Extract participant ID from:
        123456_features.csv
    """
    name = file.name

    if not name.endswith("_features.csv"):
        return None

    raw_id = name[:-len("_features.csv")]

    try:
        return int(raw_id)
    except ValueError:
        return None


def calculate_feature_quality(df):
    """
    Calculate quality characteristics across the 16
    behavioral features.
    """

    numeric_df = pd.DataFrame(index=df.index)

    for feature in FEATURE_COLUMNS:
        numeric_df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    values = numeric_df.to_numpy(dtype=np.float64)

    finite_mask = np.isfinite(values)

    total_values = values.size

    if total_values == 0:
        finite_ratio = 0.0
    else:
        finite_ratio = finite_mask.sum() / total_values

    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())

    # --------------------------------------------------------
    # Per-feature variability
    # --------------------------------------------------------

    feature_variabilities = []
    constant_features = 0

    for feature in FEATURE_COLUMNS:

        column = numeric_df[feature].to_numpy(
            dtype=np.float64
        )

        finite_values = column[np.isfinite(column)]

        if len(finite_values) == 0:
            feature_variabilities.append(0.0)
            constant_features += 1
            continue

        std_value = float(np.std(finite_values))

        feature_variabilities.append(std_value)

        if std_value <= MIN_FEATURE_VARIABILITY:
            constant_features += 1

    variability_mean = (
        float(np.mean(feature_variabilities))
        if feature_variabilities
        else 0.0
    )

    variability_median = (
        float(np.median(feature_variabilities))
        if feature_variabilities
        else 0.0
    )

    constant_feature_ratio = (
        constant_features / len(FEATURE_COLUMNS)
    )

    # --------------------------------------------------------
    # Extreme-value diagnostics
    # --------------------------------------------------------

    extreme_mask = np.zeros_like(values, dtype=bool)

    for feature_index, feature in enumerate(FEATURE_COLUMNS):

        if feature == "OVERLAP_INDICATOR":
            continue

        column = values[:, feature_index]

        extreme_mask[:, feature_index] = (
            np.isfinite(column)
            & (
                np.abs(column)
                > EXTREME_VALUE_THRESHOLD
            )
        )

    extreme_count = int(extreme_mask.sum())

    # --------------------------------------------------------
    # Quality scores
    # --------------------------------------------------------

    finite_score = min(
        finite_ratio / MIN_FINITE_RATIO,
        1.0
    )

    variability_score = (
        1.0
        if variability_median > MIN_FEATURE_VARIABILITY
        else 0.0
    )

    constant_feature_score = max(
        0.0,
        1.0 - (
            constant_feature_ratio
            / MAX_CONSTANT_FEATURE_RATIO
        )
    )

    # Combined feature-quality score
    feature_quality_score = (
        0.50 * finite_score
        + 0.30 * variability_score
        + 0.20 * constant_feature_score
    )

    return {
        "nan_values": nan_count,
        "inf_values": inf_count,
        "finite_ratio": finite_ratio,
        "feature_variability_mean": variability_mean,
        "feature_variability_median": variability_median,
        "constant_feature_count": constant_features,
        "constant_feature_ratio": constant_feature_ratio,
        "extreme_value_count": extreme_count,
        "feature_quality_score": feature_quality_score,
    }


def calculate_temporal_quality(df):

    temporal_ok = True
    sections_checked = 0
    sections_valid = 0

    try:

        grouped = df.groupby(
            "TEST_SECTION_ID",
            sort=False
        )

        for section_id, section in grouped:

            sections_checked += 1

            key_ids = pd.to_numeric(
                section["KEYSTROKE_ID"],
                errors="coerce"
            ).to_numpy()

            key_ids = key_ids[np.isfinite(key_ids)]

            if len(key_ids) <= 1:
                sections_valid += 1
                continue

            if np.any(np.diff(key_ids) < 0):
                temporal_ok = False
                continue

            sections_valid += 1

    except Exception:
        temporal_ok = False

    if sections_checked == 0:
        temporal_score = 0.0
    else:
        temporal_score = (
            sections_valid / sections_checked
        )

    return {
        "temporal_ok": temporal_ok,
        "sections_checked": sections_checked,
        "sections_valid": sections_valid,
        "temporal_score": temporal_score,
    }


def calculate_quality_score(
    rows,
    sections,
    finite_ratio,
    feature_quality,
    temporal_score
):

    # --------------------------------------------------------
    # Row quantity score
    # --------------------------------------------------------

    row_score = min(
        rows / MIN_KEYSTROKES,
        1.0
    )

    # --------------------------------------------------------
    # Section quantity score
    # --------------------------------------------------------

    section_score = min(
        sections / MIN_SECTIONS,
        1.0
    )

    # --------------------------------------------------------
    # Finite-data score
    # --------------------------------------------------------

    finite_score = min(
        finite_ratio / MIN_FINITE_RATIO,
        1.0
    )

    # --------------------------------------------------------
    # Final weighted score
    #
    # Quality only. No model metrics.
    # --------------------------------------------------------

    overall_score = (
        0.30 * row_score
        + 0.20 * section_score
        + 0.25 * finite_score
        + 0.15 * feature_quality
        + 0.10 * temporal_score
    )

    return float(overall_score)


# ============================================================
# PASS 1 — DEEP PARTICIPANT SCREENING
# ============================================================

print("\n" + "=" * 78)
print("PASS 1 — DEEP PARTICIPANT QUALITY SCREENING")
print("=" * 78)

records = []

for index, file in enumerate(files, start=1):

    filename_id = extract_filename_participant_id(file)

    record = {
        "participant_id": filename_id,
        "file": str(file),
        "file_exists": int(file.exists()),
        "rows": 0,
        "sections": 0,
        "columns": 0,
        "numeric_features": 0,
        "nan_values": 0,
        "inf_values": 0,
        "finite_ratio": 0.0,
        "sections_checked": 0,
        "sections_valid": 0,
        "temporal_score": 0.0,
        "feature_variability_mean": 0.0,
        "feature_variability_median": 0.0,
        "constant_feature_count": 0,
        "constant_feature_ratio": 0.0,
        "extreme_value_count": 0,
        "feature_quality_score": 0.0,
        "row_quality_score": 0.0,
        "sequence_quality_score": 0.0,
        "overall_quality_score": 0.0,
        "eligible": 0,
        "failure_reason": "",
    }

    # --------------------------------------------------------
    # Filename validation
    # --------------------------------------------------------

    if filename_id is None:

        record["failure_reason"] = (
            "invalid_filename_participant_id"
        )

        records.append(record)
        continue

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(file)

    except Exception as e:

        record["failure_reason"] = (
            f"read_error: {str(e)[:200]}"
        )

        records.append(record)
        continue

    record["rows"] = len(df)
    record["columns"] = len(df.columns)

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        record["failure_reason"] = (
            "missing_required_columns: "
            + ",".join(missing_columns)
        )

        records.append(record)
        continue

    record["numeric_features"] = len(FEATURE_COLUMNS)

    # --------------------------------------------------------
    # Empty file
    # --------------------------------------------------------

    if df.empty:

        record["failure_reason"] = "empty_file"

        records.append(record)
        continue

    # --------------------------------------------------------
    # Participant ID consistency
    # --------------------------------------------------------

    participant_values = (
        pd.to_numeric(
            df["PARTICIPANT_ID"],
            errors="coerce"
        )
        .dropna()
        .unique()
    )

    if len(participant_values) != 1:

        record["failure_reason"] = (
            "multiple_or_missing_participant_ids"
        )

        records.append(record)
        continue

    participant_id = int(participant_values[0])

    if participant_id != filename_id:

        record["failure_reason"] = (
            f"filename_id_mismatch: "
            f"{filename_id} != {participant_id}"
        )

        records.append(record)
        continue

    # --------------------------------------------------------
    # Basic participant statistics
    # --------------------------------------------------------

    rows = len(df)

    sections = (
        df["TEST_SECTION_ID"]
        .dropna()
        .unique()
    )

    section_count = len(sections)

    record["sections"] = section_count

    # --------------------------------------------------------
    # Feature quality
    # --------------------------------------------------------

    feature_quality = calculate_feature_quality(df)

    record.update({
        "nan_values": feature_quality["nan_values"],
        "inf_values": feature_quality["inf_values"],
        "finite_ratio": feature_quality["finite_ratio"],
        "feature_variability_mean":
            feature_quality["feature_variability_mean"],
        "feature_variability_median":
            feature_quality["feature_variability_median"],
        "constant_feature_count":
            feature_quality["constant_feature_count"],
        "constant_feature_ratio":
            feature_quality["constant_feature_ratio"],
        "extreme_value_count":
            feature_quality["extreme_value_count"],
        "feature_quality_score":
            feature_quality["feature_quality_score"],
    })

    # --------------------------------------------------------
    # Temporal quality
    # --------------------------------------------------------

    temporal_quality = calculate_temporal_quality(df)

    record.update({
        "sections_checked":
            temporal_quality["sections_checked"],
        "sections_valid":
            temporal_quality["sections_valid"],
        "temporal_score":
            temporal_quality["temporal_score"],
    })

    # --------------------------------------------------------
    # Individual quality scores
    # --------------------------------------------------------

    record["row_quality_score"] = min(
        rows / MIN_KEYSTROKES,
        1.0
    )

    record["sequence_quality_score"] = (
        0.60 * min(
            section_count / MIN_SECTIONS,
            1.0
        )
        + 0.40 * temporal_quality["temporal_score"]
    )

    record["overall_quality_score"] = calculate_quality_score(
        rows=rows,
        sections=section_count,
        finite_ratio=feature_quality["finite_ratio"],
        feature_quality=feature_quality[
            "feature_quality_score"
        ],
        temporal_score=temporal_quality[
            "temporal_score"
        ],
    )

    # --------------------------------------------------------
    # Eligibility
    # --------------------------------------------------------

    failure_reasons = []

    if rows < MIN_KEYSTROKES:
        failure_reasons.append(
            f"insufficient_keystrokes({rows})"
        )

    if section_count < MIN_SECTIONS:
        failure_reasons.append(
            f"insufficient_sections({section_count})"
        )

    if feature_quality["finite_ratio"] < MIN_FINITE_RATIO:
        failure_reasons.append(
            f"low_finite_ratio("
            f"{feature_quality['finite_ratio']:.4f})"
        )

    if (
        feature_quality["constant_feature_ratio"]
        > MAX_CONSTANT_FEATURE_RATIO
    ):
        failure_reasons.append(
            "too_many_constant_features"
        )

    if temporal_quality["temporal_score"] < 1.0:
        failure_reasons.append(
            "temporal_order_issue"
        )

    if failure_reasons:

        record["failure_reason"] = ";".join(
            failure_reasons
        )

        record["eligible"] = 0

    else:

        record["eligible"] = 1
        record["failure_reason"] = "PASS"

    records.append(record)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 1000 == 0 or index == len(files):

        elapsed = time.time() - start_time

        rate = (
            index / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"Processed {index:,}/{len(files):,} | "
            f"eligible={sum(r['eligible'] for r in records):,} | "
            f"rate={rate:,.1f} files/sec"
        )


# ============================================================
# CREATE REPORT DATAFRAME
# ============================================================

report_df = pd.DataFrame(records)

report_df = report_df.sort_values(
    "overall_quality_score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SAVE FULL REPORT
# ============================================================

full_report_path = (
    OUTPUT_DIR /
    "full_deep_quality_report_v2.csv"
)

report_df.to_csv(
    full_report_path,
    index=False
)


# ============================================================
# ELIGIBLE PARTICIPANTS
# ============================================================

eligible_df = report_df[
    report_df["eligible"] == 1
].copy()

eligible_df = eligible_df.sort_values(
    [
        "overall_quality_score",
        "rows",
        "sections",
        "finite_ratio",
        "feature_variability_median",
    ],
    ascending=False
).reset_index(drop=True)


eligible_path = (
    OUTPUT_DIR /
    "eligible_participants_ranked_v2.csv"
)

eligible_df.to_csv(
    eligible_path,
    index=False
)


# ============================================================
# FINAL TOP 5000
# ============================================================

if len(eligible_df) < TARGET_PARTICIPANTS:

    print("\n" + "=" * 78)
    print("WARNING")
    print("=" * 78)

    print(
        f"Only {len(eligible_df):,} participants "
        f"passed strict quality screening."
    )

    print(
        f"Required: {TARGET_PARTICIPANTS:,}"
    )

    print(
        "\nNo artificial participants will be added."
    )

    final_df = eligible_df.copy()

else:

    final_df = (
        eligible_df
        .head(TARGET_PARTICIPANTS)
        .copy()
    )

final_df = final_df.reset_index(drop=True)

final_path = (
    OUTPUT_DIR /
    "final_top_5000_participants_v2.csv"
)

final_df.to_csv(
    final_path,
    index=False
)


# ============================================================
# QUALITY DISTRIBUTION
# ============================================================

quality_bins = [
    0.0,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    1.00
]

quality_distribution = pd.cut(
    report_df["overall_quality_score"],
    bins=quality_bins,
    include_lowest=True
).value_counts(
    sort=False
)

quality_distribution_path = (
    OUTPUT_DIR /
    "quality_score_distribution_v2.csv"
)

quality_distribution.rename(
    "participants"
).to_csv(
    quality_distribution_path
)


# ============================================================
# FAILURE DISTRIBUTION
# ============================================================

failure_distribution = (
    report_df[
        report_df["eligible"] == 0
    ]["failure_reason"]
    .value_counts()
    .rename_axis("failure_reason")
    .reset_index(name="participants")
)

failure_distribution_path = (
    OUTPUT_DIR /
    "failure_reason_distribution_v2.csv"
)

failure_distribution.to_csv(
    failure_distribution_path,
    index=False
)


# ============================================================
# FINAL COHORT STATISTICS
# ============================================================

if len(final_df) > 0:

    cohort_stats = {
        "participants": int(len(final_df)),
        "mean_quality_score":
            float(
                final_df[
                    "overall_quality_score"
                ].mean()
            ),
        "median_quality_score":
            float(
                final_df[
                    "overall_quality_score"
                ].median()
            ),
        "minimum_quality_score":
            float(
                final_df[
                    "overall_quality_score"
                ].min()
            ),
        "maximum_quality_score":
            float(
                final_df[
                    "overall_quality_score"
                ].max()
            ),
        "mean_keystrokes":
            float(
                final_df["rows"].mean()
            ),
        "median_keystrokes":
            float(
                final_df["rows"].median()
            ),
        "minimum_keystrokes":
            int(
                final_df["rows"].min()
            ),
        "maximum_keystrokes":
            int(
                final_df["rows"].max()
            ),
        "mean_sections":
            float(
                final_df["sections"].mean()
            ),
        "median_sections":
            float(
                final_df["sections"].median()
            ),
        "minimum_sections":
            int(
                final_df["sections"].min()
            ),
        "maximum_sections":
            int(
                final_df["sections"].max()
            ),
        "mean_finite_ratio":
            float(
                final_df["finite_ratio"].mean()
            ),
        "minimum_finite_ratio":
            float(
                final_df["finite_ratio"].min()
            ),
        "mean_feature_variability":
            float(
                final_df[
                    "feature_variability_median"
                ].mean()
            ),
    }

else:

    cohort_stats = {
        "participants": 0
    }


# ============================================================
# SUMMARY
# ============================================================

elapsed = time.time() - start_time

summary = {
    "random_seed": RANDOM_SEED,
    "files_discovered": int(len(files)),
    "participants_processed": int(len(report_df)),
    "eligible_participants": int(len(eligible_df)),
    "target_participants": int(TARGET_PARTICIPANTS),
    "final_selected_participants": int(len(final_df)),
    "minimum_sections": int(MIN_SECTIONS),
    "minimum_keystrokes": int(MIN_KEYSTROKES),
    "minimum_finite_ratio": float(MIN_FINITE_RATIO),
    "max_constant_feature_ratio":
        float(MAX_CONSTANT_FEATURE_RATIO),
    "features_per_keystroke":
        int(len(FEATURE_COLUMNS)),
    "processing_time_seconds":
        float(elapsed),
    "processing_time_minutes":
        float(elapsed / 60.0),
    "selection_method":
        "quality_only_ranked_selection",
    "model_performance_used": False,
    "validation_metrics_used": False,
    "test_metrics_used": False,
    "cohort_statistics": cohort_stats,
}


summary_path = (
    OUTPUT_DIR /
    "deep_quality_summary_v2.json"
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


# ============================================================
# FINAL CONSOLE REPORT
# ============================================================

print("\n" + "=" * 78)
print("DEEP QUALITY SCREENING V2 COMPLETE")
print("=" * 78)

print(
    f"Candidates processed      : "
    f"{len(report_df):,}"
)

print(
    f"Eligible participants     : "
    f"{len(eligible_df):,}"
)

print(
    f"Target participants       : "
    f"{TARGET_PARTICIPANTS:,}"
)

print(
    f"Final Top-5000 selected   : "
    f"{len(final_df):,}"
)

print(
    f"Processing time           : "
    f"{elapsed / 60:.2f} minutes"
)

if len(final_df) > 0:

    print("\n" + "=" * 78)
    print("FINAL COHORT QUALITY")
    print("=" * 78)

    print(
        f"Mean quality score       : "
        f"{cohort_stats['mean_quality_score']:.4f}"
    )

    print(
        f"Median quality score     : "
        f"{cohort_stats['median_quality_score']:.4f}"
    )

    print(
        f"Minimum quality score    : "
        f"{cohort_stats['minimum_quality_score']:.4f}"
    )

    print(
        f"Mean keystrokes          : "
        f"{cohort_stats['mean_keystrokes']:.1f}"
    )

    print(
        f"Median keystrokes        : "
        f"{cohort_stats['median_keystrokes']:.1f}"
    )

    print(
        f"Minimum keystrokes       : "
        f"{cohort_stats['minimum_keystrokes']:,}"
    )

    print(
        f"Mean sections            : "
        f"{cohort_stats['mean_sections']:.1f}"
    )

    print(
        f"Median sections          : "
        f"{cohort_stats['median_sections']:.1f}"
    )

    print(
        f"Minimum sections         : "
        f"{cohort_stats['minimum_sections']}"
    )

    print(
        f"Mean finite ratio        : "
        f"{cohort_stats['mean_finite_ratio']:.4f}"
    )

    print(
        f"Minimum finite ratio     : "
        f"{cohort_stats['minimum_finite_ratio']:.4f}"
    )


# ============================================================
# OUTPUTS
# ============================================================

print("\n" + "=" * 78)
print("OUTPUT FILES")
print("=" * 78)

print(
    f"Full report              : "
    f"{full_report_path}"
)

print(
    f"Eligible ranked          : "
    f"{eligible_path}"
)

print(
    f"Final Top 5000           : "
    f"{final_path}"
)

print(
    f"Quality distribution     : "
    f"{quality_distribution_path}"
)

print(
    f"Failure distribution     : "
    f"{failure_distribution_path}"
)

print(
    f"Summary                  : "
    f"{summary_path}"
)


# ============================================================
# IMPORTANT RESEARCH CONTROLS
# ============================================================

print("\n" + "=" * 78)
print("RESEARCH CONTROLS")
print("=" * 78)

print("Model performance used   : NO")
print("Validation metrics used  : NO")
print("Test metrics used        : NO")
print("Participant-level        : YES")
print("Existing features changed: NO")
print("Selection based on       : DATA QUALITY ONLY")

print("\n" + "=" * 78)
print("SCREENING COMPLETE")
print("=" * 78)
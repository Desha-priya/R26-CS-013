#actual v3 

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
REPORT_DIR = Path("reports/deep_quality_v3")

TARGET_PARTICIPANTS = 5000

MIN_SECTIONS = 15
MIN_KEYSTROKES = 600

MIN_FINITE_RATIO = 0.90

RANDOM_SEED = 42

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

REPORT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

start_time = time.time()

files = sorted(FEATURE_DIR.glob("*_features.csv"))

print("=" * 70)
print("DEEP QUALITY SCREENING V3")
print("=" * 70)

print(f"Feature directory      : {FEATURE_DIR}")
print(f"Files discovered       : {len(files):,}")
print(f"Target participants    : {TARGET_PARTICIPANTS:,}")
print(f"Minimum sections      : {MIN_SECTIONS}")
print(f"Minimum keystrokes    : {MIN_KEYSTROKES}")
print(f"Minimum finite ratio  : {MIN_FINITE_RATIO}")
print(f"Random seed            : {RANDOM_SEED}")

print()
print("TEMPORAL POLICY")
print("-" * 70)
print("Primary temporal field : PRESS_TIME")
print("KEYSTROKE_ID ordering  : NOT USED FOR REJECTION")
print("RELEASE_TIME ordering  : NOT USED FOR REJECTION")
print()


# ============================================================
# SCREENING
# ============================================================

eligible = []
failed = []

for i, file in enumerate(files, start=1):

    try:

        # ----------------------------------------------------
        # Read file
        # ----------------------------------------------------

        df = pd.read_csv(file)

        filename_id = file.name.replace("_features.csv", "")

        try:
            filename_id = int(filename_id)
        except ValueError:

            failed.append({
                "participant_id": None,
                "file": file.name,
                "rows": len(df),
                "sections": 0,
                "keystrokes": len(df),
                "finite_ratio": 0.0,
                "press_time_decreases": 0,
                "failure_reason": "invalid_filename_id"
            })

            continue


        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        missing = [
            c for c in REQUIRED_COLUMNS
            if c not in df.columns
        ]

        if missing:

            failed.append({
                "participant_id": filename_id,
                "file": file.name,
                "rows": len(df),
                "sections": 0,
                "keystrokes": len(df),
                "finite_ratio": 0.0,
                "press_time_decreases": 0,
                "failure_reason":
                    "missing_columns:" + ",".join(missing)
            })

            continue


        # ----------------------------------------------------
        # Empty file
        # ----------------------------------------------------

        if df.empty:

            failed.append({
                "participant_id": filename_id,
                "file": file.name,
                "rows": 0,
                "sections": 0,
                "keystrokes": 0,
                "finite_ratio": 0.0,
                "press_time_decreases": 0,
                "failure_reason": "empty_file"
            })

            continue


        # ----------------------------------------------------
        # Participant ID validation
        # ----------------------------------------------------

        participant_ids = (
            df["PARTICIPANT_ID"]
            .dropna()
            .unique()
        )

        if len(participant_ids) != 1:

            failed.append({
                "participant_id": filename_id,
                "file": file.name,
                "rows": len(df),
                "sections": 0,
                "keystrokes": len(df),
                "finite_ratio": 0.0,
                "press_time_decreases": 0,
                "failure_reason": "multiple_participant_ids"
            })

            continue

        participant_id = int(participant_ids[0])

        if participant_id != filename_id:

            failed.append({
                "participant_id": participant_id,
                "file": file.name,
                "rows": len(df),
                "sections": 0,
                "keystrokes": len(df),
                "finite_ratio": 0.0,
                "press_time_decreases": 0,
                "failure_reason": "filename_id_mismatch"
            })

            continue


        # ----------------------------------------------------
        # Section count
        # ----------------------------------------------------

        section_count = (
            df["TEST_SECTION_ID"]
            .dropna()
            .nunique()
        )


        # ----------------------------------------------------
        # Keystroke count
        # ----------------------------------------------------

        keystroke_count = len(df)


        # ----------------------------------------------------
        # Numeric feature finite ratio
        # ----------------------------------------------------

        feature_numeric = df[FEATURE_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce"
        )

        values = feature_numeric.to_numpy(
            dtype=np.float64
        )

        finite_count = np.isfinite(values).sum()
        total_values = values.size

        if total_values > 0:
            finite_ratio = finite_count / total_values
        else:
            finite_ratio = 0.0


        # ----------------------------------------------------
        # PRESS_TIME validation
        #
        # This is the ONLY temporal ordering check.
        #
        # We do NOT check:
        #   KEYSTROKE_ID
        #   RELEASE_TIME
        #
        # because these are not reliable chronological
        # indicators in this dataset.
        # ----------------------------------------------------

        press_times = pd.to_numeric(
            df["PRESS_TIME"],
            errors="coerce"
        )

        press_time_decreases = 0

        # Evaluate within each section.
        for section_id, section in df.groupby(
            "TEST_SECTION_ID",
            sort=False
        ):

            times = pd.to_numeric(
                section["PRESS_TIME"],
                errors="coerce"
            ).to_numpy()

            times = times[np.isfinite(times)]

            if len(times) > 1:

                decreases = np.sum(
                    np.diff(times) < 0
                )

                press_time_decreases += int(
                    decreases
                )


        # ----------------------------------------------------
        # Determine eligibility
        # ----------------------------------------------------

        reasons = []

        if section_count < MIN_SECTIONS:
            reasons.append(
                f"insufficient_sections({section_count})"
            )

        if keystroke_count < MIN_KEYSTROKES:
            reasons.append(
                f"insufficient_keystrokes({keystroke_count})"
            )

        if finite_ratio < MIN_FINITE_RATIO:
            reasons.append(
                f"low_finite_ratio({finite_ratio:.4f})"
            )

        if press_time_decreases > 0:
            reasons.append(
                f"press_time_order_issue({press_time_decreases})"
            )


        # ----------------------------------------------------
        # Quality score
        #
        # Higher = better.
        #
        # This is DATA QUALITY ONLY.
        # No model performance is involved.
        # ----------------------------------------------------

        section_score = min(
            section_count / MIN_SECTIONS,
            1.0
        )

        keystroke_score = min(
            keystroke_count / MIN_KEYSTROKES,
            1.0
        )

        finite_score = min(
            finite_ratio,
            1.0
        )

        temporal_score = (
            1.0
            if press_time_decreases == 0
            else 0.0
        )

        overall_quality = (
            0.25 * section_score +
            0.25 * keystroke_score +
            0.35 * finite_score +
            0.15 * temporal_score
        )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        record = {
            "participant_id": participant_id,
            "file": str(file),
            "rows": keystroke_count,
            "sections": section_count,
            "keystrokes": keystroke_count,
            "finite_ratio": finite_ratio,
            "press_time_decreases":
                press_time_decreases,
            "section_score": section_score,
            "keystroke_score": keystroke_score,
            "finite_score": finite_score,
            "temporal_score": temporal_score,
            "overall_quality_score":
                overall_quality
        }


        if len(reasons) == 0:

            record["eligible"] = 1
            record["failure_reason"] = ""

            eligible.append(record)

        else:

            record["eligible"] = 0
            record["failure_reason"] = ";".join(
                reasons
            )

            failed.append(record)


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if i % 5000 == 0:

            elapsed = time.time() - start_time

            rate = i / elapsed

            print(
                f"Processed {i:,}/{len(files):,} | "
                f"eligible={len(eligible):,} | "
                f"rate={rate:.1f} files/sec"
            )


    except Exception as e:

        failed.append({
            "participant_id": None,
            "file": file.name,
            "rows": 0,
            "sections": 0,
            "keystrokes": 0,
            "finite_ratio": 0.0,
            "press_time_decreases": 0,
            "eligible": 0,
            "failure_reason":
                f"read_error:{str(e)}"
        })


# ============================================================
# DATAFRAMES
# ============================================================

eligible_df = pd.DataFrame(eligible)
failed_df = pd.DataFrame(failed)


# ============================================================
# RANK ELIGIBLE PARTICIPANTS
# ============================================================

if len(eligible_df) > 0:

    eligible_df = (
        eligible_df
        .sort_values(
            [
                "overall_quality_score",
                "finite_ratio",
                "keystrokes",
                "sections"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# SELECT FINAL COHORT
# ============================================================

final_count = min(
    TARGET_PARTICIPANTS,
    len(eligible_df)
)

selected_df = (
    eligible_df
    .head(final_count)
    .copy()
)

selected_df["cohort_rank"] = np.arange(
    1,
    len(selected_df) + 1
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

full_report = pd.concat(
    [
        eligible_df.assign(status="eligible"),
        failed_df.assign(status="failed")
    ],
    ignore_index=True
)

full_report.to_csv(
    REPORT_DIR / "full_deep_quality_report_v3.csv",
    index=False
)

eligible_df.to_csv(
    REPORT_DIR / "eligible_participants_ranked_v3.csv",
    index=False
)

selected_df.to_csv(
    REPORT_DIR / "final_top_5000_participants_v3.csv",
    index=False
)


# ============================================================
# FAILURE DISTRIBUTION
# ============================================================

if len(failed_df) > 0:

    failure_distribution = (
        failed_df["failure_reason"]
        .fillna("")
        .value_counts()
        .rename_axis("failure_reason")
        .reset_index(name="participants")
    )

else:

    failure_distribution = pd.DataFrame(
        columns=[
            "failure_reason",
            "participants"
        ]
    )

failure_distribution.to_csv(
    REPORT_DIR /
    "failure_reason_distribution_v3.csv",
    index=False
)


# ============================================================
# QUALITY DISTRIBUTION
# ============================================================

if len(eligible_df) > 0:

    quality_distribution = (
        eligible_df[
            "overall_quality_score"
        ]
        .describe()
        .to_frame()
        .reset_index()
    )

else:

    quality_distribution = pd.DataFrame()

quality_distribution.to_csv(
    REPORT_DIR /
    "quality_score_distribution_v3.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

elapsed = time.time() - start_time

summary = {

    "files_processed": len(files),

    "eligible_participants":
        len(eligible_df),

    "target_participants":
        TARGET_PARTICIPANTS,

    "final_selected":
        len(selected_df),

    "mean_quality_score":
        (
            float(
                selected_df[
                    "overall_quality_score"
                ].mean()
            )
            if len(selected_df)
            else None
        ),

    "median_quality_score":
        (
            float(
                selected_df[
                    "overall_quality_score"
                ].median()
            )
            if len(selected_df)
            else None
        ),

    "minimum_quality_score":
        (
            float(
                selected_df[
                    "overall_quality_score"
                ].min()
            )
            if len(selected_df)
            else None
        ),

    "mean_keystrokes":
        (
            float(
                selected_df[
                    "keystrokes"
                ].mean()
            )
            if len(selected_df)
            else None
        ),

    "median_keystrokes":
        (
            float(
                selected_df[
                    "keystrokes"
                ].median()
            )
            if len(selected_df)
            else None
        ),

    "minimum_keystrokes":
        (
            int(
                selected_df[
                    "keystrokes"
                ].min()
            )
            if len(selected_df)
            else None
        ),

    "mean_sections":
        (
            float(
                selected_df[
                    "sections"
                ].mean()
            )
            if len(selected_df)
            else None
        ),

    "median_sections":
        (
            float(
                selected_df[
                    "sections"
                ].median()
            )
            if len(selected_df)
            else None
        ),

    "minimum_sections":
        (
            int(
                selected_df[
                    "sections"
                ].min()
            )
            if len(selected_df)
            else None
        ),

    "mean_finite_ratio":
        (
            float(
                selected_df[
                    "finite_ratio"
                ].mean()
            )
            if len(selected_df)
            else None
        ),

    "minimum_finite_ratio":
        (
            float(
                selected_df[
                    "finite_ratio"
                ].min()
            )
            if len(selected_df)
            else None
        ),

    "random_seed":
        RANDOM_SEED,

    "temporal_reference":
        "PRESS_TIME",

    "keystroke_id_used_for_temporal_rejection":
        False,

    "release_time_used_for_temporal_rejection":
        False,

    "model_performance_used":
        False,

    "validation_metrics_used":
        False,

    "test_metrics_used":
        False,

    "processing_seconds":
        elapsed
}


with open(
    REPORT_DIR /
    "deep_quality_summary_v3.json",
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

print()
print("=" * 70)
print("DEEP QUALITY SCREENING V3 COMPLETE")
print("=" * 70)

print(
    f"Files processed             : "
    f"{len(files):,}"
)

print(
    f"Eligible participants       : "
    f"{len(eligible_df):,}"
)

print(
    f"Target participants         : "
    f"{TARGET_PARTICIPANTS:,}"
)

print(
    f"Final selected              : "
    f"{len(selected_df):,}"
)

print(
    f"Processing time             : "
    f"{elapsed / 60:.2f} minutes"
)

print()
print("=" * 70)
print("TEMPORAL POLICY")
print("=" * 70)

print("PRESS_TIME ordering         : PRIMARY")
print("KEYSTROKE_ID ordering       : IGNORED")
print("RELEASE_TIME ordering       : IGNORED")

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    "Full report                 : "
    f"{REPORT_DIR / 'full_deep_quality_report_v3.csv'}"
)

print(
    "Eligible ranked             : "
    f"{REPORT_DIR / 'eligible_participants_ranked_v3.csv'}"
)

print(
    "Final Top 5000              : "
    f"{REPORT_DIR / 'final_top_5000_participants_v3.csv'}"
)

print(
    "Failure distribution        : "
    f"{REPORT_DIR / 'failure_reason_distribution_v3.csv'}"
)

print(
    "Quality distribution        : "
    f"{REPORT_DIR / 'quality_score_distribution_v3.csv'}"
)

print(
    "Summary                     : "
    f"{REPORT_DIR / 'deep_quality_summary_v3.json'}"
)

print()
print("=" * 70)
print("RESEARCH CONTROLS")
print("=" * 70)

print("Model performance used      : NO")
print("Validation metrics used    : NO")
print("Test metrics used          : NO")
print("Participant-level           : YES")
print("Existing features changed  : NO")
print("Selection based on         : DATA QUALITY ONLY")

print()
print("=" * 70)
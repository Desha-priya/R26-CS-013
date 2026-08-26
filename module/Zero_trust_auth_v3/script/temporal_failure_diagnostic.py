from pathlib import Path
import pandas as pd
import numpy as np
import time
import json

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = Path("processed/features")
OUTPUT_DIR = Path("reports/temporal_diagnostic_v2")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only inspect participants that failed temporal screening.
# If the failure report is not found, automatically inspect
# all feature files.
FAILURE_REPORT = Path(
    "reports/deep_quality_v2/full_deep_quality_report_v2.csv"
)

KEY_ID = "KEYSTROKE_ID"
SECTION_ID = "TEST_SECTION_ID"
PRESS_TIME = "PRESS_TIME"
RELEASE_TIME = "RELEASE_TIME"
PARTICIPANT_ID = "PARTICIPANT_ID"

# ============================================================
# SETUP
# ============================================================

start_time = time.time()

print("=" * 70)
print("TEMPORAL FAILURE DIAGNOSTIC V2")
print("=" * 70)

print(f"\nFeature directory : {FEATURE_DIR}")
print(f"Output directory  : {OUTPUT_DIR}")
print(f"Failure report    : {FAILURE_REPORT}")

# ============================================================
# LOAD TEMPORAL-FAILURE PARTICIPANTS
# ============================================================

failed_ids = None

if FAILURE_REPORT.exists():

    print("\nLoading deep-quality report...")

    report = pd.read_csv(FAILURE_REPORT)

    print(f"Report rows       : {len(report):,}")

    if "failure_reason" in report.columns:

        temporal_mask = (
            report["failure_reason"]
            .fillna("")
            .astype(str)
            .str.contains("temporal_order_issue", regex=False)
        )

        temporal_rows = report.loc[temporal_mask].copy()

        if "participant_id" in temporal_rows.columns:

            failed_ids = set(
                pd.to_numeric(
                    temporal_rows["participant_id"],
                    errors="coerce"
                )
                .dropna()
                .astype(np.int64)
                .tolist()
            )

    print(
        f"Temporal-failure participants: "
        f"{len(failed_ids) if failed_ids is not None else 0:,}"
    )

# ============================================================
# FIND FEATURE FILES
# ============================================================

files = sorted(FEATURE_DIR.glob("*_features.csv"))

print(f"\nFeature files discovered: {len(files):,}")

if failed_ids is not None and len(failed_ids) > 0:

    selected_files = []

    for file in files:

        try:
            pid = int(
                file.name.replace("_features.csv", "")
            )
        except ValueError:
            continue

        if pid in failed_ids:
            selected_files.append(file)

    files = selected_files

    print(
        f"Files selected for diagnostic: "
        f"{len(files):,}"
    )

else:

    print(
        "No temporal-failure list found. "
        "Inspecting all feature files."
    )

# ============================================================
# RESULTS
# ============================================================

participant_results = []
section_results = []

global_decrease_total = 0
within_section_decrease_total = 0
press_time_decrease_total = 0
release_time_decrease_total = 0

participants_with_within_id_problem = 0
participants_with_press_problem = 0
participants_with_release_problem = 0

participants_clean_within_section = 0

# ============================================================
# PROCESS FILES
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING TEMPORAL STRUCTURE")
print("=" * 70)

for index, file in enumerate(files, start=1):

    try:

        # ----------------------------------------------------
        # Participant ID
        # ----------------------------------------------------

        try:
            participant_id = int(
                file.name.replace("_features.csv", "")
            )
        except ValueError:
            participant_id = -1

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        df = pd.read_csv(file)

        required = [
            PARTICIPANT_ID,
            SECTION_ID,
            KEY_ID,
            PRESS_TIME,
            RELEASE_TIME
        ]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:

            participant_results.append({
                "participant_id": participant_id,
                "file": file.name,
                "rows": len(df),
                "sections": 0,
                "global_key_id_decreases": np.nan,
                "within_section_key_id_decreases": np.nan,
                "press_time_decreases": np.nan,
                "release_time_decreases": np.nan,
                "sections_with_key_id_decrease": np.nan,
                "sections_with_press_time_decrease": np.nan,
                "sections_with_release_time_decrease": np.nan,
                "max_key_id_backward_jump": np.nan,
                "max_press_time_backward_jump": np.nan,
                "max_release_time_backward_jump": np.nan,
                "within_section_temporal_clean": False,
                "failure": "missing_required_columns",
                "details": ",".join(missing)
            })

            continue

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        key_ids = pd.to_numeric(
            df[KEY_ID],
            errors="coerce"
        )

        press_times = pd.to_numeric(
            df[PRESS_TIME],
            errors="coerce"
        )

        release_times = pd.to_numeric(
            df[RELEASE_TIME],
            errors="coerce"
        )

        # ----------------------------------------------------
        # GLOBAL KEYSTROKE ID ORDER
        # ----------------------------------------------------

        valid_key = key_ids.dropna().to_numpy()

        global_key_decreases = 0
        max_global_backward_jump = 0.0

        if len(valid_key) > 1:

            diffs = np.diff(valid_key)

            negative = diffs[diffs < 0]

            global_key_decreases = len(negative)

            if len(negative) > 0:
                max_global_backward_jump = float(
                    abs(negative.min())
                )

        global_decrease_total += global_key_decreases

        # ----------------------------------------------------
        # WITHIN-SECTION CHECKS
        # ----------------------------------------------------

        within_key_decreases = 0
        press_decreases = 0
        release_decreases = 0

        sections_with_key_problem = 0
        sections_with_press_problem = 0
        sections_with_release_problem = 0

        max_key_backward_jump = 0.0
        max_press_backward_jump = 0.0
        max_release_backward_jump = 0.0

        section_count = (
            df[SECTION_ID]
            .dropna()
            .nunique()
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We DO NOT sort before checking.
        #
        # We want to know whether the original
        # feature file itself preserves temporal order.
        # ----------------------------------------------------

        for section_id, section in df.groupby(
            SECTION_ID,
            sort=False
        ):

            section = section.reset_index(drop=True)

            # -----------------------------------------------
            # KEYSTROKE ID
            # -----------------------------------------------

            section_keys = pd.to_numeric(
                section[KEY_ID],
                errors="coerce"
            ).dropna().to_numpy()

            key_problem = 0

            if len(section_keys) > 1:

                key_diffs = np.diff(section_keys)

                negative = key_diffs[key_diffs < 0]

                key_problem = len(negative)

                within_key_decreases += key_problem

                if len(negative) > 0:

                    sections_with_key_problem += 1

                    max_key_backward_jump = max(
                        max_key_backward_jump,
                        float(abs(negative.min()))
                    )

            # -----------------------------------------------
            # PRESS TIME
            # -----------------------------------------------

            section_press = pd.to_numeric(
                section[PRESS_TIME],
                errors="coerce"
            ).dropna().to_numpy()

            press_problem = 0

            if len(section_press) > 1:

                press_diffs = np.diff(section_press)

                negative = press_diffs[press_diffs < 0]

                press_problem = len(negative)

                press_decreases += press_problem

                if len(negative) > 0:

                    sections_with_press_problem += 1

                    max_press_backward_jump = max(
                        max_press_backward_jump,
                        float(abs(negative.min()))
                    )

            # -----------------------------------------------
            # RELEASE TIME
            # -----------------------------------------------

            section_release = pd.to_numeric(
                section[RELEASE_TIME],
                errors="coerce"
            ).dropna().to_numpy()

            release_problem = 0

            if len(section_release) > 1:

                release_diffs = np.diff(section_release)

                negative = release_diffs[
                    release_diffs < 0
                ]

                release_problem = len(negative)

                release_decreases += release_problem

                if len(negative) > 0:

                    sections_with_release_problem += 1

                    max_release_backward_jump = max(
                        max_release_backward_jump,
                        float(abs(negative.min()))
                    )

            # -----------------------------------------------
            # SECTION-LEVEL RECORD
            # -----------------------------------------------

            section_results.append({
                "participant_id": participant_id,
                "section_id": section_id,
                "rows": len(section),
                "key_id_decreases": key_problem,
                "press_time_decreases": press_problem,
                "release_time_decreases": release_problem
            })

        # ----------------------------------------------------
        # GLOBAL COUNTERS
        # ----------------------------------------------------

        within_section_decrease_total += (
            within_key_decreases
        )

        press_time_decrease_total += press_decreases
        release_time_decrease_total += release_decreases

        if within_key_decreases > 0:
            participants_with_within_id_problem += 1
        else:
            participants_clean_within_section += 1

        if press_decreases > 0:
            participants_with_press_problem += 1

        if release_decreases > 0:
            participants_with_release_problem += 1

        # ----------------------------------------------------
        # PARTICIPANT RESULT
        # ----------------------------------------------------

        participant_results.append({
            "participant_id": participant_id,
            "file": file.name,
            "rows": len(df),
            "sections": section_count,

            "global_key_id_decreases":
                global_key_decreases,

            "within_section_key_id_decreases":
                within_key_decreases,

            "press_time_decreases":
                press_decreases,

            "release_time_decreases":
                release_decreases,

            "sections_with_key_id_decrease":
                sections_with_key_problem,

            "sections_with_press_time_decrease":
                sections_with_press_problem,

            "sections_with_release_time_decrease":
                sections_with_release_problem,

            "max_key_id_backward_jump":
                max_key_backward_jump,

            "max_global_key_id_backward_jump":
                max_global_backward_jump,

            "max_press_time_backward_jump":
                max_press_backward_jump,

            "max_release_time_backward_jump":
                max_release_backward_jump,

            "within_section_temporal_clean":
                within_key_decreases == 0,

            "press_time_temporal_clean":
                press_decreases == 0,

            "release_time_temporal_clean":
                release_decreases == 0,

            "failure": ""
        })

    except Exception as e:

        participant_results.append({
            "participant_id": participant_id,
            "file": file.name,
            "rows": 0,
            "sections": 0,
            "global_key_id_decreases": np.nan,
            "within_section_key_id_decreases": np.nan,
            "press_time_decreases": np.nan,
            "release_time_decreases": np.nan,
            "sections_with_key_id_decrease": np.nan,
            "sections_with_press_time_decrease": np.nan,
            "sections_with_release_time_decrease": np.nan,
            "max_key_id_backward_jump": np.nan,
            "max_global_key_id_backward_jump": np.nan,
            "max_press_time_backward_jump": np.nan,
            "max_release_time_backward_jump": np.nan,
            "within_section_temporal_clean": False,
            "press_time_temporal_clean": False,
            "release_time_temporal_clean": False,
            "failure": "processing_error",
            "details": str(e)
        })

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if index % 5000 == 0 or index == len(files):

        elapsed = time.time() - start_time

        rate = index / elapsed if elapsed > 0 else 0

        print(
            f"Processed {index:,}/{len(files):,} | "
            f"rate={rate:.1f} files/sec"
        )

# ============================================================
# SAVE PARTICIPANT REPORT
# ============================================================

participant_df = pd.DataFrame(
    participant_results
)

participant_path = (
    OUTPUT_DIR /
    "temporal_participant_diagnostic_v2.csv"
)

participant_df.to_csv(
    participant_path,
    index=False
)

# ============================================================
# SAVE SECTION REPORT
# ============================================================

section_df = pd.DataFrame(
    section_results
)

section_path = (
    OUTPUT_DIR /
    "temporal_section_diagnostic_v2.csv"
)

section_df.to_csv(
    section_path,
    index=False
)

# ============================================================
# CLASSIFICATION
# ============================================================

if len(participant_df) > 0:

    valid = participant_df[
        participant_df["failure"] == ""
    ].copy()

else:

    valid = participant_df.copy()

total_valid = len(valid)

global_only = 0
within_section_problem = 0
press_problem = 0
fully_temporally_clean = 0

if total_valid > 0:

    global_only_mask = (
        (valid["global_key_id_decreases"] > 0)
        &
        (valid["within_section_key_id_decreases"] == 0)
    )

    global_only = int(global_only_mask.sum())

    within_section_problem = int(
        (
            valid["within_section_key_id_decreases"] > 0
        ).sum()
    )

    press_problem = int(
        (
            valid["press_time_decreases"] > 0
        ).sum()
    )

    fully_temporally_clean = int(
        (
            (valid["within_section_key_id_decreases"] == 0)
            &
            (valid["press_time_decreases"] == 0)
            &
            (valid["release_time_decreases"] == 0)
        ).sum()
    )

# ============================================================
# SUMMARY
# ============================================================

elapsed = time.time() - start_time

summary = {
    "files_processed": len(files),
    "valid_files": total_valid,

    "global_key_id_decrease_total":
        int(global_decrease_total),

    "within_section_key_id_decrease_total":
        int(within_section_decrease_total),

    "press_time_decrease_total":
        int(press_time_decrease_total),

    "release_time_decrease_total":
        int(release_time_decrease_total),

    "participants_with_within_section_key_problem":
        int(participants_with_within_id_problem),

    "participants_with_press_time_problem":
        int(participants_with_press_problem),

    "participants_with_release_time_problem":
        int(participants_with_release_problem),

    "participants_clean_within_section":
        int(participants_clean_within_section),

    "global_only_temporal_problem_participants":
        int(global_only),

    "within_section_problem_participants":
        int(within_section_problem),

    "press_time_problem_participants":
        int(press_problem),

    "fully_temporally_clean_participants":
        int(fully_temporally_clean),

    "elapsed_seconds":
        elapsed
}

summary_path = (
    OUTPUT_DIR /
    "temporal_diagnostic_summary_v2.json"
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
# TOP PROBLEMATIC PARTICIPANTS
# ============================================================

if total_valid > 0:

    worst = (
        valid
        .sort_values(
            [
                "within_section_key_id_decreases",
                "press_time_decreases",
                "global_key_id_decreases"
            ],
            ascending=False
        )
        .head(100)
    )

    worst.to_csv(
        OUTPUT_DIR /
        "worst_temporal_participants_v2.csv",
        index=False
    )

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL DIAGNOSTIC COMPLETE")
print("=" * 70)

print(
    f"\nFiles processed                    : "
    f"{len(files):,}"
)

print(
    f"Global KEYSTROKE_ID decreases     : "
    f"{global_decrease_total:,}"
)

print(
    f"Within-section KEYSTROKE_ID ↓    : "
    f"{within_section_decrease_total:,}"
)

print(
    f"PRESS_TIME decreases              : "
    f"{press_time_decrease_total:,}"
)

print(
    f"RELEASE_TIME decreases            : "
    f"{release_time_decrease_total:,}"
)

print("\n" + "-" * 70)

print(
    f"Participants with within-section "
    f"KEYSTROKE_ID problems             : "
    f"{participants_with_within_id_problem:,}"
)

print(
    f"Participants with PRESS_TIME problems: "
    f"{participants_with_press_problem:,}"
)

print(
    f"Participants with RELEASE_TIME problems: "
    f"{participants_with_release_problem:,}"
)

print(
    f"Participants clean within sections : "
    f"{participants_clean_within_section:,}"
)

print("\n" + "-" * 70)

print(
    f"GLOBAL-ONLY ordering problems      : "
    f"{global_only:,}"
)

print(
    f"WITHIN-SECTION ordering problems   : "
    f"{within_section_problem:,}"
)

print(
    f"PRESS_TIME ordering problems       : "
    f"{press_problem:,}"
)

print(
    f"FULLY temporally clean             : "
    f"{fully_temporally_clean:,}"
)

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"Participant diagnostic : "
    f"{participant_path}"
)

print(
    f"Section diagnostic     : "
    f"{section_path}"
)

print(
    f"Worst participants     : "
    f"{OUTPUT_DIR / 'worst_temporal_participants_v2.csv'}"
)

print(
    f"Summary                : "
    f"{summary_path}"
)

print(
    f"\nProcessing time        : "
    f"{elapsed / 60:.2f} minutes"
)

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if (
    total_valid > 0
    and within_section_problem == 0
    and press_problem == 0
):

    print(
        """
RESULT: The temporal failures appear to be
GLOBAL ordering artifacts.

KEYSTROKE_ID is not globally monotonic, but
the actual within-section temporal structure
is clean.

The original screening rule should NOT reject
participants based on global ordering.
"""
    )

elif (
    total_valid > 0
    and within_section_problem < total_valid * 0.10
):

    print(
        """
RESULT: Most participants appear temporally
valid within sections.

A small minority have genuine within-section
ordering problems.

The screening rule should probably be revised
to distinguish minor anomalies from genuine
corruption.
"""
    )

else:

    print(
        """
RESULT: Genuine within-section temporal problems
are widespread.

Do NOT simply remove the temporal quality check.

The feature-generation/order pipeline should be
investigated before expanding the cohort.
"""
    )

print("=" * 70)
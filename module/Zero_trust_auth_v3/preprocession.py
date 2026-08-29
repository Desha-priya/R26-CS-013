from pathlib import Path
import pandas as pd
import numpy as np
import csv
import re
import time


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path("data")
OUTPUT_DIR = Path("processed/features")
REPORT_DIR = Path("reports")

ROLLING_WINDOW = 10

# A pause is defined relative to the current participant's
# natural typing rhythm.
#
# We calculate the threshold later from each section's IKI
# distribution rather than choosing an arbitrary 500 ms.
PAUSE_PERCENTILE = 90

MIN_KEYSTROKES_PER_SECTION = 10

REQUIRED_COLUMNS = [
    "PARTICIPANT_ID",
    "TEST_SECTION_ID",
    "SENTENCE",
    "USER_INPUT",
    "KEYSTROKE_ID",
    "PRESS_TIME",
    "RELEASE_TIME",
    "LETTER",
    "KEYCODE",
]

NUMERIC_COLUMNS = [
    "PARTICIPANT_ID",
    "TEST_SECTION_ID",
    "KEYSTROKE_ID",
    "PRESS_TIME",
    "RELEASE_TIME",
    "KEYCODE",
]


# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_participant_id_from_filename(file_path):
    """
    Extract participant ID from filenames such as:

        5_keystrokes.txt
        517947_keystrokes.txt
    """

    match = re.match(
        r"^(\d+)_keystrokes\.txt$",
        file_path.name
    )

    if not match:
        return None

    return int(match.group(1))


def safe_divide(a, b):
    """
    Safe element-wise division.
    """

    return a / (b.replace(0, np.nan))


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(section):
    """
    Extract behavioral features from ONE TEST_SECTION_ID.

    Important:
    We never calculate transitions across sentence/section
    boundaries.
    """

    section = section.copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    section = section.sort_values(
        ["PRESS_TIME", "KEYSTROKE_ID"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Basic dwell time
    # --------------------------------------------------------

    section["DWELL_TIME"] = (
        section["RELEASE_TIME"] -
        section["PRESS_TIME"]
    )

    # --------------------------------------------------------
    # Previous event information
    # --------------------------------------------------------

    section["PREV_PRESS_TIME"] = (
        section["PRESS_TIME"].shift(1)
    )

    section["PREV_RELEASE_TIME"] = (
        section["RELEASE_TIME"].shift(1)
    )

    section["PREV_DWELL_TIME"] = (
        section["DWELL_TIME"].shift(1)
    )

    # Two events before
    section["PREV2_PRESS_TIME"] = (
        section["PRESS_TIME"].shift(2)
    )

    # --------------------------------------------------------
    # 1. DWELL TIME
    # --------------------------------------------------------

    # Already calculated above.

    # --------------------------------------------------------
    # 2. PRESS-TO-PRESS INTERVAL / IKI
    # --------------------------------------------------------

    section["PRESS_INTERVAL"] = (
        section["PRESS_TIME"] -
        section["PREV_PRESS_TIME"]
    )

    # --------------------------------------------------------
    # 3. RELEASE-TO-RELEASE INTERVAL
    # --------------------------------------------------------

    section["RELEASE_INTERVAL"] = (
        section["RELEASE_TIME"] -
        section["PREV_RELEASE_TIME"]
    )

    # --------------------------------------------------------
    # 4. RELEASE-TO-NEXT-PRESS LATENCY
    #
    # Negative = overlapping keypress
    # Zero     = next key pressed exactly at release
    # Positive = gap between keys
    # --------------------------------------------------------

    section["RELEASE_PRESS_LATENCY"] = (
        section["PRESS_TIME"] -
        section["PREV_RELEASE_TIME"]
    )

    # --------------------------------------------------------
    # 5. OVERLAP DURATION
    # --------------------------------------------------------

    section["OVERLAP_DURATION"] = (
        -section["RELEASE_PRESS_LATENCY"]
    ).clip(lower=0)

    # --------------------------------------------------------
    # 6. OVERLAP INDICATOR
    # --------------------------------------------------------

    section["OVERLAP_INDICATOR"] = (
        section["OVERLAP_DURATION"] > 0
    ).astype(np.int8)

    # --------------------------------------------------------
    # 7. DIGRAPH DWELL DIFFERENCE
    # --------------------------------------------------------

    section["DWELL_DIFFERENCE"] = (
        section["DWELL_TIME"] -
        section["PREV_DWELL_TIME"]
    )

    # --------------------------------------------------------
    # 8. DIGRAPH DWELL RATIO
    # --------------------------------------------------------

    section["DWELL_RATIO"] = (
        section["DWELL_TIME"] /
        (section["PREV_DWELL_TIME"] + 1e-6)
    )

    # --------------------------------------------------------
    # 9. TRIGRAPH PRESS INTERVAL
    #
    # Current press - press two keys ago.
    # --------------------------------------------------------

    section["TRIGRAPH_PRESS_INTERVAL"] = (
        section["PRESS_TIME"] -
        section["PREV2_PRESS_TIME"]
    )

    # ========================================================
    # ROLLING BEHAVIORAL FEATURES
    # ========================================================

    # --------------------------------------------------------
    # 10. MEAN DWELL
    # --------------------------------------------------------

    section["MEAN_DWELL"] = (
        section["DWELL_TIME"]
        .rolling(
            ROLLING_WINDOW,
            min_periods=3
        )
        .mean()
    )

    # --------------------------------------------------------
    # 11. DWELL STANDARD DEVIATION
    # --------------------------------------------------------

    section["STD_DWELL"] = (
        section["DWELL_TIME"]
        .rolling(
            ROLLING_WINDOW,
            min_periods=3
        )
        .std()
    )

    # --------------------------------------------------------
    # 12. MEDIAN DWELL
    # --------------------------------------------------------

    section["MEDIAN_DWELL"] = (
        section["DWELL_TIME"]
        .rolling(
            ROLLING_WINDOW,
            min_periods=3
        )
        .median()
    )

    # --------------------------------------------------------
    # 13. MEAN IKI
    # --------------------------------------------------------

    section["MEAN_IKI"] = (
        section["PRESS_INTERVAL"]
        .rolling(
            ROLLING_WINDOW,
            min_periods=3
        )
        .mean()
    )

    # --------------------------------------------------------
    # 14. IKI STANDARD DEVIATION
    # --------------------------------------------------------

    section["STD_IKI"] = (
        section["PRESS_INTERVAL"]
        .rolling(
            ROLLING_WINDOW,
            min_periods=3
        )
        .std()
    )

    # --------------------------------------------------------
    # 15. IKI COEFFICIENT OF VARIATION
    # --------------------------------------------------------

    section["IKI_CV"] = (
        section["STD_IKI"] /
        (section["MEAN_IKI"] + 1e-6)
    )

    # --------------------------------------------------------
    # 16. LOCAL PAUSE FREQUENCY
    #
    # First determine the natural IKI distribution for this
    # section. A pause is an IKI above the 90th percentile.
    # --------------------------------------------------------

    valid_iki = section["PRESS_INTERVAL"].dropna()

    if len(valid_iki) >= 3:

        pause_threshold = np.percentile(
            valid_iki,
            PAUSE_PERCENTILE
        )

        section["PAUSE_EVENT"] = (
            section["PRESS_INTERVAL"] >
            pause_threshold
        ).astype(np.int8)

        section["LOCAL_PAUSE_FREQUENCY"] = (
            section["PAUSE_EVENT"]
            .rolling(
                ROLLING_WINDOW,
                min_periods=3
            )
            .mean()
        )

    else:

        section["PAUSE_EVENT"] = np.nan
        section["LOCAL_PAUSE_FREQUENCY"] = np.nan

    return section


# ============================================================
# PROCESS ONE PARTICIPANT FILE
# ============================================================

def process_participant_file(file_path):

    filename_id = get_participant_id_from_filename(file_path)

    result = {
        "file": file_path.name,
        "filename_participant_id": filename_id,
        "rows_loaded": 0,
        "rows_final": 0,
        "duplicates_removed": 0,
        "invalid_timestamp_rows": 0,
        "sections_total": 0,
        "sections_valid": 0,
        "status": "OK",
        "error": "",
    }

    try:

        # ----------------------------------------------------
        # Read
        # ----------------------------------------------------

        df = pd.read_csv(
            file_path,
            sep="\t",
            encoding="utf-8",
            low_memory=False
        )

        result["rows_loaded"] = len(df)

        # ----------------------------------------------------
        # Validate columns
        # ----------------------------------------------------

        missing = [
            c for c in REQUIRED_COLUMNS
            if c not in df.columns
        ]

        if missing:

            result["status"] = "MISSING_COLUMNS"

            result["error"] = (
                "Missing: " +
                ", ".join(missing)
            )

            return result

        # ----------------------------------------------------
        # Convert numerical columns
        # ----------------------------------------------------

        for col in NUMERIC_COLUMNS:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Validate filename ID
        # ----------------------------------------------------

        actual_ids = (
            df["PARTICIPANT_ID"]
            .dropna()
            .unique()
        )

        if len(actual_ids) == 1:

            actual_id = int(actual_ids[0])

            if filename_id != actual_id:

                result["status"] = "ID_MISMATCH"

                result["error"] = (
                    f"Filename ID={filename_id}, "
                    f"data ID={actual_id}"
                )

                return result

        else:

            result["status"] = "INVALID_PARTICIPANT_ID"

            result["error"] = (
                f"Found {len(actual_ids)} participant IDs"
            )

            return result

        # ----------------------------------------------------
        # Remove exact duplicates
        # ----------------------------------------------------

        before = len(df)

        df = df.drop_duplicates()

        result["duplicates_removed"] = (
            before - len(df)
        )

        # ----------------------------------------------------
        # Remove missing essential fields
        # ----------------------------------------------------

        essential = [
            "PARTICIPANT_ID",
            "TEST_SECTION_ID",
            "KEYSTROKE_ID",
            "PRESS_TIME",
            "RELEASE_TIME"
        ]

        df = df.dropna(
            subset=essential
        )

        # ----------------------------------------------------
        # Initial timestamp validation
        # ----------------------------------------------------

        invalid = (
            df["RELEASE_TIME"] <
            df["PRESS_TIME"]
        )

        result["invalid_timestamp_rows"] = (
            int(invalid.sum())
        )

        df = df.loc[~invalid].copy()

        # ----------------------------------------------------
        # Remove impossible negative/zero dwell
        #
        # Zero dwell can technically occur in browser logs,
        # so we DO NOT automatically remove it here.
        # Negative dwell is impossible.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Process each test section independently
        # ----------------------------------------------------

        processed_sections = []

        section_groups = df.groupby(
            "TEST_SECTION_ID",
            sort=False
        )

        result["sections_total"] = (
            df["TEST_SECTION_ID"].nunique()
        )

        for section_id, section in section_groups:

            if len(section) < MIN_KEYSTROKES_PER_SECTION:
                continue

            features = extract_features(section)

            processed_sections.append(
                features
            )

        if not processed_sections:

            result["status"] = "NO_VALID_SECTIONS"

            result["error"] = (
                "No sections met minimum length."
            )

            return result

        result["sections_valid"] = len(
            processed_sections
        )

        final_df = pd.concat(
            processed_sections,
            ignore_index=True
        )

        # ----------------------------------------------------
        # Final ordering
        # ----------------------------------------------------

        final_df = final_df.sort_values(
            [
                "PARTICIPANT_ID",
                "TEST_SECTION_ID",
                "PRESS_TIME",
                "KEYSTROKE_ID"
            ]
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Save selected columns
        # ----------------------------------------------------

        output_columns = [
            # identifiers
            "PARTICIPANT_ID",
            "TEST_SECTION_ID",
            "KEYSTROKE_ID",

            # original key information
            "LETTER",
            "KEYCODE",

            # timestamps retained for auditability
            "PRESS_TIME",
            "RELEASE_TIME",

            # 16 behavioral features
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

        final_df = final_df[
            output_columns
        ]

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR /
            f"{filename_id}_features.parquet"
        )

        try:

            final_df.to_parquet(
                output_file,
                index=False
            )

        except ImportError:

            # If pyarrow/fastparquet isn't installed,
            # save CSV instead.

            output_file = (
                OUTPUT_DIR /
                f"{filename_id}_features.csv"
            )

            final_df.to_csv(
                output_file,
                index=False
            )

        result["rows_final"] = len(final_df)

        return result

    except Exception as e:

        result["status"] = "ERROR"
        result["error"] = str(e)

        return result


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("AALTO PARTICIPANT-BY-PARTICIPANT PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Find participant files
    # --------------------------------------------------------

    files = sorted(
        RAW_DIR.glob("*_keystrokes.txt")
    )

    if not files:

        raise FileNotFoundError(
            f"No *_keystrokes.txt files found in {RAW_DIR}"
        )

    print(
        f"\nFound {len(files):,} participant files."
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    reports = []

    for index, file_path in enumerate(files, start=1):

        print(
            f"\n[{index:,}/{len(files):,}] "
            f"{file_path.name}"
        )

        result = process_participant_file(
            file_path
        )

        reports.append(result)

        print(
            f"Status: {result['status']} | "
            f"Rows: {result['rows_loaded']:,} → "
            f"{result['rows_final']:,}"
        )

        if result["error"]:

            print(
                f"WARNING: {result['error']}"
            )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_df = pd.DataFrame(reports)

    report_file = (
        REPORT_DIR /
        "preprocessing_report.csv"
    )

    report_df.to_csv(
        report_file,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    successful = (
        report_df["status"] == "OK"
    ).sum()

    failed = len(report_df) - successful

    total_rows = (
        report_df["rows_loaded"]
        .sum()
    )

    final_rows = (
        report_df["rows_final"]
        .sum()
    )

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Participant files : {len(files):,}"
    )

    print(
        f"Successful         : {successful:,}"
    )

    print(
        f"Failed              : {failed:,}"
    )

    print(
        f"Raw rows            : {total_rows:,}"
    )

    print(
        f"Processed rows      : {final_rows:,}"
    )

    print(
        f"Processing time     : {elapsed / 60:.2f} minutes"
    )

    print(
        f"\nReport saved to:"
        f"\n{report_file}"
    )

    print(
        f"\nFeature files saved to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
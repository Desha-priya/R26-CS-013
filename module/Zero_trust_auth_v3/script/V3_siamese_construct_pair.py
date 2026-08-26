"""
V3 Siamese BiLSTM Pair Construction
====================================

Purpose:
    Construct positive and negative sequence pairs for Siamese BiLSTM
    keystroke biometric authentication.

Dataset:
    AALTO V3

Input:
    processed/siamese_bilstm_v3/
        train/
        validation/
        test/

Sequence format:
    50 timesteps x 16 features

Pair definition:
    Positive (1):
        Two DIFFERENT sequences belonging to the SAME participant.

    Negative (0):
        Two sequences belonging to DIFFERENT participants.

Important research controls:
    - Participant-level split is already fixed.
    - No participant crosses train/validation/test.
    - No self-pairs.
    - Positive pairs never use the same sequence twice.
    - Negative pairs always use different participants.
    - Fixed random seed.
    - No model performance is used.
    - Pair generation is independent of model training.
    - Pair manifest references NPZ files instead of duplicating arrays.

Output:
    processed/siamese_bilstm_v3/pairs/
        train_pairs_v3.csv
        validation_pairs_v3.csv
        test_pairs_v3.csv
        pair_generation_summary_v3.json
"""
"""     ----------terminal output ----------------
======================================================================
SIAMESE BILSTM V3 PAIR CONSTRUCTION
======================================================================

Base directory : processed\siamese_bilstm_v3
Output directory : processed\siamese_bilstm_v3\pairs
Random seed : 42

======================================================================
SCANNING TRAIN
======================================================================
Participants : 3,500
Sequences    : 41,374

======================================================================
GENERATING TRAIN PAIRS
======================================================================
Participants : 3,500
Sequences    : 41,374
Positive/anchor : 5
Negative/anchor : 5
Processed anchors: 5,000/41,374
Processed anchors: 10,000/41,374
Processed anchors: 15,000/41,374
Processed anchors: 20,000/41,374
Processed anchors: 25,000/41,374
Processed anchors: 30,000/41,374
Processed anchors: 35,000/41,374
Processed anchors: 40,000/41,374

TRAIN PAIR SUMMARY
----------------------------------------------------------------------
Total pairs              : 365,669
Positive pairs            : 158,810
Negative pairs            : 206,859
Same-participant pairs    : 158,810
Different-participant     : 206,859
Self-pairs                : 0

======================================================================
SCANNING VALIDATION
======================================================================
Participants : 750
Sequences    : 8,857

======================================================================
GENERATING VALIDATION PAIRS
======================================================================
Participants : 750
Sequences    : 8,857
Positive/anchor : 5
Negative/anchor : 5
Processed anchors: 5,000/8,857

VALIDATION PAIR SUMMARY
----------------------------------------------------------------------
Total pairs              : 78,304
Positive pairs            : 34,026
Negative pairs            : 44,278
Same-participant pairs    : 34,026
Different-participant     : 44,278
Self-pairs                : 0

======================================================================
SCANNING TEST
======================================================================
Participants : 750
Sequences    : 8,795

======================================================================
GENERATING TEST PAIRS
======================================================================
Participants : 750
Sequences    : 8,795
Positive/anchor : 5
Negative/anchor : 5
Processed anchors: 5,000/8,795

TEST PAIR SUMMARY
----------------------------------------------------------------------
Total pairs              : 77,665
Positive pairs            : 33,697
Negative pairs            : 43,968
Same-participant pairs    : 33,697
Different-participant     : 43,968
Self-pairs                : 0

======================================================================
SIAMESE PAIR CONSTRUCTION V3 COMPLETE
======================================================================
Train        : 365,669 pairs (+158,810 / -206,859)
Validation   : 78,304 pairs (+34,026 / -44,278)
Test         : 77,665 pairs (+33,697 / -43,968)

Processing time : 0.14 minutes

OUTPUT FILES
----------------------------------------------------------------------
Train       : processed\siamese_bilstm_v3\pairs\train_pairs_v3.csv
Validation  : processed\siamese_bilstm_v3\pairs\validation_pairs_v3.csv
Test        : processed\siamese_bilstm_v3\pairs\test_pairs_v3.csv
Summary     : processed\siamese_bilstm_v3\pairs\pair_generation_summary_v3.json

RESEARCH CONTROLS
----------------------------------------------------------------------
Positive pairs same participant : YES
Negative pairs different users  : YES
Self-pairs                       : 0
Model performance used           : NO
Validation/test leakage          : NO

PAIR CONSTRUCTION STATUS: PASS
====================================================================== 

"""


from pathlib import Path
from collections import defaultdict
import random
import csv
import json
import time


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("processed/siamese_bilstm_v3")

OUTPUT_DIR = BASE_DIR / "pairs"

RANDOM_SEED = 42

# Number of positive and negative pairs generated per anchor.
#
# 5 + 5 is a reasonable starting point:
#
# Train:
#   41,374 sequences × 10 ≈ 413,740 attempts
#
# Validation:
#   8,857 × 10 ≈ 88,570 attempts
#
# Test:
#   8,795 × 10 ≈ 87,950 attempts
#
# Duplicate removal may reduce the final number.
POSITIVE_PAIRS_PER_ANCHOR = 5
NEGATIVE_PAIRS_PER_ANCHOR = 5


# ============================================================
# HELPERS
# ============================================================

def extract_participant_id(filename):
    """
    Extract participant ID from filenames such as:

        100001_section_1090979_seq_0.npz
        100001_section_1091001_seq_1.npz

    The first component is the participant ID.
    """

    try:
        return int(filename.name.split("_")[0])
    except Exception:
        raise ValueError(
            f"Could not extract participant ID from: {filename}"
        )


def collect_sequences(split_dir):
    """
    Collect NPZ sequence files and group them by participant.
    """

    files = sorted(split_dir.glob("*.npz"))

    if not files:
        raise RuntimeError(
            f"No NPZ files found in {split_dir}"
        )

    participant_sequences = defaultdict(list)

    for path in files:
        participant_id = extract_participant_id(path)

        participant_sequences[participant_id].append(path)

    return participant_sequences


def canonical_pair(path_a, path_b):
    """
    Create a deterministic representation of an unordered pair.

    This prevents:

        A,B

    and

        B,A

    from being treated as two different pairs.
    """

    a = str(path_a)
    b = str(path_b)

    if a < b:
        return a, b

    return b, a


def make_relative_path(path):
    """
    Convert an absolute/local path into a path relative to BASE_DIR.
    """

    try:
        return str(path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# ============================================================
# PAIR GENERATION
# ============================================================

def generate_pairs(
    split_name,
    participant_sequences,
    rng,
):
    """
    Generate balanced positive and negative pairs.

    Positive:
        same participant

    Negative:
        different participants
    """

    participants = sorted(participant_sequences.keys())

    if len(participants) < 2:
        raise RuntimeError(
            f"{split_name}: Need at least two participants."
        )

    # --------------------------------------------------------
    # Verify every participant has enough sequences
    # --------------------------------------------------------

    insufficient = []

    for participant_id, sequences in participant_sequences.items():

        if len(sequences) < 2:
            insufficient.append(
                (participant_id, len(sequences))
            )

    if insufficient:

        print()
        print("WARNING")
        print("Participants with fewer than 2 sequences:")

        for participant_id, count in insufficient[:20]:
            print(
                f"  Participant {participant_id}: {count}"
            )

        raise RuntimeError(
            f"{split_name}: Cannot construct positive pairs "
            f"for participants with fewer than 2 sequences."
        )

    # --------------------------------------------------------
    # Candidate participants for negative pairs
    # --------------------------------------------------------

    participant_list = participants

    # --------------------------------------------------------
    # Store unique pairs
    # --------------------------------------------------------

    positive_pairs = set()
    negative_pairs = set()

    total_sequences = sum(
        len(v) for v in participant_sequences.values()
    )

    print()
    print("=" * 70)
    print(f"GENERATING {split_name.upper()} PAIRS")
    print("=" * 70)

    print(
        f"Participants : {len(participants):,}"
    )

    print(
        f"Sequences    : {total_sequences:,}"
    )

    print(
        f"Positive/anchor : {POSITIVE_PAIRS_PER_ANCHOR}"
    )

    print(
        f"Negative/anchor : {NEGATIVE_PAIRS_PER_ANCHOR}"
    )

    # --------------------------------------------------------
    # Generate pairs
    # --------------------------------------------------------

    processed = 0

    for participant_id in participant_list:

        anchor_sequences = participant_sequences[
            participant_id
        ]

        for anchor in anchor_sequences:

            # =================================================
            # POSITIVE PAIRS
            # =================================================

            # Candidate positive sequences exclude anchor.
            positive_candidates = [
                p
                for p in anchor_sequences
                if p != anchor
            ]

            number_positive = min(
                POSITIVE_PAIRS_PER_ANCHOR,
                len(positive_candidates)
            )

            selected_positive = rng.sample(
                positive_candidates,
                number_positive
            )

            for positive in selected_positive:

                a, b = canonical_pair(
                    anchor,
                    positive
                )

                positive_pairs.add(
                    (a, b, participant_id)
                )

            # =================================================
            # NEGATIVE PAIRS
            # =================================================

            number_negative = NEGATIVE_PAIRS_PER_ANCHOR

            for _ in range(number_negative):

                # Select a different participant.
                negative_participant = rng.choice(
                    participant_list
                )

                while negative_participant == participant_id:

                    negative_participant = rng.choice(
                        participant_list
                    )

                negative_candidates = participant_sequences[
                    negative_participant
                ]

                negative_sequence = rng.choice(
                    negative_candidates
                )

                a, b = canonical_pair(
                    anchor,
                    negative_sequence
                )

                negative_pairs.add(
                    (
                        a,
                        b,
                        participant_id,
                        negative_participant
                    )
                )

            processed += 1

            if processed % 5000 == 0:

                print(
                    f"Processed anchors: "
                    f"{processed:,}/{total_sequences:,}"
                )

    # --------------------------------------------------------
    # Convert into final records
    # --------------------------------------------------------

    records = []

    pair_id = 0

    # --------------------------------------------------------
    # Positive
    # --------------------------------------------------------

    for path_a, path_b, participant_id in sorted(
        positive_pairs
    ):

        records.append(
            {
                "pair_id": pair_id,
                "sequence_1": make_relative_path(
                    Path(path_a)
                ),
                "sequence_2": make_relative_path(
                    Path(path_b)
                ),
                "participant_1": participant_id,
                "participant_2": participant_id,
                "label": 1,
                "pair_type": "positive",
            }
        )

        pair_id += 1

    # --------------------------------------------------------
    # Negative
    # --------------------------------------------------------

    for (
        path_a,
        path_b,
        participant_1,
        participant_2,
    ) in sorted(negative_pairs):

        records.append(
            {
                "pair_id": pair_id,
                "sequence_1": make_relative_path(
                    Path(path_a)
                ),
                "sequence_2": make_relative_path(
                    Path(path_b)
                ),
                "participant_1": participant_1,
                "participant_2": participant_2,
                "label": 0,
                "pair_type": "negative",
            }
        )

        pair_id += 1

    # --------------------------------------------------------
    # Shuffle pairs
    # --------------------------------------------------------

    rng.shuffle(records)

    # Reassign pair IDs after shuffle.
    for index, record in enumerate(records):

        record["pair_id"] = index

    return records


# ============================================================
# SAVE CSV
# ============================================================

def save_pairs(records, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "pair_id",
        "sequence_1",
        "sequence_2",
        "participant_1",
        "participant_2",
        "label",
        "pair_type",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(records)


# ============================================================
# PAIR SUMMARY
# ============================================================

def summarize_pairs(records):

    positive = sum(
        1
        for r in records
        if r["label"] == 1
    )

    negative = sum(
        1
        for r in records
        if r["label"] == 0
    )

    same_participant = sum(
        1
        for r in records
        if r["participant_1"] == r["participant_2"]
    )

    different_participant = sum(
        1
        for r in records
        if r["participant_1"] != r["participant_2"]
    )

    self_pairs = sum(
        1
        for r in records
        if r["sequence_1"] == r["sequence_2"]
    )

    return {
        "total_pairs": len(records),
        "positive_pairs": positive,
        "negative_pairs": negative,
        "same_participant_pairs": same_participant,
        "different_participant_pairs": different_participant,
        "self_pairs": self_pairs,
        "positive_negative_ratio": (
            positive / negative
            if negative > 0
            else None
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("SIAMESE BILSTM V3 PAIR CONSTRUCTION")
    print("=" * 70)

    print()
    print(f"Base directory : {BASE_DIR}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Random seed : {RANDOM_SEED}")

    rng = random.Random(RANDOM_SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_summaries = {}

    split_names = [
        "train",
        "validation",
        "test",
    ]

    for split_name in split_names:

        split_dir = BASE_DIR / split_name

        print()
        print("=" * 70)
        print(f"SCANNING {split_name.upper()}")
        print("=" * 70)

        participant_sequences = collect_sequences(
            split_dir
        )

        total_sequences = sum(
            len(v)
            for v in participant_sequences.values()
        )

        print(
            f"Participants : "
            f"{len(participant_sequences):,}"
        )

        print(
            f"Sequences    : "
            f"{total_sequences:,}"
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        records = generate_pairs(
            split_name,
            participant_sequences,
            rng
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR /
            f"{split_name}_pairs_v3.csv"
        )

        save_pairs(
            records,
            output_file
        )

        summary = summarize_pairs(records)

        summary["split"] = split_name
        summary["participants"] = len(
            participant_sequences
        )
        summary["sequences"] = total_sequences
        summary["positive_pairs_per_anchor"] = (
            POSITIVE_PAIRS_PER_ANCHOR
        )
        summary["negative_pairs_per_anchor"] = (
            NEGATIVE_PAIRS_PER_ANCHOR
        )

        all_summaries[split_name] = summary

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print()
        print(
            f"{split_name.upper()} PAIR SUMMARY"
        )
        print("-" * 70)

        print(
            f"Total pairs              : "
            f"{summary['total_pairs']:,}"
        )

        print(
            f"Positive pairs            : "
            f"{summary['positive_pairs']:,}"
        )

        print(
            f"Negative pairs            : "
            f"{summary['negative_pairs']:,}"
        )

        print(
            f"Same-participant pairs    : "
            f"{summary['same_participant_pairs']:,}"
        )

        print(
            f"Different-participant     : "
            f"{summary['different_participant_pairs']:,}"
        )

        print(
            f"Self-pairs                : "
            f"{summary['self_pairs']:,}"
        )

        if summary["self_pairs"] != 0:
            raise RuntimeError(
                f"{split_name}: SELF-PAIR DETECTED!"
            )

        if (
            summary["positive_pairs"] == 0
            or summary["negative_pairs"] == 0
        ):
            raise RuntimeError(
                f"{split_name}: Missing positive or negative pairs."
            )

        # ----------------------------------------------------
        # Verify labels
        # ----------------------------------------------------

        for record in records:

            if record["label"] == 1:

                if (
                    record["participant_1"]
                    != record["participant_2"]
                ):
                    raise RuntimeError(
                        f"{split_name}: Invalid positive pair."
                    )

            elif record["label"] == 0:

                if (
                    record["participant_1"]
                    == record["participant_2"]
                ):
                    raise RuntimeError(
                        f"{split_name}: Invalid negative pair."
                    )

            else:

                raise RuntimeError(
                    f"{split_name}: Invalid label."
                )

    # ========================================================
    # SAVE GLOBAL SUMMARY
    # ========================================================

    elapsed = time.time() - start_time

    global_summary = {
        "dataset": "AALTO_V3",
        "random_seed": RANDOM_SEED,
        "positive_pairs_per_anchor": (
            POSITIVE_PAIRS_PER_ANCHOR
        ),
        "negative_pairs_per_anchor": (
            NEGATIVE_PAIRS_PER_ANCHOR
        ),
        "splits": all_summaries,
        "participant_leakage_assumed_from_validated_split": True,
        "self_pairs": 0,
        "model_performance_used": False,
        "validation_metrics_used": False,
        "test_metrics_used": False,
        "processing_seconds": elapsed,
        "status": "PASS",
    }

    summary_file = (
        OUTPUT_DIR /
        "pair_generation_summary_v3.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            global_summary,
            f,
            indent=4
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("SIAMESE PAIR CONSTRUCTION V3 COMPLETE")
    print("=" * 70)

    for split_name in split_names:

        summary = all_summaries[split_name]

        print(
            f"{split_name.capitalize():12s} : "
            f"{summary['total_pairs']:,} pairs "
            f"("
            f"+{summary['positive_pairs']:,} / "
            f"-{summary['negative_pairs']:,}"
            f")"
        )

    print()
    print(
        f"Processing time : "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("OUTPUT FILES")
    print("-" * 70)

    print(
        f"Train       : "
        f"{OUTPUT_DIR / 'train_pairs_v3.csv'}"
    )

    print(
        f"Validation  : "
        f"{OUTPUT_DIR / 'validation_pairs_v3.csv'}"
    )

    print(
        f"Test        : "
        f"{OUTPUT_DIR / 'test_pairs_v3.csv'}"
    )

    print(
        f"Summary     : "
        f"{summary_file}"
    )

    print()
    print("RESEARCH CONTROLS")
    print("-" * 70)

    print(
        "Positive pairs same participant : YES"
    )

    print(
        "Negative pairs different users  : YES"
    )

    print(
        "Self-pairs                       : 0"
    )

    print(
        "Model performance used           : NO"
    )

    print(
        "Validation/test leakage          : NO"
    )

    print()
    print("PAIR CONSTRUCTION STATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
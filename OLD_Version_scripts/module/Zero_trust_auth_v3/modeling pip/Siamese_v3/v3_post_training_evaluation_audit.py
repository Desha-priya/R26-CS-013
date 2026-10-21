"""
======================================================================
AALTO V3 - POST-TRAINING EVALUATION AUDIT
======================================================================

Purpose
-------
Audit the EXISTING trained Siamese BiLSTM checkpoint without retraining.

Research controls
-----------------
- Exact architecture copied from V3_Train_Siamese_biLstm.py
- Existing checkpoint only; NO weight updates
- Train-only scaler only
- Validation threshold selection only
- Test threshold is frozen from validation
- Test data is never used for model/threshold selection
- Participant-level leakage audit
- Pair-label consistency audit
- Exact checkpoint/state-dict architecture audit
- Validation/test metrics and EER
- Sequence embedding cache for faster evaluation

IMPORTANT
---------
This script does NOT modify the trained model.
It does NOT retrain or fine-tune anything.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

BASE_DIR = Path("processed/siamese_bilstm_v3")

TRAIN_PAIR_FILE = BASE_DIR / "pairs" / "train_pairs_v3.csv"
VALIDATION_PAIR_FILE = BASE_DIR / "pairs" / "validation_pairs_v3.csv"
TEST_PAIR_FILE = BASE_DIR / "pairs" / "test_pairs_v3.csv"

SCALER_FILE = (
    BASE_DIR
    / "reports"
    / "normalization_audit"
    / "train_only_scaler_v3.npz"
)

CHECKPOINT_FILE = (
    BASE_DIR
    / "model_training_v3"
    / "checkpoints"
    / "best_siamese_bilstm_v3.pt"
)

OUTPUT_DIR = (
    BASE_DIR / "post_training_evaluation_v3"
)

REPORT_DIR = OUTPUT_DIR / "reports"


# ======================================================================
# EXACT TRAINING ARCHITECTURE
# ======================================================================

SEQUENCE_LENGTH = 50
FEATURE_COUNT = 16

LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
EMBEDDING_SIZE = 64
DROPOUT = 0.30

MARGIN = 1.0


# ======================================================================
# DEVICE
# ======================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================================================
# EXACT MODEL FROM TRAINING SCRIPT
# ======================================================================

class BiLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=FEATURE_COUNT,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT
            if LSTM_LAYERS > 1
            else 0.0,
        )

        self.embedding = nn.Sequential(
            nn.Linear(
                LSTM_HIDDEN_SIZE * 2,
                128
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                128,
                EMBEDDING_SIZE
            ),
        )

    def forward(self, x):
        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        embedding = self.embedding(
            last_output
        )

        embedding = nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding


class SiameseBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # ONE shared encoder.
        self.encoder = BiLSTMEncoder()

    def forward(self, x1, x2):
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)

        distance = torch.norm(
            embedding1 - embedding2,
            p=2,
            dim=1
        )

        return (
            embedding1,
            embedding2,
            distance
        )


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        positive_loss = (
            label * torch.pow(distance, 2)
        )

        negative_loss = (
            (1 - label)
            * torch.pow(
                torch.clamp(
                    self.margin - distance,
                    min=0.0
                ),
                2
            )
        )

        loss = (
            positive_loss
            + negative_loss
        )

        return loss.mean()


# ======================================================================
# DIRECTORIES
# ======================================================================

def create_directories():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ======================================================================
# SCALER
# ======================================================================

def load_train_scaler():
    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"TRAIN-ONLY SCALER NOT FOUND:\n{SCALER_FILE}"
        )

    scaler = np.load(
        SCALER_FILE,
        allow_pickle=False
    )

    mean_keys = [
        "mean",
        "means",
        "feature_mean",
        "feature_means",
        "train_mean",
        "train_means",
    ]

    std_keys = [
        "std",
        "stds",
        "feature_std",
        "feature_stds",
        "train_std",
        "train_stds",
    ]

    mean = None
    std = None

    for key in mean_keys:
        if key in scaler:
            mean = scaler[key]
            break

    for key in std_keys:
        if key in scaler:
            std = scaler[key]
            break

    if mean is None or std is None:
        raise RuntimeError(
            "Could not identify mean/std arrays in scaler. "
            f"Available arrays: {list(scaler.keys())}"
        )

    mean = np.asarray(
        mean,
        dtype=np.float32
    ).reshape(-1)

    std = np.asarray(
        std,
        dtype=np.float32
    ).reshape(-1)

    if len(mean) != FEATURE_COUNT:
        raise ValueError(
            f"Scaler mean has {len(mean)} features; "
            f"expected {FEATURE_COUNT}."
        )

    if len(std) != FEATURE_COUNT:
        raise ValueError(
            f"Scaler std has {len(std)} features; "
            f"expected {FEATURE_COUNT}."
        )

    if not np.all(np.isfinite(mean)):
        raise ValueError("Scaler mean contains NaN/Inf.")

    if not np.all(np.isfinite(std)):
        raise ValueError("Scaler std contains NaN/Inf.")

    if np.any(std <= 0):
        raise ValueError(
            "Scaler contains zero/negative standard deviation."
        )

    return mean, std


# ======================================================================
# NPZ LOADER
# ======================================================================

def load_npz_sequence(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Sequence file not found:\n{path}"
        )

    data = np.load(
        path,
        allow_pickle=False
    )

    preferred_keys = [
        "X",
        "x",
        "sequence",
        "features",
        "data",
        "arr_0",
    ]

    array = None

    for key in preferred_keys:
        if key in data:
            candidate = data[key]

            if isinstance(candidate, np.ndarray):
                array = candidate
                break

    if array is None:
        candidates = []

        for key in data.keys():
            candidate = data[key]

            if (
                isinstance(candidate, np.ndarray)
                and candidate.ndim == 2
            ):
                candidates.append(candidate)

        if len(candidates) == 1:
            array = candidates[0]

        elif len(candidates) > 1:
            raise ValueError(
                f"Multiple possible sequence arrays in {path}: "
                f"{list(data.keys())}"
            )

    if array is None:
        raise ValueError(
            f"Could not identify sequence array in {path}. "
            f"Keys: {list(data.keys())}"
        )

    array = np.asarray(
        array,
        dtype=np.float32
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURE_COUNT
    )

    if array.shape != expected_shape:
        raise ValueError(
            f"Invalid shape in {path}: "
            f"expected {expected_shape}, got {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"NaN/Inf detected in {path}"
        )

    return array


def normalize_sequence(sequence, mean, std):
    sequence = (
        sequence - mean
    ) / std

    if not np.all(np.isfinite(sequence)):
        raise ValueError(
            "Normalization produced NaN/Inf."
        )

    return sequence.astype(np.float32)


# ======================================================================
# PAIR FILE AUDIT
# ======================================================================

REQUIRED_COLUMNS = [
    "pair_id",
    "sequence_1",
    "sequence_2",
    "participant_1",
    "participant_2",
    "label",
    "pair_type",
]


def load_pair_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Pair file not found:\n{path}"
        )

    df = pd.read_csv(path)

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    if len(df) == 0:
        raise ValueError(
            f"Empty pair file: {path}"
        )

    labels = set(
        df["label"].astype(int).unique().tolist()
    )

    if labels != {0, 1}:
        raise ValueError(
            f"Expected labels {{0,1}} in {path}, got {labels}"
        )

    return df


def audit_pair_consistency(df, name):
    labels = df["label"].astype(int)
    p1 = df["participant_1"].astype(int)
    p2 = df["participant_2"].astype(int)

    positive_different = int(
        ((labels == 1) & (p1 != p2)).sum()
    )

    negative_same = int(
        ((labels == 0) & (p1 == p2)).sum()
    )

    invalid_pair_type = 0

    if "pair_type" in df.columns:
        expected_positive = (
            (labels == 1)
            & (df["pair_type"].astype(str) != "positive")
        )

        expected_negative = (
            (labels == 0)
            & (df["pair_type"].astype(str) != "negative")
        )

        invalid_pair_type = int(
            expected_positive.sum()
            + expected_negative.sum()
        )

    print()
    print(name)
    print(
        f"Positive pairs with different users : "
        f"{positive_different}"
    )
    print(
        f"Negative pairs with same user       : "
        f"{negative_same}"
    )
    print(
        f"Invalid pair_type/label combinations : "
        f"{invalid_pair_type}"
    )

    if (
        positive_different != 0
        or negative_same != 0
        or invalid_pair_type != 0
    ):
        raise ValueError(
            f"PAIR CONSISTENCY FAILED for {name}"
        )


# ======================================================================
# PARTICIPANT LEAKAGE
# ======================================================================

def participant_set_from_pairs(df):
    return set(
        df["participant_1"].astype(int).tolist()
    ) | set(
        df["participant_2"].astype(int).tolist()
    )


def audit_participant_leakage(
    train_df,
    validation_df,
    test_df
):
    train_users = participant_set_from_pairs(train_df)
    validation_users = participant_set_from_pairs(validation_df)
    test_users = participant_set_from_pairs(test_df)

    train_validation = train_users & validation_users
    train_test = train_users & test_users
    validation_test = validation_users & test_users

    print()
    print("=" * 70)
    print("PARTICIPANT LEAKAGE AUDIT")
    print("=" * 70)

    print(
        f"Train participants      : {len(train_users):,}"
    )
    print(
        f"Validation participants : {len(validation_users):,}"
    )
    print(
        f"Test participants       : {len(test_users):,}"
    )

    print(
        f"Train ∩ Validation : {len(train_validation)}"
    )
    print(
        f"Train ∩ Test       : {len(train_test)}"
    )
    print(
        f"Validation ∩ Test  : {len(validation_test)}"
    )

    if (
        train_validation
        or train_test
        or validation_test
    ):
        raise ValueError(
            "PARTICIPANT LEAKAGE DETECTED."
        )

    print("Participant leakage : PASS")

    return {
        "train_participants": len(train_users),
        "validation_participants": len(validation_users),
        "test_participants": len(test_users),
        "train_validation_intersection": len(train_validation),
        "train_test_intersection": len(train_test),
        "validation_test_intersection": len(validation_test),
    }


# ======================================================================
# EXACT CHECKPOINT LOADING
# ======================================================================

def expected_state_keys():
    return set(
        SiameseBiLSTM().state_dict().keys()
    )


def inspect_and_load_model():
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_FILE}"
        )

    print()
    print("=" * 70)
    print("LOADING EXACT TRAINING MODEL")
    print("=" * 70)

    print(
        "Checkpoint:",
        CHECKPOINT_FILE
    )

    print(
        "Device:",
        DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    model = SiameseBiLSTM().to(DEVICE)

    actual_keys = set(
        state_dict.keys()
    )

    expected_keys = expected_state_keys()

    missing = sorted(
        expected_keys - actual_keys
    )

    unexpected = sorted(
        actual_keys - expected_keys
    )

    if missing or unexpected:
        print()
        print("EXPECTED MODEL KEYS:")
        for key in sorted(expected_keys):
            print(" ", key)

        print()
        print("CHECKPOINT KEYS:")
        for key in sorted(actual_keys):
            print(" ", key)

        raise RuntimeError(
            "\nCheckpoint/model architecture mismatch.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}\n"
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Checkpoint epoch       : "
        f"{checkpoint.get('epoch', 'N/A')}"
    )

    print(
        f"Checkpoint val loss    : "
        f"{checkpoint.get('validation_loss', 'N/A')}"
    )

    print(
        f"Model parameters       : "
        f"{parameter_count:,}"
    )

    print(
        "Architecture match     : PASS"
    )

    print(
        "Model loaded           : PASS"
    )

    print(
        "Model mode             : EVAL"
    )

    return model, checkpoint


# ======================================================================
# FAST EMBEDDING CACHE
# ======================================================================

def unique_sequence_paths(df):
    paths = set(
        df["sequence_1"].astype(str).tolist()
    )

    paths.update(
        df["sequence_2"].astype(str).tolist()
    )

    return sorted(paths)


def build_embedding_cache(
    model,
    df,
    mean,
    std,
    batch_size=512
):
    paths = unique_sequence_paths(df)

    print()
    print(
        f"Unique sequences to encode : {len(paths):,}"
    )

    cache = {}

    model.eval()

    batch_arrays = []
    batch_paths = []

    def flush_batch():
        if not batch_arrays:
            return

        batch = np.stack(
            batch_arrays,
            axis=0
        )

        tensor = torch.from_numpy(
            batch
        ).to(
            DEVICE,
            non_blocking=True
        )

        with torch.no_grad():
            embeddings = model.encoder(
                tensor
            )

        embeddings = (
            embeddings
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        for path_text, embedding in zip(
            batch_paths,
            embeddings
        ):
            cache[path_text] = embedding

        batch_arrays.clear()
        batch_paths.clear()

    for index, path_text in enumerate(paths, start=1):
        path = BASE_DIR / path_text

        sequence = load_npz_sequence(
            path
        )

        sequence = normalize_sequence(
            sequence,
            mean,
            std
        )

        batch_arrays.append(sequence)
        batch_paths.append(path_text)

        if len(batch_arrays) >= batch_size:
            flush_batch()

        if (
            index % 1000 == 0
            or index == len(paths)
        ):
            print(
                f"Encoded sequences: "
                f"{index:,}/{len(paths):,}"
            )

    flush_batch()

    if len(cache) != len(paths):
        raise RuntimeError(
            "Embedding cache size mismatch."
        )

    return cache


# ======================================================================
# PAIR DISTANCES FROM CACHED EMBEDDINGS
# ======================================================================

def compute_pair_distances(df, cache):
    sequence_1 = df["sequence_1"].astype(str)
    sequence_2 = df["sequence_2"].astype(str)

    labels = (
        df["label"]
        .astype(int)
        .to_numpy()
    )

    missing = []

    for path in pd.unique(
        pd.concat(
            [sequence_1, sequence_2],
            ignore_index=True
        )
    ):
        if path not in cache:
            missing.append(path)

    if missing:
        raise RuntimeError(
            f"{len(missing)} pair sequences missing "
            "from embedding cache."
        )

    e1 = np.stack(
        [
            cache[path]
            for path in sequence_1
        ],
        axis=0
    )

    e2 = np.stack(
        [
            cache[path]
            for path in sequence_2
        ],
        axis=0
    )

    distances = np.linalg.norm(
        e1 - e2,
        axis=1
    )

    if not np.all(
        np.isfinite(distances)
    ):
        raise ValueError(
            "Non-finite pair distances detected."
        )

    return distances, labels


# ======================================================================
# METRICS
# ======================================================================

def pair_metrics(
    distances,
    labels,
    threshold
):
    predictions = (
        distances <= threshold
    ).astype(np.int8)

    genuine = labels == 1
    impostor = labels == 0

    tp = int(
        np.sum(
            predictions[genuine] == 1
        )
    )

    fn = int(
        np.sum(
            predictions[genuine] == 0
        )
    )

    fp = int(
        np.sum(
            predictions[impostor] == 1
        )
    )

    tn = int(
        np.sum(
            predictions[impostor] == 0
        )
    )

    tpr = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    far = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    frr = (
        fn / (tp + fn)
        if tp + fn
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn)
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "far": float(far),
        "frr": float(frr),
        "tpr": float(tpr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_validation_threshold(
    distances,
    labels
):
    thresholds = np.linspace(
        float(distances.min()),
        float(distances.max()),
        501
    )

    best = None

    for threshold in thresholds:
        metrics = pair_metrics(
            distances,
            labels,
            threshold
        )

        if (
            best is None
            or metrics["accuracy"]
            > best["accuracy"]
        ):
            best = metrics

    return best


def find_eer(
    distances,
    labels
):
    thresholds = np.linspace(
        float(distances.min()),
        float(distances.max()),
        2001
    )

    best = None

    for threshold in thresholds:
        metrics = pair_metrics(
            distances,
            labels,
            threshold
        )

        difference = abs(
            metrics["far"]
            - metrics["frr"]
        )

        if (
            best is None
            or difference
            < best["difference"]
        ):
            best = {
                "threshold": float(threshold),
                "far": metrics["far"],
                "frr": metrics["frr"],
                "difference": float(
                    difference
                ),
            }

    best["eer"] = (
        best["far"]
        + best["frr"]
    ) / 2.0

    return best


# ======================================================================
# MAIN
# ======================================================================

def main():
    start_time = time.time()

    create_directories()

    print("=" * 70)
    print("POST-TRAINING EVALUATION AUDIT V3")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print("Model training : NO")
    print("Weight updates : NO")
    print("Test threshold tuning : NO")
    print("Validation threshold : YES")
    print("Test threshold : FROZEN VALIDATION THRESHOLD")

    # --------------------------------------------------------------
    # Required files
    # --------------------------------------------------------------

    required_files = [
        TRAIN_PAIR_FILE,
        VALIDATION_PAIR_FILE,
        TEST_PAIR_FILE,
        SCALER_FILE,
        CHECKPOINT_FILE,
    ]

    missing_files = [
        str(p)
        for p in required_files
        if not p.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required input files missing:\n"
            + "\n".join(missing_files)
        )

    print()
    print(
        "All required input files found."
    )

    # --------------------------------------------------------------
    # Load pair manifests
    # --------------------------------------------------------------

    train_df = load_pair_file(
        TRAIN_PAIR_FILE
    )

    validation_df = load_pair_file(
        VALIDATION_PAIR_FILE
    )

    test_df = load_pair_file(
        TEST_PAIR_FILE
    )

    print()
    print("=" * 70)
    print("PAIR FILE SUMMARY")
    print("=" * 70)

    for name, df in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df),
    ]:
        print()
        print(name)
        print(
            f"Rows       : {len(df):,}"
        )
        print(
            f"Positive   : "
            f"{int((df['label'] == 1).sum()):,}"
        )
        print(
            f"Negative   : "
            f"{int((df['label'] == 0).sum()):,}"
        )

    # --------------------------------------------------------------
    # Pair consistency
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("PAIR PARTICIPANT CONSISTENCY AUDIT")
    print("=" * 70)

    audit_pair_consistency(
        validation_df,
        "Validation"
    )

    audit_pair_consistency(
        test_df,
        "Test"
    )

    # --------------------------------------------------------------
    # Participant leakage
    # --------------------------------------------------------------

    leakage = audit_participant_leakage(
        train_df,
        validation_df,
        test_df
    )

    # --------------------------------------------------------------
    # Exact model
    # --------------------------------------------------------------

    model, checkpoint = inspect_and_load_model()

    # --------------------------------------------------------------
    # Train-only scaler
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING TRAIN-ONLY NORMALIZATION")
    print("=" * 70)

    mean, std = load_train_scaler()

    print(
        f"Feature count : {len(mean)}"
    )

    print(
        "Validation used for scaler : NO"
    )

    print(
        "Test used for scaler       : NO"
    )

    # --------------------------------------------------------------
    # Validation embedding evaluation
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION EMBEDDING EVALUATION")
    print("=" * 70)

    validation_cache = build_embedding_cache(
        model,
        validation_df,
        mean,
        std
    )

    validation_distances, validation_labels = (
        compute_pair_distances(
            validation_df,
            validation_cache
        )
    )

    print(
        f"Validation distances : "
        f"{len(validation_distances):,}"
    )

    validation_threshold = (
        find_best_validation_threshold(
            validation_distances,
            validation_labels
        )
    )

    validation_eer = find_eer(
        validation_distances,
        validation_labels
    )

    print()
    print(
        f"Validation threshold : "
        f"{validation_threshold['threshold']:.6f}"
    )

    print(
        f"Validation accuracy  : "
        f"{validation_threshold['accuracy']:.6f}"
    )

    print(
        f"Validation FAR       : "
        f"{validation_threshold['far']:.6f}"
    )

    print(
        f"Validation FRR       : "
        f"{validation_threshold['frr']:.6f}"
    )

    print(
        f"Validation EER       : "
        f"{validation_eer['eer']:.6f}"
    )

    # --------------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    print(
        "IMPORTANT:"
    )

    print(
        "Test data is NOT used to select "
        "the threshold."
    )

    test_cache = build_embedding_cache(
        model,
        test_df,
        mean,
        std
    )

    test_distances, test_labels = (
        compute_pair_distances(
            test_df,
            test_cache
        )
    )

    frozen_threshold = (
        validation_threshold["threshold"]
    )

    test_metrics = pair_metrics(
        test_distances,
        test_labels,
        frozen_threshold
    )

    test_eer = find_eer(
        test_distances,
        test_labels
    )

    print()
    print(
        f"Test threshold : "
        f"{frozen_threshold:.6f}"
    )

    print(
        f"Test accuracy  : "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Test FAR       : "
        f"{test_metrics['far']:.6f}"
    )

    print(
        f"Test FRR       : "
        f"{test_metrics['frr']:.6f}"
    )

    print(
        f"Test TPR       : "
        f"{test_metrics['tpr']:.6f}"
    )

    print(
        f"Test EER       : "
        f"{test_eer['eer']:.6f}"
    )

    # --------------------------------------------------------------
    # Save report
    # --------------------------------------------------------------

    total_time = (
        time.time() - start_time
    )

    summary = {
        "dataset": "AALTO_V3",
        "audit": {
            "training_performed": False,
            "weight_updates": False,
            "test_threshold_tuning": False,
            "validation_threshold_selection": True,
            "test_threshold_frozen_from_validation": True,
        },
        "checkpoint": {
            "path": str(CHECKPOINT_FILE),
            "epoch": checkpoint.get(
                "epoch"
            ),
            "validation_loss": checkpoint.get(
                "validation_loss"
            ),
            "architecture_match": True,
        },
        "model": {
            "architecture": "Siamese BiLSTM",
            "sequence_length": SEQUENCE_LENGTH,
            "feature_count": FEATURE_COUNT,
            "hidden_size": LSTM_HIDDEN_SIZE,
            "lstm_layers": LSTM_LAYERS,
            "embedding_size": EMBEDDING_SIZE,
            "dropout": DROPOUT,
            "margin": MARGIN,
            "shared_encoder_weights": True,
        },
        "data": {
            "train_pairs": len(train_df),
            "validation_pairs": len(validation_df),
            "test_pairs": len(test_df),
        },
        "participant_leakage": leakage,
        "normalization": {
            "source": "TRAIN ONLY",
            "validation_used": False,
            "test_used": False,
        },
        "threshold": {
            "selected_from": "VALIDATION",
            "value": frozen_threshold,
        },
        "validation": {
            "accuracy": validation_threshold[
                "accuracy"
            ],
            "far": validation_threshold[
                "far"
            ],
            "frr": validation_threshold[
                "frr"
            ],
            "eer": validation_eer[
                "eer"
            ],
        },
        "test": {
            "accuracy": test_metrics[
                "accuracy"
            ],
            "far": test_metrics[
                "far"
            ],
            "frr": test_metrics[
                "frr"
            ],
            "tpr": test_metrics[
                "tpr"
            ],
            "eer": test_eer[
                "eer"
            ],
            "threshold": frozen_threshold,
        },
        "research_controls": {
            "participant_level_split": True,
            "shared_siamese_weights": True,
            "train_only_normalization": True,
            "validation_used_for_threshold": True,
            "test_used_for_threshold": False,
            "test_used_for_training": False,
            "model_performance_used_for_cohort": False,
        },
        "processing_seconds": total_time,
        "processing_minutes": total_time / 60.0,
        "status": "PASS",
    }

    summary_file = (
        REPORT_DIR
        / "post_training_evaluation_audit_v3_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("POST-TRAINING EVALUATION AUDIT V3 COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Best checkpoint epoch : "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Validation threshold  : "
        f"{frozen_threshold:.6f}"
    )

    print()
    print("FINAL TEST")
    print("-" * 70)

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"FAR      : "
        f"{test_metrics['far']:.6f}"
    )

    print(
        f"FRR      : "
        f"{test_metrics['frr']:.6f}"
    )

    print(
        f"TPR      : "
        f"{test_metrics['tpr']:.6f}"
    )

    print(
        f"EER      : "
        f"{test_eer['eer']:.6f}"
    )

    print()
    print("OUTPUT")
    print("-" * 70)

    print(
        "Summary:",
        summary_file
    )

    print()
    print(
        f"Processing time : "
        f"{total_time / 60:.2f} minutes"
    )

    print()
    print("=" * 70)
    print("RESEARCH CONTROLS")
    print("=" * 70)

    print(
        "Exact training architecture : YES"
    )
    print(
        "Shared Siamese weights       : YES"
    )
    print(
        "Train-only normalization     : YES"
    )
    print(
        "Validation threshold         : YES"
    )
    print(
        "Test threshold tuning        : NO"
    )
    print(
        "Test used for training       : NO"
    )
    print(
        "Participant leakage          : NO"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
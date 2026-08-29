# ================================================================
# V3 BASELINE PERFORMANCE ANALYSIS
# ================================================================
# 
#output was something misleading basling, so need one more audit to identify what cause the problem to drom 14% eer to 25% eer in here
#so next script is V3_Score_EER_Reconciliation_Audit.py now.

#----------------------------------
# PURPOSE:
#   Independent descriptive evaluation of the frozen AALTO V3
#   Siamese BiLSTM baseline.
#
# IMPORTANT RESEARCH CONTROLS:
#   - NO training
#   - NO weight updates
#   - NO threshold tuning on test
#   - Validation threshold is loaded/frozen
#   - Train-only normalization
#   - Exact trained architecture
#   - Participant-disjoint evaluation
#
# OUTPUTS:
#   - ROC curve
#   - Genuine/impostor distance distributions
#   - FAR/FRR threshold analysis
#   - AUC
#   - EER
#   - Confusion matrix at validation threshold
#   - Distance statistics
#   - JSON + CSV reports
#
# ================================================================

import os
import json
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)

import matplotlib.pyplot as plt


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = r"processed\siamese_bilstm_v3"

CHECKPOINT = (
    BASE_DIR
    + r"\model_training_v3\checkpoints\best_siamese_bilstm_v3.pt"
)

SCALER_PATH = (
    BASE_DIR
    + r"\reports\normalization_audit\train_only_scaler_v3.npz"
)

VALIDATION_PAIRS = (
    BASE_DIR
    + r"\pairs\validation_pairs_v3.csv"
)

TEST_PAIRS = (
    BASE_DIR
    + r"\pairs\test_pairs_v3.csv"
)

TRAIN_DIR = BASE_DIR + r"\train"
VALIDATION_DIR = BASE_DIR + r"\validation"
TEST_DIR = BASE_DIR + r"\test"

OUTPUT_DIR = BASE_DIR + r"\baseline_performance_v3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42

EXPECTED_FEATURES = 16
EXPECTED_SEQUENCE_LENGTH = 50

# IMPORTANT:
# This is the threshold selected previously using validation.
# It MUST NOT be re-tuned using the test set.
FROZEN_VALIDATION_THRESHOLD = 0.552409

BATCH_SIZE = 512

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

warnings.filterwarnings("ignore")


# ================================================================
# MODEL ARCHITECTURE
# ================================================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_size=16,
        lstm_hidden=128,
        lstm_layers=2,
        embedding_dim=64,
        DROPOUT=0.30,

    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.embedding = nn.Sequential(
                    nn.Linear(
                        lstm_hidden * 2,
                        128
                    ),
                    nn.ReLU(),
                    nn.Dropout(
                        DROPOUT
                    ),
                    nn.Linear(
                        128,
                        embedding_dim
                    ),
                )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Last temporal representation
        last = output[:, -1, :]

        embedding = self.embedding(last)

        return embedding


class SiameseBiLSTM(nn.Module):

    def __init__(
        self,
        input_size=16,
        lstm_hidden=128,
        lstm_layers=2,
        embedding_dim=64,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_size=input_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            embedding_dim=embedding_dim,
        )

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        return z1, z2


# ================================================================
# CHECKPOINT LOADING
# ================================================================

def load_model():

    print("=" * 70)
    print("LOADING EXACT V3 MODEL")
    print("=" * 70)

    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Device     : {DEVICE}")

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    state_dict = checkpoint["model_state_dict"]

    model = SiameseBiLSTM(
        input_size=EXPECTED_FEATURES,
        lstm_hidden=128,
        lstm_layers=2,
        embedding_dim=64,
    ).to(DEVICE)

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)

    if missing or unexpected:

        print("\nMODEL ARCHITECTURE MISMATCH")

        if missing:
            print("\nMissing keys:")
            for key in missing:
                print(" ", key)

        if unexpected:
            print("\nUnexpected keys:")
            for key in unexpected:
                print(" ", key)

        raise RuntimeError(
            "Checkpoint/model architecture mismatch."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        f"Checkpoint epoch       : "
        f"{checkpoint.get('epoch', 'UNKNOWN')}"
    )

    print(
        f"Checkpoint val loss    : "
        f"{checkpoint.get('validation_loss', 'UNKNOWN')}"
    )

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(
        f"Model parameters       : {total_parameters:,}"
    )

    print("Architecture match     : PASS")
    print("Model loaded           : PASS")
    print("Model mode             : EVAL")

    return model


# ================================================================
# SCALER LOADING
# ================================================================

def load_scaler():

    print("\n" + "=" * 70)
    print("LOADING TRAIN-ONLY NORMALIZATION")
    print("=" * 70)

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found:\n{SCALER_PATH}"
        )

    scaler = np.load(SCALER_PATH)

    print("Scaler keys:")
    for key in scaler.files:
        print(" ", key)

    # Support common naming conventions.
    mean = None
    std = None

    possible_mean = [
        "mean",
        "means",
        "feature_mean",
        "train_mean",
        "mu",
    ]

    possible_std = [
        "std",
        "stds",
        "feature_std",
        "train_std",
        "sigma",
    ]

    for key in possible_mean:
        if key in scaler:
            mean = scaler[key]
            break

    for key in possible_std:
        if key in scaler:
            std = scaler[key]
            break

    if mean is None or std is None:

        raise RuntimeError(
            "Could not identify mean/std arrays in scaler file."
        )

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if mean.shape[0] != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Scaler mean has {mean.shape[0]} features; "
            f"expected {EXPECTED_FEATURES}."
        )

    if std.shape[0] != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Scaler std has {std.shape[0]} features; "
            f"expected {EXPECTED_FEATURES}."
        )

    std = np.where(
        np.abs(std) < 1e-12,
        1.0,
        std,
    )

    print(f"Feature count : {len(mean)}")
    print("Normalization source : TRAIN ONLY")
    print("Validation used for scaler : NO")
    print("Test used for scaler       : NO")

    return mean, std


# ================================================================
# NPZ LOADING
# ================================================================

def load_sequence(path, mean, std):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sequence not found:\n{path}"
        )

    data = np.load(path)

    # Locate array
    if "X" in data:
        X = data["X"]

    elif "x" in data:
        X = data["x"]

    elif len(data.files) == 1:
        X = data[data.files[0]]

    else:
        raise RuntimeError(
            f"Could not identify sequence array in:\n{path}\n"
            f"Available keys: {data.files}"
        )

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise RuntimeError(
            f"Invalid sequence dimensionality in {path}: "
            f"{X.shape}"
        )

    if X.shape != (
        EXPECTED_SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            f"Unexpected sequence shape in {path}: "
            f"{X.shape}; expected "
            f"({EXPECTED_SEQUENCE_LENGTH}, "
            f"{EXPECTED_FEATURES})"
        )

    if not np.isfinite(X).all():

        raise RuntimeError(
            f"Invalid NaN/Inf values in:\n{path}"
        )

    X = (X - mean) / std

    return X.astype(np.float32)


# ================================================================
# RESOLVE PAIR PATH
# ================================================================

def resolve_sequence_path(relative_path, split):

    relative_path = str(relative_path)

    # Normal pair paths are:
    # validation/xxxxx.npz
    # test/xxxxx.npz
    # train/xxxxx.npz

    if split == "validation":
        root = VALIDATION_DIR

    elif split == "test":
        root = TEST_DIR

    elif split == "train":
        root = TRAIN_DIR

    else:
        raise ValueError(split)

    filename = os.path.basename(relative_path)

    candidate = os.path.join(
        root,
        filename,
    )

    if os.path.exists(candidate):
        return candidate

    # Fallback: preserve nested path if present
    candidate = os.path.join(
        BASE_DIR,
        relative_path.replace("/", os.sep),
    )

    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"Could not resolve sequence:\n"
        f"{relative_path}\n"
        f"Split: {split}"
    )


# ================================================================
# UNIQUE SEQUENCE COLLECTION
# ================================================================

def collect_unique_sequences(pair_df):

    unique_paths = set()

    for value in pair_df["sequence_1"]:
        unique_paths.add(str(value))

    for value in pair_df["sequence_2"]:
        unique_paths.add(str(value))

    return sorted(unique_paths)


# ================================================================
# EMBEDDING EXTRACTION
# ================================================================

@torch.no_grad()
def encode_sequences(
    model,
    pair_df,
    split,
    mean,
    std,
):

    print("\n" + "=" * 70)
    print(f"{split.upper()} EMBEDDING EXTRACTION")
    print("=" * 70)

    unique_paths = collect_unique_sequences(
        pair_df
    )

    print(
        f"Unique sequences to encode : "
        f"{len(unique_paths):,}"
    )

    embeddings = {}

    for start in range(
        0,
        len(unique_paths),
        BATCH_SIZE,
    ):

        batch_paths = unique_paths[
            start:start + BATCH_SIZE
        ]

        batch_arrays = []

        for rel_path in batch_paths:

            full_path = resolve_sequence_path(
                rel_path,
                split,
            )

            X = load_sequence(
                full_path,
                mean,
                std,
            )

            batch_arrays.append(X)

        batch = np.stack(
            batch_arrays,
            axis=0,
        )

        batch_tensor = torch.from_numpy(
            batch
        ).to(DEVICE)

        z = model.forward_once(
            batch_tensor
        )

        z = z.detach().cpu().numpy()

        for path, embedding in zip(
            batch_paths,
            z,
        ):

            embeddings[path] = embedding

        processed = min(
            start + len(batch_paths),
            len(unique_paths),
        )

        if (
            processed % 1000 == 0
            or processed == len(unique_paths)
        ):

            print(
                f"Encoded sequences: "
                f"{processed:,}/"
                f"{len(unique_paths):,}"
            )

    return embeddings


# ================================================================
# DISTANCE CALCULATION
# ================================================================

def calculate_distances(
    pair_df,
    embeddings,
):

    distances = []

    for _, row in pair_df.iterrows():

        path1 = str(row["sequence_1"])
        path2 = str(row["sequence_2"])

        z1 = embeddings[path1]
        z2 = embeddings[path2]

        distance = np.linalg.norm(
            z1 - z2
        )

        distances.append(
            float(distance)
        )

    return np.asarray(
        distances,
        dtype=np.float64,
    )


# ================================================================
# THRESHOLD METRICS
# ================================================================

def calculate_threshold_metrics(
    distances,
    labels,
    threshold,
):

    # Siamese distance:
    # LOW distance = SAME USER
    # HIGH distance = DIFFERENT USER

    predictions = (
        distances <= threshold
    ).astype(int)

    labels = np.asarray(labels).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    total_negative = tn + fp
    total_positive = tp + fn

    far = (
        fp / total_negative
        if total_negative > 0
        else 0.0
    )

    frr = (
        fn / total_positive
        if total_positive > 0
        else 0.0
    )

    tpr = (
        tp / total_positive
        if total_positive > 0
        else 0.0
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "far": float(far),
        "frr": float(frr),
        "tpr": float(tpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ================================================================
# ROC / EER
# ================================================================

def calculate_roc_eer(
    distances,
    labels,
):

    labels = np.asarray(labels).astype(int)

    # Similarity score:
    # smaller distance = more likely genuine
    #
    # Therefore invert distance for ROC.

    similarity = -distances

    fpr, tpr, thresholds = roc_curve(
        labels,
        similarity,
    )

    auc = roc_auc_score(
        labels,
        similarity,
    )

    fnr = 1.0 - tpr

    eer_index = np.nanargmin(
        np.abs(fpr - fnr)
    )

    eer = (
        fpr[eer_index]
        + fnr[eer_index]
    ) / 2.0

    eer_threshold_similarity = (
        thresholds[eer_index]
    )

    eer_distance_threshold = (
        -eer_threshold_similarity
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "fnr": fnr,
        "thresholds": thresholds,
        "auc": float(auc),
        "eer": float(eer),
        "eer_distance_threshold": float(
            eer_distance_threshold
        ),
    }


# ================================================================
# DISTANCE STATISTICS
# ================================================================

def distance_statistics(
    distances,
    labels,
):

    labels = np.asarray(labels)

    genuine = distances[
        labels == 1
    ]

    impostor = distances[
        labels == 0
    ]

    def stats(values):

        return {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }

    return {
        "genuine": stats(genuine),
        "impostor": stats(impostor),
    }


# ================================================================
# THRESHOLD SWEEP
# ================================================================

def threshold_sweep(
    distances,
    labels,
):

    minimum = float(
        np.min(distances)
    )

    maximum = float(
        np.max(distances)
    )

    thresholds = np.linspace(
        minimum,
        maximum,
        201,
    )

    rows = []

    for threshold in thresholds:

        metrics = calculate_threshold_metrics(
            distances,
            labels,
            threshold,
        )

        rows.append(metrics)

    return pd.DataFrame(rows)


# ================================================================
# PLOT DISTANCE DISTRIBUTION
# ================================================================

def plot_distance_distribution(
    distances,
    labels,
):

    genuine = distances[
        np.asarray(labels) == 1
    ]

    impostor = distances[
        np.asarray(labels) == 0
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        genuine,
        bins=80,
        alpha=0.65,
        density=True,
        label="Genuine",
    )

    plt.hist(
        impostor,
        bins=80,
        alpha=0.65,
        density=True,
        label="Impostor",
    )

    plt.axvline(
        FROZEN_VALIDATION_THRESHOLD,
        linestyle="--",
        linewidth=2,
        label=(
            "Frozen validation threshold "
            f"({FROZEN_VALIDATION_THRESHOLD:.6f})"
        ),
    )

    plt.xlabel(
        "Euclidean embedding distance"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "AALTO V3 Siamese BiLSTM\n"
        "Genuine vs Impostor Distance Distribution"
    )

    plt.legend()
    plt.grid(alpha=0.25)

    path = os.path.join(
        OUTPUT_DIR,
        "genuine_vs_impostor_distance_distribution.png",
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


# ================================================================
# PLOT ROC
# ================================================================

def plot_roc(
    roc_data,
):

    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    auc = roc_data["auc"]

    plt.figure(figsize=(8, 7))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.6f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "AALTO V3 Siamese BiLSTM ROC Curve"
    )

    plt.legend()
    plt.grid(alpha=0.25)

    path = os.path.join(
        OUTPUT_DIR,
        "roc_curve.png",
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


# ================================================================
# PLOT FAR / FRR
# ================================================================

def plot_far_frr(
    threshold_df,
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        threshold_df["threshold"],
        threshold_df["far"],
        label="FAR",
        linewidth=2,
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["frr"],
        label="FRR",
        linewidth=2,
    )

    plt.axvline(
        FROZEN_VALIDATION_THRESHOLD,
        linestyle="--",
        linewidth=2,
        label="Frozen validation threshold",
    )

    plt.xlabel(
        "Distance threshold"
    )

    plt.ylabel(
        "Rate"
    )

    plt.title(
        "AALTO V3 FAR / FRR Threshold Analysis"
    )

    plt.legend()
    plt.grid(alpha=0.25)

    path = os.path.join(
        OUTPUT_DIR,
        "far_frr_threshold_analysis.png",
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


# ================================================================
# ANALYZE SPLIT
# ================================================================

def analyze_split(
    model,
    pair_file,
    split,
    mean,
    std,
):

    print("\n" + "=" * 70)
    print(f"ANALYZING {split.upper()}")
    print("=" * 70)

    df = pd.read_csv(
        pair_file
    )

    required_columns = {
        "sequence_1",
        "sequence_2",
        "label",
        "pair_type",
        "participant_1",
        "participant_2",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"Missing columns in {pair_file}: "
            f"{sorted(missing)}"
        )

    labels = df["label"].astype(int).values

    print(
        f"Pairs : {len(df):,}"
    )

    print(
        f"Positive : {np.sum(labels == 1):,}"
    )

    print(
        f"Negative : {np.sum(labels == 0):,}"
    )

    embeddings = encode_sequences(
        model,
        df,
        split,
        mean,
        std,
    )

    distances = calculate_distances(
        df,
        embeddings,
    )

    metrics = calculate_threshold_metrics(
        distances,
        labels,
        FROZEN_VALIDATION_THRESHOLD,
    )

    roc_data = calculate_roc_eer(
        distances,
        labels,
    )

    stats = distance_statistics(
        distances,
        labels,
    )

    print("\nRESULTS")
    print("-" * 70)

    print(
        f"Accuracy : {metrics['accuracy']:.6f}"
    )

    print(
        f"FAR      : {metrics['far']:.6f}"
    )

    print(
        f"FRR      : {metrics['frr']:.6f}"
    )

    print(
        f"TPR      : {metrics['tpr']:.6f}"
    )

    print(
        f"AUC      : {roc_data['auc']:.6f}"
    )

    print(
        f"EER      : {roc_data['eer']:.6f}"
    )

    print(
        f"EER distance threshold : "
        f"{roc_data['eer_distance_threshold']:.6f}"
    )

    print("\nCONFUSION MATRIX")

    print(
        f"TN : {metrics['tn']}"
    )

    print(
        f"FP : {metrics['fp']}"
    )

    print(
        f"FN : {metrics['fn']}"
    )

    print(
        f"TP : {metrics['tp']}"
    )

    print("\nDISTANCE STATISTICS")

    print(
        "Genuine:"
    )

    for key, value in stats["genuine"].items():
        print(
            f"  {key:10s}: {value:.6f}"
            if isinstance(value, float)
            else f"  {key:10s}: {value}"
        )

    print(
        "Impostor:"
    )

    for key, value in stats["impostor"].items():
        print(
            f"  {key:10s}: {value:.6f}"
            if isinstance(value, float)
            else f"  {key:10s}: {value}"
        )

    return {
        "pairs": int(len(df)),
        "positive_pairs": int(
            np.sum(labels == 1)
        ),
        "negative_pairs": int(
            np.sum(labels == 0)
        ),
        "threshold_metrics": metrics,
        "roc_metrics": {
            "auc": roc_data["auc"],
            "eer": roc_data["eer"],
            "eer_distance_threshold":
                roc_data["eer_distance_threshold"],
        },
        "distance_statistics": stats,
        "distances": distances,
        "labels": labels,
        "roc_data": roc_data,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("AALTO V3 BASELINE PERFORMANCE ANALYSIS")
    print("=" * 70)

    print(
        "\nRESEARCH CONTROLS"
    )

    print(
        "Model training              : NO"
    )

    print(
        "Weight updates              : NO"
    )

    print(
        "Validation threshold tuning : NO"
    )

    print(
        "Test threshold tuning       : NO"
    )

    print(
        "Frozen validation threshold : "
        f"{FROZEN_VALIDATION_THRESHOLD}"
    )

    print(
        "Train-only normalization    : YES"
    )

    print(
        "Device                      : "
        f"{DEVICE}"
    )

    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------

    required_files = [
        CHECKPOINT,
        SCALER_PATH,
        VALIDATION_PAIRS,
        TEST_PAIRS,
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required input missing:\n{path}"
            )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    model = load_model()

    # ------------------------------------------------------------
    # Load scaler
    # ------------------------------------------------------------

    mean, std = load_scaler()

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    validation_results = analyze_split(
        model,
        VALIDATION_PAIRS,
        "validation",
        mean,
        std,
    )

    # ------------------------------------------------------------
    # Test
    # ------------------------------------------------------------

    test_results = analyze_split(
        model,
        TEST_PAIRS,
        "test",
        mean,
        std,
    )

    # ------------------------------------------------------------
    # Threshold sweep
    # ------------------------------------------------------------

    threshold_df = threshold_sweep(
        test_results["distances"],
        test_results["labels"],
    )

    threshold_csv = os.path.join(
        OUTPUT_DIR,
        "test_threshold_sweep.csv",
    )

    threshold_df.to_csv(
        threshold_csv,
        index=False,
    )

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    distance_plot = plot_distance_distribution(
        test_results["distances"],
        test_results["labels"],
    )

    roc_plot = plot_roc(
        test_results["roc_data"],
    )

    far_frr_plot = plot_far_frr(
        threshold_df,
    )

    # ------------------------------------------------------------
    # Save distance-level report
    # ------------------------------------------------------------

    distance_report = pd.DataFrame({
        "distance":
            test_results["distances"],
        "label":
            test_results["labels"],
    })

    distance_report["pair_type"] = np.where(
        distance_report["label"] == 1,
        "positive",
        "negative",
    )

    distance_csv = os.path.join(
        OUTPUT_DIR,
        "test_pair_distances.csv",
    )

    distance_report.to_csv(
        distance_csv,
        index=False,
    )

    # ------------------------------------------------------------
    # JSON summary
    # ------------------------------------------------------------

    summary = {
        "experiment": "AALTO_V3_Siamese_BiLSTM",
        "seed": SEED,

        "model": {
            "checkpoint": CHECKPOINT,
            "architecture": "Siamese BiLSTM",
            "input_features": EXPECTED_FEATURES,
            "sequence_length":
                EXPECTED_SEQUENCE_LENGTH,
            "lstm_hidden": 128,
            "lstm_layers": 2,
            "bidirectional": True,
            "embedding_dimension": 64,
            "parameters": 585920,
        },

        "research_controls": {
            "model_training": False,
            "weight_updates": False,
            "validation_threshold_tuning": False,
            "test_threshold_tuning": False,
            "test_used_for_training": False,
            "train_only_normalization": True,
            "participant_leakage": False,
        },

        "threshold": {
            "source": "validation",
            "frozen_for_test": True,
            "value":
                FROZEN_VALIDATION_THRESHOLD,
        },

        "validation": {
            "pairs":
                validation_results["pairs"],
            "positive_pairs":
                validation_results["positive_pairs"],
            "negative_pairs":
                validation_results["negative_pairs"],
            "threshold_metrics":
                validation_results[
                    "threshold_metrics"
                ],
            "roc_metrics":
                validation_results[
                    "roc_metrics"
                ],
            "distance_statistics":
                validation_results[
                    "distance_statistics"
                ],
        },

        "test": {
            "pairs":
                test_results["pairs"],
            "positive_pairs":
                test_results["positive_pairs"],
            "negative_pairs":
                test_results["negative_pairs"],
            "threshold_metrics":
                test_results[
                    "threshold_metrics"
                ],
            "roc_metrics":
                test_results[
                    "roc_metrics"
                ],
            "distance_statistics":
                test_results[
                    "distance_statistics"
                ],
        },

        "outputs": {
            "threshold_sweep":
                threshold_csv,
            "distance_report":
                distance_csv,
            "distance_distribution_plot":
                distance_plot,
            "roc_plot":
                roc_plot,
            "far_frr_plot":
                far_frr_plot,
        },

        "processing_seconds":
            time.time() - start_time,
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "baseline_performance_summary_v3.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # ------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("AALTO V3 BASELINE PERFORMANCE ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nFINAL TEST BASELINE")
    print("-" * 70)

    print(
        f"Accuracy : "
        f"{test_results['threshold_metrics']['accuracy']:.6f}"
    )

    print(
        f"FAR      : "
        f"{test_results['threshold_metrics']['far']:.6f}"
    )

    print(
        f"FRR      : "
        f"{test_results['threshold_metrics']['frr']:.6f}"
    )

    print(
        f"TPR      : "
        f"{test_results['threshold_metrics']['tpr']:.6f}"
    )

    print(
        f"AUC      : "
        f"{test_results['roc_metrics']['auc']:.6f}"
    )

    print(
        f"EER      : "
        f"{test_results['roc_metrics']['eer']:.6f}"
    )

    print(
        f"Threshold: "
        f"{FROZEN_VALIDATION_THRESHOLD:.6f}"
    )

    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    print("\nSummary:")
    print(summary_path)

    print(
        f"\nProcessing time : "
        f"{(time.time() - start_time) / 60:.2f} minutes"
    )

    print("\n" + "=" * 70)
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
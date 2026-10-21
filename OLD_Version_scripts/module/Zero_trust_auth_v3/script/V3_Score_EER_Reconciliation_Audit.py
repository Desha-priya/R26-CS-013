import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("processed/siamese_bilstm_v3")

CHECKPOINT = (
    BASE_DIR
    / "model_training_v3"
    / "checkpoints"
    / "best_siamese_bilstm_v3.pt"
)

SCALER = (
    BASE_DIR
    / "reports"
    / "normalization_audit"
    / "train_only_scaler_v3.npz"
)

VAL_PAIRS = (
    BASE_DIR
    / "pairs"
    / "validation_pairs_v3.csv"
)

TEST_PAIRS = (
    BASE_DIR
    / "pairs"
    / "test_pairs_v3.csv"
)

VAL_DIR = BASE_DIR / "validation"
TEST_DIR = BASE_DIR / "test"

OUTPUT_DIR = BASE_DIR / "score_eer_reconciliation_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPECTED_FEATURES = 16
EXPECTED_SEQUENCE_LENGTH = 50

# Frozen validation threshold from the actual V3 evaluation
FROZEN_THRESHOLD = 0.552409


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MODEL
# EXACT ARCHITECTURE FROM V3 CHECKPOINT
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_size=16,
        hidden_size=128,
        num_layers=2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # IMPORTANT:
        # This exact structure is required by the checkpoint.
        #
        # embedding.0 -> Linear(256, 128)
        # embedding.1 -> ReLU
        # embedding.2 -> Dropout
        # embedding.3 -> Linear(128, 64)

        self.embedding = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Last timestep representation
        last_output = output[:, -1, :]

        embedding = self.embedding(last_output)

        return embedding


class SiameseBiLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder(
            input_size=EXPECTED_FEATURES,
            hidden_size=128,
            num_layers=2,
        )

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        return z1, z2


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING EXACT V3 MODEL")
    print("=" * 70)

    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Device     : {DEVICE}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = SiameseBiLSTM().to(DEVICE)

    state_dict = checkpoint["model_state_dict"]

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing or unexpected:

        print("\nMODEL ARCHITECTURE MISMATCH")

        print("\nMissing keys:")
        for key in missing:
            print(" ", key)

        print("\nUnexpected keys:")
        for key in unexpected:
            print(" ", key)

        raise RuntimeError(
            "Checkpoint/model architecture mismatch."
        )

    model.eval()

    print(f"Checkpoint epoch : {checkpoint.get('epoch')}")
    print(
        f"Checkpoint val loss : "
        f"{checkpoint.get('validation_loss')}"
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Model parameters : {total_params:,}")
    print("Architecture match : PASS")
    print("Model loaded : PASS")
    print("Model mode : EVAL")

    return model, checkpoint


# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler():

    print("\n" + "=" * 70)
    print("LOADING TRAIN-ONLY NORMALIZATION")
    print("=" * 70)

    scaler = np.load(SCALER)

    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    print("Scaler keys:")
    for key in scaler.files:
        print(" ", key)

    print(f"Feature count : {len(mean)}")
    print("Normalization source : TRAIN ONLY")
    print("Validation used for scaler : NO")
    print("Test used for scaler : NO")

    if len(mean) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"got {len(mean)}"
        )

    std = np.where(
        std < 1e-12,
        1.0,
        std
    )

    return mean, std


# ============================================================
# LOAD NPZ
# ============================================================

def load_sequence(path, mean, std):

    data = np.load(path)

    # Most V3 files should contain X.
    if "X" in data:
        X = data["X"]
    else:
        # Fallback to first array
        X = data[data.files[0]]

    X = np.asarray(X, dtype=np.float32)

    if X.shape != (
        EXPECTED_SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):
        raise ValueError(
            f"Invalid shape {X.shape} in {path}"
        )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X = (X - mean) / std

    return X.astype(np.float32)


# ============================================================
# RESOLVE PAIR PATH
# ============================================================

def resolve_sequence_path(sequence_name, split):

    if split == "validation":
        root = VAL_DIR
    elif split == "test":
        root = TEST_DIR
    else:
        raise ValueError(split)

    sequence_name = str(sequence_name)

    # Pair files contain paths like:
    #
    # validation/123_section_x_seq_0.npz
    #
    # test/123_section_x_seq_0.npz

    parts = Path(sequence_name).parts

    if len(parts) > 1:
        relative = Path(*parts[1:])
    else:
        relative = Path(sequence_name)

    path = root / relative

    if not path.exists():

        # fallback: maybe already just filename
        alternative = root / Path(sequence_name).name

        if alternative.exists():
            return alternative

        raise FileNotFoundError(
            f"Could not resolve sequence:\n"
            f"{sequence_name}\n"
            f"Expected:\n{path}"
        )

    return path


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model,
    sequence_names,
    split,
    mean,
    std,
):

    print(
        f"\nUnique sequences to encode : "
        f"{len(sequence_names):,}"
    )

    cache = {}

    total = len(sequence_names)

    for i, name in enumerate(sequence_names, 1):

        path = resolve_sequence_path(
            name,
            split,
        )

        X = load_sequence(
            path,
            mean,
            std,
        )

        tensor = torch.from_numpy(
            X
        ).unsqueeze(0).to(DEVICE)

        embedding = model.forward_once(
            tensor
        )

        embedding = (
            embedding
            .detach()
            .cpu()
            .numpy()[0]
            .astype(np.float64)
        )

        cache[str(name)] = embedding

        if (
            i % 1000 == 0
            or i == total
        ):
            print(
                f"Encoded sequences: "
                f"{i:,}/{total:,}"
            )

    return cache


# ============================================================
# SCORE FUNCTIONS
# ============================================================

def euclidean_distance(a, b):

    return float(
        np.linalg.norm(a - b)
    )


def squared_euclidean_distance(a, b):

    diff = a - b

    return float(
        np.sum(diff * diff)
    )


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def cosine_distance(a, b):

    return 1.0 - cosine_similarity(a, b)


# ============================================================
# EER
# ============================================================

def calculate_eer(labels, scores, higher_is_genuine=True):

    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if not higher_is_genuine:
        scores = -scores

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
    )

    fnr = 1.0 - tpr

    index = np.nanargmin(
        np.abs(fpr - fnr)
    )

    eer = float(
        (fpr[index] + fnr[index]) / 2.0
    )

    threshold = float(
        thresholds[index]
    )

    return eer, threshold


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    labels,
    scores,
    threshold,
    higher_is_genuine,
):

    labels = np.asarray(labels)

    if higher_is_genuine:
        predictions = (
            scores >= threshold
        )
    else:
        predictions = (
            scores <= threshold
        )

    predictions = predictions.astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = (
        (tp + tn)
        / max(
            tp + tn + fp + fn,
            1,
        )
    )

    far = (
        fp
        / max(
            fp + tn,
            1,
        )
    )

    frr = (
        fn
        / max(
            fn + tp,
            1,
        )
    )

    tpr = (
        tp
        / max(
            tp + fn,
            1,
        )
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


# ============================================================
# ANALYZE ONE SPLIT
# ============================================================

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

    df = pd.read_csv(pair_file)

    print(f"Pairs : {len(df):,}")
    print(
        f"Positive : "
        f"{(df['label'] == 1).sum():,}"
    )
    print(
        f"Negative : "
        f"{(df['label'] == 0).sum():,}"
    )

    sequence_1 = df["sequence_1"].astype(str)
    sequence_2 = df["sequence_2"].astype(str)

    unique_sequences = sorted(
        set(sequence_1)
        | set(sequence_2)
    )

    embeddings = extract_embeddings(
        model,
        unique_sequences,
        split,
        mean,
        std,
    )

    labels = df["label"].astype(int).to_numpy()

    euclidean = []
    squared = []
    cosine_sim = []
    cosine_dist = []

    for s1, s2 in zip(
        sequence_1,
        sequence_2,
    ):

        a = embeddings[str(s1)]
        b = embeddings[str(s2)]

        euclidean.append(
            euclidean_distance(a, b)
        )

        squared.append(
            squared_euclidean_distance(a, b)
        )

        cosine_sim.append(
            cosine_similarity(a, b)
        )

        cosine_dist.append(
            cosine_distance(a, b)
        )

    euclidean = np.asarray(
        euclidean,
        dtype=np.float64,
    )

    squared = np.asarray(
        squared,
        dtype=np.float64,
    )

    cosine_sim = np.asarray(
        cosine_sim,
        dtype=np.float64,
    )

    cosine_dist = np.asarray(
        cosine_dist,
        dtype=np.float64,
    )

    results = {}

    # --------------------------------------------------------
    # EUCLIDEAN DISTANCE
    # --------------------------------------------------------

    eer, eer_threshold = calculate_eer(
        labels,
        euclidean,
        higher_is_genuine=False,
    )

    auc = roc_auc_score(
        labels,
        -euclidean,
    )

    results["euclidean_distance"] = {
        "auc": float(auc),
        "eer": float(eer),
        "eer_distance_threshold": float(
            eer_threshold
        ),
        "frozen_threshold_result": evaluate_threshold(
            labels,
            euclidean,
            FROZEN_THRESHOLD,
            higher_is_genuine=False,
        ),
    }

    # --------------------------------------------------------
    # SQUARED EUCLIDEAN
    # --------------------------------------------------------

    eer_sq, threshold_sq = calculate_eer(
        labels,
        squared,
        higher_is_genuine=False,
    )

    auc_sq = roc_auc_score(
        labels,
        -squared,
    )

    results["squared_euclidean_distance"] = {
        "auc": float(auc_sq),
        "eer": float(eer_sq),
        "eer_distance_threshold": float(
            threshold_sq
        ),
        "frozen_threshold_result": evaluate_threshold(
            labels,
            squared,
            FROZEN_THRESHOLD,
            higher_is_genuine=False,
        ),
    }

    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    eer_cos, threshold_cos = calculate_eer(
        labels,
        cosine_sim,
        higher_is_genuine=True,
    )

    auc_cos = roc_auc_score(
        labels,
        cosine_sim,
    )

    results["cosine_similarity"] = {
        "auc": float(auc_cos),
        "eer": float(eer_cos),
        "eer_similarity_threshold": float(
            threshold_cos
        ),
        "frozen_threshold_result": evaluate_threshold(
            labels,
            cosine_sim,
            FROZEN_THRESHOLD,
            higher_is_genuine=True,
        ),
    }

    # --------------------------------------------------------
    # COSINE DISTANCE
    # --------------------------------------------------------

    eer_cd, threshold_cd = calculate_eer(
        labels,
        cosine_dist,
        higher_is_genuine=False,
    )

    auc_cd = roc_auc_score(
        labels,
        -cosine_dist,
    )

    results["cosine_distance"] = {
        "auc": float(auc_cd),
        "eer": float(eer_cd),
        "eer_distance_threshold": float(
            threshold_cd
        ),
        "frozen_threshold_result": evaluate_threshold(
            labels,
            cosine_dist,
            FROZEN_THRESHOLD,
            higher_is_genuine=False,
        ),
    }

    # --------------------------------------------------------
    # DISTANCE STATISTICS
    # --------------------------------------------------------

    genuine = euclidean[labels == 1]
    impostor = euclidean[labels == 0]

    print("\nEUCLIDEAN DISTANCE")
    print(
        f"Genuine mean   : {np.mean(genuine):.6f}"
    )
    print(
        f"Genuine median : {np.median(genuine):.6f}"
    )
    print(
        f"Impostor mean  : {np.mean(impostor):.6f}"
    )
    print(
        f"Impostor median: {np.median(impostor):.6f}"
    )

    print("\nSCORE COMPARISON")
    print("-" * 70)

    for name, result in results.items():

        print(
            f"{name:30s} "
            f"AUC={result['auc']:.6f} "
            f"EER={result['eer']:.6f}"
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AALTO V3 SCORE / EER RECONCILIATION AUDIT")
    print("=" * 70)

    print("\nRESEARCH CONTROLS")
    print("Model training              : NO")
    print("Weight updates              : NO")
    print("Validation threshold tuning: NO")
    print("Test threshold tuning      : NO")
    print(
        f"Frozen validation threshold: "
        f"{FROZEN_THRESHOLD:.6f}"
    )
    print("Train-only normalization   : YES")
    print(f"Device                     : {DEVICE}")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    model, checkpoint = load_model()

    mean, std = load_scaler()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_results = analyze_split(
        model=model,
        pair_file=VAL_PAIRS,
        split="validation",
        mean=mean,
        std=std,
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_results = analyze_split(
        model=model,
        pair_file=TEST_PAIRS,
        split="test",
        mean=mean,
        std=std,
    )

    # --------------------------------------------------------
    # RECONCILIATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EER RECONCILIATION")
    print("=" * 70)

    print("\nVALIDATION")

    for name, result in validation_results.items():

        print(
            f"{name:30s} "
            f"EER={result['eer']:.6f}"
        )

    print("\nTEST")

    for name, result in test_results.items():

        print(
            f"{name:30s} "
            f"EER={result['eer']:.6f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    summary = {
        "dataset": "AALTO_V3",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),
        "checkpoint_validation_loss": checkpoint.get(
            "validation_loss"
        ),
        "frozen_validation_threshold": FROZEN_THRESHOLD,
        "device": str(DEVICE),
        "model_training": False,
        "weight_updates": False,
        "validation_threshold_tuning": False,
        "test_threshold_tuning": False,
        "train_only_normalization": True,
        "validation": validation_results,
        "test": test_results,
    }

    summary_file = (
        OUTPUT_DIR
        / "score_eer_reconciliation_summary_v3.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("SCORE / EER RECONCILIATION COMPLETE")
    print("=" * 70)

    print("\nSummary:")
    print(summary_file)


if __name__ == "__main__":
    main()
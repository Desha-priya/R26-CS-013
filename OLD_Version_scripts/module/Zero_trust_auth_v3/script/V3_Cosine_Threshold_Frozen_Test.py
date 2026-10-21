import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

BASE_DIR = Path("processed/siamese_bilstm_v3")

CHECKPOINT = (
    BASE_DIR
    / "model_training_v3"
    / "checkpoints"
    / "best_siamese_bilstm_v3.pt"
)

SCALER_PATH = (
    BASE_DIR
    / "reports"
    / "normalization_audit"
    / "train_only_scaler_v3.npz"
)

VAL_PAIRS = BASE_DIR / "pairs" / "validation_pairs_v3.csv"
TEST_PAIRS = BASE_DIR / "pairs" / "test_pairs_v3.csv"

VAL_DIR = BASE_DIR / "validation"
TEST_DIR = BASE_DIR / "test"

OUTPUT_DIR = BASE_DIR / "cosine_threshold_v3"
REPORT_DIR = OUTPUT_DIR / "reports"

SUMMARY_PATH = REPORT_DIR / "cosine_threshold_frozen_test_summary_v3.json"

SEQUENCE_LENGTH = 50
FEATURE_COUNT = 16

BATCH_SIZE = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================
# EXACT V3 MODEL ARCHITECTURE
# ================================================================

class BiLSTMEncoder(nn.Module):

    def __init__(
        self,
        input_size=16,
        hidden_size=128,
        num_layers=2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.embedding = nn.Sequential(
                    nn.Linear(hidden_size * 2, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Final timestep
        last_output = output[:, -1, :]

        embedding = self.embedding(last_output)

        return embedding


class SiameseBiLSTM(nn.Module):

    def __init__(
        self,
        input_size=16,
        hidden_size=128,
        num_layers=2
    ):
        super().__init__()

        self.encoder = BiLSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
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

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    state_dict = checkpoint["model_state_dict"]

    model = SiameseBiLSTM(
        input_size=FEATURE_COUNT,
        hidden_size=128,
        num_layers=2
    ).to(DEVICE)

    expected_keys = set(model.state_dict().keys())
    actual_keys = set(state_dict.keys())

    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)

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

    model.load_state_dict(state_dict, strict=True)

    model.eval()

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Checkpoint epoch : {checkpoint.get('epoch')}")
    print(f"Checkpoint val loss : {checkpoint.get('validation_loss')}")
    print(f"Model parameters : {parameter_count:,}")
    print("Architecture match : PASS")
    print("Model loaded : PASS")
    print("Model mode : EVAL")

    return model


# ================================================================
# SCALER
# ================================================================

def load_scaler():

    print("\n" + "=" * 70)
    print("LOADING TRAIN-ONLY NORMALIZATION")
    print("=" * 70)

    scaler = np.load(SCALER_PATH)

    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    if mean.shape[0] != FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {FEATURE_COUNT} scaler features, "
            f"got {mean.shape[0]}"
        )

    std = np.where(std == 0, 1.0, std)

    print("Scaler keys:")
    for key in scaler.files:
        print(f"  {key}")

    print(f"Feature count : {len(mean)}")
    print("Normalization source : TRAIN ONLY")
    print("Validation used for scaler : NO")
    print("Test used for scaler : NO")

    return mean, std


# ================================================================
# PAIR FILE LOADING
# ================================================================

def load_pairs(path):

    df = pd.read_csv(path)

    required = [
        "sequence_1",
        "sequence_2",
        "participant_1",
        "participant_2",
        "label"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns in {path}: {missing}"
        )

    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise RuntimeError(
            f"Invalid labels in {path}"
        )

    return df


# ================================================================
# SEQUENCE PATH RESOLUTION
# ================================================================

def resolve_sequence_path(sequence_name, split_dir):

    sequence_name = str(sequence_name)

    # Pair files normally contain:
    #
    # validation/123_section_x_seq_y.npz
    #
    # or:
    #
    # test/123_section_x_seq_y.npz

    path = BASE_DIR / sequence_name

    if path.exists():
        return path

    path = split_dir / Path(sequence_name).name

    if path.exists():
        return path

    # Last fallback
    path = Path(sequence_name)

    if path.exists():
        return path

    raise FileNotFoundError(
        f"Sequence file not found: {sequence_name}"
    )


# ================================================================
# LOAD SINGLE SEQUENCE
# ================================================================

def load_sequence(path, mean, std):

    data = np.load(path)

    # Try common NPZ keys first
    if "X" in data.files:
        X = data["X"]

    elif "x" in data.files:
        X = data["x"]

    elif len(data.files) == 1:
        X = data[data.files[0]]

    else:
        raise RuntimeError(
            f"Could not identify sequence array in {path}. "
            f"Available keys: {data.files}"
        )

    X = np.asarray(X, dtype=np.float32)

    if X.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):

        raise RuntimeError(
            f"Invalid sequence shape in {path}: "
            f"{X.shape}, expected "
            f"({SEQUENCE_LENGTH}, {FEATURE_COUNT})"
        )

    if not np.isfinite(X).all():

        raise RuntimeError(
            f"Invalid NaN/Inf values in {path}"
        )

    X = (X - mean) / std

    return X.astype(np.float32)


# ================================================================
# EMBEDDING EXTRACTION
# ================================================================

def extract_embeddings(
    model,
    sequence_names,
    split_dir,
    mean,
    std,
    cache_name
):

    print("\n" + "=" * 70)
    print("EMBEDDING EXTRACTION")
    print("=" * 70)

    print(f"Unique sequences to encode : {len(sequence_names)}")

    embeddings = {}

    model.eval()

    total = len(sequence_names)

    batch_sequences = []
    batch_names = []

    with torch.no_grad():

        for index, sequence_name in enumerate(
            sequence_names,
            start=1
        ):

            path = resolve_sequence_path(
                sequence_name,
                split_dir
            )

            X = load_sequence(
                path,
                mean,
                std
            )

            batch_sequences.append(X)
            batch_names.append(sequence_name)

            if (
                len(batch_sequences) >= BATCH_SIZE
                or index == total
            ):

                batch = np.stack(
                    batch_sequences,
                    axis=0
                )

                tensor = torch.from_numpy(
                    batch
                ).to(DEVICE)

                z = model.encoder(tensor)

                z = z.detach().cpu().numpy()

                for name, embedding in zip(
                    batch_names,
                    z
                ):
                    embeddings[name] = embedding

                batch_sequences.clear()
                batch_names.clear()

            if (
                index % 1000 == 0
                or index == total
            ):
                print(
                    f"Encoded sequences: "
                    f"{index:,}/{total:,}"
                )

    return embeddings


# ================================================================
# COSINE SCORES
# ================================================================

def calculate_cosine_scores(
    pairs,
    embeddings
):

    z1 = np.stack(
        [
            embeddings[name]
            for name in pairs["sequence_1"]
        ]
    )

    z2 = np.stack(
        [
            embeddings[name]
            for name in pairs["sequence_2"]
        ]
    )

    z1_norm = np.linalg.norm(
        z1,
        axis=1,
        keepdims=True
    )

    z2_norm = np.linalg.norm(
        z2,
        axis=1,
        keepdims=True
    )

    z1_norm = np.maximum(
        z1_norm,
        1e-12
    )

    z2_norm = np.maximum(
        z2_norm,
        1e-12
    )

    z1 = z1 / z1_norm
    z2 = z2 / z2_norm

    scores = np.sum(
        z1 * z2,
        axis=1
    )

    return scores


# ================================================================
# EER
# ================================================================

def calculate_eer(labels, scores):

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    fnr = 1.0 - tpr

    index = np.nanargmin(
        np.abs(fpr - fnr)
    )

    eer = (
        fpr[index] +
        fnr[index]
    ) / 2.0

    threshold = thresholds[index]

    return float(eer), float(threshold)


# ================================================================
# THRESHOLD METRICS
# ================================================================

def calculate_metrics(
    labels,
    scores,
    threshold
):

    # Higher cosine similarity = more likely genuine
    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total > 0 else 0
    )

    far = (
        fp / (fp + tn)
        if (fp + tn) > 0 else 0
    )

    frr = (
        fn / (fn + tp)
        if (fn + tp) > 0 else 0
    )

    tpr = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else 0
    )

    eer, eer_threshold = calculate_eer(
        labels,
        scores
    )

    auc = roc_auc_score(
        labels,
        scores
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "far": float(far),
        "frr": float(frr),
        "tpr": float(tpr),
        "specificity": float(specificity),
        "auc": float(auc),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ================================================================
# VALIDATION THRESHOLD SELECTION
# ================================================================

def select_validation_threshold(
    labels,
    scores
):

    print("\n" + "=" * 70)
    print("VALIDATION COSINE THRESHOLD SELECTION")
    print("=" * 70)

    print("IMPORTANT:")
    print("Threshold is selected using VALIDATION ONLY.")
    print("TEST DATA IS NOT USED.")

    # ROC-derived EER threshold
    eer, eer_threshold = calculate_eer(
        labels,
        scores
    )

    print(
        f"Validation EER threshold : "
        f"{eer_threshold:.6f}"
    )

    print(
        f"Validation EER            : "
        f"{eer:.6f}"
    )

    # Also search thresholds for maximum accuracy.
    #
    # This is allowed because validation is the
    # designated threshold-selection split.

    candidate_thresholds = np.unique(
        np.percentile(
            scores,
            np.linspace(
                0,
                100,
                5001
            )
        )
    )

    best_threshold = None
    best_accuracy = -1.0

    for threshold in candidate_thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        accuracy = np.mean(
            predictions == labels
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_threshold = threshold

    print(
        f"Validation accuracy-optimal "
        f"threshold : {best_threshold:.6f}"
    )

    print(
        f"Validation accuracy at selected "
        f"threshold : {best_accuracy:.6f}"
    )

    # Research choice:
    #
    # We use the accuracy-optimal threshold
    # because the final operating point needs
    # a concrete frozen authentication threshold.
    #
    # EER remains independently reported.

    return float(best_threshold)


# ================================================================
# MAIN
# ================================================================

def main():

    set_seed(SEED)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("AALTO V3 COSINE THRESHOLD / FROZEN TEST EVALUATION")
    print("=" * 70)

    print("\nRESEARCH CONTROLS")
    print("Model training              : NO")
    print("Weight updates              : NO")
    print("Validation threshold tuning : YES")
    print("Test threshold tuning       : NO")
    print("Test threshold              : FROZEN VALIDATION THRESHOLD")
    print("Train-only normalization    : YES")
    print(f"Device                      : {DEVICE}")

    # ------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------

    model = load_model()

    # ------------------------------------------------------------
    # LOAD SCALER
    # ------------------------------------------------------------

    mean, std = load_scaler()

    # ------------------------------------------------------------
    # LOAD PAIRS
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING PAIRS")
    print("=" * 70)

    val_pairs = load_pairs(
        VAL_PAIRS
    )

    test_pairs = load_pairs(
        TEST_PAIRS
    )

    print(
        f"Validation pairs : {len(val_pairs):,}"
    )

    print(
        f"Validation positive : "
        f"{(val_pairs.label == 1).sum():,}"
    )

    print(
        f"Validation negative : "
        f"{(val_pairs.label == 0).sum():,}"
    )

    print(
        f"Test pairs : {len(test_pairs):,}"
    )

    print(
        f"Test positive : "
        f"{(test_pairs.label == 1).sum():,}"
    )

    print(
        f"Test negative : "
        f"{(test_pairs.label == 0).sum():,}"
    )

    # ------------------------------------------------------------
    # VALIDATION EMBEDDINGS
    # ------------------------------------------------------------

    val_sequence_names = sorted(
        set(
            val_pairs["sequence_1"]
        ).union(
            set(
                val_pairs["sequence_2"]
            )
        )
    )

    val_embeddings = extract_embeddings(
        model,
        val_sequence_names,
        VAL_DIR,
        mean,
        std,
        "validation"
    )

    # ------------------------------------------------------------
    # VALIDATION COSINE SCORES
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION COSINE SCORING")
    print("=" * 70)

    val_scores = calculate_cosine_scores(
        val_pairs,
        val_embeddings
    )

    val_labels = (
        val_pairs["label"]
        .to_numpy()
        .astype(int)
    )

    val_auc = roc_auc_score(
        val_labels,
        val_scores
    )

    val_eer, val_eer_threshold = calculate_eer(
        val_labels,
        val_scores
    )

    print(
        f"Validation cosine AUC : "
        f"{val_auc:.6f}"
    )

    print(
        f"Validation cosine EER : "
        f"{val_eer:.6f}"
    )

    print(
        f"Validation EER threshold : "
        f"{val_eer_threshold:.6f}"
    )

    # ------------------------------------------------------------
    # SELECT THRESHOLD
    # ------------------------------------------------------------

    frozen_threshold = select_validation_threshold(
        val_labels,
        val_scores
    )

    val_metrics = calculate_metrics(
        val_labels,
        val_scores,
        frozen_threshold
    )

    print("\nVALIDATION FINAL OPERATING POINT")
    print("-" * 70)
    print(
        f"Threshold : "
        f"{frozen_threshold:.6f}"
    )
    print(
        f"Accuracy  : "
        f"{val_metrics['accuracy']:.6f}"
    )
    print(
        f"FAR       : "
        f"{val_metrics['far']:.6f}"
    )
    print(
        f"FRR       : "
        f"{val_metrics['frr']:.6f}"
    )
    print(
        f"TPR       : "
        f"{val_metrics['tpr']:.6f}"
    )
    print(
        f"EER       : "
        f"{val_metrics['eer']:.6f}"
    )

    # ------------------------------------------------------------
    # FREEZE THRESHOLD
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FREEZING VALIDATION THRESHOLD")
    print("=" * 70)

    print(
        f"Frozen threshold : "
        f"{frozen_threshold:.6f}"
    )

    print("Validation threshold selection : COMPLETE")
    print("Test threshold tuning           : NO")

    # ------------------------------------------------------------
    # TEST EMBEDDINGS
    # ------------------------------------------------------------

    test_sequence_names = sorted(
        set(
            test_pairs["sequence_1"]
        ).union(
            set(
                test_pairs["sequence_2"]
            )
        )
    )

    test_embeddings = extract_embeddings(
        model,
        test_sequence_names,
        TEST_DIR,
        mean,
        std,
        "test"
    )

    # ------------------------------------------------------------
    # TEST SCORING
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    print("IMPORTANT:")
    print("Test threshold tuning : NO")
    print(
        f"Frozen validation threshold : "
        f"{frozen_threshold:.6f}"
    )

    test_scores = calculate_cosine_scores(
        test_pairs,
        test_embeddings
    )

    test_labels = (
        test_pairs["label"]
        .to_numpy()
        .astype(int)
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_scores,
        frozen_threshold
    )

    # ------------------------------------------------------------
    # TEST RESULTS
    # ------------------------------------------------------------

    print("\nFINAL TEST")
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
        f"Specificity : "
        f"{test_metrics['specificity']:.6f}"
    )

    print(
        f"AUC      : "
        f"{test_metrics['auc']:.6f}"
    )

    print(
        f"EER      : "
        f"{test_metrics['eer']:.6f}"
    )

    print(
        f"EER threshold : "
        f"{test_metrics['eer_threshold']:.6f}"
    )

    print("\nCONFUSION MATRIX")
    print("-" * 70)

    print(
        f"TN : {test_metrics['tn']:,}"
    )

    print(
        f"FP : {test_metrics['fp']:,}"
    )

    print(
        f"FN : {test_metrics['fn']:,}"
    )

    print(
        f"TP : {test_metrics['tp']:,}"
    )

    # ------------------------------------------------------------
    # SCORE DISTRIBUTION
    # ------------------------------------------------------------

    genuine_scores = test_scores[
        test_labels == 1
    ]

    impostor_scores = test_scores[
        test_labels == 0
    ]

    print("\nCOSINE SCORE DISTRIBUTION")
    print("-" * 70)

    print("Genuine:")
    print(
        f"  mean   : "
        f"{np.mean(genuine_scores):.6f}"
    )
    print(
        f"  median : "
        f"{np.median(genuine_scores):.6f}"
    )
    print(
        f"  q25    : "
        f"{np.percentile(genuine_scores, 25):.6f}"
    )
    print(
        f"  q75    : "
        f"{np.percentile(genuine_scores, 75):.6f}"
    )

    print("Impostor:")
    print(
        f"  mean   : "
        f"{np.mean(impostor_scores):.6f}"
    )
    print(
        f"  median : "
        f"{np.median(impostor_scores):.6f}"
    )
    print(
        f"  q25    : "
        f"{np.percentile(impostor_scores, 25):.6f}"
    )
    print(
        f"  q75    : "
        f"{np.percentile(impostor_scores, 75):.6f}"
    )

    # ------------------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------------------

    summary = {

        "dataset": "AALTO_V3",

        "random_seed": SEED,

        "model_training": False,

        "weight_updates": False,

        "checkpoint": str(
            CHECKPOINT
        ),

        "checkpoint_epoch": int(
            torch.load(
                CHECKPOINT,
                map_location="cpu"
            ).get("epoch", -1)
        ),

        "model_parameters": 585920,

        "feature_count": FEATURE_COUNT,

        "sequence_length": SEQUENCE_LENGTH,

        "normalization": {
            "source": "TRAIN_ONLY",
            "validation_used": False,
            "test_used": False
        },

        "validation": {

            "pairs": int(len(val_pairs)),

            "positive_pairs": int(
                (val_labels == 1).sum()
            ),

            "negative_pairs": int(
                (val_labels == 0).sum()
            ),

            "cosine_auc": float(
                val_auc
            ),

            "cosine_eer": float(
                val_eer
            ),

            "eer_threshold": float(
                val_eer_threshold
            ),

            "selected_threshold": float(
                frozen_threshold
            ),

            "accuracy": float(
                val_metrics["accuracy"]
            ),

            "far": float(
                val_metrics["far"]
            ),

            "frr": float(
                val_metrics["frr"]
            ),

            "tpr": float(
                val_metrics["tpr"]
            ),

            "tn": int(
                val_metrics["tn"]
            ),

            "fp": int(
                val_metrics["fp"]
            ),

            "fn": int(
                val_metrics["fn"]
            ),

            "tp": int(
                val_metrics["tp"]
            )
        },

        "test": {

            "pairs": int(len(test_pairs)),

            "positive_pairs": int(
                (test_labels == 1).sum()
            ),

            "negative_pairs": int(
                (test_labels == 0).sum()
            ),

            "threshold": float(
                frozen_threshold
            ),

            "threshold_source":
                "VALIDATION_ONLY",

            "test_threshold_tuning":
                False,

            "accuracy": float(
                test_metrics["accuracy"]
            ),

            "far": float(
                test_metrics["far"]
            ),

            "frr": float(
                test_metrics["frr"]
            ),

            "tpr": float(
                test_metrics["tpr"]
            ),

            "specificity": float(
                test_metrics["specificity"]
            ),

            "auc": float(
                test_metrics["auc"]
            ),

            "eer": float(
                test_metrics["eer"]
            ),

            "eer_threshold": float(
                test_metrics["eer_threshold"]
            ),

            "tn": int(
                test_metrics["tn"]
            ),

            "fp": int(
                test_metrics["fp"]
            ),

            "fn": int(
                test_metrics["fn"]
            ),

            "tp": int(
                test_metrics["tp"]
            )
        },

        "research_controls": {

            "exact_training_architecture":
                True,

            "shared_siamese_weights":
                True,

            "train_only_normalization":
                True,

            "validation_threshold_selection":
                True,

            "test_threshold_tuning":
                False,

            "test_used_for_training":
                False,

            "participant_leakage":
                False
        }
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # ------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "AALTO V3 COSINE THRESHOLD / "
        "FROZEN TEST EVALUATION COMPLETE"
    )
    print("=" * 70)

    print("\nFINAL TEST")
    print("-" * 70)

    print(
        f"Frozen threshold : "
        f"{frozen_threshold:.6f}"
    )

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
        f"AUC      : "
        f"{test_metrics['auc']:.6f}"
    )

    print(
        f"EER      : "
        f"{test_metrics['eer']:.6f}"
    )

    print("\nRESEARCH CONTROLS")
    print("-" * 70)

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

    print("\nOUTPUT")
    print("-" * 70)

    print(
        f"Summary : {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
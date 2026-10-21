# ============================================================
# siamese_bilstm_service.py
# ============================================================
#
# NeuraShield — Siamese BiLSTM Behavioral Authentication
#
# PURPOSE
# -------
# This module is the ONLY behavioral-biometric ML component
# used by the new Zero-Trust authentication architecture.
#
# FLOW
# ----
# Raw 50 x 16 keystroke sequence
#          |
#          v
# Train-only normalization
#          |
#          v
# Exact V3 Siamese BiLSTM encoder
#          |
#          v
# 64-dimensional L2-normalized embedding
#          |
#          v
# Cosine similarity
#          |
#          v
# Adaptive Risk Engine
#
# IMPORTANT
# ---------
# This module does NOT:
#   - train the model
#   - update model weights
#   - maintain a personal ML model
#   - use Isolation Forest
#   - use One-Class SVM
#   - use old feature weights
#   - calculate final Zero-Trust risk
#   - select production thresholds
#
# The Risk Engine decides how the behavioral score should
# affect authentication depending on context and alerts.
# ============================================================

import os
import logging
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("SiameseBiLSTMService")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "processed/siamese_bilstm_v3"

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "model_training_v3",
    "checkpoints",
    "best_siamese_bilstm_v3.pt"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "normalization_audit",
    "train_only_scaler_v3.npz"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

FEATURE_COUNT = 16
SEQUENCE_LENGTH = 50

LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2

EMBEDDING_SIZE = 64

DROPOUT = 0.30


# ============================================================
# EXACT V3 FEATURE ORDER
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


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# EXACT V3 BiLSTM ENCODER
# ============================================================

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

        # EXACT V3 TRAINING REPRESENTATION:
        # last timestep of BiLSTM output.
        last_output = output[:, -1, :]

        embedding = self.embedding(
            last_output
        )

        # EXACT V3 TRAINING NORMALIZATION.
        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding


# ============================================================
# SIAMESE MODEL WRAPPER
# ============================================================

class SiameseBiLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        # IMPORTANT:
        # Siamese architecture means both branches use
        # this ONE shared encoder.
        self.encoder = BiLSTMEncoder()

    def forward_once(self, x):

        return self.encoder(x)

    def forward(self, x1, x2):

        embedding1 = self.forward_once(x1)
        embedding2 = self.forward_once(x2)

        # Euclidean distance retained only for compatibility
        # with the original V3 training architecture.
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


# ============================================================
# BEHAVIORAL AUTHENTICATION SERVICE
# ============================================================

class SiameseBiLSTMService:

    def __init__(
        self,
        checkpoint_path: str = CHECKPOINT_PATH,
        scaler_path: str = SCALER_PATH,
        device: torch.device = DEVICE
    ):

        self.checkpoint_path = checkpoint_path
        self.scaler_path = scaler_path
        self.device = device

        self.model = None

        self.mean = None
        self.std = None

        self.feature_count = None
        self.sequence_length = None

        self.loaded = False

        self._load_scaler()

        self._load_model()

        logger.info(
            "Siamese BiLSTM behavioral service ready."
        )


    # ========================================================
    # LOAD TRAIN-ONLY NORMALIZATION
    # ========================================================

    def _load_scaler(self):

        if not os.path.exists(self.scaler_path):

            raise FileNotFoundError(
                f"Train-only scaler not found:\n"
                f"{self.scaler_path}"
            )

        scaler = np.load(
            self.scaler_path,
            allow_pickle=False
        )

        required_keys = [
            "mean",
            "std",
            "feature_count",
            "sequence_length"
        ]

        for key in required_keys:

            if key not in scaler:

                raise RuntimeError(
                    f"Scaler is missing required key: {key}"
                )

        self.mean = scaler["mean"].astype(
            np.float32
        )

        self.std = scaler["std"].astype(
            np.float32
        )

        self.feature_count = int(
            scaler["feature_count"]
        )

        self.sequence_length = int(
            scaler["sequence_length"]
        )

        # ----------------------------------------------------
        # HARD ARCHITECTURE CHECK
        # ----------------------------------------------------

        if self.feature_count != FEATURE_COUNT:

            raise RuntimeError(
                "Scaler feature count mismatch.\n"
                f"Expected: {FEATURE_COUNT}\n"
                f"Found:    {self.feature_count}"
            )

        if self.sequence_length != SEQUENCE_LENGTH:

            raise RuntimeError(
                "Scaler sequence length mismatch.\n"
                f"Expected: {SEQUENCE_LENGTH}\n"
                f"Found:    {self.sequence_length}"
            )

        if self.mean.shape != (FEATURE_COUNT,):

            raise RuntimeError(
                f"Invalid scaler mean shape: "
                f"{self.mean.shape}"
            )

        if self.std.shape != (FEATURE_COUNT,):

            raise RuntimeError(
                f"Invalid scaler std shape: "
                f"{self.std.shape}"
            )

        # Avoid division by zero.
        self.std = np.where(
            self.std == 0.0,
            1.0,
            self.std
        ).astype(np.float32)

        logger.info(
            "Train-only normalization loaded."
        )

        logger.info(
            f"Features: {self.feature_count}"
        )

        logger.info(
            f"Sequence length: {self.sequence_length}"
        )


    # ========================================================
    # LOAD EXACT V3 CHECKPOINT
    # ========================================================

    def _load_model(self):

        if not os.path.exists(
            self.checkpoint_path
        ):

            raise FileNotFoundError(
                f"V3 checkpoint not found:\n"
                f"{self.checkpoint_path}"
            )

        logger.info(
            "Loading exact V3 Siamese BiLSTM checkpoint."
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device
        )

        if not isinstance(
            checkpoint,
            dict
        ):

            raise RuntimeError(
                "Unexpected checkpoint format."
            )

        if "model_state_dict" not in checkpoint:

            raise RuntimeError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        state_dict = checkpoint[
            "model_state_dict"
        ]

        self.model = SiameseBiLSTM()

        # ----------------------------------------------------
        # EXACT ARCHITECTURE VERIFICATION
        # ----------------------------------------------------

        expected_keys = set(
            self.model.state_dict().keys()
        )

        checkpoint_keys = set(
            state_dict.keys()
        )

        missing = sorted(
            expected_keys - checkpoint_keys
        )

        unexpected = sorted(
            checkpoint_keys - expected_keys
        )

        if missing or unexpected:

            raise RuntimeError(
                "\n"
                "================================================\n"
                "CHECKPOINT ARCHITECTURE MISMATCH\n"
                "================================================\n"
                f"Missing keys:\n{missing}\n\n"
                f"Unexpected keys:\n{unexpected}\n"
                "================================================\n"
            )

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        # ----------------------------------------------------
        # INFORMATION
        # ----------------------------------------------------

        checkpoint_epoch = checkpoint.get(
            "epoch",
            None
        )

        validation_loss = checkpoint.get(
            "validation_loss",
            None
        )

        parameter_count = sum(
            p.numel()
            for p in self.model.parameters()
        )

        logger.info(
            f"Checkpoint epoch: {checkpoint_epoch}"
        )

        logger.info(
            f"Validation loss: {validation_loss}"
        )

        logger.info(
            f"Model parameters: {parameter_count}"
        )

        logger.info(
            "Architecture match: PASS"
        )

        logger.info(
            "Model mode: EVAL"
        )

        self.loaded = True


    # ========================================================
    # VALIDATE RAW SEQUENCE
    # ========================================================

    def _validate_sequence(
        self,
        sequence
    ):

        array = np.asarray(
            sequence,
            dtype=np.float32
        )

        if array.shape != (
            SEQUENCE_LENGTH,
            FEATURE_COUNT
        ):

            raise ValueError(
                "Invalid behavioral sequence shape.\n"
                f"Expected: "
                f"({SEQUENCE_LENGTH}, {FEATURE_COUNT})\n"
                f"Received: {array.shape}"
            )

        if not np.isfinite(
            array
        ).all():

            raise ValueError(
                "Behavioral sequence contains "
                "NaN or Inf values."
            )

        return array


    # ========================================================
    # NORMALIZE SEQUENCE
    # ========================================================

    def _normalize_sequence(
        self,
        sequence
    ):

        array = self._validate_sequence(
            sequence
        )

        normalized = (
            array - self.mean
        ) / self.std

        if not np.isfinite(
            normalized
        ).all():

            raise ValueError(
                "Normalization produced "
                "NaN or Inf values."
            )

        return normalized.astype(
            np.float32
        )


    # ========================================================
    # EXTRACT EMBEDDING
    # ========================================================

    def get_embedding(
        self,
        sequence
    ):

        if not self.loaded:

            raise RuntimeError(
                "Siamese BiLSTM service is not loaded."
            )

        normalized = (
            self._normalize_sequence(
                sequence
            )
        )

        tensor = torch.from_numpy(
            normalized
        ).unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():

            embedding = self.model.encoder(
                tensor
            )

        result = embedding[
            0
        ].detach().cpu().numpy()

        # Final safety check.
        if not np.isfinite(
            result
        ).all():

            raise RuntimeError(
                "Model produced invalid embedding."
            )

        return result.astype(
            np.float32
        )


    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    @staticmethod
    def cosine_similarity(
        embedding1,
        embedding2
    ):

        a = np.asarray(
            embedding1,
            dtype=np.float32
        )

        b = np.asarray(
            embedding2,
            dtype=np.float32
        )

        denominator = (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )

        if denominator <= 1e-12:

            raise ValueError(
                "Cannot calculate cosine similarity "
                "for zero-norm embedding."
            )

        similarity = float(
            np.dot(a, b)
            /
            denominator
        )

        return float(
            np.clip(
                similarity,
                -1.0,
                1.0
            )
        )


    # ========================================================
    # COMPARE TWO SEQUENCES
    # ========================================================

    def compare(
        self,
        sequence1,
        sequence2
    ):

        embedding1 = self.get_embedding(
            sequence1
        )

        embedding2 = self.get_embedding(
            sequence2
        )

        similarity = self.cosine_similarity(
            embedding1,
            embedding2
        )

        return {
            "cosine_similarity": similarity,
            "cosine_distance": float(
                1.0 - similarity
            ),
            "embedding_dimension": EMBEDDING_SIZE,
        }


    # ========================================================
    # COMPARE LIVE SEQUENCE WITH ENROLLED EMBEDDING
    # ========================================================

    def score_against_embedding(
        self,
        sequence,
        enrolled_embedding
    ):

        live_embedding = self.get_embedding(
            sequence
        )

        enrolled = np.asarray(
            enrolled_embedding,
            dtype=np.float32
        )

        if enrolled.shape != (
            EMBEDDING_SIZE,
        ):

            raise ValueError(
                "Invalid enrolled embedding shape.\n"
                f"Expected: ({EMBEDDING_SIZE},)\n"
                f"Received: {enrolled.shape}"
            )

        similarity = self.cosine_similarity(
            live_embedding,
            enrolled
        )

        return {
            "cosine_similarity": similarity,
            "cosine_distance": float(
                1.0 - similarity
            ),
            "embedding_dimension": EMBEDDING_SIZE,
            "model": "Siamese_BiLSTM_V3",
            "feature_count": FEATURE_COUNT,
            "sequence_length": SEQUENCE_LENGTH,
        }


    # ========================================================
    # HEALTH INFORMATION
    # ========================================================

    def health(self):

        return {
            "loaded": self.loaded,
            "device": str(self.device),
            "model": "Siamese_BiLSTM_V3",
            "feature_count": FEATURE_COUNT,
            "sequence_length": SEQUENCE_LENGTH,
            "embedding_size": EMBEDDING_SIZE,
            "checkpoint": self.checkpoint_path,
            "normalization": "train_only",
            "scaler": self.scaler_path,
            "training": False,
        }


# ============================================================
# SINGLETON SERVICE
# ============================================================

behavioral_auth = SiameseBiLSTMService()


# ============================================================
# SIMPLE STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("NEURASHIELD — SIAMESE BiLSTM BEHAVIORAL AUTHENTICATION")
    print("=" * 70)

    print()
    print("MODEL")
    print("-" * 70)

    print(
        f"Device           : {DEVICE}"
    )

    print(
        f"Features         : {FEATURE_COUNT}"
    )

    print(
        f"Sequence length  : {SEQUENCE_LENGTH}"
    )

    print(
        f"Embedding size   : {EMBEDDING_SIZE}"
    )

    print(
        "Normalization     : TRAIN ONLY"
    )

    print(
        "Training          : NO"
    )

    print()
    print("HEALTH")
    print("-" * 70)

    health = behavioral_auth.health()

    for key, value in health.items():

        print(
            f"{key:<20}: {value}"
        )

    print()
    print("=" * 70)
    print("SERVICE LOAD TEST: PASS")
    print("=" * 70)
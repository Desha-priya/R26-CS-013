# ============================================================
# KEYSTROKE AGENT V3  (corrected for live NeuraShield)
# ============================================================
# - Siamese BiLSTM V3 only (no Isolation Forest / SVM)
# - Train-only scaler
# - Auto-enroll on first 50-key window
# - Timestamps normalized to MILLISECONDS (Aalto-compatible)
# - Returns cosine SIMILARITY for RiskEngineV3
# ============================================================

import os
import threading
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = r"processed\siamese_bilstm_v3"

MODEL_PATH = os.path.join(
    BASE_DIR, "model_training_v3", "checkpoints", "best_siamese_bilstm_v3.pt"
)
SCALER_PATH = os.path.join(
    BASE_DIR, "reports", "normalization_audit", "train_only_scaler_v3.npz"
)

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

SEQUENCE_LENGTH = 50
FEATURE_COUNT = 16
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
EMBEDDING_SIZE = 64
DROPOUT = 0.30

ROLLING_WINDOW = 10
PAUSE_PERCENTILE = 90

# Cosine threshold from V3 validation (L2-normalized embeddings)
BASELINE_COSINE_THRESHOLD = 0.847201
SUSPICIOUS_WINDOWS_REQUIRED = 2
SCORE_HISTORY_SIZE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL (exact V3 architecture)
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
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        self.embedding = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, EMBEDDING_SIZE),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        embedding = self.embedding(last_output)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class SiameseBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BiLSTMEncoder()

    def forward_one(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        distance = torch.norm(e1 - e2, p=2, dim=1)
        return e1, e2, distance


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint missing model_state_dict")

    model = SiameseBiLSTM().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print(f"[MODEL] Loaded epoch={checkpoint.get('epoch')} device={DEVICE}")
    print("[MODEL] Old IF/SVM path: DISABLED")
    return model


class TrainOnlyScaler:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler not found: {path}")
        data = np.load(path, allow_pickle=False)
        self.mean = data["mean"].astype(np.float32)
        self.std = data["std"].astype(np.float32)
        self.std = np.where(self.std == 0, 1.0, self.std)
        print("[SCALER] Train-only scaler loaded")

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean) / self.std


# ============================================================
# EVENTS + FEATURES
# ============================================================

class KeyEvent:
    def __init__(self, key, press_time_ms, release_time_ms):
        self.key = key
        self.press_time = float(press_time_ms)      # ALWAYS ms
        self.release_time = float(release_time_ms)  # ALWAYS ms


def _to_milliseconds(press_time, release_time):
    """
    Aalto training used millisecond timings.
    time.perf_counter() is seconds → convert.
    Already-ms values (large) are left as-is.
    """
    press_time = float(press_time)
    release_time = float(release_time)

    # Relative seconds (perf_counter) or small values → ms
    if press_time < 1e10 and release_time < 1e10:
        # Heuristic: if dwell looks like seconds, convert
        dwell = abs(release_time - press_time)
        if dwell < 10:  # e.g. 0.12s typing dwell
            return press_time * 1000.0, release_time * 1000.0

    return press_time, release_time


def extract_features(events):
    if len(events) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    section = pd.DataFrame({
        "PRESS_TIME": [e.press_time for e in events],
        "RELEASE_TIME": [e.release_time for e in events],
    }).sort_values("PRESS_TIME").reset_index(drop=True)

    section["DWELL_TIME"] = section["RELEASE_TIME"] - section["PRESS_TIME"]
    section["PREV_PRESS_TIME"] = section["PRESS_TIME"].shift(1)
    section["PREV_RELEASE_TIME"] = section["RELEASE_TIME"].shift(1)
    section["PREV_DWELL_TIME"] = section["DWELL_TIME"].shift(1)
    section["PREV2_PRESS_TIME"] = section["PRESS_TIME"].shift(2)

    section["PRESS_INTERVAL"] = section["PRESS_TIME"] - section["PREV_PRESS_TIME"]
    section["RELEASE_INTERVAL"] = section["RELEASE_TIME"] - section["PREV_RELEASE_TIME"]
    section["RELEASE_PRESS_LATENCY"] = section["PRESS_TIME"] - section["PREV_RELEASE_TIME"]
    section["OVERLAP_DURATION"] = (-section["RELEASE_PRESS_LATENCY"]).clip(lower=0)
    section["OVERLAP_INDICATOR"] = (section["OVERLAP_DURATION"] > 0).astype(np.float32)
    section["DWELL_DIFFERENCE"] = section["DWELL_TIME"] - section["PREV_DWELL_TIME"]
    section["DWELL_RATIO"] = section["DWELL_TIME"] / (section["PREV_DWELL_TIME"] + 1e-6)
    section["TRIGRAPH_PRESS_INTERVAL"] = section["PRESS_TIME"] - section["PREV2_PRESS_TIME"]

    section["MEAN_DWELL"] = section["DWELL_TIME"].rolling(ROLLING_WINDOW, min_periods=3).mean()
    section["STD_DWELL"] = section["DWELL_TIME"].rolling(ROLLING_WINDOW, min_periods=3).std()
    section["MEDIAN_DWELL"] = section["DWELL_TIME"].rolling(ROLLING_WINDOW, min_periods=3).median()
    section["MEAN_IKI"] = section["PRESS_INTERVAL"].rolling(ROLLING_WINDOW, min_periods=3).mean()
    section["STD_IKI"] = section["PRESS_INTERVAL"].rolling(ROLLING_WINDOW, min_periods=3).std()
    section["IKI_CV"] = section["STD_IKI"] / (section["MEAN_IKI"] + 1e-6)

    valid_iki = section["PRESS_INTERVAL"].dropna()
    if len(valid_iki) >= 3:
        pause_threshold = np.percentile(valid_iki, PAUSE_PERCENTILE)
        section["PAUSE_EVENT"] = (section["PRESS_INTERVAL"] > pause_threshold).astype(np.float32)
        section["LOCAL_PAUSE_FREQUENCY"] = (
            section["PAUSE_EVENT"].rolling(ROLLING_WINDOW, min_periods=3).mean()
        )
    else:
        section["LOCAL_PAUSE_FREQUENCY"] = np.nan

    return section[FEATURE_COLUMNS]


def sanitize_features(features):
    X = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ============================================================
# AGENT
# ============================================================

class KeystrokeAgentV3:
    def __init__(self, authenticated_user, reference_embedding=None):
        self.authenticated_user = authenticated_user
        self.model = load_model()
        self.scaler = TrainOnlyScaler(SCALER_PATH)

        self.events = deque(maxlen=SEQUENCE_LENGTH * 3)  # keep a bit more history
        self.score_history = deque(maxlen=SCORE_HISTORY_SIZE)
        self.suspicious_windows = 0
        self.total_inferences = 0
        self.total_suspicious = 0
        self.last_result = None
        self.lock = threading.Lock()

        self.reference_embedding = None
        if reference_embedding is not None:
            self.set_reference_embedding(reference_embedding)

    def set_reference_embedding(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.shape[0] != EMBEDDING_SIZE:
            raise ValueError(f"Embedding must be size {EMBEDDING_SIZE}")
        n = np.linalg.norm(embedding)
        if n <= 1e-12:
            raise ValueError("Zero-norm embedding")
        self.reference_embedding = embedding / n
        print(f"[AGENT] Reference embedding set for user={self.authenticated_user}")

    def create_embedding(self, events):
        features = extract_features(events)
        if len(features) < SEQUENCE_LENGTH:
            raise ValueError("Not enough events")

        features = features.iloc[-SEQUENCE_LENGTH:]
        X = sanitize_features(features)
        X = self.scaler.transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        tensor = torch.from_numpy(X.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.model.forward_one(tensor)
        return emb.squeeze(0).cpu().numpy()

    def add_event(self, key, press_time, release_time):
        """
        press_time / release_time:
          - from time.perf_counter() (seconds) → auto-converted to ms
          - or already ms → left as ms
        """
        press_ms, release_ms = _to_milliseconds(press_time, release_time)

        # basic sanity: ignore impossible dwells
        dwell = release_ms - press_ms
        if dwell <= 0 or dwell > 5000:  # >5s hold ignored
            return {"status": "ignored_event", "reason": "invalid_dwell"}

        event = KeyEvent(key, press_ms, release_ms)

        with self.lock:
            self.events.append(event)

            if len(self.events) < SEQUENCE_LENGTH:
                return {
                    "status": "warming_up",
                    "events": len(self.events),
                    "required": SEQUENCE_LENGTH,
                    "authenticated_user": self.authenticated_user,
                }

            # ---------- AUTO-ENROLL ----------
            # First full window becomes this user's reference.
            # This is NOT manual training. Model already trained offline.
            if self.reference_embedding is None:
                emb = self.create_embedding(list(self.events)[-SEQUENCE_LENGTH:])
                self.set_reference_embedding(emb)
                return {
                    "status": "enrolled",
                    "authenticated_user": self.authenticated_user,
                    "events": len(self.events),
                    "message": "Reference embedding created from first 50-key window",
                }

            # ---------- LIVE SCORE ----------
            return self.process_window(list(self.events)[-SEQUENCE_LENGTH:])

    def process_window(self, events=None):
        if events is None:
            events = list(self.events)[-SEQUENCE_LENGTH:]

        if len(events) < SEQUENCE_LENGTH:
            return {
                "status": "warming_up",
                "events": len(events),
                "required": SEQUENCE_LENGTH,
            }

        if self.reference_embedding is None:
            return {"status": "reference_embedding_required"}

        embedding = self.create_embedding(events)
        similarity = cosine_similarity(embedding, self.reference_embedding)

        self.total_inferences += 1
        self.score_history.append(similarity)

        suspicious = similarity < BASELINE_COSINE_THRESHOLD
        if suspicious:
            self.suspicious_windows += 1
            self.total_suspicious += 1
        else:
            self.suspicious_windows = 0

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "inference_complete",
            "authenticated_user": self.authenticated_user,
            # IMPORTANT: RiskEngineV3 must use this field
            "similarity": float(similarity),
            "behavioral_similarity": float(similarity),
            "baseline_threshold": BASELINE_COSINE_THRESHOLD,
            "suspicious": bool(suspicious),
            "consecutive_suspicious_windows": self.suspicious_windows,
            "reauthentication_hint": self.suspicious_windows >= SUSPICIOUS_WINDOWS_REQUIRED,
            "model": "Siamese_BiLSTM_V3",
        }
        self.last_result = result
        return result

    def reset_session(self):
        with self.lock:
            self.events.clear()
            self.score_history.clear()
            self.suspicious_windows = 0
            self.last_result = None
            # keep reference_embedding unless you want full re-enroll

    def clear_reference(self):
        """Force re-enroll on next 50 keys."""
        self.reference_embedding = None
        self.reset_session()

    def get_status(self):
        return {
            "authenticated_user": self.authenticated_user,
            "events_buffered": len(self.events),
            "required_events": SEQUENCE_LENGTH,
            "reference_loaded": self.reference_embedding is not None,
            "total_inferences": self.total_inferences,
            "baseline_threshold": BASELINE_COSINE_THRESHOLD,
            "model": "Siamese_BiLSTM_V3",
            "device": str(DEVICE),
        }


if __name__ == "__main__":
    print("Loading agent...")
    agent = KeystrokeAgentV3("USER-001")
    print(agent.get_status())
    print("Agent ready.")
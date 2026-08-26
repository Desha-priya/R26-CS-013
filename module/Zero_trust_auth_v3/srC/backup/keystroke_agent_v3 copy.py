# ============================================================
# keystroke_agent_v3.py
#
# NeuraShield
# LIVE V3 BEHAVIORAL BIOMETRIC AGENT
#
# Responsibilities:
#   1. Capture live keyboard timing
#   2. Convert timing to milliseconds
#   3. Extract EXACT 16 V3 features
#   4. Build 50-event sequences
#   5. Apply TRAIN-ONLY normalization
#   6. Generate Siamese BiLSTM V3 embeddings
#   7. Calculate cosine similarity
#   8. Send behavioral evidence to NeuraShield Platform
#
# Does NOT:
#   - train
#   - update weights
#   - use Isolation Forest
#   - use One-Class SVM
#   - use old behavioral profiles
#   - make the final Zero-Trust decision
#
# Final decision is made by RiskEngineV3.
# ============================================================

import os
import time
import logging
import threading
from collections import deque
from datetime import datetime, timezone

import numpy as np
import requests
import torch
import torch.nn as nn

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE = "http://localhost:8000"

USER_ID = "9001"

MODEL_PATH = (
    r"processed\siamese_bilstm_v3\model_training_v3"
    r"\checkpoints\best_siamese_bilstm_v3.pt"
)

SCALER_PATH = (
    r"processed\siamese_bilstm_v3\reports"
    r"\normalization_audit\train_only_scaler_v3.npz"
)

PROFILE_PATH = (
    r"models\keystroke_v3_profile.npz"
)


# ============================================================
# EXACT TRAINING CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 50
FEATURE_COUNT = 16

LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
EMBEDDING_SIZE = 64
DROPOUT = 0.30

COSINE_THRESHOLD = 0.847201

REFERENCE_SEQUENCE_COUNT = 3

ROLLING_WINDOW = 10
PAUSE_PERCENTILE = 90

SCORE_INTERVAL = 5

MIN_EVENTS_FOR_ENROLLMENT = 150


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


logger = logging.getLogger("keystroke_agent_v3")


# ============================================================
# EXACT 16 FEATURES
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
# MODEL
# EXACT CHECKPOINT ARCHITECTURE
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

        self.encoder = BiLSTMEncoder()

    def forward(self, x1, x2):

        e1 = self.encoder(x1)

        e2 = self.encoder(x2)

        distance = torch.norm(
            e1 - e2,
            p=2,
            dim=1
        )

        return e1, e2, distance


# ============================================================
# AGENT
# ============================================================

class KeystrokeAgentV3:

    def __init__(
        self,
        user_id=USER_ID,
        platform_url=API_BASE,
    ):

        self.user_id = str(user_id)

        self.platform_url = (
            platform_url.rstrip("/")
        )

        self.running = False

        self.press_times = {}

        self.events = deque(
            maxlen=500
        )

        self.reference_sequences = []

        self.last_similarity = None
        self.last_risk = None
        self.last_decision = None
        self.last_context = "normal_browsing"

        self.n_scored = 0

        self.session_start = datetime.now()

        self.model = None
        self.mean = None
        self.std = None

        self.listener = None

        self._load_model()
        self._load_scaler()
        self._load_profile()


    # ========================================================
    # MODEL
    # ========================================================

    def _load_model(self):

        print("[MODEL] Loading exact Siamese BiLSTM V3")

        self.model = SiameseBiLSTM().to(
            DEVICE
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        if "model_state_dict" not in checkpoint:

            raise RuntimeError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        self.model.eval()

        params = sum(
            p.numel()
            for p in self.model.parameters()
        )

        print(
            f"[MODEL] Checkpoint epoch : "
            f"{checkpoint.get('epoch')}"
        )

        print(
            f"[MODEL] Validation loss : "
            f"{checkpoint.get('validation_loss')}"
        )

        print(
            f"[MODEL] Parameters : "
            f"{params:,}"
        )

        print(
            "[MODEL] Architecture match : PASS"
        )

        print(
            "[MODEL] Model mode : EVAL"
        )

        print(
            f"[MODEL] Device : {DEVICE}"
        )

        print(
            "[MODEL] Training/weight updates : DISABLED"
        )


    # ========================================================
    # SCALER
    # ========================================================

    def _load_scaler(self):

        scaler = np.load(
            SCALER_PATH
        )

        self.mean = scaler["mean"].astype(
            np.float32
        )

        self.std = scaler["std"].astype(
            np.float32
        )

        self.std = np.where(
            self.std < 1e-8,
            1.0,
            self.std
        )

        if len(self.mean) != FEATURE_COUNT:

            raise RuntimeError(
                f"Scaler feature count mismatch. "
                f"Expected {FEATURE_COUNT}, "
                f"got {len(self.mean)}."
            )

        print(
            "[SCALER] Train-only normalization loaded"
        )

        print(
            f"[SCALER] Feature count : "
            f"{len(self.mean)}"
        )


    # ========================================================
    # PROFILE
    # ========================================================

    def _load_profile(self):

        if not os.path.exists(
            PROFILE_PATH
        ):

            print(
                "[PROFILE] No V3 reference profile found"
            )

            return

        try:

            data = np.load(
                PROFILE_PATH,
                allow_pickle=True
            )

            refs = data[
                "reference_sequences"
            ]

            self.reference_sequences = [

                np.asarray(
                    x,
                    dtype=np.float32
                )

                for x in refs
            ]

            print(
                f"[PROFILE] Loaded "
                f"{len(self.reference_sequences)} "
                f"reference sequences"
            )

        except Exception as exc:

            print(
                f"[PROFILE] Load failed: {exc}"
            )


    def _save_profile(self):

        directory = os.path.dirname(
            PROFILE_PATH
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True
            )

        np.savez(
            PROFILE_PATH,
            reference_sequences=np.asarray(
                self.reference_sequences,
                dtype=np.float32
            )
        )

        print(
            "[PROFILE] V3 reference sequences saved"
        )


    def clear_reference(self):

        self.reference_sequences = []

        if os.path.exists(
            PROFILE_PATH
        ):

            try:
                os.remove(
                    PROFILE_PATH
                )

            except OSError:
                pass

        print(
            "[PROFILE] Reference profile cleared"
        )


    # ========================================================
    # KEYBOARD CAPTURE
    #
    # IMPORTANT:
    # perf_counter() = seconds
    #
    # Aalto/training = milliseconds
    #
    # Therefore:
    #
    #     seconds * 1000 = milliseconds
    # ========================================================

    def _on_press(self, key):

        try:

            self.press_times[
                str(key)
            ] = (
                time.perf_counter()
                * 1000.0
            )

        except Exception:
            pass


    def _on_release(self, key):

        try:

            release_time = (
                time.perf_counter()
                * 1000.0
            )

            key_id = str(key)

            press_time = (
                self.press_times.pop(
                    key_id,
                    None
                )
            )

            if press_time is None:
                return

            dwell = (
                release_time -
                press_time
            )

            # Reject impossible/noisy timings.
            if not (
                1.0 <= dwell <= 3000.0
            ):
                return

            self.events.append({

                "PRESS_TIME":
                    press_time,

                "RELEASE_TIME":
                    release_time,

                "KEYSTROKE_ID":
                    len(self.events),

            })

        except Exception as exc:

            logger.debug(
                f"[CAPTURE] {exc}"
            )


    # ========================================================
    # EXACT V3 FEATURE EXTRACTION
    # ========================================================

    def _extract_features(
        self,
        events
    ):

        if len(events) < SEQUENCE_LENGTH:
            return None

        rows = list(events)

        rows.sort(
            key=lambda x: (
                x["PRESS_TIME"],
                x["KEYSTROKE_ID"]
            )
        )

        press = np.asarray(
            [
                x["PRESS_TIME"]
                for x in rows
            ],
            dtype=np.float64
        )

        release = np.asarray(
            [
                x["RELEASE_TIME"]
                for x in rows
            ],
            dtype=np.float64
        )

        dwell = (
            release - press
        )

        prev_press = np.roll(
            press,
            1
        )

        prev_release = np.roll(
            release,
            1
        )

        prev_dwell = np.roll(
            dwell,
            1
        )

        prev2_press = np.roll(
            press,
            2
        )

        prev_press[0] = np.nan
        prev_release[0] = np.nan
        prev_dwell[0] = np.nan

        prev2_press[:2] = np.nan


        # 1
        press_interval = (
            press - prev_press
        )

        # 2
        release_interval = (
            release - prev_release
        )

        # 3
        release_press_latency = (
            press - prev_release
        )

        # 4
        overlap_duration = np.maximum(
            -release_press_latency,
            0.0
        )

        # 5
        overlap_indicator = (
            overlap_duration > 0
        ).astype(np.float32)

        # 6
        dwell_difference = (
            dwell - prev_dwell
        )

        # 7
        dwell_ratio = (
            dwell /
            (prev_dwell + 1e-6)
        )

        # 8
        trigraph_press_interval = (
            press - prev2_press
        )


        # ====================================================
        # ROLLING FEATURES
        # ====================================================

        mean_dwell = np.full(
            len(rows),
            np.nan
        )

        std_dwell = np.full(
            len(rows),
            np.nan
        )

        median_dwell = np.full(
            len(rows),
            np.nan
        )

        mean_iki = np.full(
            len(rows),
            np.nan
        )

        std_iki = np.full(
            len(rows),
            np.nan
        )

        local_pause_frequency = np.full(
            len(rows),
            np.nan
        )


        valid_iki = press_interval[
            np.isfinite(
                press_interval
            )
        ]

        if len(valid_iki) >= 3:

            pause_threshold = np.percentile(
                valid_iki,
                PAUSE_PERCENTILE
            )

        else:

            pause_threshold = np.nan


        for i in range(
            len(rows)
        ):

            start = max(
                0,
                i - ROLLING_WINDOW + 1
            )

            d = dwell[
                start:i + 1
            ]

            iki = press_interval[
                start:i + 1
            ]

            d = d[
                np.isfinite(d)
            ]

            iki = iki[
                np.isfinite(iki)
            ]

            if len(d) >= 3:

                mean_dwell[i] = (
                    np.mean(d)
                )

                std_dwell[i] = (
                    np.std(
                        d,
                        ddof=1
                    )
                )

                median_dwell[i] = (
                    np.median(d)
                )


            if len(iki) >= 3:

                mean_iki[i] = (
                    np.mean(iki)
                )

                std_iki[i] = (
                    np.std(
                        iki,
                        ddof=1
                    )
                )

                if np.isfinite(
                    pause_threshold
                ):

                    local_pause_frequency[i] = (
                        np.mean(
                            iki >
                            pause_threshold
                        )
                    )


        iki_cv = (
            std_iki /
            (mean_iki + 1e-6)
        )


        # ====================================================
        # EXACT FEATURE ORDER
        # ====================================================

        features = np.column_stack([

            dwell,

            press_interval,

            release_interval,

            release_press_latency,

            overlap_duration,

            overlap_indicator,

            dwell_difference,

            dwell_ratio,

            trigraph_press_interval,

            mean_dwell,

            std_dwell,

            median_dwell,

            mean_iki,

            std_iki,

            iki_cv,

            local_pause_frequency,

        ])


        # ====================================================
        # FINITE VALUE REPAIR
        # ====================================================

        for j in range(
            features.shape[1]
        ):

            column = features[:, j]

            finite = column[
                np.isfinite(column)
            ]

            if len(finite):

                replacement = float(
                    np.median(finite)
                )

            else:

                replacement = 0.0

            column[
                ~np.isfinite(column)
            ] = replacement

            features[:, j] = column


        sequence = features[
            -SEQUENCE_LENGTH:
        ]


        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_COUNT
        ):

            return None

        return sequence.astype(
            np.float32
        )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize(
        self,
        sequence
    ):

        return (
            sequence -
            self.mean
        ) / self.std


    # ========================================================
    # EMBEDDING
    # ========================================================

    @torch.no_grad()
    def _embedding(
        self,
        sequence
    ):

        normalized = self._normalize(
            sequence
        )

        x = torch.tensor(
            normalized,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        embedding = self.model.encoder(
            x
        )

        return (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
        )


    # ========================================================
    # ENROLLMENT
    # ========================================================

    def _try_enrollment(self):

        if len(self.events) < (
            MIN_EVENTS_FOR_ENROLLMENT
        ):

            return False

        all_events = list(
            self.events
        )

        candidates = []

        positions = [
            0,
            50,
            100
        ]

        for start in positions:

            end = (
                start +
                SEQUENCE_LENGTH
            )

            if end > len(all_events):
                continue

            seq = self._extract_features(
                all_events[start:end]
            )

            if seq is not None:

                candidates.append(
                    seq
                )


        for seq in candidates:

            if len(
                self.reference_sequences
            ) >= REFERENCE_SEQUENCE_COUNT:

                break

            self.reference_sequences.append(
                seq
            )


        if self.reference_sequences:

            self._save_profile()

            print(
                f"[PROFILE] Enrolled "
                f"{len(self.reference_sequences)} "
                f"reference sequences"
            )
            # after successful _try_enrollment / _save_profile
            
            try:
                requests.post(
                    f"{self.platform_url}/api/behavioral-score",
                    json={
                        "status": "enrolled",
                        "user_id": self.user_id,
                        "reference_count": len(self.reference_sequences),
                    },
                    timeout=5,
                )
            except Exception:
                pass

            return True

        return False


    # ========================================================
    # COSINE
    # ========================================================

    @staticmethod
    def _cosine_similarity(
        a,
        b
    ):

        denominator = (
            np.linalg.norm(a) *
            np.linalg.norm(b)
        )

        if denominator <= 1e-12:
            return 0.0

        return float(
            np.dot(a, b) /
            denominator
        )


    def _score_sequence(
        self,
        live_sequence
    ):

        if not self.reference_sequences:
            return None

        live_embedding = (
            self._embedding(
                live_sequence
            )
        )

        similarities = []

        for reference in (
            self.reference_sequences
        ):

            reference_embedding = (
                self._embedding(
                    reference
                )
            )

            similarity = (
                self._cosine_similarity(
                    live_embedding,
                    reference_embedding
                )
            )

            similarities.append(
                similarity
            )


        similarity = float(
            np.median(
                similarities
            )
        )

        return {

            "cosine_similarity":
                similarity,

            "reference_similarities":
                similarities,

            "threshold":
                COSINE_THRESHOLD,

            "authenticated":
                similarity >=
                COSINE_THRESHOLD,

        }


    # ========================================================
    # CONTEXT
    # ========================================================

    def _get_context(self):

        try:

            response = requests.get(
                f"{self.platform_url}"
                "/api/session-activity",
                timeout=2
            )

            if response.status_code == 200:

                return (
                    response.json()
                    .get(
                        "current_context",
                        "normal_browsing"
                    )
                )

        except Exception:
            pass

        return "normal_browsing"


    # ========================================================
    # SEND BEHAVIORAL EVIDENCE
    # ========================================================

    def _send_behavioral_result(
        self,
        payload
    ):

        try:

            response = requests.post(
                f"{self.platform_url}"
                "/api/behavioral-score",
                json=payload,
                timeout=5
            )

            response.raise_for_status()

            return response.json()

        except Exception as exc:

            logger.error(
                "[AGENT] Platform communication "
                f"failed: {exc}"
            )

            return None


    # ========================================================
    # SCORE
    # ========================================================

    def score(self):

        if not self.reference_sequences:

            enrolled = (
                self._try_enrollment()
            )

            if enrolled:

                return {
                    "status": "enrolled",
                    "reference_count":
                        len(
                            self.reference_sequences
                        )
                }

            return {
                "status": "warming_up",
                "buffer_size":
                    len(self.events)
            }


        if len(self.events) < (
            SEQUENCE_LENGTH
        ):

            return {
                "status": "warming_up",
                "buffer_size":
                    len(self.events)
            }


        live_events = list(
            self.events
        )[-SEQUENCE_LENGTH:]


        sequence = self._extract_features(
            live_events
        )

        if sequence is None:

            return {
                "status": "feature_error"
            }


        scoring = self._score_sequence(
            sequence
        )

        if scoring is None:

            return {
                "status": "no_reference"
            }


        similarity = float(
            scoring[
                "cosine_similarity"
            ]
        )

        authenticated = bool(
            scoring[
                "authenticated"
            ]
        )

        context = self._get_context()


        behavioral_risk = float(
            np.clip(
                1.0 - similarity,
                0.0,
                1.0
            )
        )


        # Let the platform/RiskEngine make
        # the final adaptive decision.

        payload = {
                "status": "inference_complete",        

                "user_id": self.user_id,
                "behavioral_model": "siamese_bilstm_v3",
                "similarity": similarity,                # dashboard + platform
                "cosine_similarity": similarity,
                "behavioral_threshold": COSINE_THRESHOLD,
                "behavioral_risk": behavioral_risk,
                "authenticated": authenticated,
                "context": context,
                "reference_count": len(self.reference_sequences),
                "reference_similarities": scoring["reference_similarities"],
                "sequence_length": SEQUENCE_LENGTH,
                "feature_count": FEATURE_COUNT,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        
        result = (
            self._send_behavioral_result(
                payload
            )
        )


        self.last_similarity = (
            similarity
        )

        self.last_context = (
            context
        )


        if result:

            self.last_risk = (
                result.get(
                    "overall_risk"
                )
            )

            self.last_decision = (
                result.get(
                    "decision"
                )
            )

        else:

            self.last_risk = (
                behavioral_risk
            )

            self.last_decision = (
                "ALLOW"
                if authenticated
                else "REAUTHENTICATE"
            )


        self.n_scored += 1


        logger.info(
            "[V3 SCORE] "
            f"similarity={similarity:.4f} | "
            f"threshold={COSINE_THRESHOLD:.6f} | "
            f"context={context} | "
            f"decision={self.last_decision}"
        )


        return {

            "status":
                "inference_complete",

            "similarity":
                similarity,

            "behavioral_risk":
                behavioral_risk,

            "authenticated":
                authenticated,

            "context":
                context,

            "risk_decision":
                self.last_decision,

            "overall_risk":
                self.last_risk,

        }


    # ========================================================
    # BACKGROUND SCORING LOOP
    # ========================================================

    def _score_loop(self):

        while self.running:

            try:

                self.score()

            except Exception as exc:

                logger.exception(
                    f"[AGENT] Scoring error: {exc}"
                )

            time.sleep(
                SCORE_INTERVAL
            )


    # ========================================================
    # START
    # ========================================================

    def start(self):

        if not PYNPUT_AVAILABLE:

            raise RuntimeError(
                "pynput is required for "
                "live keyboard capture."
            )

        if self.running:
            return

        self.running = True

        self.listener = (
            keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
        )

        self.listener.daemon = True

        self.listener.start()


        threading.Thread(
            target=self._score_loop,
            daemon=True,
            name="V3ScoreLoop"
        ).start()


        print(
            "[AGENT] V3 started"
        )


    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False

        if self.listener:

            self.listener.stop()

            self.listener = None

        print(
            "[AGENT] V3 stopped"
        )


    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):

        uptime = int(
            (
                datetime.now() -
                self.session_start
            ).total_seconds()
        )

        return {

            "user_id":
                self.user_id,

            "running":
                self.running,

            "behavioral_model":
                "siamese_bilstm_v3",

            "feature_count":
                FEATURE_COUNT,

            "sequence_length":
                SEQUENCE_LENGTH,

            "embedding_size":
                EMBEDDING_SIZE,

            "cosine_threshold":
                COSINE_THRESHOLD,

            "reference_count":
                len(
                    self.reference_sequences
                ),

            "buffer_size":
                len(self.events),

            "last_similarity":
                self.last_similarity,

            "last_risk":
                self.last_risk,

            "last_decision":
                self.last_decision,

            "last_context":
                self.last_context,

            "scores_generated":
                self.n_scored,

            "uptime_s":
                uptime,

            "train_only_normalization":
                True,

            "training_enabled":
                False,

            "weight_updates":
                False,

            "old_behavioral_models":
                False,

            "timing_unit":
                "milliseconds",

        }


# ============================================================
# SELF TEST
# ============================================================

def self_test():

    print("=" * 70)
    print("KEYSTROKE AGENT V3 SELF TEST")
    print("=" * 70)

    test_agent = KeystrokeAgentV3(
        user_id="SELF-TEST"
    )

    print(
        f"Feature count : {FEATURE_COUNT}"
    )

    print(
        f"Sequence length : "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Embedding size : "
        f"{EMBEDDING_SIZE}"
    )

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Cosine threshold : "
        f"{COSINE_THRESHOLD}"
    )


    # Synthetic millisecond events.

    events = []

    base = 1_000_000.0

    for i in range(60):

        press = (
            base +
            i * 180.0
        )

        release = (
            press +
            100.0
        )

        events.append({

            "PRESS_TIME":
                press,

            "RELEASE_TIME":
                release,

            "KEYSTROKE_ID":
                i,

        })


    sequence = (
        test_agent._extract_features(
            events
        )
    )

    assert sequence is not None

    assert sequence.shape == (
        50,
        16
    )

    print(
        "[FEATURE] Shape : "
        f"{sequence.shape}"
    )


    normalized = (
        test_agent._normalize(
            sequence
        )
    )

    assert normalized.shape == (
        50,
        16
    )

    assert np.isfinite(
        normalized
    ).all()

    print(
        "[SCALER] Shape : "
        f"{normalized.shape}"
    )


    embedding = (
        test_agent._embedding(
            sequence
        )
    )

    assert embedding.shape == (
        64,
    )

    norm = np.linalg.norm(
        embedding
    )

    print(
        "[EMBEDDING] Shape : "
        f"{embedding.shape}"
    )

    print(
        "[EMBEDDING] L2 norm : "
        f"{norm:.6f}"
    )

    assert abs(
        norm - 1.0
    ) < 1e-4


    print(
        "[TIMING] Live unit : milliseconds"
    )

    print(
        "[MODEL] Architecture : PASS"
    )

    print(
        "[SELF TEST] PASS"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s"
    )

    self_test()

    print(
        "\nStarting V3 agent..."
    )

    agent = KeystrokeAgentV3()

    agent.start()

    try:

        while True:

            time.sleep(10)

            print(
                agent.get_status()
            )

    except KeyboardInterrupt:

        print(
            "\nStopping V3 agent..."
        )

        agent.stop()
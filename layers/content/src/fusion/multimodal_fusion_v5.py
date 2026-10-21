# =============================================================
# NeuraShield — Layer 4 — Multimodal Fusion Engine (v5)

import sys as _sys
# Status lines use symbols (check marks, box drawing). When stdout/stderr are a
# file, a pipe or a packaged exe's console, Windows defaults to cp1252 and the
# first such print raises UnicodeEncodeError - which used to abort the model
# loading block. Never let a log line kill the layer.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

import requests

from src.common.security import get_api_key, key_fingerprint

# ══════════════════════════════════════════════════════════════

VIDEO_FAKE_IS_RAW = False  
# --- Calibrated decision thresholds (from eval; safe defaults) ---
THRESHOLDS = {
    "text":  0.20,    # CALIBRATED: eval_text.py F1-optimal threshold (EER 2.15%, F1 98.0% on 6000 held-out rows)
    "video": 0.4176,  # CALIBRATED: eval_video_facecrop.py EER threshold (15.68% EER, face-crop pipeline — NOT the old
    "audio": 0.002,   # CALIBRATED: eval_audio.py EER threshold (9.00% EER, official ASVspoof eval, 71237 files)
}

RELIABILITY = {
    "text":  0.98,   
    "video": 0.84,   # CALIBRATED: face-crop pipeline, 1 - EER = 1 - 0.1568.
    "audio": 0.91,   # CALIBRATED: 1 - EER = 1 - 0.09 (official eval set)
}

# --- Base modality weights (attack-vector priors from the brief) ---

BASE_WEIGHTS = {"text": 0.40, "video": 0.40, "audio": 0.20}

# --- Fusion tuning ---
CONSISTENCY_BONUS = 0.10   # max additive boost when modalities agree
NOISY_OR_FLOOR    = True   # let one very-confident modality alert alone
HIGH_CONF_ALERT   = 0.80   # a modality this confident on 'threat' can alert alone

# --- Decision thresholds on the FUSED score ---
FUSED_HIGH   = 0.65
FUSED_MEDIUM = 0.40

# --- Layer D (Zero-Trust Continuous Auth) alert endpoint ---
LAYER_D_ALERT_URL = os.environ.get(
    "NEURASHIELD_LAYER_D_URL", "http://localhost:8000/api/platform/alert")

NEURASHIELD_API_KEY = get_api_key()
_LAYER_D_HEADERS = {"X-API-Key": NEURASHIELD_API_KEY}
print(f"[fusion] Layer D auth key fingerprint: {key_fingerprint(NEURASHIELD_API_KEY)}")

ALERT_SOURCE = "content_threat_detection"
ALERT_LAYER  = "Layer 4" 

ALERT_LAYER_NUMERIC = 4


# ══════════════════════════════════════════════════════════════
#  MODEL LOADING  (lazy, with public readiness flags for app.py)
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("  NeuraShield — Multimodal Fusion Engine (v5)")
print("=" * 60)

import torch

# ---- Model locations -------------------------------------------------
# Resolve model files independently of the current working directory, so the
# layer loads the same models whether it is started from source, by the
# NeuraShield supervisor, or from the packaged exe (where files live under the
# PyInstaller bundle, not next to the process's cwd).
def _layer_roots():
    roots = []
    env = os.environ.get("NEURASHIELD_CONTENT_ROOT")
    if env:
        roots.append(env)
    roots.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    meipass = getattr(__import__("sys"), "_MEIPASS", None)
    if meipass:
        roots += [os.path.join(meipass, "layers", "content"), meipass]
    roots.append(os.getcwd())
    out = []
    for r in roots:
        if r and r not in out:
            out.append(r)
    return out


def _resolve_model(rel_path: str) -> str:
    for root in _layer_roots():
        cand = os.path.join(root, rel_path)
        if os.path.exists(cand):
            return cand
    return os.path.join(_layer_roots()[0], rel_path)


# ---- Text: DistilBERT ----
_TEXT_MODEL_DIR = Path(_resolve_model("models/text/distilbert_phishing"))
_text_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_text_tokenizer = None
_text_model = None

def _load_text_model():
    global _text_tokenizer, _text_model, text_model
    if _text_model is not None:
        return
    from transformers import (DistilBertTokenizerFast,
                              DistilBertForSequenceClassification)
    _text_tokenizer = DistilBertTokenizerFast.from_pretrained(str(_TEXT_MODEL_DIR))
    _text_model = DistilBertForSequenceClassification.from_pretrained(str(_TEXT_MODEL_DIR))
    _text_model.to(_text_device)
    _text_model.eval()
    text_model = _text_model   # publish readiness
    print("[Fusion] DistilBERT phishing model loaded")

text_model = True if _TEXT_MODEL_DIR.exists() else None

# ---- Video: MobileNetV2 CNN (.h5) ----
def _build_mobilenet_deepfake(K):
    """The exact architecture the model was trained with (train_cnn_v2.py):
    MobileNetV2(224, no top) -> GAP -> Dropout(.4) -> Dense(128, relu)
    -> Dropout(.3) -> Dense(1, sigmoid)."""
    base = K.applications.MobileNetV2(weights=None, include_top=False,
                                      input_shape=(224, 224, 3))
    x = K.layers.GlobalAveragePooling2D()(base.output)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Dense(128, activation="relu")(x)
    x = K.layers.Dropout(0.3)(x)
    out = K.layers.Dense(1, activation="sigmoid")(x)
    return K.models.Model(inputs=base.input, outputs=out)


def _load_weights_from_h5(model, path):
    """Copy the weights out of a Keras .h5 file layer by layer, using the file's
    own recorded weight order. Independent of the Keras version that wrote the
    file (a model saved by Keras 3.14 cannot be deserialised by Keras 3.1 /
    TF 2.16, but its weight arrays are plain HDF5 datasets)."""
    import h5py
    by_name = {l.name: l for l in model.layers}
    loaded = 0
    with h5py.File(path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        names = [n.decode() if isinstance(n, bytes) else n for n in g.attrs["layer_names"]]
        for ln in names:
            wn = [w.decode() if isinstance(w, bytes) else w
                  for w in g[ln].attrs.get("weight_names", [])]
            if not wn:
                continue
            by_name[ln].set_weights([np.asarray(g[ln][w]) for w in wn])
            loaded += 1
    expected = sum(1 for l in model.layers if l.weights)
    if loaded != expected:
        raise ValueError(f"only {loaded}/{expected} weighted layers found in {path}")


def _load_video_model(path):
    errors = []
    # 1) the file as-is, with the installed Keras
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False), "tf.keras"
    except Exception as e:
        errors.append(f"tf.keras: {str(e)[:90]}")
    # 2) Keras-2 compatibility package
    try:
        import tf_keras
        return tf_keras.models.load_model(path, compile=False), "tf_keras"
    except Exception as e:
        errors.append(f"tf_keras: {str(e)[:90]}")
    # 3) rebuild the known architecture, read the weights from the file
    for mod in ("tf_keras", "tensorflow.keras"):
        try:
            K = __import__(mod, fromlist=["x"])
            m = _build_mobilenet_deepfake(K)
            _load_weights_from_h5(m, path)
            return m, f"rebuilt architecture + h5 weights ({mod})"
        except Exception as e:
            errors.append(f"rebuild/{mod}: {str(e)[:90]}")
    raise RuntimeError("; ".join(errors))


video_model = None
try:
    _cnn_path = _resolve_model("models/image/deepfake_cnn_model.h5")
    if os.path.exists(_cnn_path):
        video_model, _how = _load_video_model(_cnn_path)
        print(f"  ✓ CNN video model loaded (MobileNetV2) via {_how}")
    else:
        print(f"  ✗ CNN model not found at {_cnn_path}")
except Exception as e:
    print(f"  ✗ CNN video model error: {e}")
    video_model = None

# ---- Audio: ResNet18 on mel spectrogram (.pth) ----
import torch.nn as nn
from torchvision import models as tv_models

class ResNetSpoof(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = tv_models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                     padding=3, bias=False)
        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)

audio_model = None
_AUDIO_MODEL_PATH = _resolve_model("models/audio/spoof_resnet18.pth")
_audio_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    if os.path.exists(_AUDIO_MODEL_PATH):
        audio_model = ResNetSpoof()
        audio_model.load_state_dict(
            torch.load(_AUDIO_MODEL_PATH, map_location=_audio_device,
                       weights_only=True))
        audio_model.to(_audio_device).eval()
        print(f"  ✓ Audio ResNet18 model loaded ({_audio_device})")
    else:
        print(f"  ✗ Audio model not found at {_AUDIO_MODEL_PATH}")
except Exception as e:
    print(f"  ✗ Audio model error: {e}")
    audio_model = None


# ---- Video face-crop preprocessing ----
FACE_MARGIN = 0.35

try:
    import cv2 as _cv2_probe
    _face_cascade = _cv2_probe.CascadeClassifier(
        _cv2_probe.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception as e:
    print(f"  ✗ Face cascade load error: {e}")
    _face_cascade = None


def _crop_face(frame, cv2):
    """Detect the largest face, crop with margin, fall back to full frame."""
    if _face_cascade is None:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                           minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1 = min(frame.shape[1], x + w + mx)
    y1 = min(frame.shape[0], y + h + my)
    crop = frame[y0:y1, x0:x1]
    return crop if crop.size > 0 else frame

# ══════════════════════════════════════════════════════════════
#  PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════════

def predict_text(email_text: str) -> dict:
    if not email_text or not str(email_text).strip():
        return {"prediction": "unknown", "risk_score": 0.0, "confidence": 0.0}
    _load_text_model()
    inputs = _text_tokenizer(str(email_text).strip(), return_tensors="pt",
                             truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(_text_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _text_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    p_phishing = float(probs[1])   # label map: 0=normal, 1=phishing
    pred = "phishing" if p_phishing >= THRESHOLDS["text"] else "normal"
    conf = float(abs(p_phishing - 0.5) * 2)
    return {"prediction": pred, "risk_score": p_phishing, "confidence": conf}


def predict_video(video_path: str, n_frames: int = 15) -> dict:
    if video_model is None:
        return {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}
    import cv2
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"prediction": "error", "risk_score": 0.5, "confidence": 0.0}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return {"prediction": "error", "risk_score": 0.5, "confidence": 0.0}
        frames = []
        for idx in np.linspace(0, total - 1, n_frames, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = _crop_face(frame, cv2)
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = (frame.astype(np.float32) / 127.5) - 1.0
            frames.append(frame)
        cap.release()
        if not frames:
            return {"prediction": "error", "risk_score": 0.5, "confidence": 0.0}
        batch = np.stack(frames, axis=0)
        raw = video_model.predict(batch, verbose=0).flatten()
        # SINGLE source of truth for orientation:
        fake_probs = raw if VIDEO_FAKE_IS_RAW else (1.0 - raw)
        risk = float(np.mean(fake_probs))
        conf = float(abs(risk - 0.5) * 2)
        pred = "fake" if risk >= THRESHOLDS["video"] else "real"
        return {"prediction": pred, "risk_score": round(risk, 4),
                "confidence": round(conf, 4), "frames_used": len(frames)}
    
    except Exception as e:
        print(f"  Video prediction error: {e}")
        return {"prediction": "error", "risk_score": 0.5, "confidence": 0.0}


def _rms_energy(waveform) -> float:
    """Root-mean-square amplitude — a standard, simple voice-activity
    signal. Near-silent audio has RMS close to 0."""
    return float(torch.sqrt(torch.mean(waveform ** 2)).item())


_SILENCE_RMS_THRESHOLD = 0.01   # empirically reasonable for [-1,1]-scaled audio;
                                # tune if real ambient noise sits above this


def predict_audio(audio_path: str) -> dict:
    if audio_model is None:
        return {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}
    try:
        import torchaudio, soundfile as sf
        waveform, sr = sf.read(audio_path)
        waveform = torch.tensor(waveform).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.mean(dim=1, keepdim=True).T

        # Voice-activity gate — check BEFORE resampling/model inference.
        if _rms_energy(waveform) < _SILENCE_RMS_THRESHOLD:
            return {"prediction": "no_speech", "risk_score": 0.0,
                    "confidence": 0.0, "detail": "No voice activity detected"}

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        max_len = 16000 * 4
        if waveform.shape[1] > max_len:
            waveform = waveform[:, :max_len]
        else:
            waveform = torch.nn.functional.pad(
                waveform, (0, max_len - waveform.shape[1]))
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=80)(waveform)
        mel = torch.log(mel + 1e-6)
       
        mel_std = mel.std()
        mel = (mel - mel.mean()) / torch.clamp(mel_std, min=1e-3)
        mel = mel.unsqueeze(0).to(_audio_device)
        with torch.no_grad():
            risk = torch.sigmoid(audio_model(mel)).item()
        pred = "fake" if risk >= THRESHOLDS["audio"] else "real"
        conf = float(abs(risk - 0.5) * 2)
        return {"prediction": pred, "risk_score": round(risk, 4),
                "confidence": round(conf, 4)}
    except Exception as e:
        print(f"  Audio prediction error: {e}")
        return {"prediction": "error", "risk_score": 0.5, "confidence": 0.0}


# ══════════════════════════════════════════════════════════════
#  NOVEL FUSION
# ══════════════════════════════════════════════════════════════

def _is_active(r):
    return r.get("prediction", "unknown") not in ("unknown", "error", "no_speech")

def _threat_vote(key, r):
    """Does this modality independently call a threat?"""
    if key == "text":
        return r.get("prediction") == "phishing"
    return r.get("prediction") == "fake"


def fuse_scores(text_result, video_result, audio_result, weights=None):
    """
    Uncertainty-aware, reliability-weighted, consistency-boosted fusion.
    Returns the alert dict (same shape as v4, plus a 'fusion_detail' block).
    """
    results = {"text": text_result, "video": video_result, "audio": audio_result}
    active = [k for k, r in results.items() if _is_active(r)]

    if not active:
        final_risk = 0.0
        detail = {"note": "no active modalities"}
    else:
        base = weights or BASE_WEIGHTS
       
        eff = {}
        for k in active:
            conf = float(results[k].get("confidence", 0.0))
            eff[k] = base[k] * RELIABILITY[k] * (0.5 + 0.5 * conf)
        tot = sum(eff.values()) or 1e-9

        weighted_risk = sum(results[k]["risk_score"] * (eff[k] / tot)
                            for k in active)

        # Cross-modal consistency: independent agreement is strong evidence.
        votes = [k for k in active if _threat_vote(k, results[k])]
        agree = len(votes)
        if agree >= 2:
           
            mean_conf = np.mean([results[k]["confidence"] for k in votes])
            boost = CONSISTENCY_BONUS * (agree - 1) / max(1, len(active) - 1) * (0.5 + 0.5 * mean_conf)
        else:
            boost = 0.0

        final_risk = weighted_risk + boost

        if NOISY_OR_FLOOR:
            for k in active:
                if _threat_vote(k, results[k]) and results[k]["risk_score"] >= HIGH_CONF_ALERT:
                    final_risk = max(final_risk, results[k]["risk_score"])

        final_risk = float(min(1.0, max(0.0, final_risk)))

        detail = {
            "active_modalities": active,
            "effective_weights": {k: round(eff[k] / tot, 4) for k in active},
            "reliability_used": {k: RELIABILITY[k] for k in active},
            "agreeing_modalities": votes,
            "consistency_boost": round(boost, 4),
            "weighted_risk_before_boost": round(weighted_risk, 4),
        }

    final_risk = round(final_risk, 4)

    if final_risk >= FUSED_HIGH:
        decision = "HIGH_RISK"
        recommendation = "ALERT — notify Layer D (Zero-Trust) to tighten authentication"

    elif final_risk >= FUSED_MEDIUM:
        decision = "MEDIUM_RISK"
        recommendation = "MONITOR — flag for secondary review"

    else:
        decision = "LOW_RISK"
        recommendation = "SAFE — no action required"

    # --- Alert logic: fused decision OR any single high-confidence modality ---
    single_high = any(
        _is_active(results[k]) and _threat_vote(k, results[k])
        and results[k]["risk_score"] >= HIGH_CONF_ALERT
        for k in ("text", "video", "audio")
    )
    should_alert = decision in ("HIGH_RISK", "MEDIUM_RISK") or single_high

    if should_alert:
        severity = "high" if (decision == "HIGH_RISK" or single_high) else "medium"
        parts = []
        for k in ("text", "video", "audio"):
            r = results[k]
            if _is_active(r) and _threat_vote(k, r):
                parts.append(f"{k}:{r.get('prediction')}({r.get('risk_score',0):.2f})")
        message = " | ".join(parts) if parts else f"Multimodal threat (score {final_risk})"
        send_alert_to_layer_d(severity=severity, message=message,
                              final_risk=final_risk, decision=decision)

    return {
        "timestamp": datetime.now().isoformat(),
        "layer": "MultimodalContentThreatDetection",
        "final_risk_score": final_risk,
        "final_decision": decision,
        "recommendation": recommendation,
        "modality_scores": results,
        "weights_used": weights or BASE_WEIGHTS,
        "fusion_detail": detail,
    }


# ══════════════════════════════════════════════════════════════
#  ALERT ROUTING → LAYER D
# ══════════════════════════════════════════════════════════════
_PENDING_ALERTS_DIR = Path("outputs/alerts/pending")
_ALERT_RETRY_ATTEMPTS = 3
_ALERT_RETRY_BACKOFF_SECONDS = (1, 3, 8)


def _queue_failed_alert(payload: dict, reason: str) -> str:
    _PENDING_ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = _PENDING_ALERTS_DIR / f"pending_{ts}.json"
    path.write_text(json.dumps({**payload, "_delivery_failure_reason": reason,
                                "_queued_at": datetime.now().isoformat()}, indent=2))
    print(f"[Layer 4→Platform] alert delivery failed after retries — "
          f"queued at {path} for later replay ({reason})")
    return str(path)


def flush_pending_alerts() -> int:
    """Re-attempt delivery of any previously-queued failed alerts.
    Safe to call periodically — delivered alerts are removed, still-
    failing ones are left queued for the next attempt."""
    if not _PENDING_ALERTS_DIR.exists():
        return 0
    delivered = 0
    for f in sorted(_PENDING_ALERTS_DIR.glob("pending_*.json")):
        try:
            payload = json.loads(f.read_text())
            payload.pop("_delivery_failure_reason", None)
            payload.pop("_queued_at", None)
            resp = requests.post(LAYER_D_ALERT_URL, json=payload,
                                 headers=_LAYER_D_HEADERS, timeout=5)
            if resp.status_code == 200:
                f.unlink()
                delivered += 1
        except Exception:
            continue
    if delivered:
        print(f"[Layer 4→Platform] flushed {delivered} previously-queued alert(s)")
    return delivered


def send_alert_to_layer_d(severity: str, message: str,
                          final_risk: float, decision: str) -> bool:
    """POST a structured alert to Layer D (Zero-Trust Continuous Auth).
    Retries on transient failure; queues durably if still failing."""
    payload = {
        "source": ALERT_SOURCE,          # 'content_threat_detection'
        "severity": severity,            # 'high' | 'medium'
        "layer": ALERT_LAYER_NUMERIC,    # int, required by the platform
        "target_layer": "D",             # informational only, ignored by platform
        "final_risk_score": final_risk,
        "decision": decision,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }

    last_error = None
    for attempt in range(_ALERT_RETRY_ATTEMPTS):
        try:
            resp = requests.post(LAYER_D_ALERT_URL, json=payload,
                                 headers=_LAYER_D_HEADERS, timeout=5)
            if resp.status_code == 200:
                print(f"[Layer 4→Platform] alert sent ({severity}): {message}")
                return True
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_error = str(e)

        if attempt < _ALERT_RETRY_ATTEMPTS - 1:
            wait = _ALERT_RETRY_BACKOFF_SECONDS[min(attempt, len(_ALERT_RETRY_BACKOFF_SECONDS)-1)]
            print(f"[Layer 4→Platform] delivery attempt {attempt+1} failed "
                 f"({last_error}) — retrying in {wait}s...")
            time.sleep(wait)

    _queue_failed_alert(payload, last_error or "unknown error")
    return False


# ══════════════════════════════════════════════════════════════
#  TAMPER-EVIDENT AUDIT LOG (GRC control)
# ══════════════════════════════════════════════════════════════
import hashlib
import hmac as _hmac_mod

_AUDIT_KEY = os.environ.get("NEURASHIELD_AUDIT_KEY", "").encode() or os.urandom(32)
_AUDIT_CHAIN_FILE = Path("outputs/alerts/.chain_state")

try:
    import threading as _threading_mod
    _audit_chain_lock = _threading_mod.Lock()
except Exception:
    _audit_chain_lock = None

class _NullLock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _get_last_signature() -> str:
    if _AUDIT_CHAIN_FILE.exists():
        try:
            return _AUDIT_CHAIN_FILE.read_text().strip()
        except Exception:
            pass
    return "GENESIS"


def _set_last_signature(sig: str):
    _AUDIT_CHAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _AUDIT_CHAIN_FILE.write_text(sig)


def _sign_alert(alert_content: str, prev_signature: str) -> str:
    payload = (prev_signature + "|" + alert_content).encode()
    return _hmac_mod.new(_AUDIT_KEY, payload, hashlib.sha256).hexdigest()


def verify_audit_chain(alert_dir: str = "outputs/alerts") -> dict:
    """Re-walk every saved alert and confirm each signature matches
    what it should be given the previous one. Never raises."""
    d = Path(alert_dir)
    files = sorted(d.glob("alert_*.json")) if d.exists() else []
    prev_sig = "GENESIS"
    verified, broken = 0, []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            broken.append({"file": str(f), "reason": f"unreadable: {e}"})
            continue
        stored_sig = data.get("_audit_signature")
        stored_prev = data.get("_audit_prev_signature")
        content_copy = {k: v for k, v in data.items() if not k.startswith("_audit")}
        content_str = json.dumps(content_copy, sort_keys=True)
        if stored_prev != prev_sig:
            broken.append({"file": str(f), "reason": "chain link broken — "
                          "previous signature does not match (file reordered, "
                          "deleted, or inserted out of sequence)"})
        elif stored_sig != _sign_alert(content_str, prev_sig):
            broken.append({"file": str(f), "reason": "signature mismatch — "
                          "file content was modified after signing"})
        else:
            verified += 1
        prev_sig = stored_sig or prev_sig
    return {
        "total_files": len(files),
        "verified": verified,
        "broken": broken,
        "chain_intact": len(broken) == 0,
    }


# ══════════════════════════════════════════════════════════════
#  CSV AUDIT EXPORT — for manual, human review

import csv as _csv_mod
import hashlib as _hashlib_export  # (hashlib already imported above; alias avoids shadowing confusion)


def export_audit_csv(csv_path: str = None, alert_dir: str = "outputs/alerts") -> dict:

    d = Path(alert_dir)
    files = sorted(d.glob("alert_*.json")) if d.exists() else []

    if csv_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"outputs/alerts/audit_export_{ts}.csv"
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    prev_sig = "GENESIS"
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            rows.append({
                "row_valid": "NO", "reason": f"unreadable file: {e}",
                "alert_id": f.stem, "event_type": "", "timestamp": "",
                "decision": "", "risk_score": "", "text_result": "",
                "video_result": "", "audio_result": "", "note": "",
                "signature_fingerprint": "",
            })
            continue

        stored_sig = data.get("_audit_signature", "")
        stored_prev = data.get("_audit_prev_signature", "")
        content_copy = {k: v for k, v in data.items() if not k.startswith("_audit")}
        content_str = json.dumps(content_copy, sort_keys=True)

        if stored_prev != prev_sig:
            row_valid, reason = "NO", "chain link broken (reordered/deleted/inserted)"
        elif stored_sig != _sign_alert(content_str, prev_sig):
            row_valid, reason = "NO", "content modified after signing"
        else:
            row_valid, reason = "YES", ""
        prev_sig = stored_sig or prev_sig

        mods = data.get("modality_scores", {}) or {}

        def _mod_summary(m):
            if not m or m.get("prediction") in (None, "unknown", "error"):
                return ""
            return f"{m.get('prediction')} ({m.get('risk_score', 0):.2f})"

        rows.append({
            "row_valid": row_valid,
            "reason": reason,
            "alert_id": data.get("alert_id", f.stem),
            "event_type": data.get("event_type", ""),
            "timestamp": data.get("timestamp", ""),
            "decision": data.get("final_decision", ""),
            "risk_score": (f"{data['final_risk_score']:.4f}"
                          if isinstance(data.get("final_risk_score"), (int, float)) else ""),
            "text_result": _mod_summary(mods.get("text")),
            "video_result": _mod_summary(mods.get("video")),
            "audio_result": _mod_summary(mods.get("audio")),
            "note": data.get("note", "") or data.get("acknowledged_alert_id", ""),
            "signature_fingerprint": stored_sig[-12:] if stored_sig else "",
        })

    fieldnames = ["row_valid", "reason", "alert_id", "event_type", "timestamp",
                 "decision", "risk_score", "text_result", "video_result",
                 "audio_result", "note", "signature_fingerprint"]
    with open(csv_path, "w", newline="") as f:
        writer = _csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Sign the exported CSV's exact bytes so it stays verifiable even
    # once separated from the original JSON files/this running process.
    csv_bytes = csv_path.read_bytes()
    csv_sha256 = _hashlib_export.sha256(csv_bytes).hexdigest()
    export_signature = _hmac_mod.new(_AUDIT_KEY, csv_sha256.encode(), hashlib.sha256).hexdigest()

    chain_intact = all(r["row_valid"] == "YES" for r in rows)
    manifest = {
        "csv_file": csv_path.name,
        "csv_sha256": csv_sha256,
        "export_signature": export_signature,
        "generated_at": datetime.now().isoformat(),
        "total_rows": len(rows),
        "chain_intact_at_export_time": chain_intact,
        "note": ("This manifest proves the CSV file has not been edited since "
                "export. Re-run verify_csv_export(csv_path, manifest_path) — "
                "if the CSV's hash no longer matches csv_sha256, or the "
                "signature doesn't verify against the audit key, the file "
                "was altered after this manifest was generated."),
    }
    manifest_path = csv_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "ok": True,
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "rows": len(rows),
        "chain_intact": chain_intact,
    }


def verify_csv_export(csv_path: str, manifest_path: str = None) -> dict:
    """
    Confirms an exported CSV hasn't been modified since export, by
    recomputing its SHA-256 and checking it against the signed value
    in the manifest. Never raises.
    """
    csv_path = Path(csv_path)
    if manifest_path is None:
        manifest_path = csv_path.with_suffix(".manifest.json")
    manifest_path = Path(manifest_path)

    if not csv_path.exists():
        return {"ok": False, "detail": "CSV file not found"}
    if not manifest_path.exists():
        return {"ok": False, "detail": "Manifest file not found"}

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        return {"ok": False, "detail": f"Manifest unreadable: {e}"}

    actual_sha256 = _hashlib_export.sha256(csv_path.read_bytes()).hexdigest()
    expected_sha256 = manifest.get("csv_sha256", "")
    expected_sig = manifest.get("export_signature", "")
    recomputed_sig = _hmac_mod.new(_AUDIT_KEY, expected_sha256.encode(), hashlib.sha256).hexdigest()

    hash_matches = actual_sha256 == expected_sha256
    signature_matches = expected_sig == recomputed_sig

    return {
        "ok": hash_matches and signature_matches,
        "csv_unmodified_since_export": hash_matches,
        "signature_valid": signature_matches,
        "detail": ("CSV verified — unmodified since export" if hash_matches and signature_matches
                  else "CSV FAILED verification — file may have been edited since export, "
                       "or the audit key used to verify differs from the one used to sign it"),
    }


def save_alert(alert: dict, filename: str = None) -> str:
   
    os.makedirs("outputs/alerts", exist_ok=True)
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"outputs/alerts/alert_{ts}.json"

    alert_id = Path(filename).stem   # e.g. "alert_20260101_120000_123456"

    lock_ctx = _audit_chain_lock if _audit_chain_lock else _NullLock()
    with lock_ctx:
        prev_sig = _get_last_signature()
        enriched = dict(alert)
        enriched.setdefault("event_type", "threat_detection")
        enriched["alert_id"] = alert_id
        content_str = json.dumps(enriched, sort_keys=True)
        signature = _sign_alert(content_str, prev_sig)
        signed_alert = dict(enriched)
        signed_alert["_audit_prev_signature"] = prev_sig
        signed_alert["_audit_signature"] = signature
        with open(filename, "w") as f:
            json.dump(signed_alert, f, indent=2)
        _set_last_signature(signature)
    return filename


_SNOOZE_MINUTES = 5


def _alert_categories(alert: dict) -> list:
   
    mods = alert.get("modality_scores", {}) or {}
    cats = []
    if mods.get("text", {}).get("prediction") == "phishing":
        cats.append("phishing")
    if mods.get("video", {}).get("prediction") == "fake":
        cats.append("deepfake_video")
    if mods.get("audio", {}).get("prediction") == "fake":
        cats.append("cloned_voice")
    return cats


def acknowledge_alert(alert_id: str, note: str = "") -> dict:
    
    original_path = Path("outputs/alerts") / f"{alert_id}.json"
    if not original_path.exists():
        return {"ok": False, "detail": f"Alert '{alert_id}' not found"}

    try:
        original = json.loads(original_path.read_text())
        categories = _alert_categories(original)
    except Exception:
        categories = []

    now = datetime.now()
    record = {
        "event_type": "alert_acknowledged",
        "acknowledged_alert_id": alert_id,
        "note": note,
        "timestamp": now.isoformat(),
        "snoozed_categories": categories,
        "snoozed_until": (now + timedelta(minutes=_SNOOZE_MINUTES)).isoformat(),
    }
    path = save_alert(record)
    return {"ok": True, "detail": "Acknowledged", "acknowledgment_id": Path(path).stem,
           "snoozed_categories": categories, "snoozed_for_minutes": _SNOOZE_MINUTES}


_NOTIFICATION_BASELINE_FILE = Path("outputs/alerts/.notification_baseline")


def _get_notification_baseline() -> str:
    if _NOTIFICATION_BASELINE_FILE.exists():
        try:
            return _NOTIFICATION_BASELINE_FILE.read_text().strip()
        except Exception:
            pass
    # First time this runs: everything saved so far becomes "historical",
    # only alerts from this moment forward count as new notifications.
    now = datetime.now().isoformat()
    _NOTIFICATION_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _NOTIFICATION_BASELINE_FILE.write_text(now)
    return now


def list_unacknowledged_alerts(limit: int = 20, min_decision: str = "MEDIUM_RISK") -> list:
 
    d = Path("outputs/alerts")
    if not d.exists():
        return []

    baseline = _get_notification_baseline()
    now = datetime.now()
    rank = {"LOW_RISK": 0, "MEDIUM_RISK": 1, "HIGH_RISK": 2}
    min_rank = rank.get(min_decision, 1)

    detections, acknowledged_ids = [], set()
    snoozed_categories = set()
    for f in sorted(d.glob("alert_*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if data.get("event_type") == "alert_acknowledged":
            acknowledged_ids.add(data.get("acknowledged_alert_id"))
            snooze_until_str = data.get("snoozed_until")
            if snooze_until_str:
                try:
                    if datetime.fromisoformat(snooze_until_str) > now:
                        snoozed_categories.update(data.get("snoozed_categories", []))
                except Exception:
                    pass
        elif (data.get("final_decision")
              and rank.get(data["final_decision"], 0) >= min_rank
              and data.get("alert_id")      
              and data.get("timestamp", "") > baseline):   
            detections.append(data)

    def _is_snoozed(alert):
        cats = _alert_categories(alert)
        return bool(cats) and any(c in snoozed_categories for c in cats)

    detections = [a for a in detections if not _is_snoozed(a)]

    pending = [a for a in detections if a.get("alert_id") not in acknowledged_ids]
    pending.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
    return pending[:limit]


# Backward-compat alias so older app.py imports don't break
send_alert_to_platform = lambda source, severity, message: send_alert_to_layer_d(
    severity=severity, message=message, final_risk=0.0, decision="MEDIUM_RISK")


if __name__ == "__main__":
    print("\n--- Fusion v5 self-test (synthetic, no models needed) ---")
    t = {"prediction": "phishing", "risk_score": 0.91, "confidence": 0.82}
    v = {"prediction": "fake",     "risk_score": 0.78, "confidence": 0.56}
    a = {"prediction": "unknown",  "risk_score": 0.50, "confidence": 0.00}
    alert = fuse_scores(t, v, a)
    print(json.dumps(alert, indent=2))
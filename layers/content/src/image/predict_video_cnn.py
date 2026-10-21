# =============================================================
# NeuraShield — Multimodal Content Threat Detection Layer
# Script  : src/image/predict_video_cnn.py
# Purpose : Load MobileNetV2 CNN and predict real vs deepfake.
#
# IMPORTANT — label mapping fix:
#   ImageDataGenerator reads folders alphabetically:
#     fake/ → class index 0
#     real/ → class index 1
#   So raw CNN output near 0.0 = FAKE, near 1.0 = REAL
#   We flip: risk_score = 1.0 - raw_output
#   So risk_score near 1.0 = fake (high threat)
#      risk_score near 0.0 = real (safe)
#
# ⚠ STATUS: NOT the file actually used at runtime. app.py imports
#   predict_video() from src/fusion/multimodal_fusion_v5.py instead,
#   which is the validated, currently-calibrated implementation — it
#   also adds face-cropping (eval_video_facecrop.py: +24.8pp recall on
#   fakes) and mobilenet-style [-1,1] normalisation, neither of which
#   this file does. This module previously had its flip logic disabled
#   by a stray comment directly contradicting the note above (fixed
#   below). Kept as a simpler, single-purpose reference implementation
#   — if nothing else in your codebase imports it, it's safe to delete.
#   If you keep it, its risk_score will differ from production's
#   predict_video() because of the preprocessing differences above.
# =============================================================

import cv2
import numpy as np
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

# Model loaded once and reused for all predictions
_video_model = None

def load_video_model():
    global _video_model
    if _video_model is not None:
        return _video_model
    try:
        import tensorflow as tf
        model_path = "models/image/deepfake_cnn_model.h5"
        if not Path(model_path).exists():
            print(f"  ✗ CNN model not found: {model_path}")
            return None
        _video_model = tf.keras.models.load_model(model_path)
        print(f"  ✓ CNN video model loaded (MobileNetV2)")
        return _video_model
    except Exception as e:
        print(f"  ✗ Could not load CNN model: {e}")
        return None


def predict_video_cnn(video_path: str, n_frames: int = 15) -> dict:
    """
    Predict whether a video is deepfake using MobileNetV2 CNN.

    Process:
      1. Extract 15 evenly-spaced frames from the video
      2. Resize to 224x224, convert BGR to RGB, normalise 0-1
      3. Batch predict through CNN
      4. Average fake probabilities across all frames

    Label mapping fix:
      ImageDataGenerator assigned: fake=0, real=1 (alphabetical)
      Raw model output near 0 means fake, near 1 means real.
      We flip with: risk_score = 1.0 - raw_output
      Now risk_score near 1.0 = fake threat, near 0.0 = safe real.
    """
    model = load_video_model()
    if model is None:
        return {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"prediction": "error", "risk_score": 0.5, "confidence": 0.0}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return {"prediction": "error", "risk_score": 0.5, "confidence": 0.0}

        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames  = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)

        cap.release()

        if not frames:
            return {"prediction": "error", "risk_score": 0.5, "confidence": 0.0}

        batch     = np.stack(frames, axis=0)
        raw_preds = model.predict(batch, verbose=0).flatten()

        # CONFIRMED empirically in eval_video_facecrop.py (mobilenet/flipped
        # config won): ImageDataGenerator assigned fake=0, real=1, so the
        # raw model output is HIGH for real, LOW for fake. We flip it so
        # risk_score is high for fake, matching every other modality in
        # this project (risk_score near 1.0 = threat) and matching the
        # module docstring above.
        # Keep this in sync with VIDEO_FAKE_IS_RAW in multimodal_fusion_v5.py
        # — that file is the canonical, currently-used implementation.
        fake_probs = 1.0 - raw_preds
        risk_score = float(np.mean(fake_probs))
        confidence = float(abs(risk_score - 0.5) * 2)
        prediction = "fake" if risk_score >= 0.5 else "real"

        return {
            "prediction" : prediction,
            "risk_score" : round(risk_score, 4),
            "confidence" : round(confidence, 4),
            "frames_used": len(frames)
        }

    except Exception as e:
        print(f"  Video prediction error: {e}")
        return {"prediction": "error", "risk_score": 0.5, "confidence": 0.0}


if __name__ == "__main__":
    print("=" * 55)
    print("  CNN Video Prediction Test")
    print("=" * 55)
    tests = [
        ("data/video/raw/original",    "real"),
        ("data/video/raw/manipulated", "fake"),
    ]
    for folder, expected in tests:
        videos = list(Path(folder).glob("*.mp4"))
        if not videos:
            print(f"  No videos found in {folder}")
            continue
        for vp in videos[:3]:
            r  = predict_video_cnn(str(vp))
            ok = "✓" if r["prediction"] == expected else "✗"
            print(f"  {ok} [{expected}] predicted={r['prediction']} "
                  f"risk={r['risk_score']*100:.1f}% "
                  f"conf={r['confidence']*100:.1f}%  {vp.name}")
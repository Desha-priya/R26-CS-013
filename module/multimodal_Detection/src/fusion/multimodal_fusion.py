# =============================================================
# NeuraShield -* Multimodal Content Threat Detection Layer
# Script  : src/fusion/multimodal_fusion.py
# Purpose : Combine confidence scores from all three models
#           (text, video, audio) into a single threat risk score,
#           then output a JSON alert for the other NeuraShield layers.
#
# Fusion strategy:
#   Each model returns a probability/confidence between 0.0 and 1.0
#   that its input is a threat (phishing / deepfake / cloned voice).
#   We combine these using a WEIGHTED AVERAGE:
#       risk = 0.40 × text_score + 0.40 × video_score + 0.20 × audio_score
#
#   Weights reflect how much evidence each modality provides:
#     - Text and video are the strongest indicators (0.40 each)
#     - Audio is supportive evidence (0.20)
#   These weights can be tuned as more evaluation data is collected.
#
# Run from project root:
#   python src/fusion/multimodal_fusion.py
# =============================================================

import joblib
import numpy as np
import pandas as pd
import json
import os
import librosa
import cv2
from datetime import datetime

# ── Load all three trained models ───────────────────────────
print("=" * 60)
print("  NeuraShield -* Multimodal Fusion Engine")
print("=" * 60)

# Each model was saved after training -* load them here
try:
    text_model      = joblib.load("models/text/phishing_text_classifier_v3.pkl")
    text_vectorizer = joblib.load("models/text/tfidf_vectorizer_v3.pkl")
    print("  ✓ Text model loaded")
except Exception as e:
    print(f"  ✗ Text model not found: {e}")
    text_model = None

try:
    video_model = joblib.load("models/image/deepfake_video_model_v5.pkl")
    print("  ✓ Video model loaded")
except Exception as e:
    print(f"  ✗ Video model not found: {e}")
    video_model = None

try:
    audio_model  = joblib.load("models/audio/audio_deepfake_model.pkl")
    audio_scaler = joblib.load("models/audio/audio_scaler.pkl")
    print("  ✓ Audio model loaded")
except Exception as e:
    print(f"  ✗ Audio model not found: {e}")
    audio_model = None


# ── Helper: extract video features from a file ──────────────
def extract_video_features(video_path: str) -> dict | None:
    """
    Sample the first 30 frames of a video and compute the same
    visual features that were used during training.
    Returns a dict of feature values, or None if the file fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    mean_r, mean_g, mean_b, edges = [], [], [], []
    frame_idx = 0
    width, height = 0, 0

    while frame_idx < 30:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        bgr = cv2.mean(frame)
        mean_r.append(bgr[2])
        mean_g.append(bgr[1])
        mean_b.append(bgr[0])
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge  = cv2.Canny(gray, 100, 200)
        edges.append(np.sum(edge > 0) / (height * width))
        frame_idx += 1

    cap.release()
    if frame_idx == 0:
        return None

    return {
        "width"      : width,
        "height"     : height,
        "mean_r"     : float(np.mean(mean_r)),
        "mean_g"     : float(np.mean(mean_g)),
        "mean_b"     : float(np.mean(mean_b)),
        "std_color"  : float(np.std(mean_r + mean_g + mean_b)),
        "edge_ratio" : float(np.mean(edges)),
        "frames_used": frame_idx
    }


# ── Helper: extract MFCC features from an audio file ────────
def extract_audio_features(audio_path: str, n_mfcc=40) -> np.ndarray | None:
    """
    Load audio and compute the same 160-dimensional MFCC feature
    vector used during training.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=4.0, mono=True)
        mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        return np.concatenate([
            np.mean(mfcc,       axis=1),
            np.std(mfcc,        axis=1),
            np.mean(mfcc_delta, axis=1),
            np.std(mfcc_delta,  axis=1),
        ])
    except Exception:
        return None


# ── Core prediction functions ────────────────────────────────

def predict_text(email_text: str) -> dict:
    """
    Predict whether an email is phishing.
    Returns prediction label and a risk score 0.0–1.0.
    """
    if not email_text or not email_text.strip():
        return {"prediction": "unknown", "risk_score": 0.0, "confidence": 0.0}

    X   = text_vectorizer.transform([email_text])
    pred = text_model.predict(X)[0]
    prob = text_model.predict_proba(X)[0]

    # Get phishing risk score
    classes = list(text_model.classes_)
    phishing_idx = classes.index(1) if 1 in classes else classes.index('phishing') if 'phishing' in classes else 0
    risk_score = float(prob[phishing_idx])

    return {
        "prediction" : "phishing" if pred in [1, 'phishing'] else "normal",
        "risk_score" : round(risk_score, 4),
        "confidence" : round(float(max(prob)), 4)
    }


def predict_video(video_path: str) -> dict:
    """
    Extract features from a video file and predict real vs deepfake.
    Returns prediction label and a risk score 0.0–1.0.
    """
    if video_model is None:
        return {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}

    feats = extract_video_features(video_path)
    if feats is None:
        return {"prediction": "error", "risk_score": 0.5, "confidence": 0.0}

    # Build a dataframe with the same column order the model was trained on
    required_cols = list(video_model.feature_names_in_) if hasattr(video_model, 'feature_names_in_') else list(feats.keys())
    X = pd.DataFrame([feats])[required_cols]

    pred = video_model.predict(X)[0]
    prob = video_model.predict_proba(X)[0]

    classes   = list(video_model.classes_)
    fake_idx  = classes.index('fake') if 'fake' in classes else 0
    risk_score = float(prob[fake_idx])

    return {
        "prediction" : str(pred),
        "risk_score" : round(risk_score, 4),
        "confidence" : round(float(max(prob)), 4)
    }


def predict_audio(audio_path: str) -> dict:
    """
    Extract MFCC features from an audio file and predict real vs cloned voice.
    Returns prediction label and a risk score 0.0–1.0.
    """
    if audio_model is None:
        return {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}

    feats = extract_audio_features(audio_path)
    if feats is None:
        return {"prediction": "error", "risk_score": 0.5, "confidence": 0.0}

    X_scaled = audio_scaler.transform([feats])
    pred     = audio_model.predict(X_scaled)[0]
    prob     = audio_model.predict_proba(X_scaled)[0]

    classes  = list(audio_model.classes_)
    fake_idx = classes.index('fake') if 'fake' in classes else 0
    risk_score = float(prob[fake_idx])

    return {
        "prediction" : str(pred),
        "risk_score" : round(risk_score, 4),
        "confidence" : round(float(max(prob)), 4)
    }


# ── Fusion function ──────────────────────────────────────────
def fuse_scores(text_result, video_result, audio_result=None):
    """Balanced fusion with length awareness"""
    
    if audio_result is None:
        audio_result = {"prediction": "unknown", "risk_score": 0.0, "confidence": 0.0}

    text_risk = text_result.get('risk_score', 0.0)
    video_risk = video_result.get('risk_score', 0.0)

    # Penalty for very short text (less reliable)
    text_length = len(text_result.get('prediction', '')) if isinstance(text_result.get('prediction', ''), str) else 0
    length_factor = 0.7 if text_length < 50 else 1.0   # reduce weight for very short text

    final_risk = (text_risk * 0.5 * length_factor) + (video_risk * 0.5)
    final_risk = round(final_risk, 4)

    # More relaxed thresholds
    if final_risk >= 0.78:
        decision = "HIGH_RISK"
        recommendation = "ALERT — Send to Zero-Trust Authentication Layer"
    elif final_risk >= 0.60:
        decision = "MEDIUM_RISK"
        recommendation = "MONITOR — Flag for review"
    else:
        decision = "LOW_RISK"
        recommendation = "SAFE — No immediate action required"

    return {
        "timestamp": datetime.now().isoformat(),
        "layer": "MultimodalContentThreatDetection",
        "final_risk_score": final_risk,
        "final_decision": decision,
        "recommendation": recommendation,
        "modality_scores": {
            "text": text_result,
            "video": video_result,
            "audio": audio_result
        }
    }
# ── Save alert to JSON file ──────────────────────────────────
def save_alert(alert: dict, filename: str = None):
    """Save the alert dictionary as a JSON file in outputs/alerts/."""
    os.makedirs("outputs/alerts", exist_ok=True)
    if filename is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/alerts/alert_{ts}.json"
    with open(filename, 'w') as f:
        json.dump(alert, f, indent=2)
    return filename


# ── Quick self-test with dummy data ─────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Running fusion self-test with example data...")
    print("=" * 60)

    # Simulate what the demo will pass in
    sample_text = (
        "Dear customer, your account has been suspended due to "
        "suspicious activity. Click here immediately to restore access "
        "or your account will be permanently deleted."
    )


    # For the self-test we use dummy scores (0.0–1.0)
    # Replace with actual file paths in the demo
    dummy_text_result  = predict_text(sample_text)
    dummy_video_result = {"prediction": "fake", "risk_score": 0.82, "confidence": 0.82}
    dummy_audio_result = {"prediction": "fake", "risk_score": 0.71, "confidence": 0.71}

    alert = fuse_scores(dummy_text_result, dummy_video_result, dummy_audio_result)

    print("\nText  result :", dummy_text_result)
    print("Video result :", dummy_video_result)
    print("Audio result :", dummy_audio_result)
    print("\n--- Final Alert JSON ---")
    print(json.dumps(alert, indent=2))

    saved_path = save_alert(alert)
    print(f"\nAlert saved to: {saved_path}")
    print("\n✓ Fusion engine is ready. Next step: run demo/app.py")
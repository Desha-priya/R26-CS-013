# =============================================================
# NeuraShield — Layer C — Video: Face-Crop Recall Experiment
# File   : eval_video_facecrop.py
# Run on : YOUR machine, AFTER eval_video.py
#
# WHY THIS EXISTS
#   Your eval_video.py run showed: accuracy 74.69%, but RECALL ON
#   FAKE = 56.39% (44% of deepfakes missed). Real-clip detection was
#   near-perfect (105/108). That asymmetry is a classic signature of
#   feeding the WHOLE FRAME to a CNN that was actually trained on
#   CROPPED FACES — deepfake manipulation artifacts are localized to
#   the face; background/hair/clothing just dilutes the signal.
#
#   This is a testable hypothesis, not a guess: this script re-runs
#   the SAME held-out split (same seed) but crops to the detected
#   face before resizing to 224x224, using OpenCV's bundled Haar
#   cascade (no extra downloads needed). It reports the same metrics
#   so you can directly compare recall/accuracy against eval_video.py.
#
#   If recall improves a lot -> face-crop was the missing piece,
#   adopt it as the production preprocessing.
#   If it doesn't improve -> the model itself needs more/better
#   training data or fine-tuning; report that honestly instead.
#
# Uses the WINNING preprocessing from eval_video.py (mobilenet [-1,1]
# scaling) as the base, since that already beat /255.0.
# =============================================================

import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

MODEL_PATH = "models/image/deepfake_cnn_model.h5"
REAL_DIR   = Path("data/video/raw/original")
FAKE_DIR   = Path("data/video/raw/manipulated")
OUT_DIR    = Path("data/video/eval")
N_FRAMES   = 15
IMG_SIZE   = (224, 224)
SEED       = 42          # SAME seed as eval_video.py -> same test split
TEST_FRAC  = 0.30         # SAME as eval_video.py
SPLIT_BY_PREFIX = 0        # SAME as eval_video.py -- keep in sync

# Face crop settings
FACE_MARGIN = 0.35   # expand the detected box by this fraction on each side
                     # (deepfake artifacts often bleed slightly outside the
                     # tightest box — a small margin helps, too little
                     # crops them out, too much re-adds background noise)

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.RandomState(SEED)

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def log(m):
    print(m, flush=True)


def load_model():
    import tensorflow as tf
    m = tf.keras.models.load_model(MODEL_PATH)
    log(f"[OK] Loaded model: {MODEL_PATH}")
    return m


def list_files():
    real = sorted(REAL_DIR.glob("*.mp4"))
    fake = sorted(FAKE_DIR.glob("*.mp4"))
    log(f"[OK] Found {len(real)} real, {len(fake)} fake clips")
    return real, fake


def make_split(files, label):
    items = [(f, label) for f in files]
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_test = max(1, int(len(items) * TEST_FRAC))
    return [items[i] for i in idx[:n_test]]


def crop_face(frame):
    """Detect the largest face, crop with margin, fall back to full frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                           minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame, False   # no face found -> use full frame, flag it
    # take the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1 = min(frame.shape[1], x + w + mx)
    y1 = min(frame.shape[0], y + h + my)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame, False
    return crop, True


def extract_frames(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, 0
    frames, faces_found = [], 0
    for idx in np.linspace(0, total - 1, N_FRAMES, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        crop, found = crop_face(frame)
        if found:
            faces_found += 1
        crop = cv2.resize(crop, IMG_SIZE)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # winning preprocessing from eval_video.py: [-1,1] scaling
        crop = (crop.astype(np.float32) / 127.5) - 1.0
        frames.append(crop)
    cap.release()
    if not frames:
        return None, 0
    return np.stack(frames, axis=0), faces_found


def compute_eer(y_true, risk):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, risk)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thr[idx])


def main():
    log("=" * 64)
    log("  NeuraShield — Video Face-Crop Recall Experiment")
    log("=" * 64)

    model = load_model()
    real, fake = list_files()
    test_items = make_split(real, 0) + make_split(fake, 1)
    rng.shuffle(test_items)
    log(f"[OK] Held-out TEST set: {len(test_items)} clips (same seed as eval_video.py)")

    from sklearn.metrics import (accuracy_score, roc_auc_score,
                                 precision_recall_fscore_support,
                                 confusion_matrix)

    y_true, scores, no_face_clips = [], [], 0
    for i, (path, label) in enumerate(test_items):
        frames, faces_found = extract_frames(path)
        if frames is None:
            continue
        if faces_found == 0:
            no_face_clips += 1
        preds = model.predict(frames, verbose=0).flatten()
        risk = float(np.mean(1.0 - preds))   # confirmed flip from eval_video.py
        scores.append(risk)
        y_true.append(label)
        if (i + 1) % 20 == 0:
            log(f"     scored {i+1}/{len(test_items)} ...")

    y_true = np.array(y_true)
    scores = np.array(scores)
    if len(y_true) == 0:
        log("[FAIL] No readable clips.")
        return

    pred = (scores >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")
    eer, eer_thr = compute_eer(y_true, scores)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    log("\n" + "=" * 64)
    log("  RESULTS — Face-Crop Preprocessing")
    log("=" * 64)
    log(f"  Test clips        : {len(y_true)}")
    log(f"  Clips w/ no face detected (used full frame fallback): {no_face_clips}")
    log(f"  Accuracy @0.5      : {acc*100:.2f}%")
    log(f"  Precision          : {p*100:.2f}%")
    log(f"  Recall (catches fakes): {r*100:.2f}%   <-- compare to 56.39% baseline")
    log(f"  F1                 : {f1*100:.2f}%")
    log(f"  ROC-AUC            : {auc:.4f}")
    log(f"  EER                : {eer*100:.2f}%  (threshold {eer_thr:.4f})")
    log(f"  Confusion          : TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    log("=" * 64)
    log("\n  --- Interpretation ---")
    log("  Compare RECALL above to the eval_video.py baseline (56.39%).")
    log("  If recall improved meaningfully (e.g. +10pp or more):")
    log("    -> face-crop was a real preprocessing gap. Adopt face-crop")
    log("       in predict_video() (multimodal_fusion_v5.py) permanently.")
    log("  If recall is similar or worse:")
    log("    -> the gap isn't preprocessing — it's the model itself.")
    log("       Report both experiments honestly in the dissertation as")
    log("       evidence of a genuine limitation, and note it as future work")
    log("       (e.g. fine-tune on more diverse manipulation types).")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "face_crop", "n_test": int(len(y_true)),
        "no_face_fallback_clips": no_face_clips,
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        "roc_auc": auc, "eer": eer, "eer_threshold": eer_thr,
        "cm": cm.tolist(), "timestamp": datetime.now().isoformat(),
        "baseline_recall_no_crop": 0.5639,
    }
    (OUT_DIR / f"video_facecrop_eval_{stamp}.json").write_text(json.dumps(payload, indent=2))
    log(f"\n[SAVED] {OUT_DIR}/video_facecrop_eval_{stamp}.json")


if __name__ == "__main__":
    main()

# test_video_accuracy.py
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

os.makedirs("outputs/results", exist_ok=True)

model = tf.keras.models.load_model("models/image/deepfake_cnn_model.h5")

REAL_DIR = Path("data/video/raw/original")
FAKE_DIR = Path("data/video/raw/manipulated")

real_videos = list(REAL_DIR.glob("*.mp4"))[:50]
fake_videos = list(FAKE_DIR.glob("*.mp4"))[:50]
all_videos  = [(v, "real") for v in real_videos] + [(v, "fake") for v in fake_videos]

def predict_video(path):
    cap    = cv2.VideoCapture(str(path))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, total-1, 15, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    if not frames: return 0.5
    batch     = np.stack(frames, axis=0)
    raw_preds = model.predict(batch, verbose=0).flatten()
    return float(np.mean(1.0 - raw_preds))

y_true, y_pred = [], []
for i, (vpath, label) in enumerate(all_videos):
    print(f"[{i+1}/{len(all_videos)}] {vpath.name}")
    risk  = predict_video(vpath)
    pred  = "fake" if risk >= 0.5 else "real"
    y_true.append(label)
    y_pred.append(pred)

acc    = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Real", "Fake"])
cm     = confusion_matrix(y_true, y_pred, labels=["real", "fake"])

lines = [
    "=" * 60,
    "  NeuraShield — Video Model Accuracy Report",
    "=" * 60,
    f"  Model     : MobileNetV2 CNN (Transfer Learning)",
    f"  Dataset   : FaceForensics++ DFD subset",
    f"  Tested on : {len(real_videos)} real + {len(fake_videos)} fake videos",
    f"  Accuracy  : {acc*100:.2f}%",
    "",
    "Classification Report:",
    report,
    "Confusion Matrix:",
    f"  TN={cm[0][0]}  FP={cm[0][1]}",
    f"  FN={cm[1][0]}  TP={cm[1][1]}",
    "=" * 60
]

output = "\n".join(lines)
print(output)

with open("outputs/results/video_model_accuracy.txt", "w") as f:
    f.write(output)

print("\nSaved to: outputs/results/video_model_accuracy.txt")
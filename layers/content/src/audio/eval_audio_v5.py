# =============================================================
# NeuraShield — Layer C — Audio Spoof Model Evaluation
# File   : eval_audio.py
# Run on : YOUR machine (needs spoof_resnet18.pth + ASVspoof eval)
#
# THIS IS YOUR STRONGEST, MOST DEFENSIBLE EVALUATION.
# You have the official ASVspoof 2019 LA eval partition, which is
# the exact held-out set every paper reports on. Zero leakage.
#
# What this does that audio_acc.py did NOT:
#   1. Evaluates on the OFFICIAL EVAL protocol, not the first N
#      training files. (audio_acc.py tested on train data.)
#   2. Reports EER — the standard ASVspoof metric your examiners
#      will expect — plus accuracy, AUC, precision/recall/F1, and
#      a calibrated (EER-optimal) threshold.
#   3. Uses the EXACT preprocessing from Audio_dataset_loader_csv.py
#      so the model sees what it was trained on.
#   4. Handles the full eval set efficiently with batching and
#      progress, and saves artifacts for your dissertation.
#
# PROTOCOL FILE (official names — adjust PATHS below to your disk):
#   Eval protocol : ASVspoof2019.LA.cm.eval.trl.txt
#   Eval flac dir : ASVspoof2019_LA_eval/flac
#   (Dev is also supported — flip USE_PARTITION to 'dev'.)
#
# Protocol line format (same as your train.trn.txt):
#   SPEAKER  FILE_ID  -  SYSTEM_ID  bonafide|spoof
#   we read parts[1] = file id, parts[-1] = label
# =============================================================

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torchvision import models as tv_models

# ----------------------- CONFIG -----------------------
MODEL_PATH = "models/audio/spoof_resnet18.pth"

USE_PARTITION = "eval"     # "eval" (gold standard) or "dev"

# Adjust these to match your folder layout:
PROTO = {
    "eval": "data/audio/raw/ASVspoof2019_LA_eval/ASVspoof2019.LA.cm.eval.trl.txt",
    "dev":  "data/audio/raw/ASVspoof2019_LA_dev/ASVspoof2019.LA.cm.dev.trl.txt",
}
FLAC = {
    "eval": "data/audio/raw/ASVspoof2019_LA_eval/flac",
    "dev":  "data/audio/raw/ASVspoof2019_LA_dev/flac",
}

OUT_DIR   = Path("data/audio/processed/eval")
SR        = 16000
MAX_LEN   = SR * 4        # 4 seconds, same as training
N_FFT     = 1024
HOP       = 512
N_MELS    = 80
BATCH     = 32
# Set to an int (e.g. 4000) for a quick smoke test; None = full eval set
LIMIT     = None
# ------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetSpoof(nn.Module):
    """Identical architecture to model.py / training."""
    def __init__(self):
        super().__init__()
        self.model = tv_models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                     padding=3, bias=False)
        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)


mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)


def log(m):
    print(m, flush=True)


def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    m = ResNetSpoof()
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.to(device).eval()
    log(f"[OK] Loaded model on {device}: {MODEL_PATH}")
    return m


def parse_protocol(part):
    proto_path = PROTO[part]
    flac_dir = FLAC[part]
    if not Path(proto_path).exists():
        raise FileNotFoundError(
            f"Protocol not found: {proto_path}\n"
            f"     Fix the PROTO['{part}'] path at the top of this file.")
    items = []
    with open(proto_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            fid = parts[1]
            label = parts[-1].lower()
            y = 0 if label == "bonafide" else 1   # 1 = spoof/fake
            items.append((os.path.join(flac_dir, fid + ".flac"), y))
    log(f"[OK] Parsed {len(items)} items from {part} protocol")
    return items


def preprocess(path):
    """EXACT match to Audio_dataset_loader_csv.py preprocessing."""
    waveform, sr = sf.read(path)
    waveform = torch.tensor(waveform).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.mean(dim=1, keepdim=True).T
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)
    if waveform.shape[1] > MAX_LEN:
        waveform = waveform[:, :MAX_LEN]
    else:
        waveform = torch.nn.functional.pad(
            waveform, (0, MAX_LEN - waveform.shape[1]))
    mel = mel_transform(waveform)
    mel = torch.log(mel + 1e-6)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    return mel   # (1, 80, T)


def compute_eer(y_true, score_fake):
    """EER; positive class = spoof (1). Returns (eer, threshold)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, score_fake)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thr[idx])


def main():
    log("=" * 64)
    log(f"  NeuraShield — Audio Spoof Evaluation on ASVspoof {USE_PARTITION.upper()}")
    log("=" * 64)

    model = load_model()
    items = parse_protocol(USE_PARTITION)
    if LIMIT:
        items = items[:LIMIT]
        log(f"[WARN] LIMIT set: evaluating only first {LIMIT} (smoke test)")

    scores, y_true, batch_buf, meta_buf = [], [], [], []
    done = 0

    def flush_batch():
        nonlocal batch_buf, meta_buf
        if not batch_buf:
            return
        x = torch.stack(batch_buf).to(device)   # (B,1,80,T)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        for p, y in zip(np.atleast_1d(probs), meta_buf):
            scores.append(float(p))
            y_true.append(int(y))
        batch_buf, meta_buf = [], []

    for path, y in items:
        try:
            mel = preprocess(path)
            batch_buf.append(mel)
            meta_buf.append(y)
        except Exception as e:
            # missing/corrupt file — skip but keep going
            continue
        if len(batch_buf) >= BATCH:
            flush_batch()
        done += 1
        if done % 2000 == 0:
            log(f"     processed {done}/{len(items)} ...")
    flush_batch()

    y_true = np.array(y_true)
    scores = np.array(scores)   # P(spoof)
    if len(y_true) == 0:
        log("[FAIL] No files could be read. Check FLAC paths.")
        return

    from sklearn.metrics import (accuracy_score, roc_auc_score,
                                 precision_recall_fscore_support,
                                 confusion_matrix)
    eer, eer_thr = compute_eer(y_true, scores)

    # Metrics at the default 0.5 threshold AND at EER-optimal threshold
    def metrics_at(th):
        pred = (scores >= th).astype(int)
        acc = accuracy_score(y_true, pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
        return {"threshold": th, "accuracy": acc, "precision": p,
                "recall": r, "f1": f1, "cm": cm}

    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")

    m_def = metrics_at(0.5)
    m_eer = metrics_at(eer_thr)

    n_bona = int((y_true == 0).sum())
    n_spoof = int((y_true == 1).sum())

    log("\n" + "=" * 64)
    log(f"  RESULTS — ASVspoof 2019 LA [{USE_PARTITION}]")
    log("=" * 64)
    log(f"  Files scored : {len(y_true)}  ({n_bona} bonafide, {n_spoof} spoof)")
    log(f"  EER          : {eer*100:.2f}%   <-- headline metric")
    log(f"  ROC-AUC      : {auc:.4f}")
    log("-" * 64)
    log(f"  @ threshold 0.50 : acc={m_def['accuracy']*100:.2f}%  "
        f"P={m_def['precision']*100:.1f}  R={m_def['recall']*100:.1f}  "
        f"F1={m_def['f1']*100:.1f}")
    log(f"  @ EER thr {eer_thr:.3f} : acc={m_eer['accuracy']*100:.2f}%  "
        f"P={m_eer['precision']*100:.1f}  R={m_eer['recall']*100:.1f}  "
        f"F1={m_eer['f1']*100:.1f}")
    cm = m_eer["cm"]
    log(f"  Confusion@EER: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    log("=" * 64)
    log("  NOTE: use the EER threshold as the calibrated operating point")
    log("        in predict_audio() instead of the hard-coded 0.5.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "partition": USE_PARTITION, "model": MODEL_PATH,
        "n_files": int(len(y_true)), "n_bonafide": n_bona, "n_spoof": n_spoof,
        "eer": eer, "eer_threshold": eer_thr, "roc_auc": auc,
        "metrics_at_0.5": m_def, "metrics_at_eer": m_eer,
        "timestamp": datetime.now().isoformat(),
    }
    (OUT_DIR / f"audio_eval_{USE_PARTITION}_{stamp}.json").write_text(
        json.dumps(payload, indent=2))
    (OUT_DIR / f"audio_eval_{USE_PARTITION}_{stamp}.txt").write_text("\n".join([
        f"NeuraShield Audio Spoof Evaluation — ASVspoof 2019 LA [{USE_PARTITION}]",
        f"timestamp     : {datetime.now().isoformat()}",
        f"model         : {MODEL_PATH}",
        f"files_scored  : {len(y_true)} ({n_bona} bonafide, {n_spoof} spoof)",
        f"EER           : {eer*100:.2f}%",
        f"roc_auc       : {auc:.4f}",
        f"acc@0.5       : {m_def['accuracy']*100:.2f}%",
        f"acc@eer_thr   : {m_eer['accuracy']*100:.2f}%",
        f"eer_threshold : {eer_thr:.4f}",
        f"confusion@eer : TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}",
    ]))
    log(f"\n[SAVED] {OUT_DIR}/audio_eval_{USE_PARTITION}_{stamp}.json")
    log(f"[SAVED] {OUT_DIR}/audio_eval_{USE_PARTITION}_{stamp}.txt")


if __name__ == "__main__":
    main()

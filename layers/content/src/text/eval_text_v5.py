# =============================================================
# NeuraShield — Layer C — Text (DistilBERT) Model Evaluation
# File   : eval_text.py
# Run on : YOUR machine (needs models/text/distilbert_phishing +
#          data/text/processed/phishing_distilbert_clean.csv)
#
# WHY THIS EXISTS
#   You already have a strong DistilBERT result, but it wasn't
#   reported the same way as audio/video (no saved EER-style
#   artifact, no confirmed held-out split, no F1-optimal threshold).
#   This script reports it in the SAME format as eval_audio.py /
#   eval_video.py so all three modalities are comparable in your
#   dissertation, and it uses the EXACT preprocessing used by
#   predict_text() in multimodal_fusion_v5.py (truncation=True,
#   padding=True, max_length=256) so the reported number matches
#   what the model actually sees in production.
#
# HONESTY NOTE (same spirit as the video caveat):
#   No training script for DistilBERT was provided to me, so I do
#   not know for certain which rows were used to train it. This
#   script draws a FIXED, SEEDED, STRATIFIED held-out split
#   (test_size=0.2, random_state=42 — the same convention used
#   elsewhere in this project, e.g. train_text_model_v2.py) from
#   the cleaned dataset. If the DistilBERT model was trained on a
#   DIFFERENT split of the same file, this evaluation may overlap
#   with its training data and the number should be reported with
#   that caveat. If you still have the ORIGINAL train/test CSVs
#   used to fine-tune DistilBERT, point TEXT_CSV at the test file
#   instead and set ALREADY_SPLIT = True to skip re-splitting.
# =============================================================

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ----------------------- CONFIG -----------------------
MODEL_DIR  = "models/text/distilbert_phishing"
TEXT_CSV   = "data/text/processed/phishing_distilbert_clean.csv"   # columns: text, y
OUT_DIR    = Path("data/text/eval")

# Set True if TEXT_CSV already IS a held-out test file (no re-splitting).
ALREADY_SPLIT = False

TEST_SIZE   = 0.20
SEED        = 42
MAX_LENGTH  = 256      # must match predict_text() in multimodal_fusion_v5.py
BATCH       = 32
LIMIT       = None     # int for a quick smoke test, None = full test split
# ------------------------------------------------------

os.mkdir(OUT_DIR ) if not OUT_DIR.exists() else None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m):
    print(m, flush=True)


def load_model():
    if not Path(MODEL_DIR).exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")
    tok = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    mdl = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    mdl.to(device).eval()
    log(f"[OK] Loaded DistilBERT from {MODEL_DIR} on {device}")
    return tok, mdl


def load_split():
    if not Path(TEXT_CSV).exists():
        raise FileNotFoundError(f"Dataset not found: {TEXT_CSV}")
    df = pd.read_csv(TEXT_CSV)
    if "text" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Expected columns 'text','y' in {TEXT_CSV}, "
                         f"found {list(df.columns)}")
    df = df.dropna(subset=["text", "y"]).copy()
    df["y"] = df["y"].astype(int)

    if ALREADY_SPLIT:
        log(f"[OK] Using {TEXT_CSV} as-is (already a held-out set): {len(df)} rows")
        return df

    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["y"])
    log(f"[OK] Stratified held-out split: {len(test_df)}/{len(df)} rows "
        f"(test_size={TEST_SIZE}, seed={SEED})")
    log("[WARN] This split was drawn independently of the original DistilBERT")
    log("       training run. If training used a different split of the same")
    log("       file, some overlap with training data is possible. Report the")
    log("       caveat, or re-run with ALREADY_SPLIT=True on your true test file.")
    return test_df


def predict_batch(tok, mdl, texts):
    inputs = tok(texts, return_tensors="pt", truncation=True,
                padding=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs[:, 1]   # P(phishing)


def compute_eer(y_true, score_pos):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, score_pos)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thr[idx])


def f1_optimal_threshold(y_true, score_pos):
    from sklearn.metrics import f1_score
    best_th, best_f1 = 0.5, -1.0
    for th in np.linspace(0.05, 0.95, 91):
        pred = (score_pos >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th, best_f1


def main():
    log("=" * 64)
    log("  NeuraShield — Text (DistilBERT) Phishing Evaluation")
    log("=" * 64)

    tok, mdl = load_model()
    test_df = load_split()
    if LIMIT:
        test_df = test_df.sample(n=min(LIMIT, len(test_df)), random_state=SEED)
        log(f"[WARN] LIMIT set: evaluating only {len(test_df)} rows (smoke test)")

    texts = test_df["text"].astype(str).tolist()
    y_true = test_df["y"].to_numpy()

    scores = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        scores.extend(predict_batch(tok, mdl, chunk).tolist())
        if (i // BATCH) % 20 == 0:
            log(f"     scored {min(i+BATCH, len(texts))}/{len(texts)} ...")
    scores = np.array(scores)

    from sklearn.metrics import (accuracy_score, roc_auc_score,
                                 precision_recall_fscore_support,
                                 confusion_matrix)

    eer, eer_thr = compute_eer(y_true, scores)
    f1_thr, f1_best = f1_optimal_threshold(y_true, scores)

    def metrics_at(th):
        pred = (scores >= th).astype(int)
        acc = accuracy_score(y_true, pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
        return {"threshold": float(th), "accuracy": acc, "precision": p,
                "recall": r, "f1": f1, "cm": cm}

    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")

    m_def = metrics_at(0.5)
    m_f1  = metrics_at(f1_thr)

    n_normal = int((y_true == 0).sum())
    n_phish  = int((y_true == 1).sum())

    log("\n" + "=" * 64)
    log("  RESULTS — Text Phishing (DistilBERT)")
    log("=" * 64)
    log(f"  Rows scored  : {len(y_true)}  ({n_normal} normal, {n_phish} phishing)")
    log(f"  ROC-AUC      : {auc:.4f}")
    log(f"  EER          : {eer*100:.2f}%")
    log("-" * 64)
    log(f"  @ threshold 0.50   : acc={m_def['accuracy']*100:.2f}%  "
        f"P={m_def['precision']*100:.1f}  R={m_def['recall']*100:.1f}  "
        f"F1={m_def['f1']*100:.1f}")
    log(f"  @ F1-optimal {f1_thr:.2f} : acc={m_f1['accuracy']*100:.2f}%  "
        f"P={m_f1['precision']*100:.1f}  R={m_f1['recall']*100:.1f}  "
        f"F1={m_f1['f1']*100:.1f}")
    cm = m_f1["cm"]
    log(f"  Confusion@F1opt: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    log("=" * 64)
    log("  NOTE: RELIABILITY['text'] in multimodal_fusion_v5.py should reflect")
    log(f"        this F1 ({m_f1['f1']:.3f}), e.g. RELIABILITY['text'] = round(f1, 2)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)   # defensive re-create (Windows fix)
    payload = {
        "model": MODEL_DIR, "dataset": TEXT_CSV, "already_split": ALREADY_SPLIT,
        "n_rows": int(len(y_true)), "n_normal": n_normal, "n_phishing": n_phish,
        "roc_auc": auc, "eer": eer, "eer_threshold": eer_thr,
        "f1_optimal_threshold": f1_thr,
        "metrics_at_0.5": m_def, "metrics_at_f1_optimal": m_f1,
        "timestamp": datetime.now().isoformat(),
        "leakage_caveat": ("independently-seeded split; verify against true "
                           "training split" if not ALREADY_SPLIT else "none"),
    }
    (OUT_DIR / f"text_eval_{stamp}.json").write_text(json.dumps(payload, indent=2))
    (OUT_DIR / f"text_eval_{stamp}.txt").write_text("\n".join([
        "NeuraShield Text Phishing Evaluation (DistilBERT)",
        f"timestamp        : {datetime.now().isoformat()}",
        f"model            : {MODEL_DIR}",
        f"rows_scored      : {len(y_true)} ({n_normal} normal, {n_phish} phishing)",
        f"roc_auc          : {auc:.4f}",
        f"eer              : {eer*100:.2f}%",
        f"acc@0.5          : {m_def['accuracy']*100:.2f}%",
        f"acc@f1_optimal   : {m_f1['accuracy']*100:.2f}%",
        f"f1_optimal_thr   : {f1_thr:.3f}",
        f"f1_optimal_score : {m_f1['f1']:.4f}",
        f"confusion@f1opt  : TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}",
        f"leakage_caveat   : {payload['leakage_caveat']}",
    ]))
    log(f"\n[SAVED] {OUT_DIR}/text_eval_{stamp}.json")
    log(f"[SAVED] {OUT_DIR}/text_eval_{stamp}.txt")


if __name__ == "__main__":
    main()

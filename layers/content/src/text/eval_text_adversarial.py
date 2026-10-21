# =============================================================
# NeuraShield — Phase 1.1 — Adversarial Evaluation (Text)
# File   : eval_text_adversarial.py
# Run on : YOUR machine (needs models/text/distilbert_phishing/
#          and the same held-out CSV used by eval_text.py)
#
# WHY THIS EXISTS (the publishable point)
#   Reporting a 98% F1 on held-out data proves the model works
#   under normal conditions. It says nothing about whether it
#   survives an *attacker who knows what the model is looking for*.
#   For a security-domain paper this is the difference between a
#   "we ran a classifier" result and a "we evaluated a defensive
#   system" result. Reviewers at IEEE S&P, USENIX Security, ACSAC
#   expect adversarial evaluation as table-stakes.
#
# WHAT WE TEST (four attacks, weakest to strongest)
#   1. Homoglyph substitution  — replace Latin chars with
#      visually-identical Cyrillic/Greek chars (a real, in-the-wild
#      phishing evasion technique; see Gabrilovich & Gontmakher
#      2002, and every modern phishing report)
#   2. Character-level typos   — random insert/delete/swap; simulates
#      attackers who add noise while keeping meaning
#   3. Word-level synonym sub  — WordNet-based synonym substitution;
#      preserves meaning but shifts distribution
#   4. URL obfuscation         — replace "click here" with encoded
#      variants and legit-looking hyperlink text
#
# WHAT WE REPORT (per attack)
#   - Attack success rate (ASR): % of originally-caught phishing
#     that becomes misclassified as normal after attack
#   - Accuracy drop vs clean baseline
#   - Confidence degradation on still-caught samples
#
# The result is a defensible robustness table that goes directly
# into the paper's evaluation section.
# =============================================================

import os
import json
import random
import string
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ----------------------- CONFIG -----------------------
MODEL_DIR  = "models/text/distilbert_phishing"
TEXT_CSV   = "data/text/processed/phishing_distilbert_clean.csv"
OUT_DIR    = Path("outputs/eval/text_adversarial")
SEED       = 42
TEST_SIZE  = 0.20
MAX_LENGTH = 256
BATCH      = 32
# For each attack we sample this many originally-caught phishing
# emails and apply the perturbation. Set higher for a tighter number,
# lower for a quick smoke test.
N_ATTACK_SAMPLES = 500
# ------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m):
    print(m, flush=True)


# =============================================================
# ATTACK IMPLEMENTATIONS
# Each takes a string and returns a perturbed string. Kept
# deliberately simple and reproducible — no external libraries
# for the attacks themselves so results are exactly replayable.
# =============================================================

# Latin -> visually-identical non-Latin chars. This is a REAL
# technique used in the wild (IDN homograph attacks, evasion of
# keyword filters). Not comprehensive; a defensible sample.
_HOMOGLYPHS = {
    'a': 'а',   # Cyrillic a (U+0430)
    'e': 'е',   # Cyrillic e (U+0435)
    'o': 'о',   # Cyrillic o (U+043E)
    'p': 'р',   # Cyrillic er (U+0440)
    'c': 'с',   # Cyrillic es (U+0441)
    'y': 'у',   # Cyrillic u (U+0443)
    'x': 'х',   # Cyrillic ha (U+0445)
    'i': 'і',   # Cyrillic i (U+0456)
    's': 'ѕ',   # Cyrillic dze (U+0455)
}

def attack_homoglyph(text: str, rate: float = 0.30) -> str:
    """Replace `rate` fraction of eligible chars with homoglyphs."""
    out = []
    for ch in text:
        low = ch.lower()
        if low in _HOMOGLYPHS and random.random() < rate:
            sub = _HOMOGLYPHS[low]
            out.append(sub.upper() if ch.isupper() else sub)
        else:
            out.append(ch)
    return "".join(out)


def attack_typo(text: str, rate: float = 0.05) -> str:
    """Random char insert/delete/swap at `rate` frequency per char."""
    out = list(text)
    i = 0
    while i < len(out):
        if out[i].isalpha() and random.random() < rate:
            op = random.choice(["insert", "delete", "swap"])
            if op == "insert":
                out.insert(i + 1, random.choice(string.ascii_lowercase))
                i += 1
            elif op == "delete" and len(out) > 1:
                out.pop(i)
                i -= 1
            elif op == "swap" and i + 1 < len(out):
                out[i], out[i+1] = out[i+1], out[i]
                i += 1
        i += 1
    return "".join(out)


# Small hand-curated synonym set for the phishing-relevant vocabulary.
# This is not a full WordNet — it's a targeted set of substitutions
# that preserve meaning in the phishing context, so we can measure
# whether the model is anchoring on specific lexical cues.
_SYNONYMS = {
    "urgent": ["immediate", "critical", "pressing", "time-sensitive"],
    "verify": ["confirm", "validate", "authenticate", "check"],
    "click": ["tap", "select", "press", "follow"],
    "account": ["profile", "user record", "membership"],
    "suspended": ["locked", "restricted", "frozen", "disabled"],
    "password": ["passcode", "credentials", "login details"],
    "immediately": ["right away", "at once", "without delay"],
    "security": ["safety", "protection"],
    "important": ["essential", "crucial", "significant"],
    "confirm": ["verify", "validate"],
    "update": ["refresh", "renew", "revise"],
    "click here": ["tap this link", "follow this link", "select here"],
}

def attack_synonym(text: str, rate: float = 0.50) -> str:
    """Replace `rate` fraction of matched keywords with synonyms."""
    result = text
    # sort by length descending so multi-word keys are matched first
    for key in sorted(_SYNONYMS.keys(), key=len, reverse=True):
        if key in result.lower() and random.random() < rate:
            sub = random.choice(_SYNONYMS[key])
            # case-preserving replace at first occurrence only
            idx = result.lower().find(key)
            if idx >= 0:
                original = result[idx:idx+len(key)]
                sub_cased = sub.capitalize() if original[0].isupper() else sub
                result = result[:idx] + sub_cased + result[idx+len(key):]
    return result


def attack_url_obfuscation(text: str) -> str:
    """Rewrite obvious phishing-URL cues in less-flagged forms."""
    subs = [
        ("click here", "read more"),
        ("Click Here", "Read more"),
        ("verify your account", "review your profile"),
        ("http://", "hxxp://"),   # defanged URLs still readable to humans
        ("https://", "hxxps://"),
        ("login", "sign-in portal"),
    ]
    out = text
    for old, new in subs:
        out = out.replace(old, new)
    return out


ATTACKS = {
    "homoglyph":       attack_homoglyph,
    "typo":            attack_typo,
    "synonym":         attack_synonym,
    "url_obfuscation": attack_url_obfuscation,
}


# =============================================================
# EVAL PIPELINE
# =============================================================

def load_model():
    tok = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    mdl = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    mdl.to(device).eval()
    log(f"[OK] Loaded DistilBERT from {MODEL_DIR} on {device}")
    return tok, mdl


def load_test_split():
    df = pd.read_csv(TEXT_CSV).dropna(subset=["text", "y"]).copy()
    df["y"] = df["y"].astype(int)
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=TEST_SIZE,
                                  random_state=SEED, stratify=df["y"])
    log(f"[OK] Held-out split: {len(test_df)} rows")
    return test_df


def predict_batch(tok, mdl, texts):
    """Return P(phishing) per text."""
    inputs = tok(texts, return_tensors="pt", truncation=True,
                padding=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs[:, 1]


def predict_all(tok, mdl, texts):
    scores = []
    for i in range(0, len(texts), BATCH):
        scores.extend(predict_batch(tok, mdl, texts[i:i+BATCH]).tolist())
    return np.array(scores)


def main():
    log("=" * 68)
    log("  NeuraShield — Adversarial Text Evaluation (Phase 1.1)")
    log("=" * 68)

    tok, mdl = load_model()
    test_df = load_test_split()

    # --- Baseline pass on the FULL held-out set (clean) ---
    log("\n[1/3] Scoring clean held-out set...")
    all_texts = test_df["text"].astype(str).tolist()
    all_y = test_df["y"].to_numpy()
    clean_scores = predict_all(tok, mdl, all_texts)
    # Use F1-optimal threshold from eval_text.py; 0.20 was the value
    # from your run. Adjust if you re-ran and got a different one.
    THR = 0.20
    clean_pred = (clean_scores >= THR).astype(int)

    from sklearn.metrics import accuracy_score, recall_score
    clean_acc = accuracy_score(all_y, clean_pred)
    clean_recall_phish = recall_score(all_y, clean_pred, pos_label=1)
    log(f"     Clean accuracy   : {clean_acc*100:.2f}%")
    log(f"     Clean phish recall: {clean_recall_phish*100:.2f}%")

    # --- Pick the phishing samples the model got RIGHT for attack ---
    # We only attack originally-caught phishing. Attacking a
    # misclassified sample doesn't measure attack success — the
    # model was already wrong.
    correct_phish_mask = (all_y == 1) & (clean_pred == 1)
    correct_phish_idx = np.where(correct_phish_mask)[0]
    log(f"     Correctly-caught phishing (pool): {len(correct_phish_idx)}")

    if len(correct_phish_idx) < 20:
        log("[FAIL] Too few caught-phishing samples to attack meaningfully.")
        return

    rng = np.random.RandomState(SEED)
    n_sample = min(N_ATTACK_SAMPLES, len(correct_phish_idx))
    attack_idx = rng.choice(correct_phish_idx, n_sample, replace=False)
    attack_texts = [all_texts[i] for i in attack_idx]
    attack_clean_scores = clean_scores[attack_idx]

    log(f"\n[2/3] Attacking {n_sample} correctly-caught phishing samples...")

    results = {}
    for attack_name, attack_fn in ATTACKS.items():
        perturbed = [attack_fn(t) for t in attack_texts]
        adv_scores = predict_all(tok, mdl, perturbed)
        adv_pred = (adv_scores >= THR).astype(int)
        # Attack success = originally 1, now 0
        n_flipped = int(np.sum(adv_pred == 0))
        asr = n_flipped / n_sample
        conf_drop = float(np.mean(attack_clean_scores - adv_scores))
        # Sample lengths for reference
        mean_len_change = float(np.mean(
            [len(p) - len(o) for p, o in zip(perturbed, attack_texts)]))
        results[attack_name] = {
            "n_samples": n_sample,
            "attack_success_rate": asr,
            "n_flipped_to_normal": n_flipped,
            "mean_score_drop": conf_drop,
            "mean_length_change_chars": mean_len_change,
            "example_original": attack_texts[0][:120] + ("..." if len(attack_texts[0]) > 120 else ""),
            "example_perturbed": perturbed[0][:120] + ("..." if len(perturbed[0]) > 120 else ""),
        }
        log(f"     [{attack_name:>16}] ASR={asr*100:5.2f}%  "
            f"flipped={n_flipped:4d}/{n_sample}  "
            f"mean_score_drop={conf_drop:+.3f}")

    # --- Robustness summary ---
    log("\n[3/3] Robustness summary")
    log("=" * 68)
    log(f"  Clean baseline (F1-opt threshold {THR})")
    log(f"    Accuracy on held-out         : {clean_acc*100:.2f}%")
    log(f"    Recall on phishing class     : {clean_recall_phish*100:.2f}%")
    log(f"  Adversarial (each attack applied to caught phishing only)")
    for name, r in results.items():
        keeps = 100 - r["attack_success_rate"] * 100
        log(f"    {name:>16}  keeps {keeps:5.2f}% of detections  "
            f"(ASR {r['attack_success_rate']*100:.2f}%)")
    log("=" * 68)
    log("  Interpretation:")
    log("    ASR = attack success rate = fraction of correctly-caught")
    log("          phishing that flips to 'normal' after the attack.")
    log("    Lower ASR = more robust. Report every attack, not just")
    log("    the ones the model survives.")
    log("=" * 68)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL_DIR,
        "threshold": THR,
        "clean_accuracy": clean_acc,
        "clean_phish_recall": clean_recall_phish,
        "n_attack_samples": n_sample,
        "attacks": results,
        "timestamp": datetime.now().isoformat(),
    }
    (OUT_DIR / f"text_adversarial_{stamp}.json").write_text(
        json.dumps(payload, indent=2))
    log(f"\n[SAVED] {OUT_DIR}/text_adversarial_{stamp}.json")


if __name__ == "__main__":
    main()

# =============================================================
# NeuraShield — Layer 4 — Shared email scan pipeline
# File   : src/watchers/email_intake.py
#
# Every email intake that is NOT the original IMAP watcher (Gmail API,
# drop-folder, manual .eml upload, notification previews) funnels through
# scan_email(), so they all score and alert EXACTLY like the IMAP watcher:
#
#     predict_text()  ->  fuse_scores(text, unknown, unknown)  ->  save_alert()
#
# fuse_scores() is also what posts the alert to Layer D, so there is no
# second alerting path to drift out of sync.
#
# PRIVACY: raw email text is scored in memory and discarded. The saved alert
# holds only scores/labels/timestamps plus an "intake" tag naming the source
# (e.g. "gmail_api"). Sender and subject are NOT written to the console log
# (the original IMAP watcher prints them; these intakes deliberately don't).
# =============================================================

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.common.email_parse import build_scan_text

_UNKNOWN = {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}

# Previews / one-line notifications are weak evidence; below this many
# characters the text model is scoring noise ("New mail"), so skip them.
MIN_CHARS = int(os.environ.get("NEURASHIELD_MIN_SCAN_CHARS", "1"))


def email_threshold():
    """
    Optional email-specific phishing cut-off (env NEURASHIELD_EMAIL_PHISH_THRESHOLD,
    e.g. 0.85). UNSET by default = behaviour is exactly as before (the model's
    calibrated 0.20). Real mailboxes (newsletters, notifications) look different
    from the public phishing corpora the model was calibrated on, so a higher
    cut-off here cuts false alarms without retraining. Emails scoring below it
    are counted as safe and raise NO alert to Layer D.
    """
    raw = os.environ.get("NEURASHIELD_EMAIL_PHISH_THRESHOLD", "").strip()
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if 0.0 < v <= 1.0 else None


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def scan_email(subject: str, body: str, intake: str, stats: dict = None,
               min_chars: int = None) -> dict:
    """
    Score one email and raise the normal alert if risky.

    Returns {"scanned": bool, "phishing": bool, "risk_score": float,
             "decision": str|None, "reason": str}
    Never raises for a bad email - a single malformed message must not
    stop a watcher; the caller counts it via stats['errors'].
    """
    text = build_scan_text(subject or "", body or "")
    floor = MIN_CHARS if min_chars is None else min_chars
    if len(text) < max(1, floor):
        return {"scanned": False, "phishing": False, "risk_score": 0.0,
                "decision": None, "reason": "too short / empty"}

    # imported lazily: keeps this module importable without loading models
    from src.fusion.multimodal_fusion_v5 import (
        predict_text, fuse_scores, save_alert)

    try:
        text_res = predict_text(text)
        if text_res.get("prediction") in ("unknown", "error"):
            return {"scanned": False, "phishing": False, "risk_score": 0.0,
                    "decision": None, "reason": "text model unavailable"}
        thr = email_threshold()
        if thr is not None:
            risk = float(text_res.get("risk_score", 0.0))
            if risk < thr:
                if stats is not None:
                    stats["checked"] = stats.get("checked", 0) + 1
                    stats["below_threshold"] = stats.get("below_threshold", 0) + 1
                log(f"  [{intake}] [safe] risk={risk:.3f} (below email threshold {thr})")
                return {"scanned": True, "phishing": False, "risk_score": risk,
                        "decision": "LOW_RISK", "reason": "below email threshold"}
            text_res = dict(text_res, prediction="phishing")
        alert = fuse_scores(text_res, dict(_UNKNOWN), dict(_UNKNOWN))
        alert["intake"] = intake
        save_alert(alert)
    except Exception as e:
        if stats is not None:
            stats["errors"] = stats.get("errors", 0) + 1
        return {"scanned": False, "phishing": False, "risk_score": 0.0,
                "decision": None, "reason": f"{type(e).__name__}"}

    phishing = text_res.get("prediction") == "phishing"
    if stats is not None:
        stats["checked"] = stats.get("checked", 0) + 1
        if phishing:
            stats["phishing_found"] = stats.get("phishing_found", 0) + 1
    tag = "PHISHING" if phishing else "safe"
    log(f"  [{intake}] [{tag}] risk={text_res['risk_score']:.3f} "
        f"decision={alert['final_decision']}")
    return {"scanned": True, "phishing": phishing,
            "risk_score": float(text_res["risk_score"]),
            "decision": alert["final_decision"], "reason": ""}

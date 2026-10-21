# =============================================================
# NeuraShield — Layer C — Automatic Email Watcher

# credentials in source):
#   NEURASHIELD_MAIL_HOST      e.g. imap.gmail.com
#   NEURASHIELD_MAIL_USER      your email address
#   NEURASHIELD_MAIL_PASSWORD  app password (NOT your normal password)
#   NEURASHIELD_MAIL_FOLDER    default "INBOX"
#   NEURASHIELD_POLL_SECONDS   default 20

import os
import sys
import time
import imaplib
import email
from email.header import decode_header
from datetime import datetime

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.fusion.multimodal_fusion_v5 import predict_text, fuse_scores, save_alert
from src.common import credential_store

FOLDER = os.environ.get("NEURASHIELD_MAIL_FOLDER", "INBOX")
POLL_SECONDS = int(os.environ.get("NEURASHIELD_POLL_SECONDS", "20"))


def _resolve_credentials():
    """
    Env vars take priority (explicit, terminal/deployment-driven config).
    Falls back to whatever was configured through the dashboard's
    Automatic Monitors form (encrypted local storage, see
    src/common/credential_store.py). Resolved fresh on every connect
    attempt — NOT cached at import time — so a credential entered
    through the dashboard after this module was already imported
    still takes effect on the next (re)connect.
    """
    env_host = os.environ.get("NEURASHIELD_MAIL_HOST", "")
    env_user = os.environ.get("NEURASHIELD_MAIL_USER", "")
    env_pass = os.environ.get("NEURASHIELD_MAIL_PASSWORD", "")
    if env_host and env_user and env_pass:
        return env_host, env_user, env_pass

    stored = credential_store.load_mail_credentials()
    if stored:
        return stored["host"], stored["user"], stored["password"]

    return "", "", ""

_UNKNOWN = {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}

STATS = {
    "checked": 0,
    "phishing_found": 0,
    "last_check": None,        # ISO timestamp of the most recent poll
    "last_error": None,
}


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _decode(value) -> str:
    if value is None:
        return ""
    parts = decode_header(value)
    out = []
    for text, enc in parts:
        if isinstance(text, bytes):
            out.append(text.decode(enc or "utf-8", errors="ignore"))
        else:
            out.append(text)
    return " ".join(out)


def _extract_body(msg) -> str:
    """Plain-text body; if the email has no usable text/plain part (very
    common for real phishing, which is often HTML-only), fall back to the
    visible text of its HTML part instead of skipping it as empty. Emails
    that already had a text/plain part produce exactly the same text."""
    try:
        from src.common.email_parse import html_fallback_text
    except Exception:
        return _extract_plain_body(msg)
    # A single-part text/html message used to be fed to the model as raw
    # markup; give the model the visible text instead.
    if not msg.is_multipart() and msg.get_content_type() == "text/html":
        t = html_fallback_text(msg)
        if t:
            return t
    body = _extract_plain_body(msg)
    if body and body.strip():
        return body
    return html_fallback_text(msg) or body


def _extract_plain_body(msg) -> str:
    """Pull plain-text body from a possibly multipart email."""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if ctype == "text/plain" and "attachment" not in disp:
                try:
                    return part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", errors="ignore")
                except Exception:
                    continue
        return ""
    else:
        try:
            return msg.get_payload(decode=True).decode(
                msg.get_content_charset() or "utf-8", errors="ignore")
        except Exception:
            return str(msg.get_payload())


def connect():
    host, user, password = _resolve_credentials()
    if not (host and user and password):
        raise RuntimeError(
            "No mail credentials configured. Set NEURASHIELD_MAIL_HOST/"
            "USER/PASSWORD, or configure a mailbox in the dashboard's "
            "Automatic Monitors tab.")
    conn = imaplib.IMAP4_SSL(host)
    conn.login(user, password)
    log(f"Connected to {host} as {user}")
    return conn


def _check_new_mail_legacy(conn):
    """(old behaviour, only used when NEURASHIELD_MAIL_SCAN_EXISTING_UNREAD=1)
    Fetch UNSEEN messages, classify each, alert on threats, mark seen."""
    conn.select(FOLDER)
    status, data = conn.search(None, "UNSEEN")
    STATS["last_check"] = datetime.now().isoformat()
    if status != "OK":
        log(f"[WARN] search failed: {status}")
        STATS["last_error"] = f"IMAP search failed: {status}"
        return
    ids = data[0].split()
    if not ids:
        return

    log(f"{len(ids)} new unread message(s)")
    for mid in ids:
        status, msg_data = conn.fetch(mid, "(RFC822)")
        if status != "OK":
            continue
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)

        subject = _decode(msg.get("Subject"))
        sender  = _decode(msg.get("From"))
        body    = _extract_body(msg)
        full_text = f"{subject}\n\n{body}".strip()

        if not full_text:
            log(f"  (skipped, empty body) from={sender}")
            continue

        text_res = predict_text(full_text)
   
        alert = fuse_scores(text_res, _UNKNOWN, _UNKNOWN)
        save_alert(alert)

        STATS["checked"] += 1
        is_phishing = text_res["prediction"] == "phishing"
        if is_phishing:
            STATS["phishing_found"] += 1
        tag = "🚨 PHISHING" if is_phishing else "safe"
        log(f"  [{tag}] risk={text_res['risk_score']:.3f} "
            f"from={sender} subject={subject[:60]!r}")

    for mid in ids:
        conn.store(mid, "+FLAGS", "\\Seen")


# ── Incoming-only scanning (default) ─────────────────────────────────
# The old behaviour scanned EVERY unread message in the folder (a mailbox with
# months of unread mail was scanned in one burst) and then marked them all as
# read. Now: the first time a mailbox is connected we only remember the newest
# message's UID (baseline); afterwards only messages that ARRIVE later are
# scanned. The folder is opened READ-ONLY and messages are fetched with
# BODY.PEEK, so read/unread flags are never changed.
# Set NEURASHIELD_MAIL_SCAN_EXISTING_UNREAD=1 to get the old behaviour back.
import hashlib
import json

SCAN_EXISTING_UNREAD = os.environ.get("NEURASHIELD_MAIL_SCAN_EXISTING_UNREAD", "0") == "1"
_STATE_FILE = credential_store._STORE_DIR / "imap_state.json"


def _state_key():
    host, user, _ = _resolve_credentials()
    return hashlib.sha256(f"{host}|{user}|{FOLDER}".encode()).hexdigest()[:24]


def _load_imap_state():
    try:
        d = json.loads(_STATE_FILE.read_text())
        if d.get("key") == _state_key():
            return d
    except Exception:
        pass
    return None


def _save_imap_state(uidvalidity, last_uid):
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(
            {"key": _state_key(), "uidvalidity": uidvalidity, "last_uid": int(last_uid)}))
    except Exception:
        pass


def reset_imap_state():
    try:
        _STATE_FILE.unlink()
    except Exception:
        pass


def _uidvalidity(conn):
    try:
        typ, data = conn.response("UIDVALIDITY")
        return int(data[0]) if data and data[0] else 0
    except Exception:
        return 0


def _all_uids(conn, criteria):
    status, data = conn.uid("SEARCH", None, *criteria)
    if status != "OK":
        raise RuntimeError(f"IMAP search failed: {status}")
    return [int(x) for x in (data[0] or b"").split()]


def check_new_mail(conn):
    """Scan only mail that arrived since the mailbox was first connected."""
    if SCAN_EXISTING_UNREAD:
        return _check_new_mail_legacy(conn)
    conn.select(FOLDER, readonly=True)
    STATS["last_check"] = datetime.now().isoformat()
    validity = _uidvalidity(conn)
    state = _load_imap_state()
    if state is None or state.get("uidvalidity") != validity:
        uids = _all_uids(conn, ["ALL"])
        base = max(uids) if uids else 0
        _save_imap_state(validity, base)
        log(f"Baseline set at UID {base}: existing mail is NOT scanned, "
            "only new arrivals from now on.")
        return
    last = int(state["last_uid"])
    new = sorted(u for u in _all_uids(conn, [f"UID {last + 1}:*"]) if u > last)
    if not new:
        return
    from src.watchers.email_intake import scan_email
    log(f"{len(new)} new message(s)")
    for uid in new:
        try:
            status, msg_data = conn.uid("FETCH", str(uid), "(BODY.PEEK[])")
            if status == "OK" and msg_data and msg_data[0] and len(msg_data[0]) > 1:
                msg = email.message_from_bytes(msg_data[0][1])
                scan_email(_decode(msg.get("Subject")), _extract_body(msg),
                           "imap", stats=STATS)
        except Exception as e:
            STATS["errors"] = STATS.get("errors", 0) + 1
            log(f"[WARN] could not scan message ({type(e).__name__})")
        last = uid                      # never re-scan, even if this one failed
        _save_imap_state(validity, last)


def main(stop_event=None):
    """
    stop_event: optional threading.Event. When set, the loop exits
    cleanly on its next check (within POLL_SECONDS, and immediately
    if currently sleeping — we wait on the event itself rather than
    a plain time.sleep so a stop request doesn't have to wait out a
    full poll interval). If None, behaves exactly as before
    (runs until KeyboardInterrupt) — the standalone `python
    mail_watcher.py` CLI usage is unaffected.
    """
    log("=" * 60)
    log("  NeuraShield — Automatic Email Watcher (IMAP)")
    log(f"  Folder: {FOLDER}  |  Poll interval: {POLL_SECONDS}s")
    log("=" * 60)

    conn = connect()
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                check_new_mail(conn)
                STATS["last_error"] = None
            except (imaplib.IMAP4.abort, ConnectionError) as e:
                log(f"[WARN] connection dropped ({e}), reconnecting...")
                STATS["last_error"] = f"connection dropped: {e}"
                try:
                    conn.logout()
                except Exception:
                    pass
                if stop_event is not None and stop_event.wait(5):
                    break
                elif stop_event is None:
                    time.sleep(5)
                conn = connect()
            except Exception as e:
                log(f"[ERROR] {e} — continuing")
                STATS["last_error"] = str(e)

            if stop_event is not None:
                if stop_event.wait(POLL_SECONDS):
                    break   # stop was requested during the wait
            else:
                time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        log("Stopping watcher...")
    finally:
        log("Mail watcher stopped.")
        try:
            conn.logout()
        except Exception:
            pass


if __name__ == "__main__":
    main()
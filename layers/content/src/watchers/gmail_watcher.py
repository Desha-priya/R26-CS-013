# =============================================================
# NeuraShield — Layer 4 — Gmail Watcher (Gmail API, read-only OAuth)
# File   : src/watchers/gmail_watcher.py
#
# PURPOSE
#   Watches a Gmail inbox WITHOUT an app password: the user signs in with
#   Google once (see src/common/google_oauth.py), NeuraShield keeps only an
#   encrypted refresh token, and this watcher polls the Gmail API for new
#   inbox mail and scores each message with the text detector.
#
# WHY THE GMAIL API AND NOT IMAP
#   IMAP over OAuth needs the all-powerful https://mail.google.com/ scope
#   ("read, send, delete all your email"). The Gmail API lets us ask for
#   gmail.readonly only - least privilege for a security product.
#
# BEHAVIOUR (all deliberate)
#   - READ-ONLY: never marks mail read, moves, labels or deletes anything
#     (the original IMAP watcher marks scanned mail as read; this does not).
#   - FIRST START baseline: only mail that arrives AFTER the first start is
#     scanned, so connecting a busy inbox doesn't flood the audit log.
#   - DEDUPE across restarts: processed message ids are remembered as
#     SHA-256 hashes only (no subjects, senders or bodies are ever stored).
#   - Email text is scored in memory and discarded.
#   - Polls (default 30 s). Gmail push (Pub/Sub watch) needs a public
#     endpoint, which an on-prem tool shouldn't expose - noted as future work.
# =============================================================

import base64
import email
import hashlib
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.common import credential_store, google_oauth, oauth_store
from src.common.email_parse import extract_email
from src.watchers.email_intake import scan_email

POLL_SECONDS = int(os.environ.get("NEURASHIELD_GMAIL_POLL_SECONDS", "30"))
MAX_PER_POLL = 25
_SEEN_LIMIT = 1000
_STATE_FILE = credential_store._STORE_DIR / "gmail_state.json"
_HTTP_TIMEOUT = 15

STATS = {
    "checked": 0,
    "phishing_found": 0,
    "errors": 0,
    "last_check": None,
    "last_error": None,
}


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ── small persisted state (hashes only) ──────────────────────────────
def _hash_id(msg_id: str) -> str:
    return hashlib.sha256(msg_id.encode()).hexdigest()[:32]


def _load_state(email_addr: str) -> dict:
    try:
        d = json.loads(_STATE_FILE.read_text())
        if d.get("email") == email_addr and isinstance(d.get("seen"), list):
            return d
    except Exception:
        pass
    # new account (or first run): baseline = now
    return {"email": email_addr, "started_at": int(time.time()), "seen": []}


def _save_state(state: dict) -> None:
    try:
        state["seen"] = state["seen"][-_SEEN_LIMIT:]
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state))
    except Exception:
        pass


def reset_state() -> None:
    """Forget the baseline + seen ids (called when the user disconnects)."""
    try:
        _STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


# ── Gmail API access ─────────────────────────────────────────────────
class _Session:
    def __init__(self, http_get=None, http_post=None):
        d = oauth_store.load_gmail()
        self.client_id = d.get("client_id", "")
        self.client_secret = d.get("client_secret", "")
        self.refresh_token = d.get("refresh_token", "")
        self.email = d.get("email", "")
        self._tok = ""
        self._exp = 0.0
        self._get = http_get or _requests_get
        self._post = http_post

    def ready(self) -> bool:
        return bool(self.client_id and self.client_secret and self.refresh_token)

    def token(self, force=False) -> str:
        if force or not self._tok or time.time() > self._exp - 60:
            t = google_oauth.refresh_access_token(
                self.client_id, self.client_secret, self.refresh_token,
                http_post=self._post)
            self._tok, self._exp = t["access_token"], t["expires_at"]
        return self._tok

    def get(self, path: str, params: dict = None):
        """GET a Gmail API path; refresh once on 401. Returns parsed JSON."""
        url = google_oauth.GMAIL_API + path
        for attempt in (0, 1):
            resp = self._get(url, {"Authorization": f"Bearer {self.token(force=attempt == 1)}"},
                             params or {})
            if resp.status_code == 401 and attempt == 0:
                continue
            break
        if resp.status_code == 200:
            return resp.json()
        raise _ApiError(resp.status_code)


class _ApiError(Exception):
    def __init__(self, status):
        super().__init__(f"Gmail API returned HTTP {status}")
        self.status = status


def _requests_get(url, headers, params):
    import requests
    return requests.get(url, headers=headers, params=params,
                        timeout=_HTTP_TIMEOUT)


def fetch_profile_email(access_token: str, http_get=None) -> str:
    """Email address of the signed-in account ('' on failure)."""
    get = http_get or _requests_get
    try:
        r = get(google_oauth.GMAIL_API + "/profile",
                {"Authorization": f"Bearer {access_token}"}, {})
        if r.status_code == 200:
            return r.json().get("emailAddress", "")
    except Exception:
        pass
    return ""


def _decode_raw(raw_b64url: str) -> bytes:
    return base64.urlsafe_b64decode(raw_b64url + "=" * (-len(raw_b64url) % 4))


# ── one polling pass ─────────────────────────────────────────────────
def poll_once(sess: _Session, state: dict) -> int:
    """Scan new inbox mail. Returns the number of messages scanned."""
    q = f"in:inbox after:{state['started_at'] - 60}"
    listing = sess.get("/messages", {"q": q, "maxResults": MAX_PER_POLL})
    STATS["last_check"] = datetime.now().isoformat()
    ids = [m["id"] for m in listing.get("messages", []) if "id" in m]
    seen = set(state["seen"])
    scanned = 0
    try:
        for mid in reversed(ids):                  # oldest first
            h = _hash_id(mid)
            if h in seen:
                continue
            try:
                msg_json = sess.get(f"/messages/{mid}", {"format": "raw"})
                msg = email.message_from_bytes(_decode_raw(msg_json["raw"]))
                subject, _sender, body = extract_email(msg)
                res = scan_email(subject, body, "gmail_api", stats=STATS)
                if res["scanned"]:
                    scanned += 1
            except _ApiError as e:
                if e.status == 404:                # message vanished: skip it
                    STATS["errors"] += 1
                else:
                    raise                          # config / transient: caller decides
            except Exception:
                STATS["errors"] += 1
            # mark processed even on a bad message so we never loop on it
            state["seen"].append(h)
            seen.add(h)
    finally:
        _save_state(state)
    return scanned


def main(stop_event=None, http_get=None, http_post=None):
    """
    Same contract as mail_watcher.main(): loops until stop_event is set.
    Raises RuntimeError with a user-readable reason for non-recoverable
    problems (not signed in, token expired/revoked, API not enabled) so the
    dashboard shows a clear error instead of silently polling a dead token.
    """
    log("=" * 60)
    log("  NeuraShield — Gmail Watcher (Gmail API, read-only)")
    log(f"  Poll interval: {POLL_SECONDS}s")
    log("=" * 60)

    sess = _Session(http_get=http_get, http_post=http_post)
    if not sess.ready():
        raise RuntimeError("Not signed in to Google. Use 'Sign in with Google' "
                           "in Automatic Monitors.")
    state = _load_state(sess.email)
    _save_state(state)
    log(f"Watching Gmail inbox for {sess.email or 'signed-in account'} "
        f"(new mail only)")

    while stop_event is None or not stop_event.is_set():
        try:
            poll_once(sess, state)
            STATS["last_error"] = None
        except google_oauth.InvalidGrantError as e:
            STATS["last_error"] = str(e)
            raise RuntimeError(str(e))
        except _ApiError as e:
            msg = (f"{e} - enable the Gmail API for your Google Cloud project "
                   "and make sure this account is a test user."
                   if e.status in (400, 403) else str(e))
            STATS["last_error"] = msg
            if e.status in (400, 403):
                raise RuntimeError(msg)
            log(f"[WARN] {msg} - will retry")
        except google_oauth.OAuthError as e:
            STATS["last_error"] = str(e)
            log(f"[WARN] {e} - will retry")
        except Exception as e:
            STATS["last_error"] = f"{type(e).__name__}"
            log(f"[WARN] gmail poll failed ({type(e).__name__}) - will retry")

        if stop_event is not None:
            if stop_event.wait(POLL_SECONDS):
                break
        else:
            time.sleep(POLL_SECONDS)
    log("Gmail watcher stopped.")


if __name__ == "__main__":
    main()

# =============================================================
# NeuraShield — Layer 4 — Email Drop-Folder Watcher
# File   : src/watchers/folder_watcher.py
#
# PURPOSE
#   Zero-credential email intake that works with ANY mail client or
#   provider (Gmail, Outlook, Thunderbird, Yahoo, corporate Exchange...):
#   whatever lands in a chosen folder as an .eml (or .txt/.html) file is
#   scored automatically. No password, no OAuth, nothing leaves the machine.
#
#   Typical uses:
#     - Gmail:  open the email -> "Download message" -> saves an .eml
#     - Outlook / Thunderbird: drag a message into the folder, or add a
#       rule that saves suspicious-looking or all incoming mail there
#     - A mail gateway / forwarder that drops .eml files for scanning
#
# BEHAVIOUR (all deliberate)
#   - Never modifies, moves or deletes the user's files.
#   - Remembers processed files as content hashes only (no names/contents),
#     so a restart doesn't re-alert on the same file.
#   - Ignores symlinks, folders, files > 5 MB and other extensions.
#   - A file still being written (modified in the last 2 s) is picked up on
#     the next pass, so half-copied files are never scored.
# =============================================================

import email
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.common import credential_store, sources_config
from src.common.email_parse import extract_email, html_to_text
from src.watchers.email_intake import scan_email

ALLOWED_EXT = {".eml", ".txt", ".html", ".htm"}
MAX_BYTES = 5 * 1024 * 1024
SETTLE_SECONDS = 2.0
POLL_SECONDS = float(os.environ.get("NEURASHIELD_FOLDER_POLL_SECONDS", "3"))
_SEEN_LIMIT = 5000
_SEEN_FILE = credential_store._STORE_DIR / "folder_seen.json"

STATS = {
    "checked": 0,
    "phishing_found": 0,
    "errors": 0,
    "folder": "",
    "last_check": None,
    "last_error": None,
}


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def default_folder() -> Path:
    return Path(os.environ.get("NEURASHIELD_EML_FOLDER", "") or "inbox_watch")


def resolve_folder() -> Path:
    saved = sources_config.folder_path()
    return Path(saved) if saved else default_folder()


def validate_folder(path: str):
    """(ok, detail). Creates the default folder on demand; never creates
    arbitrary user-supplied paths."""
    p = Path(path).expanduser()
    if not p.exists():
        if p == default_folder() or str(path) == str(default_folder()):
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return False, f"Could not create folder: {type(e).__name__}"
        else:
            return False, "Folder does not exist."
    if not p.is_dir():
        return False, "That path is not a folder."
    try:
        list(p.iterdir())
    except Exception:
        return False, "Folder is not readable."
    return True, "ok"


# ── parsing ──────────────────────────────────────────────────────────
def parse_file_bytes(raw: bytes, ext: str):
    """(subject, body) from the bytes of an .eml/.txt/.html file."""
    ext = ext.lower()
    if ext == ".eml":
        msg = email.message_from_bytes(raw)
        subject, _sender, body = extract_email(msg)
        return subject, body
    text = raw.decode("utf-8", errors="ignore")
    if ext in (".html", ".htm"):
        return "", html_to_text(text)
    return "", text


def scan_eml_bytes(raw: bytes, intake: str = "eml_upload", stats: dict = None):
    """Score one .eml given as bytes (used by the manual-upload endpoint)."""
    subject, body = parse_file_bytes(raw, ".eml")
    return scan_email(subject, body, intake, stats=stats)


# ── seen-hash persistence ────────────────────────────────────────────
def _load_seen() -> list:
    try:
        d = json.loads(_SEEN_FILE.read_text())
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _save_seen(seen: list) -> None:
    try:
        _SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SEEN_FILE.write_text(json.dumps(seen[-_SEEN_LIMIT:]))
    except Exception:
        pass


# ── one pass ─────────────────────────────────────────────────────────
def scan_folder_once(folder: Path, seen: list, now: float = None) -> int:
    """Scan new files in `folder`. `seen` (list of hashes) is updated in
    place. Returns how many files were scanned."""
    now = now or time.time()
    seen_set = set(seen)
    scanned = 0
    STATS["last_check"] = datetime.now().isoformat()
    for p in sorted(folder.iterdir()):
        try:
            if p.is_symlink() or not p.is_file():
                continue
            if p.suffix.lower() not in ALLOWED_EXT:
                continue
            st = p.stat()
            if st.st_size == 0 or st.st_size > MAX_BYTES:
                continue
            if now - st.st_mtime < SETTLE_SECONDS:
                continue                       # still being written
            raw = p.read_bytes()
            h = hashlib.sha256(raw).hexdigest()[:32]
            if h in seen_set:
                continue
            subject, body = parse_file_bytes(raw, p.suffix)
            res = scan_email(subject, body, "eml_folder", stats=STATS)
            seen.append(h)
            seen_set.add(h)
            if res["scanned"]:
                scanned += 1
        except Exception:
            STATS["errors"] += 1
    return scanned


def main(stop_event=None):
    folder = resolve_folder()
    ok, detail = validate_folder(str(folder))
    if not ok:
        raise RuntimeError(f"Drop folder problem: {detail}")
    STATS["folder"] = str(folder)

    log("=" * 60)
    log("  NeuraShield — Email Drop-Folder Watcher")
    log(f"  Folder: {folder}  |  Poll interval: {POLL_SECONDS}s")
    log("=" * 60)

    seen = _load_seen()
    while stop_event is None or not stop_event.is_set():
        try:
            if scan_folder_once(folder, seen):
                _save_seen(seen)
            STATS["last_error"] = None
        except FileNotFoundError:
            STATS["last_error"] = "Drop folder disappeared."
        except Exception as e:
            STATS["last_error"] = f"{type(e).__name__}"
            log(f"[WARN] folder scan failed ({type(e).__name__}) - continuing")
        if stop_event is not None:
            if stop_event.wait(POLL_SECONDS):
                break
        else:
            time.sleep(POLL_SECONDS)
    _save_seen(seen)
    log("Folder watcher stopped.")


if __name__ == "__main__":
    main()

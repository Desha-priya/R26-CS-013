# =============================================================
# NeuraShield — Layer 4 — Windows Mail-Notification Watcher  (EXPERIMENTAL)
# File   : src/watchers/notification_watcher.py
#
# PURPOSE
#   "Watch for the notification when an email arrives." Windows keeps a
#   local database of toast notifications it has shown. When a mail app or
#   a mail website (Outlook, Windows Mail, Gmail/Outlook.com in Chrome/Edge)
#   pops up "New email - <sender> - <subject> - <preview>", that toast text
#   is in the database. This watcher reads NEW mail toasts and scores the
#   visible text. No password, no OAuth, nothing sent anywhere.
#
# HOW IT WORKS
#   %LOCALAPPDATA%\Microsoft\Windows\Notifications\wpndatabase.db (SQLite)
#     Notification        Id, HandlerId, Type, Payload (toast XML), ...
#     NotificationHandler RecordId, PrimaryId (the app that showed it)
#   Opened READ-ONLY. Only rows newer than the moment you started the
#   watcher are looked at.
#
# PRIVACY (important - read before enabling)
#   The notification database contains notifications from EVERY app (chat,
#   banking, calendar...). This watcher therefore:
#     1. drops any row whose app is not an allow-listed mail app/browser
#        BEFORE reading its text, and for browsers additionally requires the
#        toast to name a known webmail site;
#     2. never logs or stores the toast text - it is scored in memory and
#        discarded; the alert holds scores only;
#     3. is OFF by default and only runs after the user turns it on.
#
# HONEST LIMITATIONS
#   - A toast shows only sender + subject + a short preview, NOT the full
#     email, so this is an EARLY-WARNING signal with less evidence than the
#     full-body scanners (Gmail watcher / drop-folder / IMAP).
#   - Windows only. It relies on an undocumented Windows file format that
#     can change between Windows builds. If the schema isn't recognised the
#     watcher reports a clear error instead of guessing.
#   - It only sees emails the user's mail app actually notified about (focus
#     assist / Do Not Disturb suppresses toasts).
#   - Notifications are removed from the database after they expire, so a
#     toast shown and dismissed between two polls (every 5 s) can be missed.
#   - Written against the documented schema and tested on synthetic
#     databases; NOT yet validated on a live Windows machine.
# =============================================================

import os
import shutil
import sqlite3
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.watchers.email_intake import scan_email

POLL_SECONDS = float(os.environ.get("NEURASHIELD_NOTIF_POLL_SECONDS", "5"))
MIN_CHARS = int(os.environ.get("NEURASHIELD_NOTIF_MIN_CHARS", "25"))

# Native mail apps (matched as substrings of the notification handler id).
MAIL_APP_TOKENS = ("outlook", "olk", "windowscommunicationsapps",   # Mail & Calendar
                   "thunderbird", "mailbird", "emclient", "gmail")
# Browsers can show web-push for ANY site, so for these the toast must also
# name a webmail host.
BROWSER_TOKENS = ("chrome", "msedge", "edge", "firefox", "brave", "opera")
WEBMAIL_HOSTS = ("mail.google.com", "outlook.live.com", "outlook.office.com",
                 "outlook.office365.com", "mail.yahoo.com", "proton.me",
                 "mail.proton.me")

STATS = {
    "checked": 0,
    "phishing_found": 0,
    "errors": 0,
    "ignored_other_apps": 0,
    "last_check": None,
    "last_error": None,
}


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def default_db_path() -> Path:
    base = os.environ.get("LOCALAPPDATA", "")
    return Path(base) / "Microsoft" / "Windows" / "Notifications" / "wpndatabase.db"


# ── database access ──────────────────────────────────────────────────
class SchemaError(RuntimeError):
    pass


def _open_readonly(db_path: Path):
    """Open the live DB read-only; fall back to a temp copy if it is locked."""
    try:
        # Path.as_uri() gives a correct file:///C:/... URI on Windows
        con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True,
                              timeout=2)
        con.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        return con, None
    except Exception:
        tmp = tempfile.mkdtemp(prefix="ns_wpn_")
        try:
            for suffix in ("", "-wal", "-shm"):
                src = Path(str(db_path) + suffix)
                if src.exists():
                    shutil.copy2(src, Path(tmp) / (db_path.name + suffix))
            con = sqlite3.connect(str(Path(tmp) / db_path.name), timeout=2)
            con.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            return con, tmp
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise


def _columns(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _check_schema(con):
    n, h = _columns(con, "Notification"), _columns(con, "NotificationHandler")
    need_n = {"Id", "HandlerId", "Payload"}
    need_h = {"RecordId", "PrimaryId"}
    if not need_n <= n or not need_h <= h:
        raise SchemaError("Unrecognised Windows notification database layout "
                          "(this Windows build isn't supported yet).")
    return "Type" in n


def max_notification_id(db_path: Path) -> int:
    con, tmp = _open_readonly(db_path)
    try:
        _check_schema(con)
        row = con.execute("SELECT COALESCE(MAX(Id), 0) FROM Notification").fetchone()
        return int(row[0])
    finally:
        con.close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def fetch_new_rows(db_path: Path, after_id: int):
    """[(id, handler_primary_id, payload_bytes)] for toasts with Id > after_id."""
    con, tmp = _open_readonly(db_path)
    try:
        has_type = _check_schema(con)
        sql = ("SELECT n.Id, h.PrimaryId, n.Payload FROM Notification n "
               "JOIN NotificationHandler h ON n.HandlerId = h.RecordId "
               "WHERE n.Id > ? ")
        if has_type:
            sql += "AND (n.Type IS NULL OR lower(n.Type) = 'toast') "
        sql += "ORDER BY n.Id"
        return con.execute(sql, (after_id,)).fetchall()
    finally:
        con.close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# ── filtering + parsing ──────────────────────────────────────────────
def is_mail_handler(primary_id: str) -> bool:
    pid = (primary_id or "").lower()
    return any(t in pid for t in MAIL_APP_TOKENS)


def is_browser_handler(primary_id: str) -> bool:
    pid = (primary_id or "").lower()
    return any(t in pid for t in BROWSER_TOKENS)


def toast_texts(payload) -> list:
    """Visible text strings of a toast XML payload ([] if not parseable)."""
    if payload is None:
        return []
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
        for enc in ("utf-8", "utf-16"):
            try:
                xml = raw.decode(enc)
                break
            except Exception:
                xml = None
        if xml is None:
            return []
    else:
        xml = str(payload)
    xml = xml.lstrip("﻿\x00 \r\n\t")
    try:
        root = ET.fromstring(xml)
    except Exception:
        return []
    out = []
    for el in root.iter():
        if el.tag.split("}")[-1].lower() == "text" and (el.text or "").strip():
            out.append(el.text.strip())
    return out


def mail_text_from_row(primary_id: str, payload):
    """
    The text to scan for this notification, or None when it is not a mail
    notification. The payload is only parsed AFTER the app allow-list passes.
    """
    mail_app = is_mail_handler(primary_id)
    browser = is_browser_handler(primary_id)
    if not (mail_app or browser):
        return None
    texts = toast_texts(payload)
    if not texts:
        return None
    joined = "\n".join(texts)
    if browser and not mail_app:
        low = joined.lower()
        if not any(h in low for h in WEBMAIL_HOSTS):
            return None
        # drop the site-attribution line ("mail.google.com") from the scan text
        texts = [t for t in texts if t.strip().lower() not in WEBMAIL_HOSTS]
        joined = "\n".join(texts)
    return joined


# ── one pass ─────────────────────────────────────────────────────────
def scan_new_notifications(db_path: Path, last_id: int) -> int:
    """Scan toasts newer than last_id. Returns the new last_id."""
    STATS["last_check"] = datetime.now().isoformat()
    new_last = last_id
    for nid, primary_id, payload in fetch_new_rows(db_path, last_id):
        new_last = max(new_last, int(nid))
        text = mail_text_from_row(primary_id, payload)
        if text is None:
            STATS["ignored_other_apps"] += 1
            continue
        scan_email("", text, "mail_notification", stats=STATS, min_chars=MIN_CHARS)
    return new_last


def main(stop_event=None, db_path: Path = None):
    if db_path is None and sys.platform != "win32":
        raise RuntimeError("Notification watching is Windows-only.")
    db_path = db_path or default_db_path()
    if not db_path.exists():
        raise RuntimeError("Windows notification database not found "
                           "(notifications disabled, or unsupported Windows build).")
    try:
        last_id = max_notification_id(db_path)      # baseline: only NEW toasts
    except SchemaError as e:
        raise RuntimeError(str(e))
    except Exception as e:
        raise RuntimeError("Could not read the Windows notification database "
                           f"({type(e).__name__}).")

    log("=" * 60)
    log("  NeuraShield — Mail Notification Watcher (experimental)")
    log(f"  Poll interval: {POLL_SECONDS}s  |  mail apps/webmail only")
    log("=" * 60)

    while stop_event is None or not stop_event.is_set():
        try:
            last_id = scan_new_notifications(db_path, last_id)
            STATS["last_error"] = None
        except SchemaError as e:
            STATS["last_error"] = str(e)
            raise RuntimeError(str(e))
        except Exception as e:
            STATS["errors"] += 1
            STATS["last_error"] = f"{type(e).__name__}"
            log(f"[WARN] notification read failed ({type(e).__name__}) - continuing")
        if stop_event is not None:
            if stop_event.wait(POLL_SECONDS):
                break
        else:
            time.sleep(POLL_SECONDS)
    log("Notification watcher stopped.")


if __name__ == "__main__":
    main()

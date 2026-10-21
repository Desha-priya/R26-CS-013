# =============================================================
# NeuraShield — Layer 4 — Extra Email Source Manager
# File   : src/watchers/source_manager.py
#
# Starts/stops the credential-free and OAuth email intakes:
#     gmail_watcher         Gmail API (read-only OAuth)
#     folder_watcher        .eml drop-folder
#     notification_watcher  Windows mail toasts (experimental)
#
# It follows the SAME pattern as watcher_manager.py (stop via
# threading.Event, status polled from each module's STATS) and the same
# safety rule: a source failing to start is recorded as status, never raised.
# watcher_manager.get_watcher_status() merges get_status() below, so the
# existing /api/watchers/status endpoint reports these sources too.
# =============================================================

import importlib
import os
import sys
import threading
import traceback
from datetime import datetime

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.common import oauth_store, sources_config

_LOCK = threading.Lock()
_POLL_INTERVAL_SECONDS = 3

SOURCES = {
    "gmail_watcher": {
        "display_name": "Gmail Watcher (Google sign-in, read-only)",
        "module": "src.watchers.gmail_watcher",
        "flag": "gmail",
        "idle": "Not signed in",
    },
    "folder_watcher": {
        "display_name": "Email Drop-Folder Watcher",
        "module": "src.watchers.folder_watcher",
        "flag": "folder",
        "idle": "Not enabled",
    },
    "notification_watcher": {
        "display_name": "Mail Notification Watcher (experimental, Windows)",
        "module": "src.watchers.notification_watcher",
        "flag": "notifications",
        "idle": "Not enabled",
    },
}

_STATUS = {
    name: {"display_name": cfg["display_name"], "state": "disabled",
           "detail": cfg["idle"], "last_activity": None, "stats": {}}
    for name, cfg in SOURCES.items()
}
_threads = {}          # name -> (thread, stop_event)


def _set(name, **kw):
    with _LOCK:
        _STATUS[name].update(kw)


def get_status() -> dict:
    import copy
    with _LOCK:
        return copy.deepcopy(_STATUS)


def _alive(name) -> bool:
    t = _threads.get(name)
    return bool(t and t[0].is_alive())


def _runner(name, stop_event):
    try:
        mod = importlib.import_module(SOURCES[name]["module"])
    except Exception as e:
        _set(name, state="error", detail=f"Import failed: {type(e).__name__}: {e}")
        return
    _set(name, state="starting", detail="Starting...")
    try:
        mod.main(stop_event)
    except Exception as e:
        _set(name, state="error", detail=str(e) or type(e).__name__,
             last_activity=datetime.now().isoformat())
        traceback.print_exc()
    else:
        if stop_event.is_set():
            _set(name, state="disabled", detail="Stopped")


def _poller(name, stop_event):
    try:
        mod = importlib.import_module(SOURCES[name]["module"])
    except Exception:
        return
    while not stop_event.is_set():
        try:
            with _LOCK:
                cur = _STATUS[name]["state"]
            if cur in ("starting", "running") and mod.STATS.get("last_check"):
                err = mod.STATS.get("last_error")
                if err:
                    _set(name, state="error", detail=err,
                         last_activity=mod.STATS["last_check"], stats=dict(mod.STATS))
                else:
                    _set(name, state="running",
                         detail=(f"Active - {mod.STATS.get('checked', 0)} email(s) checked, "
                                 f"{mod.STATS.get('phishing_found', 0)} phishing found"),
                         last_activity=mod.STATS["last_check"], stats=dict(mod.STATS))
        except Exception:
            pass
        stop_event.wait(_POLL_INTERVAL_SECONDS)


def start(name: str, persist: bool = True) -> dict:
    """Start one source. Returns {"ok": bool, "detail": str}; never raises."""
    if name not in SOURCES:
        return {"ok": False, "detail": f"Unknown source '{name}'."}
    try:
        if _alive(name):
            return {"ok": False, "detail": "Already running."}

        if name == "gmail_watcher":
            s = oauth_store.gmail_summary()
            if not s["connected"]:
                return {"ok": False, "detail":
                        "Not signed in to Google yet. Use 'Sign in with Google'."}

        # forget stale counters/timestamps from a previous run
        mod = importlib.import_module(SOURCES[name]["module"])
        mod.STATS["last_check"] = None
        mod.STATS["last_error"] = None

        ev = threading.Event()
        t = threading.Thread(target=_runner, args=(name, ev), daemon=True,
                             name=f"neurashield-{name}")
        p = threading.Thread(target=_poller, args=(name, ev), daemon=True,
                             name=f"neurashield-{name}-poller")
        _threads[name] = (t, ev)
        _set(name, state="starting", detail="Starting...", last_activity=None)
        if persist:
            sources_config.set_enabled(SOURCES[name]["flag"], True)
        t.start()
        p.start()
        return {"ok": True, "detail": "Starting."}
    except Exception as e:
        _set(name, state="error", detail=f"{type(e).__name__}: {e}")
        return {"ok": False, "detail": f"{type(e).__name__}: {e}"}


def stop(name: str, persist: bool = True) -> dict:
    if name not in SOURCES:
        return {"ok": False, "detail": f"Unknown source '{name}'."}
    if persist:
        try:
            sources_config.set_enabled(SOURCES[name]["flag"], False)
        except Exception:
            pass
    if not _alive(name):
        _set(name, state="disabled", detail=SOURCES[name]["idle"])
        return {"ok": True, "detail": "Was not running."}
    t, ev = _threads[name]
    ev.set()
    t.join(timeout=10)
    clean = not t.is_alive()
    _set(name, state="disabled",
         detail="Stopped" if clean else "Stop requested (still shutting down)")
    return {"ok": clean, "detail": "Stopped." if clean else "Stop requested."}


def autostart() -> None:
    """Called once at API boot. Starts only sources the USER previously
    turned on. Never raises."""
    for name, cfg in SOURCES.items():
        try:
            if sources_config.is_enabled(cfg["flag"]):
                r = start(name, persist=False)
                if not r["ok"]:
                    _set(name, state="error", detail=r["detail"])
        except Exception as e:
            _set(name, state="error", detail=f"Startup failed: {type(e).__name__}")

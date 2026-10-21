# =============================================================
# NeuraShield — Layer 4 — Watcher Manager
# File   : src/watchers/watcher_manager.py
#
# PURPOSE
#   Controls the automatic watchers (mail, call monitor):
#     - starts whichever ones are already configured when the API boots
#     - lets the DASHBOARD start/stop them on demand afterward, so you
#       don't need a terminal or environment variables at all — just
#       type a mailbox into a form and flip a switch
#     - tracks live status for the dashboard to display
#
# SAFETY PRINCIPLE (unchanged from the previous version)
#   A watcher failing to start (missing credentials, missing OS
#   dependency, wrong platform) must NEVER crash or block app.py's
#   own startup or any dashboard request. Every start attempt is
#   wrapped in try/except; failures are recorded as status, not
#   raised past this module's public functions.
#
# HOW ON-DEMAND STOP WORKS
#   mail_watcher.main() and call_monitor.main() both now accept an
#   optional threading.Event. This module creates a fresh Event each
#   time a watcher is started, hands it to the watcher's main loop,
#   and calling stop_*() simply sets that Event — the watcher's own
#   loop notices within a few seconds (it waits ON the event, not a
#   plain sleep, so shutdown is prompt, not delayed to the next poll).
# =============================================================

import os
import sys
import threading
import traceback
from datetime import datetime

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.common import credential_store

_LOCK = threading.Lock()
_POLL_INTERVAL_SECONDS = 3

_STATUS = {
    "mail_watcher": {
        "display_name": "IMAP Mail Watcher",
        "state": "disabled",
        "detail": "Not configured",
        "last_activity": None,
        "stats": {},
    },
    "call_monitor": {
        "display_name": "Real-Time Call Monitor (audio + video)",
        "state": "disabled",
        "detail": "Not enabled",
        "last_activity": None,
        "stats": {},
    },
}

# Live handles to the running watcher (thread + its stop event), so we
# know whether one is already running and can stop it on request.
_mail_thread = None
_mail_stop_event = None
_call_thread = None
_call_stop_event = None


def _set(name, **kwargs):
    with _LOCK:
        _STATUS[name].update(kwargs)


def get_watcher_status() -> dict:
    with _LOCK:
        import copy
        out = copy.deepcopy(_STATUS)
    # Extra email sources (Gmail sign-in, drop-folder, notifications) report
    # through the same endpoint. Purely additive: if that module can't load
    # for any reason the original two watchers still report normally.
    try:
        from src.watchers import source_manager
        out.update(source_manager.get_status())
    except Exception:
        pass
    return out


def _is_alive(thread):
    return thread is not None and thread.is_alive()


# ─────────────────────────────────────────────────────────────
# Mail watcher
# ─────────────────────────────────────────────────────────────

def _mail_watcher_runner(stop_event):
    try:
        from src.watchers import mail_watcher as mw
    except Exception as e:
        _set("mail_watcher", state="error", detail=f"Import failed: {e}")
        return
    _set("mail_watcher", state="starting", detail="Connecting to mailbox...")
    try:
        mw.main(stop_event)
    except Exception as e:
        _set("mail_watcher", state="error",
            detail=f"{e}", last_activity=datetime.now().isoformat())
        traceback.print_exc()
    else:
        # main() returned because stop_event was set (clean stop),
        # not because of an error.
        if stop_event.is_set():
            _set("mail_watcher", state="disabled", detail="Stopped")


def _mail_watcher_poller(stop_event):
    try:
        from src.watchers import mail_watcher as mw
    except Exception:
        return
    while not stop_event.is_set():
        try:
            with _LOCK:
                current_state = _STATUS["mail_watcher"]["state"]
            if current_state in ("starting", "running") and mw.STATS.get("last_check"):
                new_state = "error" if mw.STATS.get("last_error") else "running"
                detail = (mw.STATS["last_error"] if mw.STATS.get("last_error")
                         else f"Connected — {mw.STATS['checked']} email(s) checked, "
                              f"{mw.STATS['phishing_found']} phishing found")
                _set("mail_watcher", state=new_state, detail=detail,
                    last_activity=mw.STATS["last_check"], stats=dict(mw.STATS))
        except Exception:
            pass
        stop_event.wait(_POLL_INTERVAL_SECONDS)


def start_mail_watcher(host: str = None, user: str = None,
                       password: str = None, remember: bool = True) -> dict:
    """
    Start the mail watcher. If host/user/password are given, they're
    used for this run and (if remember=True) saved encrypted for
    future starts/restarts. If omitted, whatever's already configured
    (environment variables, or previously-saved dashboard credentials)
    is used.
    Returns {"ok": bool, "detail": str} — never raises.
    """
    global _mail_thread, _mail_stop_event

    if _is_alive(_mail_thread):
        return {"ok": False, "detail": "Mail watcher is already running."}

    if host and user and password:
        if remember:
            try:
                credential_store.save_mail_credentials(host, user, password)
            except Exception as e:
                return {"ok": False, "detail": f"Could not save credentials: {e}"}
        else:
            # one-off run without persisting — set env vars for this
            # process only (still picked up by mail_watcher's resolver)
            os.environ["NEURASHIELD_MAIL_HOST"] = host
            os.environ["NEURASHIELD_MAIL_USER"] = user
            os.environ["NEURASHIELD_MAIL_PASSWORD"] = password
    else:
        # No credentials passed this call — must already be configured
        # somewhere (env vars or previously-saved dashboard config).
        env_ok = all(os.environ.get(k) for k in
                    ("NEURASHIELD_MAIL_HOST", "NEURASHIELD_MAIL_USER",
                     "NEURASHIELD_MAIL_PASSWORD"))
        if not env_ok and not credential_store.has_mail_credentials():
            return {"ok": False, "detail":
                   "No mailbox configured yet. Provide host/user/password."}

    _mail_stop_event = threading.Event()
    _mail_thread = threading.Thread(target=_mail_watcher_runner,
                                    args=(_mail_stop_event,), daemon=True,
                                    name="neurashield-mail-watcher")
    poller = threading.Thread(target=_mail_watcher_poller,
                              args=(_mail_stop_event,), daemon=True,
                              name="neurashield-mail-watcher-poller")
    _mail_thread.start()
    poller.start()
    return {"ok": True, "detail": "Mail watcher starting."}


def stop_mail_watcher() -> dict:
    global _mail_thread, _mail_stop_event
    if not _is_alive(_mail_thread):
        _set("mail_watcher", state="disabled", detail="Not running")
        return {"ok": True, "detail": "Mail watcher was not running."}
    _mail_stop_event.set()
    _mail_thread.join(timeout=10)
    stopped_cleanly = not _mail_thread.is_alive()
    _set("mail_watcher", state="disabled",
        detail="Stopped" if stopped_cleanly else "Stop requested (still shutting down)")
    return {"ok": stopped_cleanly, "detail": "Mail watcher stopped."
           if stopped_cleanly else "Stop requested, still shutting down."}


# ─────────────────────────────────────────────────────────────
# Call monitor
# ─────────────────────────────────────────────────────────────

def _call_monitor_runner(stop_event):
    try:
        from src.watchers import call_monitor as cm
    except Exception as e:
        _set("call_monitor", state="error", detail=f"Import failed: {e}")
        return
    _set("call_monitor", state="starting", detail="Initializing audio/video capture...")
    try:
        cm.main(stop_event)
    except Exception as e:
        _set("call_monitor", state="error",
            detail=f"{e}", last_activity=datetime.now().isoformat())
        traceback.print_exc()
    else:
        if stop_event.is_set():
            _set("call_monitor", state="disabled", detail="Stopped")


def _call_monitor_poller(stop_event):
    try:
        from src.watchers import call_monitor as cm
    except Exception:
        return
    while not stop_event.is_set():
        try:
            with _LOCK:
                current_state = _STATUS["call_monitor"]["state"]
            if current_state in ("starting", "running"):
                if cm.STATS.get("last_error"):
                    _set("call_monitor", state="error", detail=cm.STATS["last_error"],
                        last_activity=cm.STATS.get("last_check"), stats=dict(cm.STATS))
                elif cm.STATS.get("last_check"):
                    _set("call_monitor", state="running",
                        detail=f"Monitoring — {cm.STATS['fused_checks']} checks, "
                               f"{cm.STATS['high_risk_calls']} high-risk detected",
                        last_activity=cm.STATS["last_check"], stats=dict(cm.STATS))
        except Exception:
            pass
        stop_event.wait(_POLL_INTERVAL_SECONDS)


def start_call_monitor() -> dict:
    """No credentials needed — this captures local system audio/screen,
    it's a pure on/off toggle. Returns {"ok": bool, "detail": str}."""
    global _call_thread, _call_stop_event

    if _is_alive(_call_thread):
        return {"ok": False, "detail": "Call monitor is already running."}

    _call_stop_event = threading.Event()
    _call_thread = threading.Thread(target=_call_monitor_runner,
                                    args=(_call_stop_event,), daemon=True,
                                    name="neurashield-call-monitor")
    poller = threading.Thread(target=_call_monitor_poller,
                              args=(_call_stop_event,), daemon=True,
                              name="neurashield-call-monitor-poller")
    _call_thread.start()
    poller.start()
    return {"ok": True, "detail": "Call monitor starting."}


def stop_call_monitor() -> dict:
    global _call_thread, _call_stop_event
    if not _is_alive(_call_thread):
        _set("call_monitor", state="disabled", detail="Not running")
        return {"ok": True, "detail": "Call monitor was not running."}
    _call_stop_event.set()
    _call_thread.join(timeout=10)
    stopped_cleanly = not _call_thread.is_alive()
    _set("call_monitor", state="disabled",
        detail="Stopped" if stopped_cleanly else "Stop requested (still shutting down)")
    return {"ok": stopped_cleanly, "detail": "Call monitor stopped."
           if stopped_cleanly else "Stop requested, still shutting down."}


# ─────────────────────────────────────────────────────────────
# Boot-time auto-start (unchanged behavior from before: starts
# automatically ONLY if already configured, never blocks app.py)
# ─────────────────────────────────────────────────────────────

def start_all_watchers():
    """Called once at API startup. Never raises."""
    try:
        env_ok = all(os.environ.get(k) for k in
                    ("NEURASHIELD_MAIL_HOST", "NEURASHIELD_MAIL_USER",
                     "NEURASHIELD_MAIL_PASSWORD"))
        if env_ok or credential_store.has_mail_credentials():
            result = start_mail_watcher()
            if not result["ok"]:
                _set("mail_watcher", state="error", detail=result["detail"])
        else:
            summary = credential_store.mail_credentials_summary()
            _set("mail_watcher", state="disabled",
                detail="Not configured — set up a mailbox in Automatic Monitors, "
                       "or set NEURASHIELD_MAIL_HOST/USER/PASSWORD")
    except Exception as e:
        _set("mail_watcher", state="error", detail=f"Startup failed: {e}")

    try:
        if os.environ.get("NEURASHIELD_ENABLE_CALL_MONITOR", "0") == "1":
            result = start_call_monitor()
            if not result["ok"]:
                _set("call_monitor", state="error", detail=result["detail"])
        else:
            _set("call_monitor", state="disabled",
                detail="Not enabled — opt-in from Automatic Monitors "
                       "(captures live system audio and screen content)")
    except Exception as e:
        _set("call_monitor", state="error", detail=f"Startup failed: {e}")

    # Restart any extra email source the user previously turned on
    # (Gmail sign-in / drop-folder / notifications). Never raises.
    try:
        from src.watchers import source_manager
        source_manager.autostart()
    except Exception as e:
        print(f"[WARN] extra email sources could not auto-start: {e}")
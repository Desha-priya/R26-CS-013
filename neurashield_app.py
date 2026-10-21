"""
neurashield_app.py  —  the NeuraShield desktop application (the exe entry point)
--------------------------------------------------------------------------------
This is what the installed shortcut / service-tray launches. It:

  1. ensures the console+supervisor is running (starts it in-process if needed);
  2. opens a native desktop window showing the console, signed in automatically
     via a one-time app token — so opening the app needs NO login;
  3. a plain browser visiting http://127.0.0.1:9000 still gets the login page,
     because only this app can read the protected app-token file.

If pywebview is not available, it falls back to opening the default browser
(which will show the login page — still fully usable).

    python neurashield_app.py
"""

import os
import sys
import time
import threading
import socket
from pathlib import Path

def _install_root() -> Path:
    """Source run: the folder holding this file. Packaged run: the installed layout
    if it exists, otherwise a per-user folder (see gateway/install_root.py)."""
    if getattr(sys, "frozen", False):
        sys.path.insert(0, str(Path(__file__).resolve().parent / "gateway"))
        import install_root
        return install_root.resolve()
    return Path(__file__).resolve().parent


ROOT = _install_root()
os.environ.setdefault("NEURASHIELD_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "gateway"))

# A windowed (no-console) exe has no stdout/stderr; uvicorn's logger and every
# print() would raise. Give the process a log file instead.
if sys.stdout is None or sys.stderr is None:
    try:
        _logdir = ROOT / "data" / "logs"
        _logdir.mkdir(parents=True, exist_ok=True)
        _lf = open(_logdir / "app.log", "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = _lf
        if sys.stderr is None:
            sys.stderr = _lf
    except OSError:
        pass

CONSOLE_PORT = 9000
CONSOLE_URL = f"http://127.0.0.1:{CONSOLE_PORT}"
APP_TOKEN_FILE = ROOT / "data" / "console" / "app_token"


_MUTEX = []


def _single_instance() -> bool:
    """True if this is the first NeuraShield app process for this user. Auto-start
    at sign-in and a manual launch can overlap; the second one must not create a
    second tray icon (it just opens a window on the console that is already up)."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = ctypes.c_void_p
        h = k32.CreateMutexW(None, False, "Local\\NeuraShield.App")
        _MUTEX.append(h)                      # keep the handle for the process lifetime
        return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS
    except Exception:
        return True


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_console_in_thread():
    """Start the console + supervisor inside this process (desktop mode)."""
    import console_app
    # product.json may set a different port
    try:
        import json
        prod = json.loads((ROOT / "config" / "product.json").read_text())
        globals()["CONSOLE_PORT"] = int(prod.get("console", {}).get("port", 9000))
    except Exception:
        pass
    t = threading.Thread(target=console_app.main, daemon=True)
    t.start()


def _read_app_token(timeout=15) -> str:
    """Wait for the console to write its one-time app token, then read it."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            if APP_TOKEN_FILE.exists():
                tok = APP_TOKEN_FILE.read_text().strip()
                if tok:
                    return tok
        except OSError:
            pass
        time.sleep(0.3)
    return ""


def _app_login_url() -> str:
    """URL that signs this window in without a password: the app token (readable
    only by the current user) is traded for a one-time ticket that expires in a
    minute, so no long-lived credential ever lands in a URL or browser history.
    Falls back to the plain URL (which shows the login page)."""
    from urllib import request as _rq
    tok = _read_app_token()
    if tok:
        try:
            import json as _json
            req = _rq.Request(CONSOLE_URL + "/api/app/ticket", method="POST",
                              headers={"X-App-Token": tok})
            with _rq.urlopen(req, timeout=5) as r:
                ticket = _json.loads(r.read().decode()).get("ticket")
            if ticket:
                return f"{CONSOLE_URL}/?app_token={ticket}"
        except Exception as e:
            print(f"[app] could not get a sign-in ticket ({e}); showing login page")
    return CONSOLE_URL + "/"


def _wait_console(timeout=20) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if _port_open(CONSOLE_PORT):
            return True
        time.sleep(0.3)
    return False


def main():
    # Service mode: run the console+supervisor only, no desktop window.
    # The Windows service launches with NEURASHIELD_SERVICE=1 because a service
    # runs in session 0 and cannot show a GUI; the per-user shortcut (no flag)
    # opens the window and connects to this already-running console.
    if os.environ.get("NEURASHIELD_SERVICE") == "1":
        import console_app
        console_app.main()   # blocks, serving the console + supervisor
        return

    first_instance = _single_instance()
    if not first_instance and "--background" in sys.argv:
        return                                   # already running: nothing to do

    # 1. make sure the console is up (start it if the service isn't running)
    if not _port_open(CONSOLE_PORT):
        _start_console_in_thread()
    if not _wait_console():
        print("[app] console did not start in time")
    # 2. trusted auto-login URL for the window (one-time ticket, never the token)
    start_url = _app_login_url()

    # 2b. system-tray presence: live status colour + quick actions (first instance only).
    if first_instance:
        _start_tray()

    # Background mode (auto-start at sign-in): console + tray only, no window.
    # The tray's "Open NeuraShield" opens the window/browser on demand and its
    # "Quit" stops everything.
    if "--background" in sys.argv:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return

    # 3. open a native window if we can; otherwise the browser
    try:
        import webview  # pywebview
        webview.create_window(
            "NeuraShield", start_url,
            width=1280, height=860, min_size=(1000, 680),
            confirm_close=False,
        )
        webview.start()  # blocks until the window is closed
    except Exception as e:
        print(f"[app] native window unavailable ({e}); opening browser.")
        import webbrowser
        webbrowser.open(start_url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


def _start_tray():
    """Run the tray icon in a background thread (best-effort)."""
    try:
        sys.path.insert(0, str(ROOT / "gateway"))
        from tray import Tray
        import secrets_store  # noqa
        from urllib import request as _rq, error as _er

        api_key = None
        try:
            import json as _json
            ak = ROOT / "data" / "console" / "app_token"  # not the key; use secret store
            import secrets_store as _ss
            api_key = _ss.get_api_key()
        except Exception:
            api_key = None

        def api_get(path):
            url = CONSOLE_URL + path
            try:
                # use the console's own status via platform; read through console
                req = _rq.Request("http://127.0.0.1:8000" + path,
                                  headers={"X-API-Key": api_key or ""})
                with _rq.urlopen(req, timeout=3) as r:
                    import json as _j
                    return _j.loads(r.read().decode()), r.status
            except Exception:
                return None, 0

        def on_open():
            import webbrowser
            webbrowser.open(_app_login_url())

        def on_quit():
            # Ask the console to shut the whole platform down, then exit. The
            # request carries the per-run app token (only this user can read it).
            try:
                tok = _read_app_token(timeout=1)
                _rq.urlopen(_rq.Request(CONSOLE_URL + "/api/app/quit", method="POST",
                                        headers={"X-App-Token": tok}), timeout=3)
            except Exception:
                pass
            import os as _os
            _os._exit(0)

        tray = Tray(CONSOLE_URL, api_get, on_open, on_quit)
        threading.Thread(target=tray.start, daemon=True).start()
    except Exception as e:
        print(f"[app] tray unavailable ({e})")


try:  # makes the build include every package the bundled layers import
    import _bundled_imports  # noqa: F401
except ImportError:
    pass

if __name__ == "__main__":
    # Needed for frozen Windows builds (multiprocessing children re-enter the exe).
    import multiprocessing
    multiprocessing.freeze_support()
    # The ONE exe is also every layer: the supervisor starts layers as
    # `NeuraShield.exe --run-layer <id>`; handle that before any app code runs.
    import layer_host
    if layer_host.dispatch(sys.argv[1:]):
        sys.exit(0)
    main()

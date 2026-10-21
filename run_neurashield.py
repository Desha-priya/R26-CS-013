"""
run_neurashield.py  —  single entry point for the whole product
----------------------------------------------------------------
This is the one command that brings NeuraShield up on this computer. It starts
the console (which, in turn, resolves the shared secrets, starts every enabled
layer through the supervisor, and runs the watchdog), then opens the operator
console in the browser.

    python run_neurashield.py            # start everything + open the console
    python run_neurashield.py --no-open  # start everything, don't open a browser

On Windows the installer registers this (frozen) as a service so it starts at
boot; the per-user tray agent opens the console window. For a source/dev run,
this script is enough.
"""

import os
import sys
import time
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NEURASHIELD_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "gateway"))


def _open_browser_when_ready(url: str):
    import urllib.request
    for _ in range(40):
        try:
            urllib.request.urlopen(url, timeout=1)
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(0.5)


def main():
    no_open = "--no-open" in sys.argv
    import json
    product = json.loads((ROOT / "config" / "product.json").read_text())
    host = product.get("console", {}).get("host", "127.0.0.1")
    port = int(product.get("console", {}).get("port", 9000))
    url = f"http://{host}:{port}/"

    print("=" * 56)
    print("  NeuraShield — starting")
    print("=" * 56)
    py_all = ROOT / "runtimes" / "all" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not py_all.exists():
        print("  !! The product environment is missing (runtimes/all).")
        print("  !! Layers need their own packages (flask, tensorflow, ...) and will")
        print("  !! NOT start on this Python. Create it once with:")
        print("  !!     python scripts/bootstrap_envs.py")
        print("-" * 56)
    if not no_open:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    # console_app.main() resolves secrets, starts the supervisor + watchdog
    # (NEURASHIELD_CONSOLE_SUPERVISE defaults to 1) and runs the web server.
    import console_app
    console_app.main()


if __name__ == "__main__":
    main()

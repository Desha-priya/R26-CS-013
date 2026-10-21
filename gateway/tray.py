"""
tray.py  —  NeuraShield system-tray presence
--------------------------------------------
A tray icon that shows live platform status (green / amber / red), with a
right-click menu for quick actions, and that keeps the platform present even
when the main window is closed.

Uses pystray + Pillow if available; degrades to a no-op otherwise (the app and
console still run, just without a tray icon). Runs in its own thread.
"""

import threading
import time
import webbrowser
from urllib import request as _request


def _make_image(color):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # shield silhouette
    d.polygon([(32, 6), (56, 16), (56, 34), (32, 60), (8, 34), (8, 16)],
              fill=(22, 28, 34, 255))
    d.ellipse([24, 24, 40, 40], fill=color)
    return img


_COLORS = {
    "secure":   (73, 209, 122, 255),   # green
    "elevated": (240, 180, 63, 255),    # amber
    "alert":    (255, 93, 108, 255),    # red
    "standby":  (139, 149, 143, 255),   # grey
}


class Tray:
    def __init__(self, console_url, api_get, on_open, on_quit):
        """
        console_url : str
        api_get(path) -> (json, code)   # to read platform status for colour
        on_open()  : open the console window
        on_quit()  : stop the platform (full shutdown)
        """
        self.console_url = console_url
        self._get = api_get
        self._on_open = on_open
        self._on_quit = on_quit
        self._icon = None
        self._stop = threading.Event()
        self._state = "standby"

    def _status_color(self):
        try:
            data, code = self._get("/api/platform/status")
            if code != 200 or not isinstance(data, dict):
                return "standby"
            lvl = (data.get("platform_threat_level") or "low").lower()
            if lvl in ("high", "critical"):
                return "alert"
            if lvl == "medium":
                return "elevated"
            return "secure"
        except Exception:
            return "standby"

    def _poll(self):
        while not self._stop.wait(6.0):
            try:
                s = self._status_color()
                if s != self._state and self._icon is not None:
                    self._state = s
                    self._icon.icon = _make_image(_COLORS[s])
                    self._icon.title = f"NeuraShield — {s.capitalize()}"
            except Exception:
                pass

    def start(self):
        try:
            import pystray
            from pystray import MenuItem as Item, Menu
        except Exception as e:
            print(f"[tray] unavailable ({e}); running without a tray icon")
            return

        def _open(icon, item):
            try:
                self._on_open()
            except Exception:
                webbrowser.open(self.console_url)

        def _overview(icon, item):
            webbrowser.open(self.console_url)

        def _incidents(icon, item):
            webbrowser.open(self.console_url + "/#incidents")

        def _quit(icon, item):
            self._stop.set()
            try:
                icon.stop()
            finally:
                self._on_quit()

        menu = Menu(
            Item("Open NeuraShield", _open, default=True),
            Item("Overview", _overview),
            Item("Incidents", _incidents),
            Menu.SEPARATOR,
            Item("Quit (stop protection)", _quit),
        )
        self._icon = pystray.Icon("NeuraShield", _make_image(_COLORS["standby"]),
                                  "NeuraShield", menu)
        threading.Thread(target=self._poll, daemon=True).start()
        # pystray must run on the main thread on some platforms; the caller runs
        # this in its own thread and that is fine on Windows.
        self._icon.run()

    def stop(self):
        self._stop.set()
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass

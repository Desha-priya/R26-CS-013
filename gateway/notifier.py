"""
notifier.py  —  NeuraShield desktop notifications
-------------------------------------------------
One central notifier for the whole platform. It watches the platform for new
alerts and Zero-Trust state changes and raises native Windows toast
notifications (Action Center), so the user is told what the system is doing
without having the console open:

  * Zero-Trust enters MONITOR / step-up (REAUTHENTICATE) — "verify to continue"
  * Ransomware containment / rollback
  * Content layer: deepfake call, phishing, spoofed audio
  * Network: anomaly / exfiltration

Design:
  * Runs as a background thread inside the console process (which is always up).
  * Polls /api/platform/status every few seconds, diffs against what it already
    notified, and fires a toast per NEW item (deduplicated).
  * Respects the user's settings (master switch, per-layer, minimum severity,
    step-up prompts) read live from product.json via a callback.
  * Clicking a toast opens the console (best-effort; protocol link).

Backends, in order of preference:
  1. windows-toasts / win11toast (rich, clickable) if installed
  2. plyer.notification (cross-platform fallback)
  3. PowerShell BurntToast / raw toast XML (no extra dependency on Windows)
  4. no-op elsewhere (logged)
"""

import sys
import time
import threading
import subprocess


_SEV_RANK = {"low": 0, "info": 0, "medium": 1, "high": 2, "critical": 3}


class Notifier:
    def __init__(self, platform_get, settings_cb, console_url="http://127.0.0.1:9000"):
        """
        platform_get(path) -> (json, status_code)   # same helper the console uses
        settings_cb() -> dict  notification settings (live from product.json)
        """
        self._get = platform_get
        self._settings = settings_cb
        self._console_url = console_url
        self._seen = set()          # dedupe keys already notified
        self._last_zta = None       # last ZTA decision we announced
        self._stop = threading.Event()
        self._thread = None
        self._backend = self._pick_backend()
        # When True the console's AlertCollector owns alert notifications (richer
        # text: what, where, when) and this class only announces Zero-Trust changes.
        self.external_alerts = False

    # ---- lifecycle ----------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[notifier] started (backend: {self._backend})")

    def stop(self):
        self._stop.set()

    # ---- settings -----------------------------------------------------------
    def _cfg(self):
        s = self._settings() or {}
        return {
            "enabled": s.get("enabled", True),
            "min_severity": s.get("min_severity", "medium"),
            "stepup": s.get("stepup", True),
            "layers": s.get("layers", {"zta": True, "network": True,
                                       "ransomware": True, "content": True}),
        }

    def _allowed(self, layer_key, severity):
        c = self._cfg()
        if not c["enabled"]:
            return False
        if not c["layers"].get(layer_key, True):
            return False
        return _SEV_RANK.get(str(severity).lower(), 1) >= _SEV_RANK.get(c["min_severity"], 1)

    # ---- the watch loop -----------------------------------------------------
    def _loop(self):
        while not self._stop.wait(4.0):
            try:
                data, code = self._get("/api/platform/status")
                if code != 200 or not isinstance(data, dict):
                    continue
                self._check_zta(data)
                if not self.external_alerts:
                    self._check_alerts(data)
            except Exception:
                pass

    _LAYER_KEY = {1: "network", 2: "ransomware", 3: "zta", 4: "content"}
    _LAYER_NAME = {"network": "Network Guardian", "ransomware": "Ransomware Killer",
                   "zta": "Zero-Trust", "content": "Content Threat Detection"}

    def _check_zta(self, data):
        agent = data.get("latest_agent_score") or {}
        decision = (agent.get("decision") or "").upper()
        if not decision or decision == self._last_zta:
            return
        prev = self._last_zta
        self._last_zta = decision
        if prev is None:
            return  # don't notify the very first reading
        c = self._cfg()
        if not c["enabled"] or not c["layers"].get("zta", True):
            return
        pct = agent.get("risk_percent")
        if decision in ("REAUTHENTICATE", "STEP_UP", "STEP-UP"):
            if c["stepup"]:
                self._toast("Zero-Trust — verification needed",
                            f"Your session risk rose to {pct}%. Open NeuraShield to confirm it's you "
                            f"(face or PIN).", "high", action="open")
        elif decision == "MONITOR":
            self._toast("Zero-Trust — monitoring",
                        "Continuous verification is active for this session.", "low")
        elif decision in ("ALLOW", "AUTHENTICATED"):
            self._toast("Zero-Trust — verified",
                        "Your identity is confirmed. Monitoring continues quietly.", "low")

    def _check_alerts(self, data):
        active = data.get("active_alerts") or {}
        for source, a in active.items():
            layer_num = a.get("layer")
            key = self._LAYER_KEY.get(layer_num, "zta")
            ts = a.get("timestamp", "")
            dedupe = f"{source}|{ts}|{a.get('message','')[:40]}"
            if dedupe in self._seen:
                continue
            self._seen.add(dedupe)
            sev = a.get("severity", "medium")
            if not self._allowed(key, sev):
                continue
            title = f"{self._LAYER_NAME.get(key, 'NeuraShield')} — {sev.upper()}"
            self._toast(title, a.get("message", "A security event was detected."),
                        sev, action="open")
        # keep the dedupe set from growing forever
        if len(self._seen) > 400:
            self._seen = set(list(self._seen)[-200:])

    def notify_record(self, rec):
        """Toast for one stored alert record (see alert_store): says WHAT was
        detected, how severe, and WHEN."""
        if not self._allowed(rec.get("layer", "zta"), rec.get("severity", "medium")):
            return
        from alert_store import notification_text
        title, body = notification_text(rec)
        self._toast(title, body, rec.get("severity", "medium"), action="open")

    # ---- backends -----------------------------------------------------------
    def _pick_backend(self):
        if sys.platform != "win32":
            try:
                import plyer  # noqa
                return "plyer"
            except Exception:
                return "none"
        for name in ("win11toast", "windows_toasts", "plyer"):
            try:
                __import__(name)
                return name
            except Exception:
                continue
        return "powershell"  # always available on Windows 10/11

    def _toast(self, title, message, severity="medium", action=None):
        try:
            if self._backend == "win11toast":
                from win11toast import toast
                kw = {"app_id": "NeuraShield"}
                if action == "open":
                    kw["on_click"] = self._console_url
                threading.Thread(target=lambda: toast(title, message, **kw), daemon=True).start()
            elif self._backend == "windows_toasts":
                from windows_toasts import Toast, WindowsToaster
                t = Toast(); t.text_fields = [title, message]
                WindowsToaster("NeuraShield").show_toast(t)
            elif self._backend == "plyer":
                from plyer import notification
                notification.notify(title=title, message=message, app_name="NeuraShield", timeout=8)
            elif self._backend == "powershell":
                self._toast_powershell(title, message)
            else:
                print(f"[notifier] (no backend) {title}: {message}")
        except Exception as e:
            print(f"[notifier] toast failed ({self._backend}): {e}")

    def _toast_powershell(self, title, message):
        # Uses the built-in Windows Runtime toast API via PowerShell — no extra
        # Python package required. Best-effort; silently ignores failure.
        def esc(s):
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        ps = (
            "$ErrorActionPreference='SilentlyContinue';"
            "[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime]>$null;"
            "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
            "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
            f"$t.GetElementsByTagName('text').Item(0).AppendChild($t.CreateTextNode('{esc(title)}'))>$null;"
            f"$t.GetElementsByTagName('text').Item(1).AppendChild($t.CreateTextNode('{esc(message)}'))>$null;"
            "$n=[Windows.UI.Notifications.ToastNotification]::new($t);"
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('NeuraShield').Show($n);"
        )
        try:
            subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            pass

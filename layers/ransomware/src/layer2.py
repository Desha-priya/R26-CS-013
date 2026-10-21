"""
layer2.py
-----------
The ONE class the main NeuraShield platform (or the combined exe) needs to
start, stop and query Layer 2 - the Ransomware Killer.

    from src.layer2 import Layer2

    layer = Layer2()                   # reads config + environment variables
    layer.start()                      # starts in STANDBY: nothing touches user files yet
    layer.grant_consent()              # (or the user clicks the dashboard button) -> protection starts
    layer.status()                     # dict: running, incidents, backups, outbox
    layer.health()                     # dict: every component alive? ok=True/False
    layer.analyze_memory_features(d)   # run the trained ML models on a feature dict
    layer.stop()

Everything the layer needs is configured through environment variables
(see INTEGRATION.md): NEURASHIELD_HMAC_SECRET, NEURASHIELD_API_KEY,
NEURASHIELD_PLATFORM_URL, NEURASHIELD_HOME, NEURASHIELD_CONFIG.

enable_dashboard=True also starts the local, read-only dashboard
(127.0.0.1:5050 by default).
"""

import os
import threading

from main import RansomwareKillerService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Layer2:
    def __init__(self, config_path=None, enable_dashboard: bool = False,
                 dashboard_host: str = None, dashboard_port: int = None):
        self.service = RansomwareKillerService(config_path=config_path)
        self.enable_dashboard = enable_dashboard
        self._dashboard_host = dashboard_host
        self._dashboard_port = dashboard_port
        self._dashboard_server = None
        self._dashboard_thread = None
        self._detector = None
        self._detector_lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle

    def start(self):
        # Dashboard first: it must be reachable while the (possibly long)
        # first backup is running, and in standby before any consent.
        if self.enable_dashboard:
            self._start_dashboard()
        self.service.start()

    # -------------------------------------------------------------- consent

    def grant_consent(self, by: str = "api call"):
        """Agrees to protect (watch + back up) the configured folders. Saved:
        later starts resume protection without asking again."""
        self.service.grant_consent(by=by)

    def revoke_consent(self, by: str = "api call"):
        """Stops watching and backing up. Existing backups stay on disk."""
        self.service.revoke_consent(by=by)

    @property
    def state(self) -> str:
        """standby | baseline | armed | incident"""
        return self.service.state

    def stop(self):
        if self._dashboard_server is not None:
            try:
                self._dashboard_server.close()
            except Exception as e:
                logger.warning(f"[layer2] Dashboard did not close cleanly: {e}")
        self.service.stop()

    def _start_dashboard(self):
        """Starts the local dashboard in a background thread. A failure is logged
        and printed clearly, and never stops the protection service."""
        try:
            from src.dashboard.app import create_server, DEFAULT_HOST, DEFAULT_PORT
            host = self._dashboard_host or os.environ.get("NEURASHIELD_DASHBOARD_HOST", DEFAULT_HOST)
            port = int(self._dashboard_port or os.environ.get("NEURASHIELD_DASHBOARD_PORT", DEFAULT_PORT))
            self._dashboard_server = create_server(host, port, service=self.service)
        except Exception as e:
            logger.error(f"[layer2] Dashboard could NOT start: {e}", exc_info=True)
            print(f"\n[layer2] Dashboard could NOT start: {e}\n", flush=True)
            return

        def run():
            try:
                self._dashboard_server.run()
            except Exception as e:
                logger.error(f"[layer2] Dashboard stopped with an error: {e}", exc_info=True)

        self._dashboard_thread = threading.Thread(target=run, daemon=True, name="dashboard")
        self._dashboard_thread.start()
        url = f"http://{host}:{port}"
        logger.info(f"[layer2] Dashboard listening on {url}")
        print(f"\n[layer2] Dashboard: {url}   (use exactly this address; Ctrl+C to stop)\n", flush=True)

    # --------------------------------------------------------------- status

    def status(self) -> dict:
        info = self.service.status()
        info["dashboard"] = bool(self._dashboard_thread and self._dashboard_thread.is_alive())
        return info

    def health(self) -> dict:
        """Is every part of the layer alive? ok=False means attention needed.
        STANDBY (no consent yet) is healthy: it is a deliberate state, reported
        in "state" and "protection_active"."""
        s = self.service
        checks = {"service_threads": all(t.is_alive() for t in s._threads) if s.started_at else False,
                  "resource_monitor": s.resource_monitor._thread.is_alive() if s.started_at else False}
        if s.active:
            checks.update({
                "file_monitor": bool(s.observer and s.observer.is_alive()),
                "backup_thread": bool(s.backup_scheduler and s.backup_scheduler.is_alive()),
                "command_listener": bool(s.command_listener and s.command_listener._thread.is_alive()),
                "pid_cache": bool(s.pid_cache and s.pid_cache._thread.is_alive()),
            })
        if self.enable_dashboard:
            checks["dashboard"] = bool(self._dashboard_thread and self._dashboard_thread.is_alive())
        return {"ok": all(checks.values()), "state": s.state, "protection_active": s.active, "checks": checks}

    # ------------------------------------------------------------------- ML

    def analyze_memory_features(self, features: dict, explain: bool = True) -> dict:
        """Runs the trained Isolation Forest + Random Forest on a dict of
        CIC-MalMem-2022 style memory features. Returns is_threat, confidence
        and (for threats) a SHAP explanation.

        Note: the live file monitor does NOT call this; it is for callers that
        can provide memory-forensic features."""
        with self._detector_lock:
            if self._detector is None:
                from src.detection.predict import HybridDetector
                self._detector = HybridDetector()
        return self._detector.predict(features, compute_explanation=explain)

"""
consent.py
------------
Layer 2 copies the user's files (backups) and watches what they do. It must
never do that before someone has agreed to it. This module stores that
agreement.

  * Until consent exists the service stays in STANDBY: the dashboard and
    health check work, but no file is read, watched or copied.
  * Consent is saved (runtime/data/consent.json), so it is asked ONCE; later
    starts resume protection without asking again.
  * Consent is tied to the folders that were shown to the user. If the
    configured watch folders change, the new folders are NOT covered until
    the user agrees again.
  * Revoking stops watching and backing up immediately. Existing backups stay
    on disk (so a restore is still possible).

Who may grant it:
  - the dashboard button (a person clicks "Start protecting"),
  - Layer2.grant_consent() / `python -m main --consent` (explicit, logged), or
  - NEURASHIELD_AUTO_CONSENT=1 or config main.require_consent: false, for
    deployments where the platform's own installer already showed the user
    the consent screen. This is the integrator's decision and is logged.
"""

import json
import os
import threading
import time

from src.utils import paths
from src.utils.logger import get_logger, sanitize_for_log

logger = get_logger(__name__)


def normalize_folders(folders) -> list:
    return sorted({os.path.normcase(os.path.abspath(os.path.expanduser(str(f)))) for f in folders})


class ConsentStore:
    def __init__(self, path: str = None):
        self.path = path or paths.runtime("data/consent.json")
        self._lock = threading.Lock()

    def read(self):
        """Returns the saved consent dict, or None."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("granted") is not True:
            return None
        if not isinstance(data.get("folders"), list):
            return None
        return data

    def covers(self, folders) -> bool:
        """True if consent exists AND includes every one of these folders."""
        data = self.read()
        if data is None:
            return False
        granted = set(normalize_folders(data["folders"]))
        return set(normalize_folders(folders)) <= granted

    def grant(self, folders, by: str) -> dict:
        record = {
            "granted": True,
            "granted_at": time.time(),
            "granted_by": str(by)[:40],
            "folders": normalize_folders(folders),
        }
        with self._lock:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=1)
            os.replace(tmp, self.path)
        logger.warning(f"[consent] Protection consent GRANTED by {sanitize_for_log(by)} "
                       f"for {len(record['folders'])} folder(s).")
        return record

    def revoke(self, by: str = "unknown"):
        with self._lock:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                return
        logger.warning(f"[consent] Protection consent REVOKED by {sanitize_for_log(by)}.")

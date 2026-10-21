"""
incident_store.py
-------------------
One REPORT per ransomware incident: what was detected, what was hit, what the
layer did about it, and what was (or could not be) restored.

The service writes the report while it responds; the dashboard reads it. It is
stored as a small JSON file (data/incidents.json, newest 100 incidents), written
atomically, so the report survives a restart and the dashboard can show it even
if it runs in another process.

All text shown to a person is produced by summarize(); file names are data from
the monitored machine, so the dashboard escapes every value it displays.
"""

import json
import os
import threading
import time
import uuid

from src.utils import paths
from src.utils.logger import get_logger, sanitize_for_log

logger = get_logger(__name__)

MAX_INCIDENTS = 100
MAX_FILES_PER_INCIDENT = 1000
MAX_TIMELINE = 200

DAMAGE_TEXT = {
    "missing": "deleted or renamed",
    "emptied": "emptied",
    "format": "encrypted (file format destroyed)",
    "encrypted": "encrypted",
}


def _clock(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


class IncidentStore:
    def __init__(self, path: str = None):
        self.path = path or paths.runtime("data/incidents.json")
        self._lock = threading.RLock()
        self._items = self._load()

    # ------------------------------------------------------------ storage

    def _load(self) -> list:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [i for i in data if isinstance(i, dict) and isinstance(i.get("id"), str)]
        except (OSError, ValueError):
            return []

    def _save(self):
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._items[-MAX_INCIDENTS:], f)
            os.replace(tmp, self.path)
        except OSError as e:
            logger.warning(f"[incident] Could not save incident reports: {sanitize_for_log(e)}")

    def _find(self, incident_id):
        for item in reversed(self._items):
            if item["id"] == incident_id:
                return item
        return None

    # ------------------------------------------------------------ writing

    def open(self, *, started_at: float, severity: str, confidence: float, trigger: str,
             folders=(), process=None) -> str:
        now = time.time()
        incident_id = "INC-" + time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-" + uuid.uuid4().hex[:4]
        item = {
            "id": incident_id, "started_at": started_at, "detected_at": now, "updated_at": now,
            "ended_at": None, "status": "responding", "severity": severity,
            "confidence": round(float(confidence), 4), "trigger": trigger,
            "process": process or {"pid": None, "name": "unidentified"},
            "process_killed": False, "kill_note": None,
            "folders": sorted(set(folders))[:50],
            "files": [], "unprotected": [], "leftovers": [],
            "restore_seconds": None, "alert_ids": [], "timeline": [],
        }
        with self._lock:
            self._items.append(item)
            self._items = self._items[-MAX_INCIDENTS:]
            self._event_locked(item, f"Ransomware-like activity detected: {trigger}")
            self._save()
        return incident_id

    def _event_locked(self, item, text):
        item["timeline"].append({"t": time.time(), "text": text})
        item["timeline"] = item["timeline"][-MAX_TIMELINE:]
        item["updated_at"] = time.time()

    def event(self, incident_id, text):
        with self._lock:
            item = self._find(incident_id)
            if item:
                self._event_locked(item, text)
                self._save()

    def update(self, incident_id, **fields):
        with self._lock:
            item = self._find(incident_id)
            if not item:
                return
            for k, v in fields.items():
                if k == "confidence":
                    v = max(float(v), item.get("confidence", 0))
                item[k] = v
            item["updated_at"] = time.time()
            self._save()

    def add_alert(self, incident_id, alert_id):
        with self._lock:
            item = self._find(incident_id)
            if item and alert_id not in item["alert_ids"]:
                item["alert_ids"].append(alert_id)
                self._save()

    def add_files(self, incident_id, damaged: list):
        """damaged: [{"path": ..., "reason": "missing|emptied|format|encrypted"}]"""
        with self._lock:
            item = self._find(incident_id)
            if not item:
                return
            known = {f["path"] for f in item["files"]}
            for d in damaged:
                if d["path"] in known or len(item["files"]) >= MAX_FILES_PER_INCIDENT:
                    continue
                item["files"].append({
                    "path": d["path"], "name": os.path.basename(d["path"]),
                    "damage": DAMAGE_TEXT.get(d.get("reason"), "damaged"),
                    "restored": None, "note": None})
                known.add(d["path"])
            item["updated_at"] = time.time()
            self._save()

    def apply_restore(self, incident_id, details: dict, elapsed: float = None):
        """details: {path: {"status": "ok"|"failed", "reason": ...}} from restore_all()."""
        with self._lock:
            item = self._find(incident_id)
            if not item:
                return
            by_path = {f["path"]: f for f in item["files"]}
            for path, res in (details or {}).items():
                entry = by_path.get(path)
                if entry is None:
                    continue
                ok = res.get("status") == "ok"
                entry["restored"] = ok
                entry["note"] = None if ok else (res.get("reason") or "restore failed")
            if elapsed is not None:
                item["restore_seconds"] = round((item.get("restore_seconds") or 0) + elapsed, 3)
            item["updated_at"] = time.time()
            self._save()

    def set_extra(self, incident_id, *, unprotected=None, leftovers=None):
        with self._lock:
            item = self._find(incident_id)
            if not item:
                return
            if unprotected is not None:
                item["unprotected"] = list(unprotected)[:200]
            if leftovers is not None:
                item["leftovers"] = list(leftovers)[:200]
            self._save()

    def close(self, incident_id, status: str, note: str = None):
        with self._lock:
            item = self._find(incident_id)
            if not item:
                return
            item["status"] = status
            item["ended_at"] = time.time()
            if note:
                self._event_locked(item, note)
            self._save()

    # ------------------------------------------------------------ reading

    def get(self, incident_id):
        with self._lock:
            item = self._find(incident_id)
            return json.loads(json.dumps(item)) if item else None

    def list(self, limit: int = 20) -> list:
        """Newest first, summaries with counts (no file lists)."""
        with self._lock:
            items = list(self._items[-limit:])
        out = []
        for item in reversed(items):
            summary = summarize(item)
            row = {k: item[k] for k in ("id", "detected_at", "ended_at", "status", "severity",
                                         "confidence", "process", "process_killed")}
            row.update(summary)
            out.append(row)
        return out

    def reload(self):
        """Re-read the file (used by a dashboard running in a different process)."""
        with self._lock:
            self._items = self._load()


def summarize(item: dict) -> dict:
    files = item.get("files", [])
    restored = sum(1 for f in files if f.get("restored") is True)
    failed = sum(1 for f in files if f.get("restored") is False)
    waiting = len(files) - restored - failed
    unprotected = len(item.get("unprotected", []))
    detected = _clock(item.get("detected_at", time.time()))

    parts = [f"At {detected} NeuraShield detected ransomware-like activity ({item.get('trigger', 'unknown signal')})."]
    if files:
        parts.append(f"{len(files)} protected file(s) were damaged by the attack.")
    else:
        parts.append("No protected file was found damaged.")
    if restored:
        secs = item.get("restore_seconds")
        parts.append(f"{restored} file(s) were restored automatically from clean backups"
                     + (f" in {secs:.2f} s." if secs is not None else "."))
    if failed:
        parts.append(f"{failed} file(s) could NOT be restored (see the list).")
    if waiting and item.get("status") != "responding":
        parts.append(f"{waiting} file(s) are still waiting for restore.")
    if unprotected:
        parts.append(f"{unprotected} changed file(s) had no backup (they are new files), so they could not be restored.")
    proc = item.get("process") or {}
    if item.get("process_killed"):
        parts.append(f"The attacking program ({proc.get('name', 'unknown')}) was stopped.")
    else:
        why = item.get("kill_note") or "it could not be identified with confidence"
        parts.append(f"The attacking program was not stopped: {why}.")

    headline = {
        "responding": "Responding to an attack…",
        "contained": (f"Attack contained - {restored} file(s) restored" if not failed
                      else f"Attack contained - {restored} restored, {failed} not restored"),
        "needs_attention": "Attack contained - some files need attention",
    }.get(item.get("status"), "Incident")
    return {"headline": headline, "text": " ".join(parts),
            "counts": {"affected": len(files), "restored": restored, "failed": failed,
                       "waiting": waiting, "unprotected": unprotected,
                       "leftovers": len(item.get("leftovers", []))}}

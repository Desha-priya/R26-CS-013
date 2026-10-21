"""
incident_state.py
--------------------
Tracks which incidents (alerts) an operator has acknowledged/cleared
from the dashboard.

This is deliberately NOT a delete operation on the alert queue. The
alert queue is an append-only, HMAC-signed audit log - a real
production security system must never let an operator silently erase
evidence that an incident happened, even one that turned out to be a
false positive or was already handled. What "Clear" actually means
here is: mark this incident as acknowledged/resolved so it stops
showing in the active view, while the underlying signed record stays
intact forever in the alert queue for audit purposes.

This mirrors how real SOC (Security Operations Center) tools work:
incidents get "acknowledged", "resolved", or "closed" - never deleted.
"""

import os
import json
import threading


class IncidentStateStore:
    def __init__(self, state_path=None):
        from src.utils import paths
        self.state_path = state_path or paths.runtime("data/simulated_events/incident_state.json")
        state_path = self.state_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
        self._acknowledged_ids = self._load()

    def _load(self) -> set:
        if not os.path.exists(self.state_path):
            return set()
        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
            return set(data.get("acknowledged_ids", []))
        except (json.JSONDecodeError, OSError):
            return set()

    def _save(self):
        with open(self.state_path, "w") as f:
            json.dump({"acknowledged_ids": sorted(self._acknowledged_ids)}, f, indent=2)

    def acknowledge(self, alert_id: str):
        with self._lock:
            self._acknowledged_ids.add(alert_id)
            self._save()

    def acknowledge_many(self, alert_ids: list):
        with self._lock:
            self._acknowledged_ids.update(alert_ids)
            self._save()

    def is_acknowledged(self, alert_id: str) -> bool:
        return alert_id in self._acknowledged_ids

    def unacknowledge(self, alert_id: str):
        """Lets an operator undo an accidental acknowledgment - also
        kept as an explicit action, not a silent delete."""
        with self._lock:
            self._acknowledged_ids.discard(alert_id)
            self._save()


if __name__ == "__main__":
    # Self-contained smoke test
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        store = IncidentStateStore(state_path=path)

        assert not store.is_acknowledged("alert-1")
        store.acknowledge("alert-1")
        assert store.is_acknowledged("alert-1")

        # Confirm it survives a reload (persisted to disk correctly)
        store2 = IncidentStateStore(state_path=path)
        assert store2.is_acknowledged("alert-1")
        assert not store2.is_acknowledged("alert-2")

        store2.acknowledge_many(["alert-2", "alert-3"])
        assert store2.is_acknowledged("alert-2")
        assert store2.is_acknowledged("alert-3")

        store2.unacknowledge("alert-1")
        assert not store2.is_acknowledged("alert-1")

        print("PASSED: acknowledge, persistence, bulk-acknowledge, and undo all work correctly.")

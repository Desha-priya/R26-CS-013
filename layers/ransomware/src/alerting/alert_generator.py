"""
alert_generator.py
---------------------
Builds, signs and dispatches alerts using the schema in alert_schema.py.

Where an alert goes:
  1. Signed (HMAC-SHA256) and appended to a local JSONL queue file that
     other NeuraShield layers / the dashboard can read and verify.
  2. If a platform URL is configured, a short notice
     {"event_type", "severity"} is also POSTed to the main platform
     (see platform_client.py), for severities listed in
     alerting.platform_alert_severities (default MEDIUM/HIGH/CRITICAL).

If the platform cannot be reached (network problem, platform restarting),
the notice is kept in a small outbox file and retried later by
flush_outbox(), so a short outage does not lose alerts. The local signed
alert is always written first and never depends on the platform.
"""

import json
import os
import threading

from src.alerting.alert_schema import build_alert
from src.alerting.hmac_auth import get_secret_key, sign_payload, verify_payload
from src.alerting.platform_client import PlatformClient
from src.utils import paths
from src.utils.logger import get_logger
from src.utils.rotation import RotatingJSONLWriter

logger = get_logger(__name__)

OUTBOX_MAX_ITEMS = 500


class AlertGenerator:
    def __init__(self, config_path=None, queue_path=None):
        self.cfg = paths.load_config(config_path)
        self.queue_path = queue_path or paths.runtime("data/simulated_events/alert_queue.jsonl")
        os.makedirs(os.path.dirname(self.queue_path) or ".", exist_ok=True)
        self._secret = get_secret_key(self.cfg)

        rotation_cfg = self.cfg.get("alerting", {}).get("queue_rotation", {})
        self._writer = RotatingJSONLWriter(
            self.queue_path,
            max_bytes=rotation_cfg.get("max_bytes", 5 * 1024 * 1024),
            backup_count=rotation_cfg.get("backup_count", 3),
        )

        # Platform URL: environment variable first (never commit a real URL),
        # then config.yaml. Empty means "do not send to a platform".
        platform_url = os.environ.get("NEURASHIELD_PLATFORM_URL") or \
            self.cfg.get("alerting", {}).get("platform_base_url") or ""
        self._platform_client = None
        if platform_url.strip():
            try:
                self._platform_client = PlatformClient(platform_url)
            except ValueError as e:
                logger.error(f"[alert] Platform integration disabled: {e}")

        self._platform_severities = set(
            self.cfg.get("alerting", {}).get("platform_alert_severities", ["MEDIUM", "HIGH", "CRITICAL"])
        )

        self._outbox_path = os.path.join(os.path.dirname(self.queue_path) or ".", "platform_outbox.jsonl")
        self._outbox_lock = threading.Lock()

    # ------------------------------------------------------------------ send

    def send_alert(self, threat_type, confidence, affected_files, process_info,
                   detection_features, action_taken, explanation=None) -> dict:
        alert = build_alert(
            threat_type=threat_type,
            confidence=confidence,
            affected_files=affected_files,
            process_info=process_info,
            detection_features=detection_features,
            action_taken=action_taken,
            target_layer=self.cfg["alerting"]["target_layer"],
            explanation=explanation,
        )

        # Sign BEFORE writing: the signature covers everything except itself,
        # so later tampering (for example editing the queue file) is detectable.
        alert["signature"] = sign_payload(alert, self._secret)
        self._writer.append(json.dumps(alert))

        logger.info(
            f"[alert] Sent alert {alert['alert_id']} | "
            f"severity={alert['threat']['severity']} | "
            f"confidence={alert['threat']['confidence']} | "
            f"target={alert['target_layer']} | signed=yes"
        )

        # Platform notice: done AFTER the local write. Notification only - the
        # kill/rollback already happened, and a platform outage must never
        # undo or block an alert that was recorded locally.
        if self._platform_client and alert["threat"]["severity"] in self._platform_severities:
            result = self._platform_client.send_threat_alert(
                event_type=alert["threat"]["type"],
                severity=alert["threat"]["severity"],
            )
            alert["platform_delivery"] = result
            if result.get("status") != "sent" and result.get("retryable"):
                self._add_to_outbox(alert["threat"]["type"], alert["threat"]["severity"], alert["alert_id"])

        return alert

    # ---------------------------------------------------------------- outbox

    def _read_outbox(self) -> list:
        if not os.path.exists(self._outbox_path):
            return []
        items = []
        try:
            with open(self._outbox_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return items

    def _write_outbox(self, items: list):
        tmp = self._outbox_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for item in items[-OUTBOX_MAX_ITEMS:]:
                f.write(json.dumps(item) + "\n")
        os.replace(tmp, self._outbox_path)

    def _add_to_outbox(self, event_type: str, severity: str, alert_id: str):
        with self._outbox_lock:
            items = self._read_outbox()
            items.append({"event_type": event_type, "severity": severity, "alert_id": alert_id})
            self._write_outbox(items)
        logger.warning(f"[alert] Platform not reachable - alert {alert_id} kept in the outbox for retry.")

    def outbox_size(self) -> int:
        with self._outbox_lock:
            return len(self._read_outbox())

    def flush_outbox(self, max_items: int = 20) -> dict:
        """Tries to deliver waiting notices. Called every ~30 seconds by the service."""
        if not self._platform_client:
            return {"sent": 0, "remaining": self.outbox_size()}

        sent = 0
        with self._outbox_lock:
            items = self._read_outbox()
            if not items:
                return {"sent": 0, "remaining": 0}

            remaining = []
            platform_down = False
            for index, item in enumerate(items):
                if platform_down or index >= max_items:
                    remaining.append(item)          # not attempted this round
                    continue
                result = self._platform_client.send_threat_alert(item["event_type"], item["severity"])
                if result.get("status") == "sent":
                    sent += 1
                elif result.get("retryable"):
                    remaining.append(item)          # still failing: keep it
                    platform_down = True            # stop hammering a down platform
                # not retryable (wrong key, bad data): drop it, retrying cannot help

            if remaining != items:
                self._write_outbox(remaining)

        if sent:
            logger.info(f"[alert] Delivered {sent} queued alert notice(s) to the platform.")
        return {"sent": sent, "remaining": len(remaining)}

    # ---------------------------------------------------------------- verify

    def verify_alert(self, alert: dict) -> bool:
        """Verifies an alert's HMAC signature. Anything that consumes the
        alert queue should call this before trusting an alert."""
        signature = alert.get("signature")
        if not signature:
            return False
        payload = {k: v for k, v in alert.items() if k != "signature"}
        return verify_payload(payload, signature, self._secret)

    def read_recent_alerts(self, n: int = 10) -> list:
        if not os.path.exists(self.queue_path):
            return []
        with open(self.queue_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        alerts = []
        for line in lines[-n:]:
            try:
                alerts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return alerts


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        gen = AlertGenerator(queue_path=os.path.join(tmp, "alerts.jsonl"))
        alert = gen.send_alert(
            threat_type="ransomware_activity", confidence=0.98,
            affected_files=["/home/user/Documents/report.docx"],
            process_info={"pid": 4821, "name": "svchost32.exe", "parent_name": "powershell.exe", "cmdline": ["x"]},
            detection_features={"file_rate_5s": 12.4, "avg_entropy_5s": 7.91},
            action_taken={"process_killed": True, "kill_method": "terminate",
                          "files_restored": 1, "restore_elapsed_seconds": 0.34},
        )
        print("Verify untampered:", gen.verify_alert(alert))
        tampered = dict(alert)
        tampered["threat"] = dict(tampered["threat"])
        tampered["threat"]["confidence"] = 0.01
        print("Verify tampered (should be False):", gen.verify_alert(tampered))

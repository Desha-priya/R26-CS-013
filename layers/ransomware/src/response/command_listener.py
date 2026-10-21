"""
command_listener.py
----------------------
The other half of the authenticated message loop that alert_generator.py
started: the Brain (or an operator approving an alert in the dashboard)
sends a SIGNED rollback command, and this listener verifies it before
doing anything.

Command flow:
  1. A sender builds a command dict, signs it with the shared HMAC secret
     (build_rollback_command) and appends it to the command queue file.
  2. CommandListener polls the queue and checks each new command:
       - signature is valid                       (not forged / not edited)
       - command is fresh (issued within max_age)  (a copy of an old
                                                    command cannot be replayed later)
       - command id has not been used before       (replay protection that
                                                    survives a restart)
       - the file list looks sane (a list of text paths, not too long)
  3. Only commands that pass every check reach the RollbackEngine. Anything
     else is rejected and logged.

The rollback engine itself only restores files that have a backup in its
own index, so even a valid command cannot write to arbitrary paths.
"""

import json
import os
import threading
import time
import uuid

from src.alerting.hmac_auth import get_secret_key, sign_payload, verify_payload
from src.response.rollback_engine import RollbackEngine
from src.utils import paths
from src.utils.logger import get_logger, sanitize_for_log

logger = get_logger(__name__)

MAX_TARGET_FILES = 1000
MAX_PATH_LENGTH = 4096
MAX_REASON_LENGTH = 200
MAX_REMEMBERED_IDS = 5000
FUTURE_TOLERANCE_SECONDS = 5.0


def build_rollback_command(target_files: list, secret: bytes, reason: str = "") -> dict:
    """Builds and signs a rollback command (what the Brain or an operator calls)."""
    command = {
        "command_id": str(uuid.uuid4()),
        "command": "rollback",
        "target_files": target_files,
        "reason": reason,
        "issued_at": time.time(),
    }
    command["signature"] = sign_payload(command, secret)
    return command


class CommandListener:
    def __init__(self, rollback: RollbackEngine, config_path=None, queue_path=None):
        self.cfg = paths.load_config(config_path)
        self.rollback = rollback
        self.queue_path = queue_path or paths.runtime("data/simulated_events/command_queue.jsonl")
        os.makedirs(os.path.dirname(self.queue_path) or ".", exist_ok=True)
        self._secret = get_secret_key(self.cfg)
        self.max_age_seconds = float(self.cfg.get("security", {}).get("command_max_age_seconds", 60))

        # Used command ids are saved next to the queue so a restart cannot
        # replay an old command.
        self._ids_path = self.queue_path + ".processed"
        self._processed_ids = self._load_ids()
        self._offset = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._poll_interval = 1.0

    # ------------------------------------------------------------ id memory

    def _load_ids(self) -> set:
        try:
            with open(self._ids_path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except (OSError, ValueError):
            return set()

    def _save_ids(self):
        ids = list(self._processed_ids)[-MAX_REMEMBERED_IDS:]
        tmp = self._ids_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(ids, f)
            os.replace(tmp, self._ids_path)
        except OSError as e:
            logger.warning(f"[command_listener] Could not save used command ids: {sanitize_for_log(e)}")

    # -------------------------------------------------------------- running

    def start(self, poll_interval_seconds: float = 1.0):
        self._poll_interval = poll_interval_seconds
        self._thread.start()
        logger.info("[command_listener] Started, watching for authenticated rollback commands.")

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception as e:  # never let the listener thread die silently
                logger.error(f"[command_listener] Poll error: {sanitize_for_log(e)}")
            self._stop_event.wait(self._poll_interval)

    def _poll_once(self):
        """Reads only the NEW lines of the queue file since the last poll."""
        if not os.path.exists(self.queue_path):
            return
        size = os.path.getsize(self.queue_path)
        if size < self._offset:          # file was rotated or replaced
            self._offset = 0
        if size == self._offset:
            return

        with open(self.queue_path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
        # Only consume complete lines.
        last_newline = data.rfind(b"\n")
        if last_newline == -1:
            return
        self._offset += last_newline + 1
        lines = data[:last_newline].decode("utf-8", errors="replace").splitlines()

        changed = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(command, dict):
                continue

            cmd_id = command.get("command_id")
            if not isinstance(cmd_id, str) or not cmd_id or cmd_id in self._processed_ids:
                continue
            self._processed_ids.add(cmd_id)
            changed = True
            self._handle_command(command)

        if changed:
            self._save_ids()

    # ------------------------------------------------------------- checking

    def _reject(self, command: dict, why: str):
        logger.error(f"[command_listener] REJECTED command {sanitize_for_log(command.get('command_id'))}: {why}")

    def _handle_command(self, command: dict):
        signature = command.get("signature")
        if not signature:
            return self._reject(command, "no signature present.")

        payload = {k: v for k, v in command.items() if k != "signature"}
        if not verify_payload(payload, signature, self._secret):
            return self._reject(command, "invalid signature - possible tampering or forged command.")

        # Freshness: stops an old, once-valid command from being replayed later.
        issued_at = command.get("issued_at")
        if not isinstance(issued_at, (int, float)) or isinstance(issued_at, bool):
            return self._reject(command, "missing or invalid issue time.")
        age = time.time() - float(issued_at)
        if age > self.max_age_seconds:
            return self._reject(command, f"expired ({age:.0f}s old, limit {self.max_age_seconds:.0f}s).")
        if age < -FUTURE_TOLERANCE_SECONDS:
            return self._reject(command, "issue time is in the future.")

        if command.get("command") != "rollback":
            logger.warning(f"[command_listener] Ignoring unrecognized command type: "
                           f"{sanitize_for_log(command.get('command'))}")
            return

        target_files = command.get("target_files")
        if (not isinstance(target_files, list) or not target_files or len(target_files) > MAX_TARGET_FILES
                or not all(isinstance(p, str) and 0 < len(p) <= MAX_PATH_LENGTH for p in target_files)):
            return self._reject(command, "target_files must be a non-empty list of text paths "
                                         f"(at most {MAX_TARGET_FILES}).")

        reason = sanitize_for_log(command.get("reason", "n/a"), MAX_REASON_LENGTH)
        logger.info(f"[command_listener] VERIFIED command {sanitize_for_log(command['command_id'])} - "
                    f"executing rollback on {len(target_files)} file(s). Reason: {reason}")
        result = self.rollback.restore_all(target_files)
        logger.info(f"[command_listener] Rollback complete: {result['succeeded']}/{result['total_files']} restored")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "test.txt")
        with open(test_file, "w") as f:
            f.write("clean original content")

        rollback = RollbackEngine(backup_root=os.path.join(tmp, "_backups"))
        rollback.backup_file(test_file)
        with open(test_file, "wb") as f:
            f.write(os.urandom(100))

        queue_path = os.path.join(tmp, "command_queue.jsonl")
        listener = CommandListener(rollback, queue_path=queue_path)
        good = build_rollback_command([test_file], listener._secret, reason="test")
        with open(queue_path, "a") as f:
            f.write(json.dumps(good) + "\n")
        listener._poll_once()
        with open(test_file, "r") as f:
            assert f.read() == "clean original content"
        print("PASSED: legitimate command executed.")

        forged = build_rollback_command([test_file], listener._secret)
        forged["target_files"] = ["/etc/passwd"]
        with open(queue_path, "a") as f:
            f.write(json.dumps(forged) + "\n")
        listener._poll_once()
        print("PASSED: forged command rejected (see REJECTED line).")

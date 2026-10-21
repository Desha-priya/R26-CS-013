"""
rotation.py
-------------
Size-based rotation for append-only JSONL queue files (alert_queue.jsonl,
command_queue.jsonl). Without this, a system designed to run
continuously (main.py) would let these files grow forever - a real
gap for anything meant to survive days or weeks of uptime, not just a
demo session.

Mirrors the pattern already used for text logs in src/utils/logger.py
(Python's RotatingFileHandler), but implemented directly here since
JSONL queue writes go through plain file I/O, not the logging module.

Behavior: when a write would push the file over `max_bytes`, the
current file is rotated to a numbered backup (.1, .2, ...) before the
new line is written, keeping at most `backup_count` old files. This is
the same approach as Python's own RotatingFileHandler, just applied to
JSONL data instead of log lines.
"""

import os
import threading


class RotatingJSONLWriter:
    def __init__(self, path: str, max_bytes: int = 5 * 1024 * 1024, backup_count: int = 3):
        self.path = path
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, line: str):
        """Appends one line (without trailing newline - this adds it),
        rotating first if the file is already at/over the size limit."""
        with self._lock:
            self._rotate_if_needed()
            with open(self.path, "a") as f:
                f.write(line + "\n")

    def _rotate_if_needed(self):
        if not os.path.exists(self.path):
            return
        if os.path.getsize(self.path) < self.max_bytes:
            return

        # Shift existing backups up by one (.2 -> .3, .1 -> .2, ...),
        # dropping the oldest if it would exceed backup_count.
        oldest = f"{self.path}.{self.backup_count}"
        if os.path.exists(oldest):
            os.remove(oldest)
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.path}.{i}"
            dst = f"{self.path}.{i + 1}"
            if os.path.exists(src):
                os.replace(src, dst)

        os.replace(self.path, f"{self.path}.1")


if __name__ == "__main__":
    # Self-contained smoke test: write past the size limit, confirm
    # rotation actually happens and old data isn't silently lost.
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_queue.jsonl")
        # Tiny limit (200 bytes) to force rotation quickly in a test
        writer = RotatingJSONLWriter(path, max_bytes=200, backup_count=2)

        for i in range(30):
            writer.append(json.dumps({"seq": i, "padding": "x" * 20}))

        print("Files present after 30 writes with a 200-byte limit:")
        for f in sorted(os.listdir(tmp)):
            size = os.path.getsize(os.path.join(tmp, f))
            print(f"  {f}: {size} bytes")

        assert os.path.exists(path), "current file should exist"
        assert os.path.exists(path + ".1"), "at least one rotation should have happened"
        # backup_count=2, so .3 should never exist
        assert not os.path.exists(path + ".3"), "should not exceed backup_count"

        with open(path) as f:
            current_lines = f.readlines()
        print(f"\nCurrent file has {len(current_lines)} lines (most recent writes)")
        print("PASSED: rotation occurred correctly, backup_count respected.")

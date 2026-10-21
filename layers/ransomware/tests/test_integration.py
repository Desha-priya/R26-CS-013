"""
test_integration.py
----------------------
Unit tests for the cross-component integration modules built later in
the project: authenticated rollback commands, the rolling PID cache,
and the Zero-Trust Auth layer stub. These previously only had ad-hoc
smoke tests in their own __main__ blocks - this makes them part of the
real, repeatable test suite.

Run with:
    python -m pytest tests/test_integration.py -v
"""

import os
import sys
import json
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.response.rollback_engine import RollbackEngine
from src.response.command_listener import CommandListener, build_rollback_command
from src.monitor.process_tracker import ProcessTracker, RollingPidCache
from src.alerting.alert_generator import AlertGenerator
from src.utils.rotation import RotatingJSONLWriter
from src.utils.resource_monitor import ResourceMonitor
from src.dashboard.incident_state import IncidentStateStore


# ---------- command_listener.py ----------

def test_command_listener_executes_legitimate_command():
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "test.txt")
        with open(test_file, "w") as f:
            f.write("clean original content")

        rollback = RollbackEngine(backup_root=os.path.join(tmp, "_backups"))
        rollback.backup_file(test_file)

        with open(test_file, "wb") as f:
            f.write(os.urandom(100))  # simulate corruption

        listener = CommandListener(rollback, queue_path=os.path.join(tmp, "cmds.jsonl"))
        command = build_rollback_command([test_file], listener._secret, reason="test")
        with open(listener.queue_path, "a") as f:
            f.write(json.dumps(command) + "\n")

        listener._poll_once()

        with open(test_file, "r") as f:
            assert f.read() == "clean original content"


def test_command_listener_rejects_forged_command():
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "test.txt")
        with open(test_file, "w") as f:
            f.write("clean original content")

        rollback = RollbackEngine(backup_root=os.path.join(tmp, "_backups"))
        rollback.backup_file(test_file)

        with open(test_file, "wb") as f:
            f.write(os.urandom(100))

        listener = CommandListener(rollback, queue_path=os.path.join(tmp, "cmds.jsonl"))
        command = build_rollback_command([test_file], listener._secret, reason="test")
        command["target_files"] = ["/etc/passwd"]  # tamper AFTER signing
        with open(listener.queue_path, "a") as f:
            f.write(json.dumps(command) + "\n")

        listener._poll_once()

        # File should remain corrupted - the forged command must NOT
        # have triggered any rollback.
        with open(test_file, "rb") as f:
            content = f.read()
        assert content != b"clean original content"


def test_command_listener_ignores_duplicate_command_id():
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "test.txt")
        with open(test_file, "w") as f:
            f.write("v1")

        rollback = RollbackEngine(backup_root=os.path.join(tmp, "_backups"))
        rollback.backup_file(test_file)

        listener = CommandListener(rollback, queue_path=os.path.join(tmp, "cmds.jsonl"))
        command = build_rollback_command([test_file], listener._secret, reason="test")
        with open(listener.queue_path, "a") as f:
            f.write(json.dumps(command) + "\n")
            f.write(json.dumps(command) + "\n")  # same command_id appended twice

        listener._poll_once()
        assert len(listener._processed_ids) == 1  # only processed once, not twice


# ---------- process_tracker.py: RollingPidCache ----------

def test_rolling_pid_cache_catches_held_open_handle():
    """Matches the real finding from testing: a handle held open across
    a chunked write loop (realistic ransomware behavior) should be
    caught by the rolling cache, where a single instantaneous scan
    would likely miss it."""
    with tempfile.TemporaryDirectory() as tmp:
        tracker = ProcessTracker()
        cache = RollingPidCache(tracker, [tmp], sample_interval=0.1, max_age_seconds=5.0)
        cache.start()

        test_file = os.path.join(tmp, "held_open.txt")
        with open(test_file, "wb") as f:
            for _ in range(5):
                f.write(os.urandom(10000))
                time.sleep(0.08)  # handle held open ~0.4s total

        time.sleep(0.3)
        result = cache.get_recent_pids(tmp)
        cache.stop()

        assert os.getpid() in result


def test_rolling_pid_cache_returns_empty_for_untouched_directory():
    with tempfile.TemporaryDirectory() as tmp_watched, tempfile.TemporaryDirectory() as tmp_other:
        tracker = ProcessTracker()
        cache = RollingPidCache(tracker, [tmp_watched], sample_interval=0.1, max_age_seconds=5.0)
        cache.start()
        time.sleep(0.3)

        result = cache.get_recent_pids(tmp_other)  # a directory nothing ever touched
        cache.stop()

        assert result == []


# ---------- rotation.py ----------

def test_rotating_writer_rotates_at_size_limit():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "queue.jsonl")
        writer = RotatingJSONLWriter(path, max_bytes=200, backup_count=2)

        for i in range(30):
            writer.append(json.dumps({"seq": i, "padding": "x" * 20}))

        assert os.path.exists(path)
        assert os.path.exists(path + ".1")
        assert not os.path.exists(path + ".3")  # backup_count=2, never more than 2 old files


def test_rotating_writer_preserves_data_across_rotation():
    """Data isn't silently lost on rotation - it moves to the backup file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "queue.jsonl")
        writer = RotatingJSONLWriter(path, max_bytes=150, backup_count=1)

        for i in range(10):
            writer.append(json.dumps({"seq": i}))

        total_lines = 0
        for suffix in ("", ".1"):
            p = path + suffix
            if os.path.exists(p):
                with open(p) as f:
                    total_lines += len([l for l in f if l.strip()])
        assert total_lines == 10  # nothing silently dropped, some just moved to backup


# ---------- rollback_engine.py: global storage cap ----------

def test_rollback_engine_enforces_global_storage_cap():
    with tempfile.TemporaryDirectory() as tmp:
        engine = RollbackEngine(backup_root=os.path.join(tmp, "_backups"),
                                 max_versions=100, max_total_mb=0.01)  # 10KB cap

        for i in range(15):
            path = os.path.join(tmp, f"file_{i}.txt")
            with open(path, "w") as f:
                f.write("x" * 2000)
            engine.backup_file(path)

        total_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(engine.backup_root)
            for f in files if f.endswith(".bak")
        )
        assert total_size <= 0.01 * 1024 * 1024 + 2000  # cap respected (+ one file's slack)


# ---------- resource_monitor.py ----------

def test_resource_monitor_writes_valid_status():
    with tempfile.TemporaryDirectory() as tmp:
        status_path = os.path.join(tmp, "status.json")
        monitor = ResourceMonitor(status_path=status_path, sample_interval=0.3,
                                   cpu_target_percent=5.0, memory_target_mb=50.0)
        monitor.start()
        time.sleep(0.8)  # let at least one real sample happen
        monitor.stop()

        assert os.path.exists(status_path)
        with open(status_path) as f:
            status = json.load(f)

        assert "cpu_percent_avg" in status
        assert "memory_mb_avg" in status
        assert status["memory_mb_avg"] > 0  # a real process always uses some memory
        assert status["cpu_target_percent"] == 5.0
        assert status["memory_target_mb"] == 50.0


def test_resource_monitor_first_sample_is_not_inflated():
    """Regression test for a real measurement bug found via testing:
    sampling immediately after psutil's cpu_percent() baseline is
    primed gives a near-zero elapsed window, producing a wildly
    inflated first reading (a real run showed 355% CPU on an idle
    process). The monitor must wait a full interval before its first
    sample, not just subsequent ones."""
    with tempfile.TemporaryDirectory() as tmp:
        status_path = os.path.join(tmp, "status.json")
        monitor = ResourceMonitor(status_path=status_path, sample_interval=0.3)
        monitor.start()
        time.sleep(0.8)
        monitor.stop()

        with open(status_path) as f:
            status = json.load(f)

        # A sane upper bound - genuinely no realistic single-process
        # workload in this test should read anywhere near 355%.
        assert status["cpu_percent_avg"] < 100.0


# ---------- incident_state.py ----------

def test_incident_state_acknowledge_and_persist():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        store = IncidentStateStore(state_path=path)

        assert not store.is_acknowledged("alert-1")
        store.acknowledge("alert-1")
        assert store.is_acknowledged("alert-1")

        # Reload from disk - must survive a restart, not just live in memory
        store2 = IncidentStateStore(state_path=path)
        assert store2.is_acknowledged("alert-1")


def test_incident_state_bulk_acknowledge_and_undo():
    with tempfile.TemporaryDirectory() as tmp:
        store = IncidentStateStore(state_path=os.path.join(tmp, "state.json"))
        store.acknowledge_many(["a1", "a2", "a3"])
        assert store.is_acknowledged("a1")
        assert store.is_acknowledged("a2")
        assert store.is_acknowledged("a3")

        store.unacknowledge("a2")
        assert not store.is_acknowledged("a2")
        assert store.is_acknowledged("a1")  # others unaffected


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

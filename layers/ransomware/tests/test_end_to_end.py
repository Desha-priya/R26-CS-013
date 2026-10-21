"""
test_end_to_end.py
--------------------
The real thing in miniature: the live service watches a folder, a separate
process "encrypts" the files, and we check that Layer 2 detects it, restores
EVERY file (including the ones damaged after the first response), and
writes signed alerts. No real malware is used.
"""

import json
import os
import subprocess
import sys
import time

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ATTACK = r'''
import os, sys, time
d, n, delay = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
time.sleep(1)
for i in range(n):
    p = os.path.join(d, f"doc{i}.txt")
    with open(p, "wb") as f:
        f.write(os.urandom(20000))
    os.replace(p, p + ".locked")
    time.sleep(delay)
'''


# Rename-only attack: contents stay readable (low entropy), only the NAME changes.
# Must be caught by the file-name rule alone.
RENAME_ONLY = r'''
import os, sys, time
d, n, delay = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
time.sleep(1)
for i in range(n):
    p = os.path.join(d, f"doc{i}.txt")
    os.replace(p, p + ".crypted")
    time.sleep(delay)
'''


@pytest.mark.parametrize("script,n_files,delay", [
    (ATTACK, 60, 0.03), (ATTACK, 25, 0.15), (RENAME_ONLY, 30, 0.1)])
def test_attack_is_detected_and_all_files_restored(tmp_path, monkeypatch, script, n_files, delay):
    watch = tmp_path / "watch"
    watch.mkdir()
    cfg_path = tmp_path / "config.yaml"
    here = os.path.join(os.path.dirname(__file__), "..")
    cfg = yaml.safe_load(open(os.path.join(here, "config", "config.yaml")))
    cfg["live_monitor"]["watch_paths"] = [str(watch)]
    cfg["response"]["containment_seconds"] = 20
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setenv("NEURASHIELD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NEURASHIELD_CONFIG", str(cfg_path))
    monkeypatch.setenv("NEURASHIELD_BACKUP_ROOT", str(tmp_path / "bk"))
    monkeypatch.delenv("NEURASHIELD_PLATFORM_URL", raising=False)

    originals = {}
    for i in range(n_files):
        text = (f"Quarterly report {i}. " * 900).encode()
        (watch / f"doc{i}.txt").write_bytes(text)
        originals[f"doc{i}.txt"] = text

    from src.layer2 import Layer2
    layer = Layer2()
    layer.start()
    try:
        assert layer.state == "standby"          # nothing runs before consent
        layer.grant_consent(by="test")
        deadline0 = time.time() + 30
        while time.time() < deadline0 and layer.state != "armed":
            time.sleep(0.5)                      # wait for the baseline backup
        assert layer.state == "armed"
        assert layer.health()["ok"]
        attacker = subprocess.Popen([sys.executable, "-c", script, str(watch), str(n_files), str(delay)])
        attacker.wait(timeout=60)

        deadline = time.time() + 40
        def all_restored():
            return all((watch / n).exists() and (watch / n).read_bytes() == t for n, t in originals.items())
        while time.time() < deadline and not all_restored():
            time.sleep(1)
        assert all_restored(), "some files were not restored"
        assert layer.status()["incidents"] >= 1

        # One attack = ONE incident report that says what was hit and what was restored.
        store = layer.service.incidents
        t1 = time.time() + 40
        while time.time() < t1 and (not store.list() or store.list()[0]["status"] == "responding"):
            time.sleep(1)
        reports = store.list()
        assert len(reports) == 1, f"expected one incident report, got {len(reports)}"
        rep = reports[0]
        assert rep["status"] == "contained", rep["headline"]
        # (If the layer stops the attacker early, fewer than n_files are damaged - that is fine.)
        c = rep["counts"]
        assert c["failed"] == 0 and c["waiting"] == 0 and c["restored"] == c["affected"] >= 5, c
        full = store.get(rep["id"])
        assert full["timeline"] and all(f["restored"] for f in full["files"])
    finally:
        layer.stop()

    alerts_file = tmp_path / "home" / "data" / "simulated_events" / "alert_queue.jsonl"
    alerts = [json.loads(l) for l in alerts_file.read_text().splitlines() if l.strip()]
    assert alerts and all("signature" in a for a in alerts)


def test_attacker_process_is_identified_and_stopped(tmp_path, monkeypatch):
    """A long-running fake ransomware process: the layer must identify it by how much it
    wrote (it never holds a file open for long), kill it, and restore the files."""
    watch = tmp_path / "watch"; watch.mkdir()
    here = os.path.join(os.path.dirname(__file__), "..")
    cfg = yaml.safe_load(open(os.path.join(here, "config", "config.yaml")))
    cfg["live_monitor"]["watch_paths"] = [str(watch)]
    cfg["response"]["containment_seconds"] = 20
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("NEURASHIELD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NEURASHIELD_CONFIG", str(tmp_path / "c.yaml"))
    monkeypatch.setenv("NEURASHIELD_BACKUP_ROOT", str(tmp_path / "bk"))
    monkeypatch.delenv("NEURASHIELD_PLATFORM_URL", raising=False)
    originals = {}
    for i in range(120):
        t = (f"Quarterly report {i}. " * 900).encode()
        (watch / f"doc{i}.txt").write_bytes(t); originals[f"doc{i}.txt"] = t

    from src.layer2 import Layer2
    layer = Layer2(); layer.start()
    attacker = None
    try:
        layer.grant_consent(by="test")
        t0 = time.time()
        while time.time() - t0 < 40 and layer.state != "armed":
            time.sleep(0.5)
        assert layer.state == "armed"
        time.sleep(11)          # idle baseline of per-process write counters
        attacker = subprocess.Popen([sys.executable, "-c", ATTACK, str(watch), "120", "0.08"])
        started = time.time()
        deadline = time.time() + 40
        while time.time() < deadline and attacker.poll() is None:
            time.sleep(0.25)
        alive_for = time.time() - started
        assert attacker.poll() is not None
        t1 = time.time() + 30
        store = layer.service.incidents
        while time.time() < t1 and (not store.list() or store.list()[0]["status"] == "responding"):
            time.sleep(1)
        rep = store.list()[0]
        assert rep["process"]["pid"] == attacker.pid, (rep["process"], "attacker was", attacker.pid)
        assert rep["process_killed"], rep
        # Its script takes about 10.6 s on its own; being stopped early means the layer killed it.
        assert alive_for < 8.5, f"the attacker ran {alive_for:.1f}s: it was not stopped by the layer"
        t2 = time.time() + 30
        while time.time() < t2 and not all((watch / n).exists() and (watch / n).read_bytes() == t
                                           for n, t in originals.items()):
            time.sleep(1)
        assert all((watch / n).exists() and (watch / n).read_bytes() == t for n, t in originals.items())
    finally:
        if attacker is not None and attacker.poll() is None:
            attacker.kill()
        layer.stop()

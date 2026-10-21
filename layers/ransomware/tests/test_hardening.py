"""
test_hardening.py
-------------------
Tests for the production-hardening work: rollback-everything, protected
processes, command replay protection, platform client, outbox, log
sanitizer, model integrity, paths, HMAC key, ignore list and the dashboard
security rules.   Run:  python -m pytest tests -q
"""

import json
import os
import shutil
import subprocess
import sys
import time

import psutil
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.alerting.platform_client import PlatformClient
from src.response.command_listener import CommandListener, build_rollback_command
from src.response.process_killer import ProcessKiller, summarize_kill
from src.response.rollback_engine import RollbackEngine
from src.utils import paths
from src.utils.logger import sanitize_for_log

CLEAN = b"This is clean readable document text. " * 200


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURASHIELD_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NEURASHIELD_HMAC_SECRET", raising=False)
    monkeypatch.delenv("NEURASHIELD_PLATFORM_URL", raising=False)
    monkeypatch.delenv("NEURASHIELD_API_KEY", raising=False)
    return tmp_path / "home"


def make_engine(tmp_path):
    return RollbackEngine(backup_root=str(tmp_path / "bk"))


# ------------------------------------------------------------------ rollback

def test_find_damaged_and_restore_all(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    eng = make_engine(tmp_path)
    files = []
    for i in range(5):
        f = folder / f"f{i}.txt"
        f.write_bytes(CLEAN)
        eng.backup_file(str(f))
        files.append(str(f))
    for f in files[:3]:
        open(f, "wb").write(os.urandom(5000))
    os.remove(files[3])
    damaged = eng.find_damaged([str(folder)])
    assert sorted(damaged) == sorted(files[:4])
    result = eng.restore_all(damaged)
    assert result["failed"] == 0
    assert all(open(f, "rb").read() == CLEAN for f in files[:4])


def test_clean_backup_is_not_overwritten_by_encrypted_file(tmp_path):
    eng = make_engine(tmp_path)
    f = tmp_path / "a.txt"
    f.write_bytes(CLEAN)
    eng.backup_file(str(f))
    f.write_bytes(os.urandom(8000))
    res = eng.backup_file(str(f))
    assert res["status"] == "skipped" and "entropy" in res["reason"]
    restored = eng.restore_file(str(f))
    assert restored["status"] == "ok", restored
    assert f.read_bytes() == CLEAN


def test_restore_uses_version_before_incident(tmp_path):
    eng = make_engine(tmp_path)
    f = tmp_path / "b.txt"
    f.write_bytes(CLEAN)
    eng.backup_file(str(f))
    time.sleep(0.05)
    cutoff = time.time()
    time.sleep(0.05)
    f.write_bytes(CLEAN + b" edited after the cutoff " * 50)
    eng.backup_file(str(f))
    f.write_bytes(os.urandom(5000))
    assert eng.restore_file(str(f), before_epoch=cutoff)["status"] == "ok"
    assert f.read_bytes() == CLEAN


# ------------------------------------------------------------------- killer

def test_killer_refuses_itself_and_parent():
    k = ProcessKiller()
    assert k.kill(os.getpid())["status"] == "refused_protected"
    assert k.kill(os.getppid())["status"] in ("refused_protected", "already_gone")


def test_killer_extra_protected_name():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        name = psutil.Process(p.pid).name().lower()
        k = ProcessKiller(extra_protected=[name.upper()])
        assert k.kill(p.pid)["status"] == "refused_protected"
        assert p.poll() is None
    finally:
        p.kill()
        p.wait()


def test_killer_pid_reuse_check_and_tree_kill():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        wrong = psutil.Process(p.pid).create_time() - 1000
        assert ProcessKiller().kill(p.pid, expected_create_time=wrong)["status"] == "refused_pid_reused"
        assert p.poll() is None
        results = ProcessKiller(grace_period_seconds=2).kill_process_tree(p.pid, psutil.Process(p.pid).create_time())
        assert summarize_kill(results)["killed"]
        p.wait(timeout=5)
    finally:
        if p.poll() is None:
            p.kill()


# ------------------------------------------------------------------ commands

def make_listener(tmp_path):
    eng = make_engine(tmp_path)
    q = str(tmp_path / "cmd.jsonl")
    return eng, q, CommandListener(eng, queue_path=q)


def _prepared_file(tmp_path, eng):
    f = tmp_path / "c.txt"
    f.write_bytes(CLEAN)
    eng.backup_file(str(f))
    f.write_bytes(os.urandom(4000))
    return f


def _send(q, cmd):
    with open(q, "a") as fh:
        fh.write(json.dumps(cmd) + "\n")


def test_expired_command_rejected(tmp_path):
    eng, q, lis = make_listener(tmp_path)
    f = _prepared_file(tmp_path, eng)
    cmd = build_rollback_command([str(f)], lis._secret)
    cmd["issued_at"] = time.time() - 3600
    from src.alerting.hmac_auth import sign_payload
    cmd.pop("signature")
    cmd["signature"] = sign_payload(cmd, lis._secret)
    _send(q, cmd)
    lis._poll_once()
    assert f.read_bytes() != CLEAN


def test_replay_after_restart_rejected(tmp_path):
    eng, q, lis = make_listener(tmp_path)
    f = _prepared_file(tmp_path, eng)
    cmd = build_rollback_command([str(f)], lis._secret)
    _send(q, cmd)
    lis._poll_once()
    assert f.read_bytes() == CLEAN
    f.write_bytes(os.urandom(4000))
    lis2 = CommandListener(eng, queue_path=q)       # simulated restart
    lis2._poll_once()
    assert f.read_bytes() != CLEAN                  # replayed command did nothing


def test_bad_target_list_rejected(tmp_path):
    eng, q, lis = make_listener(tmp_path)
    for bad in ([], "notalist", [1, 2], ["x"] * 2000):
        _send(q, build_rollback_command(bad, lis._secret))
    lis._poll_once()                                # must not raise


# ------------------------------------------------------------------ platform

def test_platform_client_no_key_and_bad_url(home):
    c = PlatformClient("http://localhost:9")
    assert c.send_threat_alert("x", "HIGH")["reason"] == "no_api_key"
    for bad in ("ftp://host", "javascript:alert(1)", "", "localhost:8000"):
        with pytest.raises(ValueError):
            PlatformClient(bad)


def test_platform_client_invalid_severity_and_unreachable(home):
    c = PlatformClient("http://127.0.0.1:9", timeout_seconds=1, api_key="k")
    assert c.send_threat_alert("x", "BOGUS")["reason"] == "invalid_severity"
    r = c.send_threat_alert("x", "HIGH")
    assert r["status"] == "failed" and r["retryable"] is True


def test_alert_outbox_keeps_and_retries(home, monkeypatch):
    monkeypatch.setenv("NEURASHIELD_PLATFORM_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("NEURASHIELD_API_KEY", "k")
    from src.alerting.alert_generator import AlertGenerator
    gen = AlertGenerator(queue_path=str(home / "q.jsonl"))
    gen.send_alert("ransomware_activity", 0.99, ["/x"], {"pid": 1, "name": "a"}, {}, {})
    assert gen.outbox_size() == 1
    gen.flush_outbox()
    assert gen.outbox_size() == 1                   # platform still down: kept


# --------------------------------------------------------------- misc helpers

def test_log_sanitizer_blocks_injection():
    out = sanitize_for_log("evil\nFAKE LOG LINE\r\x1b[31m" + "A" * 1000)
    assert "\n" not in out and "\r" not in out and "\x1b" not in out
    assert len(out) <= 320


def test_model_integrity_detects_tampering(tmp_path):
    from src.detection import model_integrity as mi
    cfg = paths.load_config()
    assert all(ok for _, ok, _ in mi.verify_all(cfg))
    victim = tmp_path / "scaler.pkl"
    shutil.copy(paths.resource("models/scaler.pkl"), victim)
    with open(victim, "ab") as f:
        f.write(b"x")
    ok, _ = mi.check_file(str(victim))
    assert not ok
    with pytest.raises(mi.ModelIntegrityError):
        mi.verified_path(str(victim))


def test_paths_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURASHIELD_HOME", str(tmp_path / "h"))
    assert paths.runtime("a/b.txt") == str(tmp_path / "h" / "a" / "b.txt")
    assert os.path.isdir(tmp_path / "h" / "a")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("x: 1\n")
    monkeypatch.setenv("NEURASHIELD_CONFIG", str(cfg))
    assert paths.load_config() == {"x": 1}


def test_hmac_key_generated_once_and_env_wins(home, monkeypatch):
    from src.alerting.hmac_auth import get_secret_key
    k1 = get_secret_key({})
    assert len(k1) >= 32 and get_secret_key({}) == k1
    assert (home / "secret.key").exists()
    monkeypatch.setenv("NEURASHIELD_HMAC_SECRET", "x" * 40)
    assert get_secret_key({}) == b"x" * 40


def test_watcher_ignores_own_files(tmp_path):
    from src.monitor.file_watcher import is_ignored_path, build_ignore_dirs
    dirs = build_ignore_dirs([str(tmp_path / "skip")])
    assert is_ignored_path(str(tmp_path / "skip" / "a.txt"), dirs)
    assert is_ignored_path("/x/__pycache__/m.pyc", dirs)
    assert is_ignored_path("/x/a.txt.ns_restore_tmp", dirs)
    assert not is_ignored_path("/x/report.docx", dirs)


# ----------------------------------------------------------------- dashboard

@pytest.fixture
def client(home, monkeypatch):
    monkeypatch.setenv("NEURASHIELD_NO_AUTO_APP", "1")
    from src.dashboard.app import create_app
    app = create_app(alert_queue_path=str(home / "alerts.jsonl"),
                     command_queue_path=str(home / "cmds.jsonl"),
                     status_path=str(home / "status.json"),
                     state_path=str(home / "state.json"))
    return app.test_client(), home


H = {"X-Requested-With": "NeuraShield"}


def test_dashboard_security_headers_and_host_check(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    csp = r.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert b"fonts.googleapis" not in r.data and b"onclick" not in r.data
    assert c.get("/", headers={"Host": "evil.example"}).status_code == 400


def test_dashboard_post_needs_header_and_same_origin(client):
    c, _ = client
    assert c.post("/api/alerts/clear_all").status_code == 403
    assert c.post("/api/alerts/clear_all", headers={**H, "Origin": "http://evil.example"}).status_code == 403
    assert c.post("/api/alerts/clear_all", headers=H).status_code == 200


def test_dashboard_validates_ids_and_rollback_files(client):
    c, _ = client
    assert c.post("/api/alerts/a;b/acknowledge", headers=H).status_code == 400
    assert c.post("/api/approve_rollback", json={"files": ["/etc/passwd"]}, headers=H).status_code == 400
    assert c.post("/api/approve_rollback", json={"files": "x"}, headers=H).status_code == 400
    assert c.post("/api/approve_rollback", json=None, headers=H).status_code == 400


def test_dashboard_alert_roundtrip_verified_and_approve(client):
    c, home = client
    from src.alerting.alert_generator import AlertGenerator
    gen = AlertGenerator(queue_path=str(home / "alerts.jsonl"))
    gen.send_alert("ransomware_activity", 0.8, ["/tmp/<img src=x onerror=alert(1)>.txt"],
                   {"pid": 5, "name": "<b>evil</b>"}, {}, {"kill_skipped_reason": "low"})
    alerts = c.get("/api/alerts").json
    assert len(alerts) == 1 and alerts[0]["verified"] is True
    path = alerts[0]["affected_files"]["paths"][0]
    assert c.post("/api/approve_rollback", json={"files": [path]}, headers=H).json["status"] == "sent"
    aid = alerts[0]["alert_id"]
    assert c.post(f"/api/alerts/{aid}/acknowledge", headers=H).status_code == 200
    assert c.get("/api/alerts?show_acknowledged=false").json == []


def test_dashboard_script_escapes_output():
    js = open(paths.resource("src/dashboard/static/script.js"), encoding="utf-8").read()
    assert "function esc(" in js and "${esc(proc.name" in js
    assert "${a.process.name" not in js


def test_rapid_backups_never_share_a_version_file(tmp_path):
    """Regression: two backups inside one millisecond must not overwrite/delete each other."""
    eng = make_engine(tmp_path)
    f = tmp_path / "r.txt"
    for i in range(30):
        f.write_bytes(CLEAN + str(i).encode() * (i + 1))
        eng.backup_file(str(f))
        f.write_bytes(os.urandom(8000))
        eng.backup_file(str(f))            # guarded: discarded, must not touch the clean copy
    assert eng.restore_file(str(f))["status"] == "ok"
    for v in eng._index[os.path.normcase(os.path.abspath(str(f)))]:
        assert os.path.exists(v["version_path"])


def test_bulk_backup_does_not_slow_down_with_many_files(tmp_path):
    """Regression: every backup used to re-check every stored version on disk
    (quadratic). 1500 files must finish quickly and the status call must not block."""
    eng = make_engine(tmp_path)
    folder = tmp_path / "many"
    folder.mkdir()
    for i in range(1500):
        (folder / f"f{i}.txt").write_text(f"file {i} " * 50)
    start = time.time()
    for p in folder.iterdir():
        eng.backup_file(str(p), save=False)
    eng.flush()
    assert time.time() - start < 20
    t = time.time()
    stats = eng.stats()
    assert time.time() - t < 0.5 and stats["files_tracked"] == 1500
    # second pass: unchanged files are skipped without being copied again
    start = time.time()
    results = [eng.backup_file(str(p), save=False)["status"] for p in folder.iterdir()]
    assert set(results) == {"unchanged"} and time.time() - start < 5


def test_one_scan_finds_process_with_open_file(tmp_path):
    from src.monitor.process_tracker import ProcessTracker
    folder = tmp_path / "watched"
    folder.mkdir()
    handle = open(folder / "held.txt", "w")
    try:
        found = ProcessTracker().find_processes_touching_paths([str(folder)])
        assert os.getpid() in found[os.path.normcase(os.path.abspath(str(folder)))]
    finally:
        handle.close()


def test_pid_cache_only_scans_when_files_are_changing(tmp_path):
    from src.monitor.process_tracker import ProcessTracker, RollingPidCache
    calls = []

    class Spy(ProcessTracker):
        def find_processes_touching_paths(self, *a, **k):
            calls.append(1)
            return {}

    busy = {"v": False}
    cache = RollingPidCache(Spy(), [str(tmp_path)], activity_probe=lambda: busy["v"])
    cache._sample_once()
    assert calls == []                  # idle: no process scan at all
    busy["v"] = True
    cache._sample_once()
    assert calls == [1]

"""
test_consent.py
-----------------
Layer 2 must not read, watch or copy any file until protection has been
agreed to; consent is remembered; restarts do not re-copy unchanged files;
the dashboard shows what is going on.
"""

import os
import sys
import time

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HERE = os.path.dirname(__file__)


@pytest.fixture
def env(tmp_path, monkeypatch):
    watch = tmp_path / "watch"
    watch.mkdir()
    for i in range(12):
        (watch / f"f{i}.txt").write_text(f"document {i} " * 300)
    cfg = yaml.safe_load(open(os.path.join(HERE, "..", "config", "config.yaml")))
    cfg["live_monitor"]["watch_paths"] = [str(watch)]
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("NEURASHIELD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NEURASHIELD_CONFIG", str(cfg_path))
    monkeypatch.setenv("NEURASHIELD_BACKUP_ROOT", str(tmp_path / "bk"))
    monkeypatch.delenv("NEURASHIELD_AUTO_CONSENT", raising=False)
    monkeypatch.delenv("NEURASHIELD_PLATFORM_URL", raising=False)
    return tmp_path, watch, cfg_path


def wait_for(cond, seconds=30):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.3)
    return False


def backup_files(tmp_path):
    bk = tmp_path / "bk"
    return [p for p in bk.rglob("*.bak")] if bk.exists() else []


def test_standby_touches_nothing_until_consent(env):
    tmp_path, watch, _ = env
    from src.layer2 import Layer2
    layer = Layer2()
    layer.start()
    try:
        time.sleep(4)
        assert layer.state == "standby"
        assert layer.service.observer is None and layer.service.backup_scheduler is None
        assert backup_files(tmp_path) == []
        h = layer.health()
        assert h["ok"] and h["protection_active"] is False
    finally:
        layer.stop()


def test_consent_starts_baseline_and_is_remembered_across_restart(env):
    tmp_path, watch, _ = env
    from src.layer2 import Layer2
    layer = Layer2()
    layer.start()
    try:
        layer.grant_consent(by="test")
        assert wait_for(lambda: layer.state == "armed")
        assert len(backup_files(tmp_path)) == 12
    finally:
        layer.stop()

    layer2 = Layer2()                     # restart: no new consent needed
    layer2.start()
    try:
        assert wait_for(lambda: layer2.state == "armed")
        # unchanged files are recognised without being copied again
        assert layer2.service.backup_scheduler.progress["copied"] == 0
        assert len(backup_files(tmp_path)) == 12
    finally:
        layer2.stop()


def test_only_changed_files_are_copied_after_baseline(env):
    tmp_path, watch, _ = env
    from src.layer2 import Layer2
    layer = Layer2()
    layer.start()
    try:
        layer.grant_consent(by="test")
        assert wait_for(lambda: layer.state == "armed")
        (watch / "f3.txt").write_text("edited text " * 500)
        assert wait_for(lambda: len(backup_files(tmp_path)) == 13, 15)
    finally:
        layer.stop()


def test_changed_folders_need_new_consent(env, monkeypatch):
    tmp_path, watch, cfg_path = env
    from src.layer2 import Layer2
    layer = Layer2()
    layer.start()
    layer.grant_consent(by="test")
    assert wait_for(lambda: layer.state == "armed")
    layer.stop()

    other = tmp_path / "other"
    other.mkdir()
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["live_monitor"]["watch_paths"] = [str(watch), str(other)]
    cfg_path.write_text(yaml.safe_dump(cfg))
    layer2 = Layer2()
    layer2.start()
    try:
        time.sleep(3)
        assert layer2.state == "standby"          # the new folder was never agreed to
    finally:
        layer2.stop()


def test_revoke_stops_protection(env):
    tmp_path, watch, _ = env
    from src.layer2 import Layer2
    layer = Layer2()
    layer.start()
    try:
        layer.grant_consent(by="test")
        assert wait_for(lambda: layer.state == "armed")
        layer.revoke_consent(by="test")
        assert wait_for(lambda: layer.state == "standby", 10)
    finally:
        layer.stop()


def test_automatic_consent_setting(env, monkeypatch):
    tmp_path, watch, _ = env
    monkeypatch.setenv("NEURASHIELD_AUTO_CONSENT", "1")
    from src.layer2 import Layer2
    layer = Layer2()
    layer.start()
    try:
        assert wait_for(lambda: layer.state == "armed")
    finally:
        layer.stop()


def test_dashboard_protection_flow(env, monkeypatch):
    tmp_path, watch, _ = env
    monkeypatch.setenv("NEURASHIELD_NO_AUTO_APP", "1")
    from src.dashboard.app import create_app
    from src.layer2 import Layer2
    app = create_app(alert_queue_path=str(tmp_path / "a.jsonl"), command_queue_path=str(tmp_path / "c.jsonl"))
    c = app.test_client()
    H = {"X-Requested-With": "NeuraShield"}

    c.get("/api/protection")                                   # starts the background file count
    assert wait_for(lambda: c.get("/api/protection").json["scope"] is not None, 10)
    r = c.get("/api/protection").json
    assert r["state"] == "offline" and r["scope"]["files"] == 12 and r["consent"] is None
    assert c.post("/api/protection/consent", json={}, headers=H).status_code == 400      # needs explicit confirm
    assert c.post("/api/protection/consent", json={"confirm": True}).status_code == 403  # needs CSRF header

    layer = Layer2()
    layer.start()
    try:
        assert c.post("/api/protection/consent", json={"confirm": True}, headers=H).json["status"] == "granted"
        assert wait_for(lambda: c.get("/api/protection").json["state"] == "armed")
        p = c.get("/api/protection").json
        assert p["service"]["backup"]["files_protected"] == 12
        assert p["consent"]["covers_current_folders"] is True
        (watch / "new.txt").write_text("hello " * 100)
        assert wait_for(lambda: any(e["name"] == "new.txt" for e in c.get("/api/protection").json["service"]["monitor"]["recent"]), 10)
        assert c.get("/api/status").json["state"] == "armed"
        assert c.post("/api/protection/revoke", headers=H).json["status"] == "revoked"
        assert wait_for(lambda: c.get("/api/protection").json["state"] == "standby", 10)
    finally:
        layer.stop()


def test_integrity_check_catches_a_slow_attack_the_live_monitor_cannot_see(tmp_path, monkeypatch):
    """Four files are encrypted one by one, slowly: far below the live rate limit, so
    no live detection. The integrity check (compares files with clean backups) must still
    find, restore and REPORT them."""
    import yaml
    here = os.path.join(os.path.dirname(__file__), "..")
    watch = tmp_path / "watch"; watch.mkdir()
    cfg = yaml.safe_load(open(os.path.join(here, "config", "config.yaml")))
    cfg["live_monitor"]["watch_paths"] = [str(watch)]
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("NEURASHIELD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NEURASHIELD_CONFIG", str(tmp_path / "c.yaml"))
    monkeypatch.setenv("NEURASHIELD_BACKUP_ROOT", str(tmp_path / "bk"))
    monkeypatch.delenv("NEURASHIELD_PLATFORM_URL", raising=False)
    originals = {}
    for i in range(6):
        t = (f"slow attack victim {i}. " * 700).encode()
        (watch / f"f{i}.txt").write_bytes(t); originals[f"f{i}.txt"] = t

    from src.layer2 import Layer2
    layer = Layer2(); layer.start()
    try:
        layer.grant_consent(by="test")
        t0 = time.time()
        while time.time() - t0 < 30 and layer.state != "armed":
            time.sleep(0.5)
        assert layer.state == "armed"
        for i in range(4):
            (watch / f"f{i}.txt").write_bytes(os.urandom(18000))
            time.sleep(1.5)                      # slow: 0.67 files/s, under the 3/s limit
        svc = layer.service
        assert svc.status()["incidents"] == 0, "the live monitor was not supposed to fire here"
        svc._integrity_check()
        for n, t in originals.items():
            assert (watch / n).read_bytes() == t, n
        reports = svc.incidents.list()
        assert len(reports) == 1 and reports[0]["counts"]["restored"] == 4
        assert "integrity" in reports[0]["text"].lower() or "encrypted-looking" in reports[0]["text"]
    finally:
        layer.stop()


def test_dashboard_incident_and_scan_endpoints(tmp_path):
    from src.dashboard.app import create_app
    from src.response.incident_store import IncidentStore
    store_path = str(tmp_path / "incidents.json")
    st = IncidentStore(store_path)
    i = st.open(started_at=time.time(), severity="HIGH", confidence=0.9, trigger="test trigger", folders=["/x"])
    st.add_files(i, [{"path": "/x/a.txt", "reason": "encrypted"}])
    st.apply_restore(i, {"/x/a.txt": {"status": "ok"}}, 0.1)
    st.close(i, "contained", "done")

    app = create_app(incidents_path=store_path, alert_queue_path=str(tmp_path / "a.jsonl"),
                     command_queue_path=str(tmp_path / "c.jsonl"), state_path=str(tmp_path / "s.json"),
                     status_path=str(tmp_path / "st.json"), layer_status_path=str(tmp_path / "l.json"),
                     consent_path=str(tmp_path / "consent.json"))
    c = app.test_client()
    H = {"X-Requested-With": "NeuraShield"}
    rows = c.get("/api/incidents").get_json()
    assert len(rows) == 1 and rows[0]["counts"]["restored"] == 1 and "restored automatically" in rows[0]["text"]
    detail = c.get(f"/api/incidents/{i}").get_json()
    assert detail["files"][0]["restored"] is True and detail["timeline"]
    assert c.get("/api/incidents/bad id!").status_code in (400, 404)
    assert c.get("/api/incidents/INC-nope").status_code == 404
    # actions that need the running service are refused politely without it
    assert c.post("/api/scan", headers=H).status_code == 503
    assert c.post(f"/api/incidents/{i}/restore", headers=H).status_code == 503
    assert c.post("/api/scan/restore", headers=H).status_code == 503
    # and POSTs still need the CSRF header
    assert c.post("/api/scan").status_code == 403
    assert c.get("/api/stats").get_json()["total_files_restored"] == 1
    # acknowledging hides it
    assert c.post(f"/api/alerts/{i}/acknowledge", headers=H).status_code == 200
    assert c.get("/api/incidents").get_json() == []
    assert len(c.get("/api/incidents?show_acknowledged=true").get_json()) == 1

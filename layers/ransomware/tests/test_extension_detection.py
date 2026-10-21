"""Extension / file-name detection and monitor robustness."""
import os
import time

from src.monitor.feature_extractor import (FileEvent, SlidingWindowFeatureExtractor,
                                           looks_like_ransom_note)
from src.monitor.file_watcher import RansomwareEventHandler
from src.monitor.threshold_detector import is_threat


def _rename_burst(n, new_ext=".locked", base="/data/docs"):
    ex = SlidingWindowFeatureExtractor()
    now = time.time()
    for i in range(n):
        old = f"{base}/file{i}.docx"
        ex.add_event(FileEvent("renamed", old + new_ext, now - 0.01 * (n - i), src_path=old))
    return ex


def test_mass_extension_append_is_a_threat_even_without_entropy():
    f = _rename_burst(12).compute_features()
    assert f["appended_ext_count_10s"] == 12 and f["new_ext_distinct_10s"] == 1
    assert is_threat(f)


def test_few_renames_or_many_different_extensions_are_not():
    assert not is_threat(_rename_burst(5).compute_features())
    ex = SlidingWindowFeatureExtractor()
    now = time.time()
    for i in range(12):   # 12 different new extensions: ordinary file conversion, not a family marker
        old = f"/d/f{i}.txt"
        ex.add_event(FileEvent("renamed", f"/d/f{i}.ext{i}", now, src_path=old))
    assert not is_threat(ex.compute_features())


def test_ransom_note_names():
    assert looks_like_ransom_note("C:/x/HOW_TO_DECRYPT_FILES.txt")
    assert looks_like_ransom_note("/x/!!!READ_ME.html")
    assert not looks_like_ransom_note("/x/readme.txt")
    assert not looks_like_ransom_note("/x/decrypt_tool.exe")


def test_note_plus_encrypted_writes_is_a_threat():
    ex = SlidingWindowFeatureExtractor()
    now = time.time()
    ex.add_event(FileEvent("created", "/d/HOW_TO_DECRYPT.txt", now, entropy=4.0))
    for i in range(3):
        ex.add_event(FileEvent("modified", f"/d/a{i}.bin", now, entropy=7.9))
    assert is_threat(ex.compute_features())


def test_handler_records_old_name_and_survives_errors(tmp_path):
    ex = SlidingWindowFeatureExtractor()
    h = RansomwareEventHandler(ex, ignore_dirs=[])
    new = tmp_path / "a.docx.locked"
    new.write_bytes(os.urandom(2048))
    h._handle("renamed", str(new), str(tmp_path / "a.docx"))
    ev = ex.snapshot_events()[-1]
    assert ev.src_path.endswith("a.docx") and ev.entropy > 7
    # a failure inside processing must not escape (it would kill the observer thread)
    h.on_features_ready = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    h._handle("modified", str(new))
    assert h.errors == 1


def test_extractor_is_thread_safe_while_reading():
    import threading
    ex = SlidingWindowFeatureExtractor()
    stop = threading.Event()
    def writer():
        while not stop.is_set():
            ex.add_event(FileEvent("modified", "/d/x", time.time(), entropy=7.9))
    t = threading.Thread(target=writer); t.start()
    try:
        for _ in range(300):
            ex.compute_features(); ex.snapshot_events()
    finally:
        stop.set(); t.join()


def test_watchdog_restarts_dead_observer_and_status_has_meter(tmp_path, monkeypatch):
    import yaml
    here = os.path.join(os.path.dirname(__file__), "..")
    watch = tmp_path / "w"; watch.mkdir()
    cfg = yaml.safe_load(open(os.path.join(here, "config", "config.yaml")))
    cfg["live_monitor"]["watch_paths"] = [str(watch)]
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("NEURASHIELD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NEURASHIELD_CONFIG", str(tmp_path / "c.yaml"))
    monkeypatch.setenv("NEURASHIELD_BACKUP_ROOT", str(tmp_path / "bk"))
    monkeypatch.delenv("NEURASHIELD_PLATFORM_URL", raising=False)
    from src.layer2 import Layer2
    layer = Layer2()
    layer.start()
    try:
        layer.grant_consent(by="test")
        svc = layer.service
        t0 = time.time()
        while time.time() - t0 < 15 and svc.state not in ("armed", "baseline"):
            time.sleep(0.2)
        svc.observer.stop(); svc.observer.join(timeout=5)       # simulate the observer dying
        assert not svc.observer.is_alive()
        t0 = time.time()
        while time.time() - t0 < 10 and svc.observer_restarts == 0:
            time.sleep(0.3)
        assert svc.observer_restarts >= 1 and svc.observer.is_alive()
        (watch / "new.txt").write_text("hello " * 50)
        t0 = time.time()
        while time.time() - t0 < 10 and svc.snapshot()["monitor"]["events_total"] == 0:
            time.sleep(0.3)
        snap = svc.snapshot()["monitor"]
        assert snap["events_total"] > 0 and snap["observer_alive"] and "file_rate_5s" in snap["meter"]
    finally:
        layer.stop()

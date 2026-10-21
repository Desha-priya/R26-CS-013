"""Guards that keep encrypted copies out of the backup store."""
import io
import os
import time
import zipfile

from src.response.rollback_engine import RollbackEngine, header_status


def _docx(path, text="hello world " * 400):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", text)


def test_header_status(tmp_path):
    good = tmp_path / "a.docx"; _docx(good)
    bad = tmp_path / "b.docx"; bad.write_bytes(os.urandom(4000))
    txt = tmp_path / "c.txt"; txt.write_text("x" * 100)
    assert header_status(str(good)) is True
    assert header_status(str(bad)) is False
    assert header_status(str(txt)) is None


def test_encrypted_office_file_is_not_saved_and_is_found_and_restored(tmp_path):
    """The docx is compressed (entropy ~7.9) so the entropy guard alone cannot see
    the encryption. The format check must."""
    eng = RollbackEngine(backup_root=str(tmp_path / "bk"))
    f = tmp_path / "work" / "thesis.docx"; f.parent.mkdir()
    _docx(f)
    original = f.read_bytes()
    assert eng.backup_file(str(f))["status"] == "ok"

    f.write_bytes(os.urandom(len(original)))            # "encrypted" in place
    r = eng.backup_file(str(f))
    assert r["status"] == "skipped" and r["reason"] == "format_broken_protected"
    assert eng.guard_blocks == 1

    dmg = eng.find_damaged_detailed([str(f.parent)])
    assert dmg == [{"path": os.path.abspath(str(f)), "reason": "format"}]
    res = eng.restore_all([d["path"] for d in dmg])
    assert res["succeeded"] == 1 and f.read_bytes() == original


def test_encrypted_rename_copy_is_not_backed_up(tmp_path):
    eng = RollbackEngine(backup_root=str(tmp_path / "bk"))
    d = tmp_path / "w"; d.mkdir()
    f = d / "report.txt"; f.write_text("quarterly " * 500)
    eng.backup_file(str(f))
    enc = d / "report.txt.locked"; enc.write_bytes(os.urandom(5000))
    assert eng.backup_file(str(enc))["reason"] == "encrypted_rename_protected"
    other = d / "photo.jpg.zzzzz"; other.write_bytes(os.urandom(5000))      # known ransomware extension
    assert eng.backup_file(str(other))["reason"] == "encrypted_rename_protected"
    ok = d / "new_notes.txt"; ok.write_text("fine " * 200)
    assert eng.backup_file(str(ok))["status"] == "ok"


def test_damage_reasons(tmp_path):
    eng = RollbackEngine(backup_root=str(tmp_path / "bk"))
    d = tmp_path / "w"; d.mkdir()
    a, b, c = d / "a.txt", d / "b.txt", d / "c.txt"
    for p in (a, b, c):
        p.write_text("clean text " * 300); eng.backup_file(str(p))
    a.unlink(); b.write_bytes(b""); c.write_bytes(os.urandom(4000))
    reasons = {os.path.basename(x["path"]): x["reason"] for x in eng.find_damaged_detailed([str(d)])}
    assert reasons == {"a.txt": "missing", "b.txt": "emptied", "c.txt": "encrypted"}


def test_stats_report_last_backup_and_blocks(tmp_path):
    eng = RollbackEngine(backup_root=str(tmp_path / "bk"))
    f = tmp_path / "x.txt"; f.write_text("data " * 300)
    t0 = time.time(); eng.backup_file(str(f))
    st = eng.stats()
    assert st["last_backup_at"] >= t0 and st["guard_blocks"] == 0


def test_scheduler_waits_for_quiet_and_spaces_backups(tmp_path):
    import main as m
    eng = RollbackEngine(backup_root=str(tmp_path / "bk"))
    f = tmp_path / "doc.txt"; f.write_text("v1 " * 300)
    s = m.BackupScheduler(eng, [str(tmp_path)], interval_seconds=1, ignore_dirs=[],
                          settle_seconds=0.4, min_interval_seconds=1.2)
    s.mark_dirty(str(f)); s._dirty_pass()
    assert not eng.has_backup(str(f)), "must wait until the file has been quiet"
    time.sleep(0.5); s._dirty_pass()
    assert eng.has_backup(str(f)) and s.progress["changes_backed_up"] == 1

    f.write_text("v2 " * 300); s.mark_dirty(str(f)); time.sleep(0.5); s._dirty_pass()
    assert s.progress["changes_backed_up"] == 1, "min interval: same file not backed up again yet"
    assert s.progress["pending"] == 1
    time.sleep(0.9); s._dirty_pass()
    assert s.progress["changes_backed_up"] == 2 and s.progress["pending"] == 0
    assert eng.stats()["versions"] == 2

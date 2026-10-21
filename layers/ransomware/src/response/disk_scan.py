"""
disk_scan.py
--------------
On-demand scan of the protected folders ("Scan now" in the dashboard).

It looks for traces of ransomware that are already on disk:
  - protected files that are now missing, emptied, in a destroyed file format,
    or turned into random-looking data (compared with their clean backup)
  - encrypted copies with well-known ransomware extensions (report.docx.locked)
  - ransom notes (HOW_TO_DECRYPT.txt and similar)
and says, for each finding, whether it can be restored from a backup.

It only READS files (and the backup index). It changes nothing; restoring is a
separate, explicit action.
"""

import os
import time

from src.monitor.feature_extractor import looks_like_ransom_note
from src.response.rollback_engine import RANSOM_EXTENSIONS

MAX_FINDINGS = 500


def scan_folders(rollback, folders, ignore_dirs=(), max_seconds: float = 90.0, stop_flag=None) -> dict:
    started = time.time()
    findings, seen = [], set()
    scanned = 0
    truncated = False

    def add(path, kind, detail, restorable_path=None):
        if path in seen or len(findings) >= MAX_FINDINGS:
            return
        seen.add(path)
        target = restorable_path or path
        leftover = kind in ("ransom note", "encrypted copy")     # nothing to restore: just evidence
        findings.append({"path": path, "name": os.path.basename(path), "kind": kind, "detail": detail,
                         "leftover": leftover,
                         "restorable": False if leftover else bool(rollback.has_backup(target)),
                         "restore_path": target})

    # 1. Protected files that no longer match their clean backup.
    for d in rollback.find_damaged_detailed(folders, deep=True):
        kind = {"missing": "missing", "emptied": "emptied", "format": "format destroyed",
                "encrypted": "encrypted"}[d["reason"]]
        add(d["path"], kind, {"missing": "the file is gone (deleted or renamed)",
                              "emptied": "the file is now empty",
                              "format": "the file no longer has the signature its type requires",
                              "encrypted": "the content turned into random-looking data"}[d["reason"]])

    # 2. Ransomware leftovers.
    ignore = [os.path.normcase(os.path.abspath(i)) for i in ignore_dirs]
    for base in folders:
        if not os.path.isdir(base):
            continue
        for root, dirs, names in os.walk(base):
            norm = os.path.normcase(os.path.abspath(root))
            if any(norm == i or norm.startswith(i + os.sep) for i in ignore):
                dirs[:] = []
                continue
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for name in names:
                scanned += 1
                path = os.path.join(root, name)
                ext = os.path.splitext(name)[1].lower()
                if looks_like_ransom_note(path):
                    add(path, "ransom note", "file name looks like a ransom note")
                elif ext in RANSOM_EXTENSIONS:
                    add(path, "encrypted copy", f"has the ransomware-style extension {ext}")
            if time.time() - started > max_seconds or (stop_flag is not None and stop_flag()):
                truncated = True
                break
        if truncated:
            break

    order = {"encrypted": 0, "format destroyed": 1, "missing": 2, "emptied": 3, "encrypted copy": 4, "ransom note": 5}
    findings.sort(key=lambda f: (order.get(f["kind"], 9), f["path"]))
    damaged = [f for f in findings if f["kind"] in ("encrypted", "format destroyed", "missing", "emptied")]
    return {
        "finished_at": time.time(), "seconds": round(time.time() - started, 2), "files_scanned": scanned,
        "truncated": truncated, "findings": findings,
        "verdict": "clean" if not findings else ("damage_found" if damaged else "leftovers_found"),
        "restorable": sum(1 for f in damaged if f["restorable"]),
    }

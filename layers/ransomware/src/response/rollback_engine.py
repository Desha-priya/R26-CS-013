"""
rollback_engine.py
---------------------
Versioned backup + rollback engine.

Backups are plain file copies in a hidden, versioned folder with a
SHA-256 checksum for each copy. They are NOT encrypted at rest - the
protection is (a) the hidden, non-obvious location, and (b) the checksum,
which is verified before every restore so a tampered backup is refused.

What makes this engine safe against ransomware (added after review):

1. CLEAN-BACKUP GUARD. Backups are taken every few seconds. Without a
   guard, the backup of a file that ransomware has just encrypted would
   replace the clean copy as the "latest version". Now, if a file that was
   normal (low entropy) suddenly becomes random-looking (high entropy),
   the new copy is NOT kept as a backup. The clean copy stays.

2. VERSION CHOICE. On restore the engine picks the newest version that is
   (a) clean and (b) taken before the incident started, when the caller
   gives the incident start time. It never blindly restores "the latest".

3. FIND EVERY DAMAGED FILE. find_damaged() scans the backup index for files
   in the affected folders that are now missing, emptied, or turned into
   random-looking data, so ALL damaged files are restored - not only the
   last file that triggered the alert.

4. SAFE WRITES. The index is written atomically (temp file + replace),
   protected by a lock because the backup thread, the detection thread and
   the command listener all use it, and restores go through a temp file so a
   crash never leaves a half-written document.

5. FORMAT CHECK (added after review). Office files, PDFs, images and archives
   are ALWAYS random-looking (they are compressed), so the entropy check alone
   cannot tell a normal .docx from an encrypted one. Every well-known file type
   starts with a fixed signature ("magic bytes"); ransomware destroys it. If a
   file that used to have a valid signature no longer has one, the new copy is
   NOT kept as a backup and the file is treated as damaged.

6. RENAME CHECK. Many ransomware families encrypt "a.docx" and rename it to
   "a.docx.locked". A NEW file whose name is a protected file's name plus an
   extra extension, or which carries a well-known ransomware extension, is not
   backed up (it would only fill the backup store with encrypted copies).

Responsibilities:
- Keep up to N versions per file (max_versions) and a global size cap.
- Restore on request, verifying the checksum first.
"""

import hashlib
import json
import os
import shutil
import threading
import time
import uuid

from src.monitor.feature_extractor import file_entropy
from src.utils import paths
from src.utils.logger import get_logger, sanitize_for_log

logger = get_logger(__name__)

# A file below CLEAN_ENTROPY_MAX counts as "normal data" (text, documents).
# A file at or above SUSPICIOUS_ENTROPY_MIN looks random (encrypted/compressed).
CLEAN_ENTROPY_MAX = 7.2
SUSPICIOUS_ENTROPY_MIN = 7.5


# File signatures (first bytes) of common formats. Only types whose signature is
# fixed and reliable are listed; plain text and unknown types are never judged.
_ZIP = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_OLE = (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",)
MAGIC_BYTES = {
    ".pdf": (b"%PDF",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",), ".jpeg": (b"\xff\xd8\xff",),
    ".gif": (b"GIF87a", b"GIF89a"),
    ".bmp": (b"BM",),
    ".zip": _ZIP, ".docx": _ZIP, ".xlsx": _ZIP, ".pptx": _ZIP,
    ".odt": _ZIP, ".ods": _ZIP, ".odp": _ZIP, ".jar": _ZIP, ".apk": _ZIP,
    ".doc": _OLE, ".xls": _OLE, ".ppt": _OLE, ".msg": _OLE,
    ".7z": (b"7z\xbc\xaf\x27\x1c",),
    ".rar": (b"Rar!\x1a\x07",),
    ".gz": (b"\x1f\x8b",),
    ".rtf": (b"{\\rtf",),
    ".sqlite": (b"SQLite format 3\x00",), ".sqlite3": (b"SQLite format 3\x00",),
    ".exe": (b"MZ",), ".dll": (b"MZ",),
}

# Extensions that well-known ransomware appends or substitutes. Used to avoid
# backing up encrypted copies and by the disk scan. (Deliberately NOT listed:
# ".enc" and ".lock", which ordinary programs also use.)
RANSOM_EXTENSIONS = frozenset((
    ".locked", ".encrypted", ".crypted", ".crypt", ".cry", ".locky", ".cerber", ".zepto", ".odin",
    ".thor", ".aesir", ".zzzzz", ".kraken", ".rapid", ".xtbl", ".crypz", ".cryp1", ".micro",
    ".ecc", ".ezz", ".exx", ".wncry", ".wcry", ".wnry", ".ryk", ".conti", ".lockbit", ".akira",
    ".blackcat", ".djvu", ".royal", ".hive", ".bitlocked", ".ransom", ".fucked", ".r5a",
))


def header_status(path: str):
    """True: the file starts with the signature its extension requires.
    False: its type has a known signature and this file does not have it.
    None: unknown type (nothing to judge) or unreadable."""
    ext = os.path.splitext(path)[1].lower()
    sigs = MAGIC_BYTES.get(ext)
    if not sigs:
        return None
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return None
    if not head:
        return False
    return any(head.startswith(sig) for sig in sigs)


def _key(path: str) -> str:
    """Index key: absolute path, case-folded on Windows so C:\\Docs and c:\\docs match."""
    return os.path.normcase(os.path.abspath(path))


class RollbackEngine:
    def __init__(self, config_path=None, backup_root=None, max_versions=10, max_total_mb=None):
        self.cfg = paths.load_config(config_path)

        # Hidden folder, deliberately not called "Backup"/"Shadow".
        self.backup_root = backup_root or paths.default_backup_root()
        os.makedirs(self.backup_root, exist_ok=True)
        self.max_versions = max_versions
        # Global cap across ALL files (proposal: default 5 GB).
        self.max_total_mb = max_total_mb if max_total_mb is not None else \
            self.cfg.get("response", {}).get("max_backup_storage_mb", 5120)

        self.index_path = os.path.join(self.backup_root, "_index.json")
        self._lock = threading.RLock()
        self.guard_blocks = 0          # copies refused because they looked encrypted
        self.last_backup_at = None     # epoch of the newest stored version
        self._dirty = False
        self._index = self._load_index()
        self._total_bytes = sum(v.get("size", 0) for vs in self._index.values() for v in vs)

    # ------------------------------------------------------------ index file

    def _load_index(self) -> dict:
        if not os.path.exists(self.index_path):
            return {}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # A corrupt index must not crash the tool. Keep the bad file for
            # inspection and start with an empty index.
            bad = self.index_path + f".corrupt-{int(time.time())}"
            try:
                os.replace(self.index_path, bad)
            except OSError:
                pass
            logger.error(f"[rollback] Backup index was unreadable ({sanitize_for_log(e)}); "
                         f"moved aside and starting fresh.")
            return {}

        # Normalise keys (older index files used plain absolute paths).
        index = {}
        for original, versions in raw.items():
            for v in versions:
                v.setdefault("original", original)
            index.setdefault(_key(original), []).extend(versions)
        return index

    def _save_index(self):
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=1)
        os.replace(tmp, self.index_path)

    def flush(self):
        """Writes the index to disk if bulk backups left changes in memory."""
        with self._lock:
            if self._dirty:
                self._save_index()
                self._dirty = False

    # --------------------------------------------------------------- helpers

    def _hashed_dir_for(self, original_path: str) -> str:
        """Non-obvious sub-folder per original file (not security by itself,
        it only keeps the real path out of folder names)."""
        digest = hashlib.sha256(_key(original_path).encode()).hexdigest()[:16]
        d = os.path.join(self.backup_root, digest)
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _sha256_of_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _version_epoch(v: dict) -> float:
        if "epoch" in v:
            return float(v["epoch"])
        try:  # older entries only have "YYYYmmddTHHMMSS"
            return time.mktime(time.strptime(v["timestamp"][:15], "%Y%m%dT%H%M%S"))
        except (KeyError, ValueError):
            return 0.0

    # ------------------------------------------------------------ public API

    @staticmethod
    def _header_of_copy(copy_path: str, original_path: str):
        """header_status() of a stored copy, judged by the ORIGINAL file's extension."""
        sigs = MAGIC_BYTES.get(os.path.splitext(original_path)[1].lower())
        if not sigs:
            return None
        try:
            with open(copy_path, "rb") as f:
                head = f.read(16)
        except OSError:
            return None
        return bool(head) and any(head.startswith(sig) for sig in sigs)

    def _is_encrypted_rename(self, path: str) -> bool:
        """True for a NEW file that looks like ransomware output: it has a well-known
        ransomware extension, or it is the name of an already-protected file plus
        one extra extension (report.docx -> report.docx.locked)."""
        base, ext = os.path.splitext(path)
        if ext.lower() in RANSOM_EXTENSIONS:
            return True
        return bool(ext) and bool(self._index.get(_key(base)))

    def backup_file(self, original_path: str, guard: bool = True, save: bool = True) -> dict:
        """Copies the CURRENT state of a file in as a new version.

        guard=True turns on the clean-backup guard: a copy that turned from
        normal data into random-looking data is discarded (status "skipped").
        save=False leaves the index in memory (call flush() later); used for
        bulk passes so the index file is not rewritten after every single file.

        A file whose modified-time and size match its newest stored version is
        skipped without being read or copied (this is what makes a restart or
        a repeated pass cheap: only really changed files cost anything).

        The slow work (copy, hash, entropy) happens OUTSIDE the lock; the lock
        is held only for the quick index update, so status and restore calls
        are never stuck behind a large file copy.
        """
        try:
            st = os.stat(original_path)
        except OSError:
            return {"status": "skipped", "reason": "file does not exist", "path": original_path}
        if not os.path.isfile(original_path):
            return {"status": "skipped", "reason": "file does not exist", "path": original_path}

        key = _key(original_path)
        existing = self._index.get(key)
        if existing:
            last = existing[-1]
            if last.get("mtime_ns") == st.st_mtime_ns and last.get("src_size") == st.st_size:
                return {"status": "unchanged", "path": original_path}
        elif guard and self._is_encrypted_rename(original_path):
            self.guard_blocks += 1
            logger.warning(f"[rollback] NOT backing up {sanitize_for_log(original_path)}: it looks like an "
                           f"encrypted rename of a protected file (or has a ransomware extension).")
            return {"status": "skipped", "reason": "encrypted_rename_protected", "path": original_path}

        backup_dir = self._hashed_dir_for(original_path)
        tmp_path = os.path.join(backup_dir, f".part-{uuid.uuid4().hex}")
        now = time.time()
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(now)) + f"{int(now * 1000) % 1000:03d}"
        try:
            shutil.copy2(original_path, tmp_path)
            checksum = self._sha256_of_file(tmp_path)
            # Measure the stored copy (not the live file, which may still be changing).
            entropy = file_entropy(tmp_path)
            size = os.path.getsize(tmp_path)
            # The signature is judged on the stored copy, under the ORIGINAL name's extension.
            hdr = self._header_of_copy(tmp_path, original_path)
        except OSError as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            logger.warning(f"[rollback] Could not back up {sanitize_for_log(original_path)}: {sanitize_for_log(e)}")
            return {"status": "skipped", "reason": f"copy failed: {e}", "path": original_path}

        with self._lock:
            versions = self._index.setdefault(key, [])
            if versions:
                last = versions[-1]
                if last.get("checksum") == checksum:
                    os.remove(tmp_path)
                    last["mtime_ns"], last["src_size"] = st.st_mtime_ns, st.st_size
                    self._dirty = True
                    return {"status": "unchanged", "path": original_path}

                if guard and hdr is False and last.get("header_ok") is True:
                    os.remove(tmp_path)
                    self.guard_blocks += 1
                    logger.warning(f"[rollback] KEPT clean backup of {sanitize_for_log(original_path)}: "
                                   f"the file no longer has a valid {os.path.splitext(original_path)[1]} "
                                   f"signature (looks encrypted/corrupted), new copy discarded.")
                    return {"status": "skipped", "reason": "format_broken_protected", "path": original_path}

                last_entropy = last.get("entropy")
                if (guard and entropy is not None and last_entropy is not None
                        and last_entropy < CLEAN_ENTROPY_MAX and entropy >= SUSPICIOUS_ENTROPY_MIN):
                    os.remove(tmp_path)
                    self.guard_blocks += 1
                    logger.warning(f"[rollback] KEPT clean backup of {sanitize_for_log(original_path)}: "
                                   f"file turned random-looking (entropy {last_entropy:.2f} -> {entropy:.2f}), "
                                   f"new copy discarded.")
                    return {"status": "skipped", "reason": "entropy_jump_protected", "path": original_path}

            # Unique final name: two backups of one file inside the same millisecond
            # must never share a name (discarding one would delete the other).
            version_path = os.path.join(backup_dir, f"{os.path.basename(original_path)}.v{stamp}.bak")
            counter = 0
            while os.path.exists(version_path) or any(v.get("version_path") == version_path for v in versions):
                counter += 1
                version_path = os.path.join(backup_dir, f"{os.path.basename(original_path)}.v{stamp}-{counter}.bak")
            os.replace(tmp_path, version_path)

            versions.append({
                "version_path": version_path,
                "timestamp": stamp,
                "epoch": now,
                "checksum": checksum,
                "entropy": entropy,
                "header_ok": hdr,
                "size": size,
                "mtime_ns": st.st_mtime_ns,
                "src_size": st.st_size,
                "original": os.path.abspath(original_path),
            })
            self._total_bytes += size
            self.last_backup_at = now

            while len(versions) > self.max_versions:
                oldest = versions.pop(0)
                self._total_bytes -= oldest.get("size", 0)
                if os.path.exists(oldest["version_path"]):
                    os.remove(oldest["version_path"])

            self._enforce_storage_cap()
            if save:
                self._save_index()
                self._dirty = False
            else:
                self._dirty = True

        logger.debug(f"[rollback] Backed up: {sanitize_for_log(original_path)}")
        return {"status": "ok", "version_path": version_path, "checksum": checksum}

    def _enforce_storage_cap(self):
        """Global size cap across all files: delete the globally-oldest versions
        until the total is under the cap. (Caller holds the lock.)

        The running total (self._total_bytes) is kept up to date as versions are
        added and removed, so this is a single comparison in the normal case and
        never touches the disk unless the cap really is exceeded."""
        max_bytes = self.max_total_mb * 1024 * 1024
        if self._total_bytes <= max_bytes:
            return

        all_versions = []   # (epoch, key, version dict)
        for key, versions in self._index.items():
            for v in versions:
                all_versions.append((self._version_epoch(v), key, v))
        all_versions.sort(key=lambda x: x[0])

        removed = 0
        # Pass 1 drops extra (older) versions of files that have several; pass 2 is
        # only reached if the cap is STILL exceeded and then also removes the oldest
        # single backups, because the configured cap is a hard limit.
        for only_extra_versions in (True, False):
            for _, key, v in all_versions:
                if self._total_bytes <= max_bytes:
                    break
                if v not in self._index.get(key, []):
                    continue
                if only_extra_versions and len(self._index[key]) <= 1:
                    continue
                try:
                    if os.path.exists(v["version_path"]):
                        os.remove(v["version_path"])
                except OSError:
                    continue
                self._index[key].remove(v)
                self._total_bytes -= v.get("size", 0)
                removed += 1

        if removed:
            logger.warning(f"[rollback] Storage cap ({self.max_total_mb} MB) exceeded - "
                           f"pruned {removed} oldest backup version(s).")

    def _choose_version(self, versions: list, version_index: int, before_epoch, prefer_clean: bool):
        """Returns (version or None, explanation if None)."""
        if not versions:
            return None, "no backup available"
        if version_index != -1:
            try:
                return versions[version_index], None
            except IndexError:
                return None, "no such backup version"

        candidates = [v for v in versions if v.get("header_ok") is not False] or list(versions)
        if before_epoch is not None:
            candidates = [v for v in candidates if self._version_epoch(v) <= before_epoch]
            if not candidates:
                return None, "no backup from before the incident"
        if prefer_clean:
            clean = [v for v in candidates
                     if v.get("entropy") is None or v["entropy"] < SUSPICIOUS_ENTROPY_MIN]
            if clean:
                candidates = clean
            # If every version is random-looking (for example a .zip or .jpg),
            # that is normal for this file type: use the newest one.
        return candidates[-1], None

    def restore_file(self, original_path: str, version_index: int = -1,
                     before_epoch: float = None, prefer_clean: bool = True) -> dict:
        """Restores original_path from a backup. The checksum is verified first."""
        with self._lock:
            key = _key(original_path)
            versions = list(self._index.get(key, []))
            target, why_not = self._choose_version(versions, version_index, before_epoch, prefer_clean)
            if target is None:
                logger.warning(f"[rollback] Cannot restore {sanitize_for_log(original_path)}: {why_not}")
                return {"status": "failed", "reason": why_not, "path": original_path}

            version_path = target["version_path"]
            if not os.path.exists(version_path):
                return {"status": "failed", "reason": "backup file missing on disk", "path": original_path}

            if self._sha256_of_file(version_path) != target["checksum"]:
                logger.error(f"[rollback] Checksum mismatch for backup of "
                             f"{sanitize_for_log(original_path)} - possible tampering!")
                return {"status": "failed", "reason": "checksum mismatch (backup integrity compromised)",
                        "path": original_path}

            destination = target.get("original") or os.path.abspath(original_path)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            temp_destination = destination + ".ns_restore_tmp"
            try:
                shutil.copy2(version_path, temp_destination)
                os.replace(temp_destination, destination)
            except OSError as e:
                if os.path.exists(temp_destination):
                    try:
                        os.remove(temp_destination)
                    except OSError:
                        pass
                logger.error(f"[rollback] Restore of {sanitize_for_log(destination)} failed: {sanitize_for_log(e)}")
                return {"status": "failed", "reason": f"write failed: {e}", "path": original_path}

        logger.info(f"[rollback] Restored: {sanitize_for_log(destination)}")
        return {"status": "ok", "restored_from": version_path, "restored_at": time.time()}

    def restore_all(self, file_paths: list, before_epoch: float = None) -> dict:
        """Bulk restore. Returns timing info (Time-to-Recover target: <5s / 100 files)."""
        start = time.time()
        results = {}
        for path in file_paths:
            results[path] = self.restore_file(path, before_epoch=before_epoch)
        elapsed = time.time() - start
        succeeded = sum(1 for r in results.values() if r["status"] == "ok")
        logger.info(f"[rollback] Restored {succeeded}/{len(file_paths)} files in {elapsed:.2f}s")
        return {
            "total_files": len(file_paths),
            "succeeded": succeeded,
            "failed": len(file_paths) - succeeded,
            "elapsed_seconds": round(elapsed, 3),
            "details": results,
        }

    def find_damaged_detailed(self, folders, extra_paths=(), deep: bool = False) -> list:
        """Backed-up files, in the given folders, that look damaged now. Each item is
        {"path": ..., "reason": ...} with reason one of:
          missing  - the file is gone (deleted, or renamed to something else)
          emptied  - the file is now 0 bytes
          format   - it lost the signature its type requires (docx, pdf, jpg, ...)
          encrypted- it turned from normal data into random-looking data
        Used after a detection so every damaged file is restored.

        deep=False (default) skips a file whose modified-time and size still match
        its newest backup: such a file cannot have been rewritten, and reading
        every file again each second would be very slow in a large folder.
        deep=True reads every file (used by the manual disk scan)."""
        wanted = [os.path.normcase(os.path.abspath(f)) for f in folders]
        extra = {_key(p) for p in extra_paths}
        damaged = {}
        with self._lock:
            snapshot = {k: list(v) for k, v in self._index.items() if v}

        for key, versions in snapshot.items():
            latest = versions[-1]
            original = latest.get("original") or key
            folder = os.path.dirname(key)
            in_scope = key in extra or any(folder == w or folder.startswith(w + os.sep) for w in wanted)
            if not in_scope:
                continue

            if not os.path.exists(original):
                damaged[original] = "missing"
                continue
            try:
                size_now = os.path.getsize(original)
            except OSError:
                continue
            if size_now == 0 and latest.get("size", 0) > 0:
                damaged[original] = "emptied"
                continue

            if not deep:
                try:
                    st = os.stat(original)
                    if (latest.get("mtime_ns") == st.st_mtime_ns and latest.get("src_size") == st.st_size):
                        continue
                except OSError:
                    continue

            if header_status(original) is False and any(v.get("header_ok") is True for v in versions):
                damaged[original] = "format"
                continue

            entropy_now = file_entropy(original)
            clean_before = [v for v in versions if v.get("entropy") is not None
                            and v["entropy"] < CLEAN_ENTROPY_MAX]
            if (entropy_now is not None and entropy_now >= SUSPICIOUS_ENTROPY_MIN and clean_before):
                damaged[original] = "encrypted"
        return [{"path": p, "reason": damaged[p]} for p in sorted(damaged)]

    def find_damaged(self, folders, extra_paths=()) -> list:
        """Same as find_damaged_detailed(), paths only."""
        return [d["path"] for d in self.find_damaged_detailed(folders, extra_paths)]

    def has_backup(self, path: str) -> bool:
        with self._lock:
            return bool(self._index.get(_key(path)))

    def stats(self) -> dict:
        """Counts for the status display. Deliberately does NOT take the lock, so
        the dashboard never waits behind a backup in progress."""
        for _ in range(3):
            try:
                values = list(self._index.values())
                return {"files_tracked": len(values), "versions": sum(len(v) for v in values),
                        "backup_root": self.backup_root, "total_mb": round(self._total_bytes / 1048576, 1),
                        "last_backup_at": self.last_backup_at, "guard_blocks": self.guard_blocks}
            except RuntimeError:     # index changed while counting; try again
                continue
        return {"files_tracked": 0, "versions": 0, "backup_root": self.backup_root, "total_mb": 0}


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "important_report.docx")
        with open(test_file, "w") as f:
            f.write("This is my original, clean file content. " * 20)

        engine = RollbackEngine(backup_root=os.path.join(tmp, "_shadow_backups"))
        print("1. Backing up clean file...", engine.backup_file(test_file)["status"])

        with open(test_file, "wb") as f:
            f.write(os.urandom(2000))
        print("2. File encrypted. Backup of the encrypted file is refused:",
              engine.backup_file(test_file)["status"])
        print("3. Damaged files found:", engine.find_damaged([tmp]))
        print("4. Restore:", engine.restore_file(test_file)["status"])
        with open(test_file, "r") as f:
            assert f.read().startswith("This is my original")
        print("SMOKE TEST PASSED")

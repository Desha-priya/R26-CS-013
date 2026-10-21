r"""
reset_service.py  -  where NeuraShield keeps your data, and how to clear it
---------------------------------------------------------------------------
Everything the platform stores about you lives in a few well-known places:

  <root>\data\            console account + audit log + alert history, each layer's
                          state (Zero-Trust behavioural profile and face reference,
                          ransomware state, content and network records), logs,
                          the integrity baseline
  <root>\config\product.json   consent + which layers are on
  %ProgramData%\NeuraShield\secrets   the platform key and signing secret (DPAPI)
  ~\.neurashield_cache_x7f2   ransomware versioned file backups (YOUR files)
  HKCU ...\Run\NeuraShield    the start-with-Windows entry

<root> is the install folder, or %LOCALAPPDATA%\NeuraShield when the exe runs
outside an installed layout, or the project folder in a source run.

Deleting is restricted to those places (every path is checked to be INSIDE one of
them), and the ransomware backups - copies of the user's own files - are only ever
touched when the user explicitly asks for it.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BACKUP_DIR_NAME = ".neurashield_cache_x7f2"


def backup_root() -> Path:
    env = os.environ.get("NEURASHIELD_BACKUP_ROOT")
    return Path(env).expanduser() if env else Path.home() / BACKUP_DIR_NAME


def _inside(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _size(path: Path) -> int:
    total = 0
    try:
        if path.is_file():
            return path.stat().st_size
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


class ResetService:
    def __init__(self, root, secret_dir):
        self.root = Path(root).resolve()
        self.data = self.root / "data"
        self.secret_dir = Path(secret_dir)
        self.allowed = [self.data, self.secret_dir, self.root / "layers" / "content" / "outputs"]

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        bk = backup_root()
        return {
            "root": str(self.root),
            "data_dir": str(self.data), "data_bytes": _size(self.data),
            "config_file": str(self.root / "config" / "product.json"),
            "secrets_dir": str(self.secret_dir), "secrets_bytes": _size(self.secret_dir),
            "backups_dir": str(bk), "backups_exist": bk.exists(), "backups_bytes": _size(bk) if bk.exists() else 0,
        }

    # ---------------------------------------------------------------- delete
    def _rm(self, path: Path, extra_base=None, stats=None) -> None:
        """Delete a file or folder, only if it lies inside an allowed location.
        Files that are locked (e.g. the log this process is writing) are skipped."""
        stats = stats if stats is not None else {"removed": 0, "skipped": 0}
        bases = self.allowed + ([extra_base] if extra_base else [])
        if not path.exists() or not any(_inside(path, b) for b in bases):
            return
        if path.is_file() or path.is_symlink():
            try:
                path.unlink()
                stats["removed"] += 1
            except OSError:
                stats["skipped"] += 1
            return
        for child in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink()
                    stats["removed"] += 1
                elif child.is_dir():
                    child.rmdir()
            except OSError:
                stats["skipped"] += 1
        try:
            path.rmdir()
        except OSError:
            pass

    def clear_alerts(self) -> dict:
        st = {"removed": 0, "skipped": 0}
        for name in ("alert_history.jsonl", "alert_history.jsonl.tmp", "incident_state.json"):
            self._rm(self.data / "console" / name, stats=st)
        # the content layer's own alert files (source runs write beside the layer)
        self._rm(self.root / "layers" / "content" / "outputs" / "alerts", stats=st)
        self._rm(self.data / "content" / "outputs" / "alerts", stats=st)
        return st

    def clear_audit(self) -> dict:
        st = {"removed": 0, "skipped": 0}
        self._rm(self.data / "console" / "console_audit.log", stats=st)
        return st

    def clear_layer_data(self) -> dict:
        """Zero-Trust profile + face reference + database, ransomware state, content
        and network records. Layers must be stopped first."""
        st = {"removed": 0, "skipped": 0}
        for sub in ("core", "content", "network", "ransomware"):
            self._rm(self.data / sub, stats=st)
        return st

    def delete_backups(self) -> dict:
        """The ransomware layer's versioned copies of the user's files."""
        st = {"removed": 0, "skipped": 0}
        bk = backup_root()
        if bk.exists() and (bk.name == BACKUP_DIR_NAME or os.environ.get("NEURASHIELD_BACKUP_ROOT")):
            self._rm(bk, extra_base=bk, stats=st)
        return st

    def wipe_everything(self, keep_logs_open=True) -> dict:
        """Account, audit, history, layer data, integrity baseline, logs, key files and
        the platform secrets - everything except the ransomware backups."""
        st = {"removed": 0, "skipped": 0}
        if self.data.exists():
            for child in list(self.data.iterdir()):
                self._rm(child, stats=st)
        for name in ("api_key.dat", "hmac_secret.dat", ".gw_local_key", "api_key.dat.tmp", "hmac_secret.dat.tmp"):
            self._rm(self.secret_dir / name, stats=st)
        self._rm(self.root / "layers" / "content" / "outputs" / "alerts", stats=st)
        return st

    # --------------------------------------------------------------- product
    def reset_product(self, current: dict) -> dict:
        """product.json back to a first-run state: consent off, defaults, nothing else."""
        from install_root import DEFAULT_PRODUCT
        fresh = json.loads(json.dumps(DEFAULT_PRODUCT))
        for k in ("product_name", "version", "platform_url"):
            if current.get(k):
                fresh[k] = current[k]
        if current.get("console"):
            fresh["console"] = current["console"]
        (self.root / "config" / "product.json").write_text(json.dumps(fresh, indent=2), encoding="utf-8")
        return fresh


def restart_command() -> list:
    """Command line that starts this same program again with the same arguments."""
    if getattr(sys, "frozen", False):
        return [sys.executable] + sys.argv[1:]
    return [sys.executable] + sys.argv


def spawn_restart(delay_seconds: int = 3) -> bool:
    """Start a fresh copy of the app a few seconds from now, detached, so it comes up
    after this process has released the console port and the single-instance lock."""
    if sys.platform != "win32":
        subprocess.Popen(["sh", "-c", f"sleep {int(delay_seconds)}; exec " +
                          " ".join(f"'{a}'" for a in restart_command())], start_new_session=True)
        return True
    quoted = " ".join(f'"{a}"' for a in restart_command())
    cmdline = f'ping -n {int(delay_seconds) + 1} 127.0.0.1 >nul & start "" {quoted}'
    flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
             | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    subprocess.Popen(["cmd", "/c", cmdline], creationflags=flags, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True

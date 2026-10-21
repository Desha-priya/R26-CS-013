"""
process_killer.py
--------------------
Terminates a process identified as ransomware. Tries a graceful
terminate() first, escalates to kill() if it does not exit in time, and
returns a structured result for logging, alerting and the dashboard.

SAFETY RULES (a wrong kill can crash the computer, so these are strict):
  - Protected processes are NEVER killed: core Windows/Linux system
    processes (lsass, csrss, services, svchost, explorer, ...), any process
    whose program file lives in the Windows system folders, PID 0-4,
    PID 1, this program itself and its parent.
    The list can be extended in config.yaml: response.protected_processes
  - PID-reuse check: the caller may pass the process start time it saw
    when it chose the target. If the PID now belongs to a different
    process (the old one exited and the number was reused), nothing is killed.
  - Every refusal is returned as status "refused_protected" or
    "refused_pid_reused" and written to the log.
"""

import os
import time

import psutil

from src.utils.logger import get_logger, sanitize_for_log

logger = get_logger(__name__)

# Lower-case names. A match on the process name is enough to refuse.
DEFAULT_PROTECTED_NAMES = {
    # Windows core
    "system", "system idle process", "registry", "memory compression", "secure system",
    "smss.exe", "csrss.exe", "wininit.exe", "winlogon.exe", "services.exe", "lsass.exe",
    "lsaiso.exe", "svchost.exe", "dwm.exe", "explorer.exe", "fontdrvhost.exe",
    "taskhostw.exe", "sihost.exe", "ctfmon.exe", "searchindexer.exe", "spoolsv.exe",
    "msmpeng.exe", "nissrv.exe", "securityhealthservice.exe", "wmiprvse.exe",
    "audiodg.exe", "conhost.exe", "runtimebroker.exe", "startmenuexperiencehost.exe",
    # Linux / macOS core
    "systemd", "init", "kthreadd", "launchd", "sshd", "dbus-daemon", "networkmanager",
    "xorg", "gnome-shell",
}

# On Windows nothing that runs from these folders is killed automatically.
WINDOWS_SYSTEM_DIR_PREFIXES = ("c:\\windows\\system32", "c:\\windows\\syswow64", "c:\\windows\\winsxs")


class ProcessKiller:
    def __init__(self, grace_period_seconds: float = 2.0, extra_protected=None):
        self.grace_period_seconds = grace_period_seconds
        self.protected_names = set(DEFAULT_PROTECTED_NAMES)
        for name in (extra_protected or []):
            self.protected_names.add(str(name).lower())

    # ------------------------------------------------------------ safety check

    def protection_reason(self, proc: "psutil.Process", name: str):
        """Returns a text reason if this process must not be killed, else None."""
        pid = proc.pid
        if pid in (0, 1, 4):
            return "core system process id"
        if pid == os.getpid():
            return "this program itself"
        if pid == os.getppid():
            return "parent of this program"
        if (name or "").lower() in self.protected_names:
            return f"protected process name ({name})"
        try:
            exe = (proc.exe() or "").lower()
        except (psutil.Error, OSError):
            exe = ""
        if exe and os.name == "nt" and exe.replace("/", "\\").startswith(WINDOWS_SYSTEM_DIR_PREFIXES):
            return "program file is in a Windows system folder"
        return None

    # ----------------------------------------------------------------- killing

    def kill(self, pid: int, expected_create_time: float = None) -> dict:
        try:
            proc = psutil.Process(pid)
            name = proc.name()
        except psutil.NoSuchProcess:
            return {"status": "already_gone", "pid": pid}
        except psutil.AccessDenied:
            return {"status": "failed", "reason": "access_denied", "pid": pid}

        if expected_create_time is not None:
            try:
                if abs(proc.create_time() - float(expected_create_time)) > 1.0:
                    logger.warning(f"[response] Refused to kill PID={pid}: the number now belongs to a "
                                   f"different process (PID was reused).")
                    return {"status": "refused_pid_reused", "pid": pid, "name": name}
            except psutil.Error:
                pass

        reason = self.protection_reason(proc, name)
        if reason:
            logger.warning(f"[response] REFUSED to kill PID={pid} name={sanitize_for_log(name)}: {reason}")
            return {"status": "refused_protected", "pid": pid, "name": name, "reason": reason}

        logger.warning(f"[response] Terminating suspicious process: PID={pid} name={sanitize_for_log(name)}")

        try:
            proc.terminate()
            proc.wait(timeout=self.grace_period_seconds)
            logger.info(f"[response] Process {pid} ({sanitize_for_log(name)}) terminated gracefully.")
            return {"status": "terminated", "pid": pid, "name": name, "method": "terminate"}
        except psutil.TimeoutExpired:
            logger.warning(f"[response] Process {pid} ({sanitize_for_log(name)}) did not exit - force killing.")
            try:
                proc.kill()
                proc.wait(timeout=self.grace_period_seconds)
                logger.info(f"[response] Process {pid} ({sanitize_for_log(name)}) force-killed.")
                return {"status": "killed", "pid": pid, "name": name, "method": "kill"}
            except psutil.NoSuchProcess:
                return {"status": "killed", "pid": pid, "name": name, "method": "kill"}
            except psutil.TimeoutExpired:
                return {"status": "failed", "reason": "did_not_exit", "pid": pid, "name": name}
            except psutil.AccessDenied:
                return {"status": "failed", "reason": "access_denied", "pid": pid, "name": name}
        except psutil.NoSuchProcess:
            return {"status": "already_gone", "pid": pid, "name": name}
        except psutil.AccessDenied:
            logger.error(f"[response] Access denied killing PID={pid}. Needs elevated privileges.")
            return {"status": "failed", "reason": "access_denied", "pid": pid, "name": name}

    def kill_process_tree(self, pid: int, expected_create_time: float = None) -> list:
        """Kills a process and all its CHILDREN (ransomware droppers often start
        helpers through cmd.exe/powershell). Children go first. Each process is
        checked against the protection rules on its own."""
        results = []
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            return [{"status": "already_gone", "pid": pid}]
        except psutil.AccessDenied:
            children = []

        for child in reversed(children):
            results.append(self.kill(child.pid))
        results.append(self.kill(pid, expected_create_time=expected_create_time))
        return results


def summarize_kill(results: list) -> dict:
    """Turns a list of kill() results into the fields the alert needs."""
    main = results[-1] if results else {}
    killed = any(r.get("status") in ("terminated", "killed") for r in results)
    methods = [r.get("method") for r in results if r.get("method")]
    refused = [r for r in results if str(r.get("status", "")).startswith("refused")]
    return {
        "killed": killed,
        "method": methods[-1] if methods else None,
        "main_status": main.get("status"),
        "refused": [f"{r.get('name') or r.get('pid')}: {r.get('reason', r.get('status'))}" for r in refused],
        "count": sum(1 for r in results if r.get("status") in ("terminated", "killed")),
    }


if __name__ == "__main__":
    import subprocess
    import sys

    dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    print(f"Spawned dummy process PID={dummy.pid}")
    time.sleep(0.5)
    killer = ProcessKiller()
    print("Kill dummy:", killer.kill(dummy.pid))
    print("Try to kill myself (must be refused):", killer.kill(os.getpid()))

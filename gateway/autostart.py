r"""
autostart.py  -  start NeuraShield automatically when the user signs in
-----------------------------------------------------------------------
Protection that only exists until the next reboot is not protection. This module
registers NeuraShield in the per-user Windows "Run" key so the console, the
supervisor and the tray come up at every sign-in, in background mode (no window).
Nothing monitors until consent has been given - an auto-started console on a
machine that never consented just waits at the setup screen.

  * per user (HKCU): needs no administrator rights, works for copies that were not
    made by the installer, and the user can turn it off from Settings;
  * the installer also registers a machine-wide entry (HKLM); if that one exists
    the setting shows as enabled, and a second copy exits at start-up
    (single-instance guard in neurashield_app.py), so there is never a duplicate.
"""
import os
import sys
from pathlib import Path

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = os.environ.get("NEURASHIELD_AUTOSTART_NAME", "NeuraShield")   # env override: tests only


def supported() -> bool:
    return sys.platform == "win32"


def command(root: Path) -> str:
    """The command line Windows runs at sign-in."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --background'
    py = Path(sys.executable)
    pyw = py.with_name("pythonw.exe")
    runner = pyw if pyw.exists() else py
    return f'"{runner}" "{Path(root) / "run_neurashield.py"}" --no-open'


def _winreg():
    import winreg
    return winreg


def status(value_name: str = None) -> dict:
    value_name = value_name or VALUE_NAME
    """{'supported','enabled','scope'}  scope: 'user' | 'machine' | None"""
    if not supported():
        return {"supported": False, "enabled": False, "scope": None}
    w = _winreg()
    for hive, scope in ((w.HKEY_CURRENT_USER, "user"), (w.HKEY_LOCAL_MACHINE, "machine")):
        try:
            with w.OpenKey(hive, RUN_KEY, 0, w.KEY_READ) as k:
                w.QueryValueEx(k, value_name)
                return {"supported": True, "enabled": True, "scope": scope}
        except OSError:
            continue
    return {"supported": True, "enabled": False, "scope": None}


def enable(root: Path, value_name: str = None) -> bool:
    value_name = value_name or VALUE_NAME
    if not supported():
        return False
    w = _winreg()
    with w.CreateKey(w.HKEY_CURRENT_USER, RUN_KEY) as k:
        w.SetValueEx(k, value_name, 0, w.REG_SZ, command(root))
    return True


def disable(value_name: str = None) -> dict:
    value_name = value_name or VALUE_NAME
    """Remove the per-user entry. A machine-wide entry made by the installer is
    left alone (it needs administrator rights) and is reported back."""
    if not supported():
        return status(value_name)
    w = _winreg()
    try:
        with w.OpenKey(w.HKEY_CURRENT_USER, RUN_KEY, 0, w.KEY_SET_VALUE) as k:
            w.DeleteValue(k, value_name)
    except OSError:
        pass
    return status(value_name)

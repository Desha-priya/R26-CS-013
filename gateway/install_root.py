r"""
install_root.py  —  where does a packaged NeuraShield keep its config and data?
-------------------------------------------------------------------------------
Installed layout (made by the installer):
    <install>\gateway\NeuraShield.exe   <install>\config\layers.json   <install>\data\
If the exe is run from anywhere else (the build's dist folder, a copy on a USB
stick, ...), there is no config next to it, so use a per-user folder
    %LOCALAPPDATA%\NeuraShield
seeded from the defaults bundled inside the exe. Default state: consent NOT
given, so nothing monitors until the first-run setup is completed.
Source runs never come here (the repo root is the root).
"""
import json
import os
import shutil
import sys
from pathlib import Path

DEFAULT_PRODUCT = {
    "product_name": "NeuraShield",
    "version": "1.0.0",
    "console": {"host": "127.0.0.1", "port": 9000},
    "platform_url": "http://127.0.0.1:8000",
    "enabled_layers": {"zta": True, "network": False, "ransomware": False, "content": True},
    "consent": {"accepted": False, "accepted_at": None},
    "notifications": {"enabled": True, "min_severity": "medium", "stepup": True,
                      "layers": {"zta": True, "network": True, "ransomware": True, "content": True}},
}


def _seed(home: Path, bundle: Path):
    (home / "config").mkdir(parents=True, exist_ok=True)
    (home / "data").mkdir(parents=True, exist_ok=True)
    layers = home / "config" / "layers.json"
    src = bundle / "config" / "layers.json"
    if not layers.exists() and src.exists():
        shutil.copyfile(src, layers)
    product = home / "config" / "product.json"
    if not product.exists():
        product.write_text(json.dumps(DEFAULT_PRODUCT, indent=2), encoding="utf-8")


def resolve(exe_path=None, bundle=None, local_appdata=None) -> Path:
    exe = Path(exe_path or sys.executable).resolve()
    installed = exe.parent.parent
    if (installed / "config" / "layers.json").exists():
        return installed
    base = Path(local_appdata or os.environ.get("LOCALAPPDATA") or Path.home())
    home = base / "NeuraShield"
    _seed(home, Path(bundle or getattr(sys, "_MEIPASS", exe.parent)))
    return home

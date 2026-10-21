"""
src/common/sources_config.py
Tiny non-secret settings file for the extra email sources
(.neurashield/sources.json). Holds only on/off choices and the drop-folder
path - never credentials (those live in the encrypted oauth_store).

    {"gmail": true, "folder": {"enabled": true, "path": "C:/.../inbox_watch"},
     "notifications": false}

"enabled" means "the user turned this on, so start it again at next boot".
Nothing here ever turns anything on by itself: a source with no entry is OFF.
"""

import json
import threading

from src.common import credential_store

_CFG = credential_store._STORE_DIR / "sources.json"
_lock = threading.Lock()


def load() -> dict:
    with _lock:
        try:
            d = json.loads(_CFG.read_text())
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}


def _write(d: dict) -> None:
    _CFG.parent.mkdir(parents=True, exist_ok=True)
    _CFG.write_text(json.dumps(d, indent=2))


def set_enabled(name: str, enabled: bool, **extra) -> None:
    with _lock:
        try:
            d = json.loads(_CFG.read_text())
            if not isinstance(d, dict):
                d = {}
        except Exception:
            d = {}
        if name == "folder":
            cur = d.get("folder") if isinstance(d.get("folder"), dict) else {}
            cur["enabled"] = bool(enabled)
            cur.update({k: v for k, v in extra.items() if v is not None})
            d["folder"] = cur
        else:
            d[name] = bool(enabled)
        _write(d)


def is_enabled(name: str) -> bool:
    d = load()
    v = d.get(name)
    if isinstance(v, dict):
        return bool(v.get("enabled"))
    return bool(v)


def folder_path() -> str:
    v = load().get("folder")
    return v.get("path", "") if isinstance(v, dict) else ""

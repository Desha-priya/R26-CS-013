"""
paths.py
----------
ONE place that decides where files live. Every other module asks this
file instead of using paths like "config/config.yaml" or "logs/" that
only work when you start Python from the project folder.

Why this matters for the final single .exe:
  - Inside a packaged exe the program files (config, models, dashboard
    pages) are in a read-only bundle, and the exe may be installed in a
    folder the user cannot write to.
  - So we keep two kinds of location:
      resource(...)  -> read-only things that ship with the program
                        (config.yaml, models, dashboard templates)
      runtime(...)   -> things the program writes while running
                        (logs, alert queue, secret key, state files)

When running from source (python -m main) both point at the project
folder, so nothing changes for development, demos or tests.

Overrides (environment variables):
  NEURASHIELD_HOME     folder for runtime files (logs, queues, key)
  NEURASHIELD_CONFIG   full path to a config.yaml to use instead
"""

import os
import sys

import yaml

APP_NAME = "NeuraShield"

# src/utils/paths.py  ->  project root is two folders up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def is_frozen() -> bool:
    """True when running inside a PyInstaller-style packaged exe."""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> str:
    """Folder that holds the read-only program files."""
    override = os.environ.get("NEURASHIELD_BUNDLE_ROOT")
    if override:
        # Set by the single NeuraShield.exe host: this layer's files live in a
        # sub-folder of the shared bundle, not at its root.
        return override
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return PROJECT_ROOT


def runtime_root() -> str:
    """Folder where the program may write files."""
    override = os.environ.get("NEURASHIELD_HOME")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if is_frozen():
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
        return os.path.join(base, APP_NAME, "Layer2")
    return PROJECT_ROOT


def resource(relative: str) -> str:
    """Absolute path of a read-only program file. Absolute input is returned as is."""
    if os.path.isabs(relative):
        return relative
    return os.path.join(bundle_root(), relative)


def runtime(relative: str, create_parent: bool = True) -> str:
    """Absolute path of a writable file/folder. Parent folder is created."""
    if os.path.isabs(relative):
        path = relative
    else:
        path = os.path.join(runtime_root(), relative)
    if create_parent:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    return path


def config_file() -> str:
    """Which config.yaml to use: env override > runtime copy > bundled copy."""
    override = os.environ.get("NEURASHIELD_CONFIG")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    runtime_copy = os.path.join(runtime_root(), "config", "config.yaml")
    if os.path.exists(runtime_copy):
        return runtime_copy
    return resource(os.path.join("config", "config.yaml"))


def load_config(config_path: str = None) -> dict:
    """Reads the YAML config. Safe loader only (never executes anything)."""
    path = config_path or config_file()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_backup_root() -> str:
    """Hidden folder holding the versioned backups."""
    override = os.environ.get("NEURASHIELD_BACKUP_ROOT")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.expanduser("~/.neurashield_cache_x7f2")

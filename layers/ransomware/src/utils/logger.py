"""
logger.py
----------
Central logger for all NeuraShield Ransomware Killer components.
Logs to the console and to ONE rotating file.

Two security/robustness details:

1. ONE shared file handler. Earlier, every module created its own handler
   on the same file. On Windows a rotating file cannot be renamed while a
   second handler still has it open, so rotation could crash a long-running
   service. Now every module logs through one parent logger.

2. Log-injection protection. Text that comes from outside (file names,
   process names, alert ids, operator "reason" text) could contain line
   breaks and so fake extra log lines. The formatter turns every control
   character into a visible escape (for example a new line becomes \\n),
   so one log call is always exactly one log line.
"""

import logging
import os
import re
import threading
from logging.handlers import RotatingFileHandler

from src.utils import paths

_ROOT_NAME = "neurashield"
_LOCK = threading.Lock()
_CONFIGURED = False

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize_for_log(value, max_len: int = 300) -> str:
    """Makes any value safe to put inside a log line (one line, no control chars)."""
    text = str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", " ")
    text = _CONTROL_CHARS.sub("?", text)
    if len(text) > max_len:
        text = text[:max_len] + "...(cut)"
    return text


class _SafeFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Flatten the final message first, so every part of it is cleaned.
        message = record.getMessage()
        record.msg = sanitize_for_log(message, max_len=4000)
        record.args = ()
        return super().format(record)


def _configure_root(log_dir: str = None):
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED:
            return
        root = logging.getLogger(_ROOT_NAME)
        root.setLevel(logging.INFO)
        root.propagate = False

        fmt = _SafeFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

        try:
            folder = log_dir or paths.runtime("logs", create_parent=False)
            os.makedirs(folder, exist_ok=True)
            file_handler = RotatingFileHandler(
                os.path.join(folder, "neurashield_ransomware_killer.log"),
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)
        except OSError as e:
            # Never crash the security tool just because the log folder is
            # not writable. Console logging still works.
            root.warning(f"Could not open log file, console logging only: {e}")

        _CONFIGURED = True


def get_logger(name: str = _ROOT_NAME, log_dir: str = None) -> logging.Logger:
    _configure_root(log_dir)
    if name == _ROOT_NAME:
        return logging.getLogger(_ROOT_NAME)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")

"""
file_watcher.py
------------------
Real-time, cross-platform file system monitor using the `watchdog`
library (per the proposal). Watches configured user directories
recursively, computes entropy on write events, tags each event with
the responsible process (best-effort, since OS file events don't
natively carry a PID — see note below), and feeds everything into the
SlidingWindowFeatureExtractor.

NOTE on PID attribution: standard OS filesystem-event APIs (inotify on
Linux, ReadDirectoryChangesW on Windows) do NOT report which process
caused an event. To get PIDs, you'd normally need a kernel driver /
ETW (Windows) or auditd (Linux). For this project, we approximate by
recording the PID of whichever process currently has the most open
handles into the watched directory (via process_tracker) at the time
of the event. This is documented as a known limitation in the report.
"""

import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.monitor.feature_extractor import (
    FileEvent,
    SlidingWindowFeatureExtractor,
    file_entropy,
)
from src.monitor.process_tracker import ProcessTracker
from src.utils import paths
from src.utils.logger import get_logger, sanitize_for_log

logger = get_logger(__name__)

# NEURASHIELD_DEBUG_EVENTS=1 prints every file event the monitor receives.
DEBUG_EVENTS = os.environ.get("NEURASHIELD_DEBUG_EVENTS") == "1"

# Our own files and normal programming leftovers must never count as
# "suspicious file activity". Without this, Python writing __pycache__ files
# or this program writing its own log/backup files could raise false alarms
# when the project sits inside a watched folder such as Downloads.
DEFAULT_IGNORED_FRAGMENTS = (
    "__pycache__", ".pyc", ".pyo", ".ns_restore_tmp",
    "neurashield_ransomware_killer.log", ".neurashield_cache",
)


def build_ignore_dirs(extra=None) -> list:
    """Folders this program itself writes to (logs, queues, backups)."""
    dirs = [paths.runtime("logs", create_parent=False),
            paths.runtime("data", create_parent=False),
            paths.default_backup_root()]
    dirs += list(extra or [])
    return [os.path.normcase(os.path.abspath(d)) for d in dirs]


def is_ignored_path(path: str, ignore_dirs, ignore_fragments=DEFAULT_IGNORED_FRAGMENTS) -> bool:
    norm = os.path.normcase(os.path.abspath(path))
    if any(fragment.lower() in norm for fragment in ignore_fragments):
        return True
    return any(norm == d or norm.startswith(d + os.sep) for d in ignore_dirs)


class RansomwareEventHandler(FileSystemEventHandler):
    def __init__(self, extractor: SlidingWindowFeatureExtractor, on_features_ready=None,
                 ignore_dirs=None, ignore_fragments=None):
        super().__init__()
        self.extractor = extractor
        self.tracker = ProcessTracker()
        self.on_features_ready = on_features_ready  # callback(features: dict, last_event: FileEvent)
        self.ignore_dirs = build_ignore_dirs() if ignore_dirs is None else ignore_dirs
        self.ignore_fragments = DEFAULT_IGNORED_FRAGMENTS if ignore_fragments is None else ignore_fragments

    # counters for the status display / diagnostics
    events_seen = 0
    last_event_at = None
    errors = 0

    def _handle(self, event_type: str, src_path: str, old_path: str = None):
        # An error in here must NEVER escape: an exception inside a watchdog
        # handler can stop the observer's thread, and the layer would silently
        # stop seeing file events.
        try:
            self._handle_inner(event_type, src_path, old_path)
        except Exception as e:
            self.errors += 1
            if self.errors <= 5 or self.errors % 100 == 0:
                logger.error(f"[watcher] Error while handling a {event_type} event "
                             f"(#{self.errors}): {sanitize_for_log(e)}", exc_info=True)

    def _handle_inner(self, event_type: str, src_path: str, old_path: str = None):
        # Time stamp taken the moment the event ARRIVES, before the slow
        # entropy read: on a busy or antivirus-scanned machine the read can be
        # slow, and the attack speed must not be measured by our own delay.
        arrived = time.time()
        self.events_seen += 1
        self.last_event_at = arrived

        if is_ignored_path(src_path, self.ignore_dirs, self.ignore_fragments):
            return  # our own files / Python cache: not user activity
        if os.path.isdir(src_path):
            return  # only track file-level events

        entropy = None
        # file_entropy() returns None (not a misleading 0.0) on a read race or a
        # locked file, so a missed read is one skipped reading, not a corrupted average.
        if event_type in ("modified", "created", "renamed") and os.path.exists(src_path):
            entropy = file_entropy(src_path)

        fe = FileEvent(event_type=event_type, path=src_path, timestamp=arrived,
                       entropy=entropy, src_path=old_path)
        self.extractor.add_event(fe)
        if DEBUG_EVENTS:
            logger.info(f"[watcher] {event_type:8s} | {sanitize_for_log(src_path)} | entropy={entropy}")
        else:
            logger.debug(f"[watcher] {event_type:8s} | {sanitize_for_log(src_path)} | entropy={entropy}")

        if self.on_features_ready:
            features = self.extractor.compute_features()
            self.on_features_ready(features, fe)

    def on_created(self, event):
        if not event.is_directory:
            self._handle("created", event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle("modified", event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._handle("deleted", event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._handle("renamed", event.dest_path, event.src_path)


class FileSystemMonitor:
    def __init__(self, config_path=None, on_features_ready=None):
        self.cfg = paths.load_config(config_path)

        self.watch_paths = [
            os.path.expanduser(p) for p in self.cfg["live_monitor"]["watch_paths"]
        ]
        window = tuple([
            self.cfg["live_monitor"]["file_rate_window_seconds"], 30, 60
        ])
        self.extractor = SlidingWindowFeatureExtractor(window_seconds=window)
        self.handler = RansomwareEventHandler(self.extractor, on_features_ready=on_features_ready)
        self.observer = Observer()

    def start(self):
        for path in self.watch_paths:
            if os.path.isdir(path):
                self.observer.schedule(self.handler, path, recursive=True)
                logger.info(f"[watcher] Watching: {path}")
            else:
                logger.warning(f"[watcher] Skipping missing path: {path}")
        self.observer.start()
        logger.info("[watcher] File system monitor started.")

    def stop(self):
        self.observer.stop()
        self.observer.join()
        logger.info("[watcher] File system monitor stopped.")


if __name__ == "__main__":
    def print_features(features, last_event):
        print(f"\n[demo] Latest event: {last_event.event_type} {last_event.path}")
        print(f"[demo] file_rate_5s={features.get('file_rate_5s', 0):.2f}  "
              f"avg_entropy_5s={features.get('avg_entropy_5s', 0):.2f}")

    monitor = FileSystemMonitor(on_features_ready=print_features)
    monitor.start()
    try:
        print("Monitoring started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()

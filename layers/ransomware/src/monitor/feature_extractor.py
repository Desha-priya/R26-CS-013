"""
feature_extractor.py
----------------------
Turns raw file-system events (from file_watcher.py) into the
behavioral features described in the proposal: Shannon entropy,
file-modification rate over sliding windows, rename ratio, etc.

This is the "file-level" complement to the memory-forensic
(CIC-MalMem-2022) detector — see docs/dataset_notes.md.
"""

import math
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field

# A single high-entropy file (e.g. a legitimate .zip or video copy) is
# normal and shouldn't trigger anything on its own. Multiple distinct
# high-entropy writes within one window is a much stronger signal -
# this is the threshold used by high_entropy_count below, added
# specifically to resist an entropy-dilution evasion attack proven via
# testing (see simulation/adversarial_simulator.py): an attacker who
# interleaves genuinely encrypted writes with low-entropy padding to
# drag avg_entropy down was found to fully evade a detector that only
# checks the average. Counting distinct high-entropy events (not a
# single max) survives that dilution while still requiring more than
# one legitimate high-entropy file to trigger.
HIGH_ENTROPY_EVENT_CUTOFF = 7.5


def shannon_entropy(data: bytes) -> float:
    """
    Shannon entropy of a byte string, 0-8 bits/byte.
    Encrypted/compressed data typically sits ~7.5-8.0.
    Plain text sits much lower (~3.5-5.0).
    """
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def file_entropy(path: str, sample_bytes: int = 65536, min_reliable_bytes: int = 64) -> float:
    """Reads up to `sample_bytes` from the start of a file and computes entropy.
    Returns None (not 0.0) on read failure, an empty read, OR a read
    smaller than `min_reliable_bytes`, so a failed/raced/partial read
    doesn't silently drag down the average entropy of a window.

    Two related failure modes found via real testing:
    1. Empty read (0 bytes) - the OS event fired before any bytes were
       flushed. Treating this as entropy 0.0 asserted the file has no
       randomness, when really nothing meaningful was sampled at all.
    2. Very small partial read (a handful of bytes) - Shannon entropy
       on a tiny sample is statistically noisy and can read as
       misleadingly low purely by chance, even for genuinely random
       (encrypted-looking) data. A 64-byte floor keeps the estimate
       meaningful without requiring a full read.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(sample_bytes)
        if len(data) < min_reliable_bytes:
            return None
        return shannon_entropy(data)
    except (FileNotFoundError, PermissionError, OSError):
        return None


@dataclass
class FileEvent:
    event_type: str      # "created" | "modified" | "deleted" | "renamed"
    path: str
    timestamp: float = field(default_factory=time.time)
    process_id: int = None
    entropy: float = None
    src_path: str = None   # for "renamed": the OLD name (path is the new one)


# Ransom notes are plain text/HTML files with telling names. A bare "readme.txt"
# is far too common to count, so only names that contain one of these phrases do.
_NOTE_PHRASES = ("decrypt", "ransom", "recover_", "recovery_instruction", "restore_my_files",
                 "restore-my-files", "how_to_", "how-to-", "your_files", "files_are_encrypted",
                 "files_encrypted", "read_me_to", "unlock_", "!!!")
_NOTE_EXTENSIONS = (".txt", ".html", ".htm", ".hta", ".rtf", ".url", ".md")


def looks_like_ransom_note(path: str) -> bool:
    name = os.path.basename(path or "").lower()
    return name.endswith(_NOTE_EXTENSIONS) and any(p in name for p in _NOTE_PHRASES)


def _extension(path: str) -> str:
    return os.path.splitext(path or "")[1].lower()


class SlidingWindowFeatureExtractor:
    """
    Maintains a rolling buffer of recent file events and computes
    behavioral features over configurable time windows, matching the
    proposal's "5s / 30s / 60s sliding window" requirement.
    """

    def __init__(self, window_seconds=(5, 30, 60), max_events=500):
        self.window_seconds = window_seconds
        self.events: deque = deque(maxlen=max_events)
        # The file monitor adds events from one thread while the status writer,
        # the PID scanner and the response read them from others. Iterating a
        # deque while it is being appended to raises RuntimeError, so every
        # access goes through this lock.
        self._lock = threading.Lock()

    def add_event(self, event: FileEvent):
        with self._lock:
            self.events.append(event)

    def snapshot_events(self) -> list:
        with self._lock:
            return list(self.events)

    def _events_in_window(self, seconds: int, now: float = None):
        now = now or time.time()
        return [e for e in self.snapshot_events() if now - e.timestamp <= seconds]

    def recent_events(self, seconds: float, now: float = None) -> list:
        """Events from the last `seconds` seconds (oldest first). Used after a
        detection to find which folders and files were touched."""
        return self._events_in_window(seconds, now)

    def first_suspicious_time(self, seconds: float = 60, now: float = None):
        """Time of the earliest high-entropy event in the window, or None.
        Backups taken before this moment are the trustworthy ones."""
        times = [e.timestamp for e in self._events_in_window(seconds, now)
                 if e.entropy is not None and e.entropy >= HIGH_ENTROPY_EVENT_CUTOFF]
        return min(times) if times else None

    def compute_features(self, now: float = None) -> dict:
        now = now or time.time()
        features = {}

        for w in self.window_seconds:
            window_events = self._events_in_window(w, now)
            n = len(window_events)
            renames = sum(1 for e in window_events if e.event_type == "renamed")
            modifies = sum(1 for e in window_events if e.event_type == "modified")
            creates = sum(1 for e in window_events if e.event_type == "created")
            deletes = sum(1 for e in window_events if e.event_type == "deleted")

            entropies = [e.entropy for e in window_events if e.entropy is not None]
            avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
            max_entropy = max(entropies) if entropies else 0.0
            high_entropy_count = sum(1 for e in entropies if e >= HIGH_ENTROPY_EVENT_CUTOFF)

            features[f"file_rate_{w}s"] = n / w
            features[f"rename_ratio_{w}s"] = renames / n if n else 0.0
            features[f"modify_count_{w}s"] = modifies
            features[f"create_count_{w}s"] = creates
            features[f"delete_count_{w}s"] = deletes
            features[f"avg_entropy_{w}s"] = avg_entropy
            features[f"max_entropy_{w}s"] = max_entropy
            features[f"high_entropy_count_{w}s"] = high_entropy_count

        # Distinct process IDs touching files recently (multi-file-per-process
        # in a short window is a strong ransomware signal)
        recent = self._events_in_window(max(self.window_seconds), now)
        pids = [e.process_id for e in recent if e.process_id is not None]
        features["distinct_processes"] = len(set(pids))
        features["events_per_process"] = len(recent) / len(set(pids)) if pids else 0.0

        # Extension / file-name signals (many ransomware families rename files):
        #   appended_ext_count_10s : files renamed by ADDING an extension (a.docx -> a.docx.locked)
        #   ext_change_count_10s   : files whose extension changed in any way
        #   new_ext_distinct_10s   : how many different new extensions were used
        #   ransom_note_count_60s  : files created/changed that look like ransom notes
        last10 = self._events_in_window(10, now)
        changed = appended = 0
        new_exts = set()
        for e in last10:
            if e.event_type == "renamed" and e.src_path and e.path != e.src_path:
                old_ext, new_ext = _extension(e.src_path), _extension(e.path)
                if old_ext != new_ext:
                    changed += 1
                    new_exts.add(new_ext)
                    if e.path.lower().startswith(e.src_path.lower()):
                        appended += 1
        features["ext_change_count_10s"] = changed
        features["appended_ext_count_10s"] = appended
        features["new_ext_distinct_10s"] = len(new_exts)
        features["ransom_note_count_60s"] = sum(
            1 for e in recent if e.event_type in ("created", "modified", "renamed") and looks_like_ransom_note(e.path))

        return features

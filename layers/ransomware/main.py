"""
main.py
---------
Single entry point for real (non-demo) operation: wires the live file
monitor, periodic backup, threat detection, automated response and signed
alerting into ONE continuously-running service.

What happens when a threat is detected (see _respond):
  1. Backups are paused, so half-encrypted files never replace clean copies.
  2. The process most likely responsible is found (best effort) and, if the
     confidence is high enough and the process is not protected, killed
     together with its child processes.
  3. EVERY damaged file in the affected folders is restored - not only the
     last file that triggered the alert - using the newest clean backup
     taken before the incident began.
  4. A signed alert is written locally and a short notice is sent to the
     main platform (retried later if the platform is unreachable).

CONSENT: the service starts in STANDBY and does not read, watch or copy any
user file until protection has been agreed to (dashboard button, --consent,
Layer2.grant_consent(), or NEURASHIELD_AUTO_CONSENT=1). See src/consent.py.

Run:
    python -m main                  (service only, standby until consent exists)
    python -m main --dashboard      (service + local dashboard on 127.0.0.1:5050)
    python -m main --consent        (grant consent for the configured folders, then run)
    python -m main --check          (readiness report, then exit)
Stop with Ctrl+C.
"""

import json
import os
import sys
import time
import threading

# Keep "python main.py", "python -m main" and VS Code's Play button on the
# same import path.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from watchdog.observers import Observer

from src.alerting.alert_generator import AlertGenerator
from src.consent import ConsentStore
from src.monitor.feature_extractor import SlidingWindowFeatureExtractor
from src.monitor.file_watcher import RansomwareEventHandler, build_ignore_dirs
from src.monitor.process_tracker import ProcessTracker, RollingPidCache
from src.monitor.threshold_detector import (is_threat, compute_dynamic_confidence, build_threshold_explanation,
                                            FILE_RATE_THRESHOLD, ENTROPY_THRESHOLD, MIN_HIGH_ENTROPY_EVENTS,
                                            SUSTAINED_HIGH_ENTROPY_COUNT, EXT_APPEND_MIN_COUNT)
from src.response.command_listener import CommandListener
from src.response.process_killer import ProcessKiller, summarize_kill
from src.monitor.feature_extractor import looks_like_ransom_note
from src.response.disk_scan import scan_folders
from src.response.incident_store import IncidentStore
from src.response.rollback_engine import RollbackEngine, RANSOM_EXTENSIONS
from src.utils import paths
from src.utils.logger import get_logger, sanitize_for_log
from src.utils.resource_monitor import ResourceMonitor

logger = get_logger(__name__)


class BackupScheduler:
    """
    Keeps versioned backups of the watched folders.

      1. BASELINE: the first full pass copies every file once (progress is
         published so the dashboard can show "x of y files protected").
      2. After that, only files that CHANGED are copied: the file monitor marks
         a path "dirty". A dirty file is backed up once it has been QUIET for
         backup_settle_seconds (so a file that is still being written is not
         copied half-way), and a given file is backed up at most once every
         backup_min_interval_seconds (so frequent saves do not flood the version
         history). A slow full sweep (main.full_sweep_seconds) catches anything
         the monitor missed. The polling tick (backup_interval_seconds) only
         decides how often the queue is looked at - it is NOT the backup rate.
      3. A restart is cheap: files whose modified-time and size match their
         newest stored version are skipped without being read.
      4. It can be PAUSED during an incident so half-encrypted files are never
         saved (the engine's entropy guard is a second line of defence).
    """

    DEBOUNCE_SECONDS = 1.0          # default quiet time when none is configured
    MAX_WAIT_SECONDS = 60.0         # a file that never goes quiet is still copied after this

    def __init__(self, rollback: RollbackEngine, watch_paths: list, interval_seconds: int,
                 ignore_dirs=None, full_sweep_seconds: float = 300.0,
                 settle_seconds: float = None, min_interval_seconds: float = 0.0,
                 on_sweep=None):
        self.rollback = rollback
        self.on_sweep = on_sweep              # called after every completed full sweep (not the baseline)
        self.settle_seconds = self.DEBOUNCE_SECONDS if settle_seconds is None else float(settle_seconds)
        self.min_interval_seconds = float(min_interval_seconds)
        self._last_backup = {}                # path -> time of its last copy
        self.watch_paths = watch_paths
        self.interval_seconds = interval_seconds
        self.full_sweep_seconds = full_sweep_seconds
        self.ignore_dirs = ignore_dirs if ignore_dirs is not None else build_ignore_dirs()
        self._paused_until = 0.0
        self._stop_event = threading.Event()
        self._thread = None
        self._dirty = {}                      # path -> [first marked, last marked]
        self._dirty_lock = threading.Lock()
        self.progress = {"phase": "idle", "total": 0, "done": 0, "copied": 0,
                         "baseline_complete": False, "last_full_pass": None,
                         "pending": 0, "changes_backed_up": 0, "last_change_backup_at": None}

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="backup")
        self._thread.start()
        logger.info("[backup] Backup scheduler started (baseline first, then changed files only).")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.rollback.flush()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def pause_for(self, seconds: float):
        self._paused_until = max(self._paused_until, time.time() + seconds)
        logger.warning(f"[backup] Backups paused for {seconds:.0f}s (incident in progress).")

    @property
    def paused(self) -> bool:
        return time.time() < self._paused_until

    def mark_dirty(self, path: str):
        """Called by the file monitor: this file just changed."""
        now = time.time()
        with self._dirty_lock:
            entry = self._dirty.get(path)
            if entry is None:
                self._dirty[path] = [now, now]
            else:
                entry[1] = now                # still changing: the quiet period restarts
            if len(self._dirty) > 50000:      # runaway protection; the full sweep still covers it
                self._dirty.clear()
            self.progress["pending"] = len(self._dirty)

    # ---------------------------------------------------------------- loop

    def _loop(self):
        next_full = 0.0                       # run the baseline immediately
        while not self._stop_event.is_set():
            if not self.paused:
                try:
                    if time.time() >= next_full:
                        self._full_pass()
                        next_full = time.time() + self.full_sweep_seconds
                    else:
                        self._dirty_pass()
                except Exception as e:  # keep the thread alive no matter what
                    logger.error(f"[backup] Backup pass failed: {sanitize_for_log(e)}")
            self._stop_event.wait(self.interval_seconds)

    def _skip_name(self, name: str) -> bool:
        return name.endswith((".pyc", ".pyo", ".ns_restore_tmp"))

    def _scan(self) -> list:
        files = []
        for base in self.watch_paths:
            if not os.path.isdir(base):
                continue
            for root, dirs, names in os.walk(base):
                norm_root = os.path.normcase(os.path.abspath(root))
                if any(norm_root == d or norm_root.startswith(d + os.sep) for d in self.ignore_dirs):
                    dirs[:] = []
                    continue
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                files.extend(os.path.join(root, n) for n in names if not self._skip_name(n))
                if self._stop_event.is_set():
                    return files
        return files

    def _full_pass(self):
        first = not self.progress["baseline_complete"]
        self.progress.update(phase="scanning" if first else "sweeping", done=0, copied=0)
        files = self._scan()
        self.progress.update(total=len(files), phase="copying" if first else "sweeping")
        since_flush = 0
        for i, path in enumerate(files, 1):
            if self.paused or self._stop_event.is_set():
                break
            result = self.rollback.backup_file(path, save=False)
            if result["status"] == "ok":
                self.progress["copied"] += 1
                since_flush += 1
                time.sleep(0.002)        # breathing room for the dashboard and other threads
                if since_flush >= 200:
                    self.rollback.flush()
                    since_flush = 0
            self.progress["done"] = i
        else:
            self.progress.update(baseline_complete=True, last_full_pass=time.time(), phase="watching")
            if first:
                logger.info(f"[backup] Baseline complete: {len(files)} file(s) checked, "
                            f"{self.progress['copied']} copied.")
            elif self.on_sweep is not None:
                try:
                    self.rollback.flush()
                    self.on_sweep()
                except Exception as e:
                    logger.error(f"[backup] Integrity check failed: {sanitize_for_log(e)}", exc_info=True)
        self.rollback.flush()
        if self.paused and self.progress["phase"] != "watching":
            self.progress["phase"] = "paused"

    def _dirty_pass(self):
        now = time.time()
        with self._dirty_lock:
            ready = []
            for p, (first, last) in self._dirty.items():
                settled = now - last >= self.settle_seconds or now - first >= self.MAX_WAIT_SECONDS
                spaced = now - self._last_backup.get(p, 0.0) >= self.min_interval_seconds
                if settled and spaced:
                    ready.append(p)
            for p in ready:
                del self._dirty[p]
            self.progress["pending"] = len(self._dirty)
            if len(self._last_backup) > 20000:        # keep the memory small
                cutoff = now - max(3600.0, self.min_interval_seconds)
                self._last_backup = {k: v for k, v in self._last_backup.items() if v > cutoff}
        if not ready:
            return
        for path in ready:
            if self.paused or self._stop_event.is_set():
                return
            if os.path.isfile(path):
                result = self.rollback.backup_file(path, save=False)
                if result["status"] == "ok":
                    stamp = time.time()
                    self._last_backup[path] = stamp
                    self.progress["changes_backed_up"] += 1
                    self.progress["last_change_backup_at"] = stamp
        self.rollback.flush()


class RansomwareKillerService:
    # Possible values of .state
    STANDBY, BASELINE, ARMED, INCIDENT = "standby", "baseline", "armed", "incident"

    def __init__(self, config_path=None):
        self.cfg = paths.load_config(config_path)
        self._config_path = config_path

        self.watch_paths = [os.path.expanduser(p) for p in self.cfg["live_monitor"]["watch_paths"]]
        window = (self.cfg["live_monitor"]["file_rate_window_seconds"], 30, 60)
        self.extractor = SlidingWindowFeatureExtractor(window_seconds=window)

        extra_ignores = self.cfg["live_monitor"].get("ignore_paths", []) or []
        self.ignore_dirs = build_ignore_dirs([os.path.expanduser(p) for p in extra_ignores])

        self.rollback = RollbackEngine(config_path=config_path)
        self.killer = ProcessKiller(extra_protected=self.cfg["response"].get("protected_processes", []))
        self.tracker = ProcessTracker()
        self.alerter = AlertGenerator(config_path=config_path)
        self.consent = ConsentStore()
        self.incidents = IncidentStore()
        self._incident_lock = threading.Lock()
        self._current_incident = None          # {"id": ..., "last": time of last activity}
        self.last_scan = {"running": False, "started_at": None, "result": None}
        self._scan_lock = threading.Lock()

        # Components that only exist while protection is ACTIVE (after consent).
        self.observer = None
        self.backup_scheduler = None
        self.pid_cache = None
        self.command_listener = None

        self.resource_monitor = ResourceMonitor(
            status_path=paths.runtime("data/simulated_events/service_status.json"),
            cpu_target_percent=self.cfg.get("main", {}).get("cpu_target_percent", 5.0),
            memory_target_mb=self.cfg.get("main", {}).get("memory_target_mb", 50.0),
        )

        self._own_pid = os.getpid()
        self._respond_lock = threading.Lock()
        self._last_alert_time = 0
        self._min_seconds_between_alerts = 10   # avoid alert-spamming on one ongoing incident
        self._stop_event = threading.Event()
        self._outbox_stop = self._stop_event    # (older name, same event)
        self._active = False
        self._activate_lock = threading.Lock()
        self._incident_active = False
        self._last_incident_at = None
        self._events_total = 0
        self.observer_restarts = 0
        self._threads = []
        self.started_at = None
        self.incident_count = 0
        self.status_path = paths.runtime("data/simulated_events/layer2_status.json")

        self.handler = RansomwareEventHandler(self.extractor, on_features_ready=self._on_event,
                                              ignore_dirs=self.ignore_dirs)

    # --------------------------------------------------------------- state

    @property
    def active(self) -> bool:
        return self._active

    @property
    def state(self) -> str:
        if not self._active:
            return self.STANDBY
        if self._incident_active:
            return self.INCIDENT
        if self.backup_scheduler is not None and not self.backup_scheduler.progress["baseline_complete"]:
            return self.BASELINE
        return self.ARMED

    def _auto_consent_enabled(self) -> bool:
        if os.environ.get("NEURASHIELD_AUTO_CONSENT") == "1":
            return True
        return self.cfg.get("main", {}).get("require_consent", True) is False

    def grant_consent(self, by: str = "api"):
        self.consent.grant(self.watch_paths, by=by)

    def revoke_consent(self, by: str = "api"):
        self.consent.revoke(by=by)

    # ------------------------------------------------------------ detection

    def _on_event(self, features, last_event):
        if not self._active:
            return
        try:
            self._events_total += 1
            if last_event.event_type in ("created", "modified") and self.backup_scheduler is not None:
                self.backup_scheduler.mark_dirty(last_event.path)
            if not is_threat(features):
                return
            with self._respond_lock:
                now = time.time()
                if now - self._last_alert_time < self._min_seconds_between_alerts:
                    return  # already responding to this incident
                self._last_alert_time = now
                logger.warning(
                    f"[detect] THREAT SIGNATURE: file_rate_5s={features.get('file_rate_5s', 0):.2f} "
                    f"avg_entropy_5s={features.get('avg_entropy_5s', 0):.2f}"
                )
                self._respond(features, last_event)
        except Exception as e:
            # A bug in one response must never stop the monitor.
            self._incident_active = False
            logger.error(f"[service] Error while handling an event: {sanitize_for_log(e)}", exc_info=True)

    # ------------------------------------------------------------- response

    def _expected_written_bytes(self) -> int:
        """Size of the files that changed in the last few seconds."""
        seen, total = set(), 0
        for e in self.extractor.recent_events(8):
            if e.event_type in ("modified", "created", "renamed") and e.path and e.path not in seen:
                seen.add(e.path)
                try:
                    total += os.path.getsize(e.path)
                except OSError:
                    pass
        return total

    def _find_attacker(self, last_event):
        """Best-effort process attribution (see ProcessTracker/RollingPidCache for limits).
        1) The process whose written-bytes counter matches the files that changed (works
           even when the attacker never holds a file open long enough to be seen).
        2) Processes seen with open files in the folder."""
        watched_dir = os.path.dirname(last_event.path) or self.watch_paths[0]
        exclude = {self._own_pid, os.getppid()}
        ordered = []
        try:
            expected = self._expected_written_bytes()
            if expected > 0:
                self.pid_cache._safe_sample_writes()        # fresh counters: the last timed sample may be stale
                lo, hi = max(65536, 0.3 * expected), max(20 * expected, 100 * 1024 * 1024)
                writers = [pid for pid, b in self.pid_cache.top_writers(exclude_pids=exclude) if lo <= b <= hi]
                if writers:
                    ordered.append(writers[0])
                    logger.info(f"[response] Writer-volume attribution picked PID {writers[0]}")
        except Exception as e:
            logger.warning(f"[response] write-volume attribution failed: {sanitize_for_log(e)}")
        candidates = self.pid_cache.get_recent_pids(watched_dir, exclude_pids=exclude)
        if not candidates:
            candidates = self.tracker.find_processes_touching_path(watched_dir, exclude_pids=exclude)
        ordered += [p for p in candidates if p not in ordered]
        return ordered

    # One attack = one incident report. A new detection while the previous attack
    # is still going (or ended less than this long ago) joins the same report.
    INCIDENT_JOIN_SECONDS = 120

    def _begin_incident(self, confidence, explanation, folders, incident_start, process_info):
        now = time.time()
        trigger = (explanation or {}).get("summary", "file activity matches a ransomware pattern")
        trigger = trigger[0].lower() + trigger[1:] if trigger else trigger
        trigger = trigger.rstrip(".")
        with self._incident_lock:
            cur = self._current_incident
            if cur and now - cur["last"] < self.INCIDENT_JOIN_SECONDS and self.incidents.get(cur["id"]):
                cur["last"] = now
                self.incidents.update(cur["id"], status="responding", ended_at=None, confidence=confidence)
                self.incidents.event(cur["id"], f"Attack still active - detected again ({trigger})")
                return cur["id"]
            severity = "CRITICAL" if confidence >= 0.97 else "HIGH" if confidence >= 0.85 else "MEDIUM"
            inc_id = self.incidents.open(started_at=incident_start, severity=severity, confidence=confidence,
                                         trigger=trigger, folders=folders, process=process_info)
            self._current_incident = {"id": inc_id, "last": now}
            return inc_id

    def _touch_incident(self):
        with self._incident_lock:
            if self._current_incident:
                self._current_incident["last"] = time.time()

    def _collect_extras(self, inc_id, affected_paths):
        """Things the report should also mention: encrypted copies/ransom notes the attack
        left behind, and changed files that had NO backup (so cannot be restored)."""
        try:
            affected = {os.path.normcase(os.path.abspath(p)) for p in affected_paths}
            leftovers, unprotected, seen = [], [], set()
            for e in self.extractor.recent_events(180):
                if not e.path or e.event_type == "deleted" or not os.path.exists(e.path) or e.path in seen:
                    continue
                seen.add(e.path)
                ext = os.path.splitext(e.path)[1].lower()
                src = os.path.normcase(os.path.abspath(e.src_path)) if e.src_path else None
                if looks_like_ransom_note(e.path):
                    leftovers.append({"path": e.path, "name": os.path.basename(e.path), "kind": "ransom note"})
                elif (src and src in affected) or ext in RANSOM_EXTENSIONS:
                    leftovers.append({"path": e.path, "name": os.path.basename(e.path),
                                      "kind": "encrypted copy left by the attack"})
                elif (e.entropy is not None and e.entropy >= 7.5 and e.event_type in ("created", "modified", "renamed")
                      and not self.rollback.has_backup(e.path)):
                    unprotected.append(e.path)
            self.incidents.set_extra(inc_id, unprotected=unprotected, leftovers=leftovers)
        except Exception as e:
            logger.warning(f"[incident] Could not collect extra report details: {sanitize_for_log(e)}")

    def _restore_for_incident(self, inc_id, damaged, incident_start, label):
        """Restores the damaged files and records the outcome in the incident report."""
        self.incidents.add_files(inc_id, damaged)
        result = self.rollback.restore_all([d["path"] for d in damaged], before_epoch=incident_start)
        self.incidents.apply_restore(inc_id, result.get("details"), result.get("elapsed_seconds"))
        self.incidents.event(inc_id, f"{label}: restored {result['succeeded']} of {result['total_files']} file(s)"
                                     + (f", {result['failed']} failed" if result["failed"] else ""))
        return result

    def _respond(self, features: dict, last_event):
        self._incident_active = True
        self._last_incident_at = time.time()
        confidence = compute_dynamic_confidence(features)
        explanation = build_threshold_explanation(features)
        kill_threshold = self.cfg["response"]["kill_confidence_threshold"]

        # 1. Stop new backups right away, so encrypted copies cannot pile up.
        self.backup_scheduler.pause_for(self.cfg["response"].get("backup_pause_seconds", 120))

        # Work out which folders were touched and when the trouble started.
        recent = self.extractor.recent_events(60)
        folders = {os.path.dirname(e.path) for e in recent if e.path}
        folders.add(os.path.dirname(last_event.path))
        incident_start = self.extractor.first_suspicious_time(60)
        if incident_start is None:
            incident_start = min((e.timestamp for e in recent), default=time.time())

        inc_id = self._begin_incident(confidence, explanation, folders, incident_start, None)
        self.incidents.event(inc_id, "Backups paused so no encrypted copy can replace a clean one")

        # 2. Find and (if safe) stop the attacker.
        candidate_pids = self._find_attacker(last_event)
        kill_results = []
        kill_skipped_reason = None
        process_info = {"pid": None, "name": "unidentified", "parent_name": None, "cmdline": None}

        if candidate_pids:
            attacker_pid = candidate_pids[0]
            process_info = self.tracker.get_process_info(attacker_pid)   # before any kill
            self.incidents.update(inc_id, process={k: process_info.get(k) for k in ("pid", "name", "parent_name")})
            self.incidents.event(inc_id, f"Suspect process: {sanitize_for_log(process_info.get('name'), 80)} "
                                         f"(PID {process_info.get('pid')})")
            if confidence >= kill_threshold:
                kill_results = self.killer.kill_process_tree(
                    attacker_pid, expected_create_time=process_info.get("create_time"))
                summary = summarize_kill(kill_results)
                if not summary["killed"]:
                    why = "; ".join(summary["refused"]) or summary["main_status"] or "unknown"
                    kill_skipped_reason = f"process not terminated ({why})"
                    logger.warning(f"[response] {kill_skipped_reason}")
            else:
                kill_skipped_reason = (
                    f"confidence {confidence:.2f} below kill threshold {kill_threshold:.2f} - "
                    f"rolling back and alerting only, not killing the process on uncertain evidence"
                )
                logger.warning(f"[response] {kill_skipped_reason} (PID {attacker_pid} identified but spared)")
        else:
            kill_skipped_reason = "no process could be confidently attributed to this activity"
            logger.warning(f"[response] {kill_skipped_reason} - skipping auto-kill, rollback + alert only.")

        kill_info = summarize_kill(kill_results)
        if kill_info["killed"]:
            self.incidents.update(inc_id, process_killed=True)
            self.incidents.event(inc_id, "Attacking process stopped")
        else:
            self.incidents.update(inc_id, kill_note=kill_skipped_reason)
            self.incidents.event(inc_id, f"Process not stopped: {kill_skipped_reason}")

        # 3. Restore EVERY damaged file (data protection is lower-risk than a kill).
        restore_result = {"succeeded": 0, "failed": 0, "total_files": 0, "elapsed_seconds": 0}
        affected = []
        if self.cfg["response"]["auto_rollback"] and confidence >= self.cfg["response"]["rollback_confidence_threshold"]:
            last_path = last_event.path
            extra = [last_path[:-len(".locked")]] if last_path.endswith(".locked") else [last_path]
            damaged = self.rollback.find_damaged_detailed(
                folders, extra_paths=[p for p in extra if self.rollback.has_backup(p)])
            if damaged:
                self.incidents.event(inc_id, f"Found {len(damaged)} damaged file(s); restoring from clean backups")
                restore_result = self._restore_for_incident(inc_id, damaged, incident_start, "Restore")
                affected = [d["path"] for d in damaged]
            else:
                logger.warning("[response] No damaged backed-up files found to restore.")
                self.incidents.event(inc_id, "No damaged backed-up file found yet")

        self.incident_count += 1
        alert = self.alerter.send_alert(
            threat_type="ransomware_activity",
            confidence=confidence,
            affected_files=affected or [last_event.path],
            process_info=process_info,
            detection_features=features,
            action_taken={
                "process_killed": kill_info["killed"],
                "kill_method": kill_info["method"],
                "kill_skipped_reason": kill_skipped_reason,
                "files_restored": restore_result["succeeded"],
                "files_not_restored": restore_result["failed"],
                "restore_elapsed_seconds": restore_result.get("elapsed_seconds"),
            },
            explanation=explanation,
        )
        self.incidents.add_alert(inc_id, alert["alert_id"])
        logger.warning(f"[response] Alert {alert['alert_id']} sent, severity={alert['threat']['severity']}, "
                       f"restored {restore_result['succeeded']}/{restore_result['total_files']} file(s), signed=yes")

        # 4. The attack may still be running (kill spared or failed, or an
        # attacker that is faster than the first restore). Keep sweeping.
        attacker_pid = candidate_pids[0] if candidate_pids else None
        still_running = attacker_pid is not None and not kill_info["killed"]
        threading.Thread(
            target=self._containment_loop,
            args=(folders, incident_start, attacker_pid if still_running else None,
                  process_info.get("create_time"), process_info, kill_threshold, inc_id),
            daemon=True, name="containment",
        ).start()

    def _containment_loop(self, folders, incident_start, attacker_pid, create_time, process_info,
                          kill_threshold, inc_id=None):
        """After the first response: for up to response.containment_seconds, keep
        restoring newly damaged files (and kill the attacker if the evidence has
        become strong enough). Ends early after 5 quiet seconds in a row."""
        try:
            max_seconds = float(self.cfg["response"].get("containment_seconds", 60))
            deadline = time.time() + max_seconds
            folders = set(folders)
            total_restored = failed = 0
            killed_later = None
            quiet = 0
            try:
                while time.time() < deadline and not self._outbox_stop.is_set() and quiet < 5:
                    time.sleep(1.0)
                    recent = self.extractor.recent_events(10)
                    folders.update(os.path.dirname(e.path) for e in recent if e.path)
                    self._touch_incident()

                    if attacker_pid is not None and killed_later is None:
                        conf = compute_dynamic_confidence(self.extractor.compute_features())
                        if conf >= kill_threshold:
                            results = self.killer.kill_process_tree(attacker_pid, expected_create_time=create_time)
                            if summarize_kill(results)["killed"]:
                                killed_later = summarize_kill(results)
                                logger.warning(f"[response] Attacker stopped during containment (confidence {conf:.2f}).")
                                if inc_id:
                                    self.incidents.update(inc_id, process_killed=True)
                                    self.incidents.event(inc_id, "Attacking process stopped during containment")

                    damaged = self.rollback.find_damaged_detailed(folders)
                    if damaged:
                        quiet = 0
                        if inc_id:
                            result = self._restore_for_incident(inc_id, damaged, incident_start, "Containment sweep")
                        else:
                            result = self.rollback.restore_all([d["path"] for d in damaged], before_epoch=incident_start)
                        total_restored += result["succeeded"]
                        failed += result["failed"]
                    else:
                        quiet += 1
            except Exception as e:
                logger.error(f"[service] Containment sweep error: {sanitize_for_log(e)}", exc_info=True)
                if inc_id:
                    self.incidents.close(inc_id, "needs_attention",
                                         "The containment sweep stopped because of an internal error (see the log)")
                return

            if total_restored or killed_later:
                alert = self.alerter.send_alert(
                    threat_type="ransomware_activity",
                    confidence=compute_dynamic_confidence(self.extractor.compute_features()),
                    affected_files=sorted(folders)[:50],
                    process_info=process_info,
                    detection_features=self.extractor.compute_features(),
                    action_taken={
                        "process_killed": bool(killed_later),
                        "kill_method": (killed_later or {}).get("method"),
                        "kill_skipped_reason": None if killed_later else "attack continued; follow-up restore only",
                        "files_restored": total_restored,
                        "files_not_restored": failed,
                        "restore_elapsed_seconds": None,
                    },
                    explanation=None,
                )
                if inc_id:
                    self.incidents.add_alert(inc_id, alert["alert_id"])
                logger.warning(f"[response] Containment finished: {total_restored} more file(s) restored"
                               f"{', attacker stopped' if killed_later else ''}.")

            if inc_id:
                leftover_damage = self.rollback.find_damaged_detailed(folders)
                current = self.incidents.get(inc_id) or {}
                still_failed = [f for f in current.get("files", []) if f.get("restored") is False]
                self._collect_extras(inc_id, [f["path"] for f in current.get("files", [])])
                if leftover_damage or still_failed:
                    self.incidents.close(inc_id, "needs_attention",
                                         f"Attack quiet. {len(still_failed) + len(leftover_damage)} file(s) still need attention")
                else:
                    self.incidents.close(inc_id, "contained", "Attack quiet - incident contained, all damaged files restored")
        finally:
            self._incident_active = False

    def restore_incident(self, inc_id: str) -> dict:
        """Operator action ('Restore again'): retries every file of a report that is not restored."""
        item = self.incidents.get(inc_id)
        if not item:
            return {"status": "error", "reason": "unknown incident"}
        todo = [{"path": f["path"], "reason": "missing"} for f in item.get("files", []) if f.get("restored") is not True]
        damaged = {d["path"]: d for d in self.rollback.find_damaged_detailed(item.get("folders", []))}
        for d in damaged.values():
            if d["path"] not in {t["path"] for t in todo}:
                todo.append(d)
        if not todo:
            return {"status": "ok", "restored": 0, "failed": 0, "note": "nothing left to restore"}
        result = self._restore_for_incident(inc_id, todo, item.get("started_at"), "Manual restore")
        current = self.incidents.get(inc_id) or {}
        if not any(f.get("restored") is not True for f in current.get("files", [])):
            self.incidents.close(inc_id, "contained", "All damaged files are restored")
        return {"status": "ok", "restored": result["succeeded"], "failed": result["failed"]}

    # ------------------------------------------------------ integrity + scan

    INTEGRITY_MIN_FILES = 3

    def _integrity_check(self):
        """Second line of defence, run after every full backup sweep. The live monitor
        reacts to FAST attacks; a slow or stealthy attack can stay under its rate limit.
        This compares every protected file with its clean backup: if several files have
        turned into encrypted-looking data (or lost their file-format signature), that is
        an attack whatever its speed. Deleted/emptied files are NOT acted on here (people
        delete files); they show up in the manual scan instead."""
        if self._incident_active or not self._active:
            return
        damaged = [d for d in self.rollback.find_damaged_detailed(self.watch_paths)
                   if d["reason"] in ("encrypted", "format")]
        if len(damaged) < self.INTEGRITY_MIN_FILES:
            return
        self._handle_integrity_damage(damaged)

    def _handle_integrity_damage(self, damaged):
        with self._respond_lock:
            if self._incident_active:
                return
            self._incident_active = True
        self._last_incident_at = time.time()
        try:
            n = len(damaged)
            logger.warning(f"[integrity] {n} protected file(s) turned into encrypted-looking data "
                           f"without a fast attack - treating as a slow/stealthy attack.")
            self.backup_scheduler.pause_for(self.cfg["response"].get("backup_pause_seconds", 120))
            folders = {os.path.dirname(d["path"]) for d in damaged}
            explanation = {"method": "integrity_check", "summary":
                           f"{n} protected files turned into encrypted-looking data without a fast attack "
                           f"(slow or stealthy encryption found by the integrity check)"}
            inc_id = self._begin_incident(0.9, explanation, folders, time.time(), None)
            self.incidents.event(inc_id, "Found by the integrity check that compares every file with its clean backup")
            result = self._restore_for_incident(inc_id, damaged, None, "Restore")
            self.incident_count += 1
            alert = self.alerter.send_alert(
                threat_type="ransomware_activity", confidence=0.9,
                affected_files=[d["path"] for d in damaged][:200],
                process_info={"pid": None, "name": "unidentified", "parent_name": None, "cmdline": None},
                detection_features=self.extractor.compute_features(),
                action_taken={"process_killed": False, "kill_method": None,
                              "kill_skipped_reason": "found by integrity check; no live process attribution",
                              "files_restored": result["succeeded"], "files_not_restored": result["failed"],
                              "restore_elapsed_seconds": result.get("elapsed_seconds")},
                explanation=explanation)
            self.incidents.add_alert(inc_id, alert["alert_id"])
            self.incidents.update(inc_id, kill_note="found by the integrity check, so no live process was identified")
            self._collect_extras(inc_id, [d["path"] for d in damaged])
            self.incidents.close(inc_id, "needs_attention" if result["failed"] else "contained",
                                 "Integrity check finished")
        except Exception as e:
            logger.error(f"[integrity] response failed: {sanitize_for_log(e)}", exc_info=True)
        finally:
            self._incident_active = False

    def scan_now(self) -> dict:
        """Starts a disk scan of the protected folders in the background."""
        if not self._active:
            return {"status": "error", "reason": "Protection is not on (no consent yet)."}
        with self._scan_lock:
            if self.last_scan["running"]:
                return {"status": "running"}
            self.last_scan = {"running": True, "started_at": time.time(), "result": None}

        def work():
            result = None
            try:
                result = scan_folders(self.rollback, self.watch_paths, ignore_dirs=self.ignore_dirs,
                                      stop_flag=self._stop_event.is_set)
            except Exception as e:
                logger.error(f"[scan] failed: {sanitize_for_log(e)}", exc_info=True)
                result = {"error": "The scan failed (see the log)."}
            finally:
                with self._scan_lock:
                    self.last_scan = {"running": False, "started_at": self.last_scan["started_at"],
                                      "result": result}

        threading.Thread(target=work, daemon=True, name="scan").start()
        return {"status": "started"}

    def restore_from_scan(self) -> dict:
        """Operator action: restore every damaged file the last scan found (and can restore)."""
        result = (self.last_scan or {}).get("result") or {}
        targets = [f["restore_path"] for f in result.get("findings", [])
                   if f["kind"] in ("encrypted", "format destroyed", "missing", "emptied")
                   and f.get("restorable")]
        targets = sorted(set(targets))
        if not targets:
            return {"status": "ok", "restored": 0, "failed": 0, "note": "nothing to restore"}
        res = self.rollback.restore_all(targets)
        logger.warning(f"[scan] Operator restored {res['succeeded']} of {res['total_files']} file(s) from a scan")
        return {"status": "ok", "restored": res["succeeded"], "failed": res["failed"]}

    # --------------------------------------------------------------- outbox

    def _outbox_loop(self):
        interval = self.cfg.get("main", {}).get("outbox_retry_seconds", 30)
        while not self._stop_event.wait(interval):
            try:
                self.alerter.flush_outbox()
            except Exception as e:
                logger.error(f"[service] Outbox retry failed: {sanitize_for_log(e)}")

    # ------------------------------------------------- consent / activation

    def _consent_loop(self):
        """Watches the consent file: starts protection when it appears,
        stops it when it is revoked or no longer covers the folders."""
        while not self._stop_event.is_set():
            try:
                covered = self.consent.covers(self.watch_paths)
                if covered and not self._active:
                    self._activate()
                elif not covered and self._active:
                    logger.warning("[consent] Consent withdrawn (or folders changed) - stopping protection.")
                    self._deactivate()
            except Exception as e:
                logger.error(f"[service] Consent check failed: {sanitize_for_log(e)}", exc_info=True)
            self._stop_event.wait(1.0)

    def _activate(self):
        """Starts everything that touches user files. Only runs with consent."""
        with self._activate_lock:
            if self._active:
                return
            self.observer = Observer()
            self.backup_scheduler = BackupScheduler(
                self.rollback, self.watch_paths,
                interval_seconds=self.cfg.get("main", {}).get("backup_interval_seconds", 2),
                ignore_dirs=self.ignore_dirs,
                full_sweep_seconds=self.cfg.get("main", {}).get("full_sweep_seconds", 300),
                settle_seconds=self.cfg.get("main", {}).get("backup_settle_seconds", 5),
                min_interval_seconds=self.cfg.get("main", {}).get("backup_min_interval_seconds", 30),
                on_sweep=self._integrity_check,
            )
            self.pid_cache = RollingPidCache(
                self.tracker, self.watch_paths, sample_interval=0.5, max_age_seconds=6.0,
                activity_probe=lambda: bool(self.extractor.recent_events(3)))
            self.command_listener = CommandListener(self.rollback, config_path=self._config_path)

            self._start_observer()
            self.pid_cache.start()
            self.command_listener.start()
            self.backup_scheduler.start()
            self._active = True
            logger.warning("[service] Protection is ACTIVE (monitoring + backups running).")

    def _start_observer(self):
        """(Re)creates the file-system observer and schedules every watch folder."""
        self.observer = Observer()
        watching = 0
        for path in self.watch_paths:
            if os.path.isdir(path):
                self.observer.schedule(self.handler, path, recursive=True)
                logger.info(f"[monitor] Watching: {path}")
                watching += 1
            else:
                logger.warning(f"[monitor] Skipping missing path: {path}")
        if watching == 0:
            logger.error("[monitor] NO watch folder exists - nothing is being protected. "
                         "Check live_monitor.watch_paths.")
        self.observer.start()

    def _watchdog_loop(self):
        """If the file observer thread ever dies (it can, on Windows, after an
        unexpected OS error) protection would silently be blind. Detect that and
        restart it, loudly."""
        while not self._stop_event.wait(2.0):
            try:
                with self._activate_lock:
                    if not self._active or self.observer is None:
                        continue
                    if self.observer.is_alive():
                        continue
                    self.observer_restarts += 1
                    logger.error(f"[watchdog] The file monitor stopped unexpectedly - restarting it "
                                 f"(restart #{self.observer_restarts}).")
                    try:
                        self.observer.stop()
                    except Exception:
                        pass
                    self._start_observer()
            except Exception as e:
                logger.error(f"[watchdog] check failed: {sanitize_for_log(e)}", exc_info=True)

    def _deactivate(self):
        with self._activate_lock:
            if not self._active:
                return
            self._active = False
            for fn in (lambda: self.observer.stop(), lambda: self.observer.join(timeout=5),
                       lambda: self.backup_scheduler.stop(), lambda: self.command_listener.stop(),
                       lambda: self.pid_cache.stop()):
                try:
                    fn()
                except Exception as e:
                    logger.warning(f"[service] Shutdown step failed: {sanitize_for_log(e)}")
            self.observer = self.backup_scheduler = self.pid_cache = self.command_listener = None
            logger.warning("[service] Protection is OFF (standby).")

    # --------------------------------------------------------------- status

    def _recent_events(self, n: int = 15) -> list:
        out = []
        for e in self.extractor.snapshot_events()[-n:]:
            out.append({
                "t": e.timestamp, "type": e.event_type,
                "name": sanitize_for_log(os.path.basename(e.path or ""), 120),
                "entropy": None if e.entropy is None else round(e.entropy, 2),
            })
        out.reverse()
        return out

    def _meter(self) -> dict:
        """Live values of the detection rules next to their thresholds, so the
        dashboard can show how close the layer is to raising an incident."""
        try:
            f = self.extractor.compute_features()
        except Exception:
            return {}
        return {
            "file_rate_5s": round(f.get("file_rate_5s", 0), 2), "rate_threshold": FILE_RATE_THRESHOLD,
            "avg_entropy_5s": round(f.get("avg_entropy_5s", 0), 2), "entropy_threshold": ENTROPY_THRESHOLD,
            "high_entropy_5s": f.get("high_entropy_count_5s", 0), "high_entropy_5s_needed": MIN_HIGH_ENTROPY_EVENTS,
            "high_entropy_60s": f.get("high_entropy_count_60s", 0), "high_entropy_60s_needed": SUSTAINED_HIGH_ENTROPY_COUNT,
            "renamed_with_new_ext_10s": f.get("appended_ext_count_10s", 0), "renamed_needed": EXT_APPEND_MIN_COUNT,
            "ransom_notes_60s": f.get("ransom_note_count_60s", 0),
        }

    def snapshot(self) -> dict:
        """Everything the dashboard shows about the service (also written to disk)."""
        now = time.time()
        sched = self.backup_scheduler
        progress = dict(sched.progress) if sched else {"phase": "off", "total": 0, "done": 0, "copied": 0,
                                                        "baseline_complete": False, "last_full_pass": None}
        consent = self.consent.read()
        stats = self.rollback.stats()
        return {
            "timestamp": now,
            "pid": self._own_pid,
            "state": self.state,
            "started_at": self.started_at,
            "watch_paths": self.watch_paths,
            "consent": None if consent is None else {
                "granted_at": consent.get("granted_at"), "granted_by": consent.get("granted_by")},
            "backup": {**progress, "paused": bool(sched and sched.paused),
                       "files_protected": stats["files_tracked"], "versions": stats["versions"],
                       "backup_root": stats["backup_root"], "last_backup_at": stats.get("last_backup_at"),
                       "guard_blocks": stats.get("guard_blocks", 0),
                       "settle_seconds": getattr(sched, "settle_seconds", None) if sched else None,
                       "min_interval_seconds": getattr(sched, "min_interval_seconds", None) if sched else None},
            "monitor": {"events_total": self._events_total,
                        "observer_alive": bool(self.observer and self.observer.is_alive()),
                        "observer_restarts": self.observer_restarts,
                        "raw_events_seen": getattr(self.handler, "events_seen", 0),
                        "handler_errors": getattr(self.handler, "errors", 0),
                        "last_event_at": getattr(self.handler, "last_event_at", None),
                        "meter": self._meter(),
                        "events_last_60s": len(self.extractor.recent_events(60)),
                        "recent": self._recent_events()},
            "incidents": self.incident_count,
            "last_incident_at": self._last_incident_at,
            "platform_outbox": self.alerter.outbox_size(),
        }

    def _write_snapshot(self):
        tmp = self.status_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.snapshot(), f)
            os.replace(tmp, self.status_path)
        except (OSError, TypeError, ValueError) as e:
            logger.debug(f"[service] Could not write status file: {sanitize_for_log(e)}")

    def _status_loop(self):
        while not self._stop_event.is_set():
            try:
                self._write_snapshot()
            except Exception as e:  # never let the status thread die
                logger.error(f"[service] status snapshot failed: {sanitize_for_log(e)}", exc_info=True)
            self._stop_event.wait(1.0)

    # ------------------------------------------------------------ lifecycle

    def start(self):
        self.started_at = time.time()
        if self._auto_consent_enabled() and not self.consent.covers(self.watch_paths):
            logger.warning("[consent] Automatic consent is enabled (NEURASHIELD_AUTO_CONSENT / "
                           "main.require_consent=false) - granting.")
            self.grant_consent(by="automatic setting")
        self.resource_monitor.start()
        for target, name in ((self._outbox_loop, "outbox"), (self._status_loop, "status"),
                             (self._consent_loop, "consent"), (self._watchdog_loop, "watchdog")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)
        if self.consent.covers(self.watch_paths):
            logger.info("[service] Consent found - starting protection.")
        else:
            logger.warning("[service] STANDBY: no consent for protection yet. No files are being "
                           "read, watched or copied. Grant consent in the dashboard "
                           "(or run with --consent).")
        logger.info("[service] NeuraShield Ransomware Killer service is running.")

    def stop(self):
        self._stop_event.set()
        self._deactivate()
        self.resource_monitor.stop()
        for t in self._threads:
            t.join(timeout=3)
        self._write_snapshot_stopped()
        logger.info("[service] Stopped.")

    def _write_snapshot_stopped(self):
        try:
            os.remove(self.status_path)       # dashboard then shows "service not running"
        except OSError:
            pass

    def status(self) -> dict:
        snap = self.snapshot()
        return {
            "running": self.started_at is not None and not self._stop_event.is_set(),
            "state": snap["state"],
            "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "incidents": self.incident_count,
            "backups_paused": snap["backup"]["paused"],
            "backup_stats": self.rollback.stats(),
            "backup_progress": {k: snap["backup"][k] for k in ("phase", "total", "done", "baseline_complete")},
            "platform_outbox": self.alerter.outbox_size(),
            "watch_paths": self.watch_paths,
        }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="NeuraShield Ransomware Killer - continuous service")
    parser.add_argument("--test-duration", type=int, default=None,
                        help="Stop automatically after N seconds (for automated testing).")
    parser.add_argument("--dashboard", action="store_true",
                        help="Also start the local dashboard (127.0.0.1:5050).")
    parser.add_argument("--consent", action="store_true",
                        help="Agree to protect (watch + back up) the configured folders, then run. "
                             "Without it the service stays in standby until consent is given "
                             "in the dashboard.")
    parser.add_argument("--check", action="store_true",
                        help="Print a readiness report and exit.")
    args = parser.parse_args(argv)

    if args.check:
        from src.selfcheck import run_selfcheck
        return run_selfcheck()

    import signal

    from src.layer2 import Layer2
    layer = Layer2(enable_dashboard=args.dashboard)
    stop_requested = threading.Event()

    def _request_stop(signum, frame):
        stop_requested.set()

    # Ctrl+C (and Ctrl+Break on Windows) only set a flag; the main loop below
    # reacts within half a second.
    for name in ("SIGINT", "SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), _request_stop)
            except (ValueError, OSError):
                pass

    layer.start()
    if args.consent:
        layer.grant_consent(by="command line --consent")
    print(f"[main] Running (state: {layer.state}). Press Ctrl+C to stop.", flush=True)
    if layer.state == "standby" and not args.consent:
        print("[main] STANDBY: nothing is being protected yet. Open the dashboard and click "
              "'Start protecting', or restart with --consent.", flush=True)
    deadline = time.time() + args.test_duration if args.test_duration else None
    while not stop_requested.is_set() and (deadline is None or time.time() < deadline):
        stop_requested.wait(0.5)

    print("[main] Stopping...", flush=True)
    stopper = threading.Thread(target=layer.stop, daemon=True)
    stopper.start()
    stopper.join(timeout=15)
    if stopper.is_alive():
        logger.error("[service] Shutdown took too long - forcing exit.")
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# =============================================================
# NeuraShield — Layer 4 — Unified Real-Time Call Monitor
# =============================================================

import os
import sys
import time
import threading
from datetime import datetime

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.fusion.multimodal_fusion_v5 import fuse_scores, save_alert
from src.watchers.call_audio_monitor import (
    call_app_running, capture_and_classify as capture_and_classify_audio
)
from src.watchers.call_video_monitor import CallVideoMonitor

AUDIO_POLL_SECONDS = 4     # matches the audio model's 4s window
REQUIRE_CALL_APP   = os.environ.get("NEURASHIELD_CALL_MONITOR_REQUIRE_APP", "0") == "1"
# Set NEURASHIELD_CALL_MONITOR_REQUIRE_APP=0 to test capture/detection

ALERT_EMIT_COOLDOWN_SECONDS = int(
    os.environ.get("NEURASHIELD_CALL_MONITOR_ALERT_COOLDOWN", "60"))

_UNKNOWN = {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}

STATS = {
    "audio_updates": 0,
    "video_updates": 0,
    "fused_checks": 0,
    "high_risk_calls": 0,
    "medium_risk_calls": 0,
    "last_decision": None,        # "HIGH_RISK" | "MEDIUM_RISK" | "LOW_RISK"
    "last_risk_score": None,      # the most recent fused risk score, every cycle
    "last_check": None,           # ISO timestamp
    "last_error": None,
}

_shared_state_singleton = None

def get_shared_state():
    """Expose the running SharedState so a status poller (or the
    dashboard endpoint) can read the latest audio/video verdicts
    without needing its own capture hardware."""
    return _shared_state_singleton


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class SharedState:
    """Thread-safe holder for the latest audio and video verdicts."""
    def __init__(self):
        self._lock = threading.Lock()
        self.audio_result = dict(_UNKNOWN)
        self.video_result = dict(_UNKNOWN)
        self.audio_updates = 0
        self.video_updates = 0

    def set_audio(self, result):
        with self._lock:
            self.audio_result = result
            self.audio_updates += 1

    def set_video(self, result):
        with self._lock:
            self.video_result = result
            self.video_updates += 1

    def snapshot(self):
        with self._lock:
            return dict(self.audio_result), dict(self.video_result)


def audio_worker(state: SharedState, stop_event: threading.Event):
    """Background thread: continuously classify rolling 4s audio windows."""

    _com_initialized = False
    if sys.platform == "win32":
        try:
            import pythoncom
            pythoncom.CoInitialize()
            _com_initialized = True
        except ImportError:
            log("[audio][WARN] pywin32 not installed — COM initialization "
                "skipped, audio capture will likely fail with "
                "CO_E_NOTINITIALIZED (Error 0x800401f0). Run: pip install pywin32")
        except Exception as e:
            log(f"[audio][WARN] CoInitialize failed: {e} — continuing anyway, "
                "audio capture may still fail")

    try:
        try:
            import soundcard as sc
        except Exception as e:
            
            log(f"[audio][FATAL] Could not load audio backend: {e}")
            STATS["last_error"] = f"audio backend unavailable: {e}"
            return

        try:
            default_speaker = sc.default_speaker()
            loopback_mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
            log(f"[audio] Listening on loopback device: {default_speaker.name}")
        except Exception as e:
            # Setup failures (no default audio device, no permission, running
            # in an environment with no audio hardware/pulseaudio, COM not
            # initialized on Windows, etc.) are real and should be reported,
            # not crash this thread silently.
            log(f"[audio][FATAL] Could not initialize audio capture: {e}")
            STATS["last_error"] = f"audio capture init failed: {e}"
            return

        try:
            with loopback_mic.recorder(samplerate=48000) as mic:
                while not stop_event.is_set():
                    active = (not REQUIRE_CALL_APP) or call_app_running()
                    if not active:
                        stop_event.wait(3)
                        continue
                    try:
                        result = capture_and_classify_audio(mic)
                        state.set_audio(result)
                    except Exception as e:
                        log(f"[audio][ERROR] {e}")
                        stop_event.wait(2)
        except Exception as e:
            log(f"[audio][FATAL] Recorder failed: {e}")
            STATS["last_error"] = f"audio recorder failed: {e}"
    finally:
        if _com_initialized:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass


def video_worker(state: SharedState, stop_event: threading.Event):
    """Background thread: continuously classify rolling ~30s video windows."""
    try:
        monitor = CallVideoMonitor()
    except Exception as e:
        log(f"[video][FATAL] Could not initialize video capture: {e}")
        STATS["last_error"] = f"video capture init failed: {e}"
        return

    while not stop_event.is_set():
        active = (not REQUIRE_CALL_APP) or call_app_running()
        if not active:
            stop_event.wait(3)
            continue
        try:
            result = monitor.poll_once()
            if result:
                state.set_video(result)
        except Exception as e:
            log(f"[video][ERROR] {e}")
        stop_event.wait(2)   # CAPTURE_INTERVAL_SECONDS, matches call_video_monitor default


def main(stop_event=None):
    """
    stop_event: optional threading.Event, shared with the audio/video
    worker threads. When set (e.g. from the dashboard's Stop button
    via watcher_manager), all three loops (main + both workers) exit
    cleanly and promptly. If None, an internal event is created and
    only KeyboardInterrupt stops it — the standalone `python
    call_monitor.py` CLI usage is unaffected.
    """
    global _shared_state_singleton
    log("=" * 64)
    log("  NeuraShield — Unified Real-Time Call Monitor")
    log("  (audio voice-clone + video deepfake, fused)")
    log("=" * 64)

    state = SharedState()
    _shared_state_singleton = state
    if stop_event is None:
        stop_event = threading.Event()

    t_audio = threading.Thread(target=audio_worker, args=(state, stop_event), daemon=True)
    t_video = threading.Thread(target=video_worker, args=(state, stop_event), daemon=True)
    t_audio.start()
    t_video.start()

    # THROTTLE: without this, a single continuing threat (e.g. a
    # deepfake persisting for the length of a call) creates a new
    # persisted alert every fusion cycle (~every AUDIO_POLL_SECONDS,
    # i.e. every few seconds) — flooding both the notification banner
    # and the audit log with dozens of near-identical entries for what
    # is really ONE ongoing event. STATS/console logging still happen
    # every cycle (so live status stays accurate), but a new alert is
    # only PERSISTED (and sent toward Layer D) once per cooldown window
    # per severity level — escalating from MEDIUM to HIGH still alerts
    # immediately rather than waiting out the cooldown, since that's a
    # meaningful change even mid-cooldown.
    _rank = {"LOW_RISK": 0, "MEDIUM_RISK": 1, "HIGH_RISK": 2}
    _last_saved_time = None
    _last_saved_rank = -1

    try:
        last_fused_audio_count = -1
        while not stop_event.is_set():
            if stop_event.wait(AUDIO_POLL_SECONDS):
                break   # stop requested during the wait
            audio_res, video_res = state.snapshot()
            STATS["audio_updates"] = state.audio_updates
            STATS["video_updates"] = state.video_updates
            STATS["last_check"] = datetime.now().isoformat()

            # Only fuse+log when a new audio reading has arrived, so we
            # don't spam duplicate fused decisions on an unchanged audio
            # window while waiting for the next one.
            if state.audio_updates == last_fused_audio_count:
                continue
            last_fused_audio_count = state.audio_updates

            alert = fuse_scores(_UNKNOWN, video_res, audio_res)
            dec = alert["final_decision"]
            tag = {"HIGH_RISK": "🚨 HIGH RISK", "MEDIUM_RISK": "⚠ MEDIUM",
                  "LOW_RISK": "safe"}[dec]
            log(f"[fused] [{tag}] risk={alert['final_risk_score']:.3f} "
                f"audio={audio_res.get('prediction')} "
                f"video={video_res.get('prediction')} "
                f"(video updates so far: {state.video_updates})")

            # Live stats update EVERY cycle regardless of throttling, so
            # the dashboard's "last risk / last decision" always reflects
            # what's actually happening right now, even between saves.
            STATS["fused_checks"] += 1
            STATS["last_decision"] = dec
            STATS["last_risk_score"] = alert["final_risk_score"]

            if dec in ("HIGH_RISK", "MEDIUM_RISK"):
                now = datetime.now()
                dec_rank = _rank[dec]
                escalated = dec_rank > _last_saved_rank
                cooldown_elapsed = (_last_saved_time is None or
                                    (now - _last_saved_time).total_seconds() >= ALERT_EMIT_COOLDOWN_SECONDS)
                if escalated or cooldown_elapsed:
                    save_alert(alert)
                    _last_saved_time = now
                    _last_saved_rank = dec_rank
                    if dec == "HIGH_RISK":
                        STATS["high_risk_calls"] += 1
                    elif dec == "MEDIUM_RISK":
                        STATS["medium_risk_calls"] += 1
                else:
                    log(f"  (throttled — same/lower severity within "
                       f"{ALERT_EMIT_COOLDOWN_SECONDS}s cooldown, not re-persisted)")
            else:
                # Risk dropped back to LOW — reset so the next real
                # threat alerts immediately rather than inheriting an
                # old cooldown from an unrelated earlier event.
                _last_saved_rank = -1

    except KeyboardInterrupt:
        log("Stopping unified call monitor...")
    finally:
        stop_event.set()   # ensure worker threads exit even on error
        log("Call monitor stopped.")


if __name__ == "__main__":
    main()
# keystroke_agent.py
# NeuraShield — Background Keystroke Capture Agent
# Runs silently in background, captures keystrokes from ANY application
# Builds and continuously updates the user's behavioral profile
# Sends scores to the risk engine periodically
#
# This is the REAL production approach:
#   - No browser enrollment box needed
#   - User just works normally
#   - System learns them automatically
#   - Profile updates every 5 minutes
#   - Scoring happens every 30 seconds
#
# Install: pip install pynput psutil requests
#
# How to run alongside the FastAPI server:
#   Terminal 1: python neurashield_platform.py
#   Terminal 2: python keystroke_agent.py
#
# The agent sends data to the FastAPI server via HTTP
# This keeps it decoupled — agent can run as a Windows service

# keystroke_agent.py — FIXED VERSION
# NeuraShield — Background Keystroke Capture Agent
#
# Fixes from previous version:
#   1. Feature units confirmed in seconds (not ms) — fixes 67% false risk
#   2. Scores sent to /api/agent-score endpoint — dashboard updates live
#   3. Profile saved to disk on first build and each update
#   4. Step-up auth triggers face_liveness.py directly on this machine


import time
import threading
import logging
import requests
import numpy as np
import joblib
import os
from collections import deque
from datetime import datetime

try:
    from pynput import keyboard as pynput_keyboard, mouse as pynput_mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("[Agent] pynput not found. Run: pip install pynput")

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────
API_BASE         = "http://localhost:8000"
AGENT_USER_ID    = 9001
WINDOW_SIZE      = 40    # keystrokes per scoring window
SCORE_INTERVAL   = 30    # score every 30 seconds
UPDATE_INTERVAL  = 300   # update profile every 5 minutes
MIN_KS_TO_BUILD  = 60    # keystrokes before first profile build
MIN_KS_TO_SCORE  = 20    # keystrokes before scoring
PROFILE_PATH     = os.path.join("models", "agent_profile.pkl")


class KeystrokeAgent:

    def __init__(self, user_id: int = AGENT_USER_ID):
        self.user_id           = user_id
        self.running           = False

        # ── Keystroke buffer ──────────────────────────────
        self.buffer            = deque(maxlen=300)
        self._press_times      = {}
        self._last_release_t   = None

        # ── Profile state ─────────────────────────────────
        self.profile_built     = False
        self.n_updates         = 0
        self.baseline_features = None   # enrolled profile's feature vector

        # ── Score state (read by dashboard) ───────────────
        self.last_score        = None
        self.last_decision     = None
        self.last_context      = 'normal_browsing'
        self.n_scored          = 0
        self.session_start     = datetime.now()

        # ── Load existing profile if server was restarted ─
        self._load_saved_profile()

    # ─────────────────────────────────────────────────────
    # PROFILE PERSISTENCE
    # ─────────────────────────────────────────────────────
    def _save_profile(self, features: list):
        """Save profile to disk so it survives server restarts."""
        os.makedirs("models", exist_ok=True)
        joblib.dump({
            'user_id':   self.user_id,
            'features':  features,
            'saved_at':  datetime.now().isoformat(),
            'n_updates': self.n_updates,
        }, PROFILE_PATH)
        logger.info(f"[Agent] Profile saved to {PROFILE_PATH}")

    def _load_saved_profile(self):
        """Load previously saved profile and re-enroll into server."""
        if not os.path.exists(PROFILE_PATH):
            return
        try:
            data    = joblib.load(PROFILE_PATH)
            features = data.get('features', [])
            if features:
                # Re-enroll into server (server may have restarted)
                resp = requests.post(
                    f"{API_BASE}/api/enroll",
                    json={"user_id": self.user_id, "features": features},
                    timeout=5
                )
                if resp.status_code == 200:
                    self.profile_built     = True
                    self.baseline_features = features
                    self.n_updates         = data.get('n_updates', 0)
                    logger.info(
                        f"[Agent] Loaded saved profile "
                        f"(saved {data.get('saved_at','?')}, "
                        f"{self.n_updates} previous updates)"
                    )
        except requests.exceptions.ConnectionError:
            logger.info("[Agent] Server not ready yet — profile will re-enroll after scoring starts")
        except Exception as e:
            logger.warning(f"[Agent] Could not load saved profile: {e}")

    # ─────────────────────────────────────────────────────
    # KEYSTROKE CAPTURE
    # ─────────────────────────────────────────────────────
    def _on_press(self, key):
        try:
            self._press_times[str(key)] = time.perf_counter()
        except Exception:
            pass

    def _on_release(self, key):
        try:
            now      = time.perf_counter()
            press_t  = self._press_times.pop(str(key), None)
            if press_t is None:
                return

            # dwell in SECONDS — this is critical
            # BB-MAS training data uses seconds, not milliseconds
            dwell_s  = now - press_t

            # flight in SECONDS — time since last release
            flight_s = (press_t - self._last_release_t) if self._last_release_t else 0.0
            self._last_release_t = now

            # Sanity check — filter noise
            # Normal human dwell: 0.04s to 0.8s
            # Normal human flight: 0.0s to 2.0s
            if not (0.03 <= dwell_s <= 1.5):
                return

            flight_s = max(0.0, min(3.0, flight_s))

            self.buffer.append({
                'dwell_s':  dwell_s,
                'flight_s': flight_s,
                'time':     now,
            })

        except Exception as e:
            logger.debug(f"[Agent] Release error: {e}")

    # ─────────────────────────────────────────────────────
    # FEATURE EXTRACTION
    # ─────────────────────────────────────────────────────
    def _extract_features(self, keystrokes: list) -> list | None:
        """
        Extract 15 features matching the training data format exactly.
        All values in SECONDS to match BB-MAS training distribution.

        BB-MAS typical ranges:
          dwell_mean:  0.08 - 0.25 seconds
          flight_mean: 0.10 - 1.50 seconds
        """
        if len(keystrokes) < 5:
            return None

        dwells  = np.array([k['dwell_s']  for k in keystrokes])
        flights = np.array([k['flight_s'] for k in keystrokes
                           if k['flight_s'] > 0.03])  # ignore near-zero

        if len(dwells) < 5:
            return None

        dm = float(dwells.mean())
        ds = float(dwells.std())   if len(dwells)  > 1 else 0.0
        fm = float(flights.mean()) if len(flights) > 0 else 0.0
        fs = float(flights.std())  if len(flights) > 1 else 0.0

        overlap_rate = float(np.mean([
            1.0 if k['flight_s'] < 0.02 else 0.0
            for k in keystrokes
        ]))
        break_rate = float(np.mean([
            1.0 if k['flight_s'] > 2.0 else 0.0
            for k in keystrokes
        ]))

        features = [
            dm,                                              # dwell_mean
            ds,                                              # dwell_std
            float(dwells.min()),                             # dwell_min
            float(dwells.max()),                             # dwell_max
            float(np.median(dwells)),                        # dwell_median
            fm,                                              # flight_mean
            fs,                                              # flight_std
            float(flights.min()) if len(flights) > 0 else 0, # flight_min
            float(flights.max()) if len(flights) > 0 else 0, # flight_max
            float(np.median(flights)) if len(flights) > 0 else 0, # flight_median
            overlap_rate,                                    # overlap_rate
            break_rate,                                      # session_break_rate
            float(len(keystrokes)),                          # total_keystrokes
            ds / dm if dm > 0 else 0.0,                      # dwell_cv
            fs / fm if fm > 0 else 0.0,                      # flight_cv
        ]

        # Sanity check — if values look wrong, log a warning
        if dm > 1.0 or dm < 0.03:
            logger.warning(
                f"[Agent] Unusual dwell_mean={dm:.4f}s — "
                f"expected 0.05-0.50s. Check timing units."
            )

        return features

    # ─────────────────────────────────────────────────────
    # PROFILE BUILD AND UPDATE
    # ─────────────────────────────────────────────────────
    def _build_profile(self):
        """
        Build or update behavioral profile from current buffer.
        First call: initial enrollment.
        Subsequent calls: adaptive update — profile evolves with user.
        """
        ks = list(self.buffer)
        if len(ks) < MIN_KS_TO_BUILD:
            logger.info(f"[Agent] Need {MIN_KS_TO_BUILD} keystrokes, have {len(ks)}")
            return

        features = self._extract_features(ks[-WINDOW_SIZE:])
        if features is None:
            return

        try:
            resp = requests.post(
                f"{API_BASE}/api/enroll",
                json={"user_id": self.user_id, "features": features},
                timeout=5
            )
            if resp.status_code == 200:
                self.profile_built     = True
                self.baseline_features = features
                self.n_updates        += 1

                # Save to disk
                self._save_profile(features)

                logger.info(
                    f"[Agent] Profile {'updated' if self.n_updates > 1 else 'BUILT'} "
                    f"#{self.n_updates} | "
                    f"dwell={features[0]*1000:.0f}ms | "
                    f"flight={features[5]*1000:.0f}ms | "
                    f"from {len(ks)} keystrokes"
                )
            else:
                logger.warning(f"[Agent] Enroll failed: {resp.status_code} {resp.text[:100]}")

        except requests.exceptions.ConnectionError:
            logger.warning("[Agent] Server not reachable for enrollment")
        except Exception as e:
            logger.error(f"[Agent] Profile build error: {e}")

    # ─────────────────────────────────────────────────────
    # SCORING
    # ─────────────────────────────────────────────────────
    def _score(self):
        """
        Score current typing window against enrolled profile.
        Gets context from session monitor automatically.
        Sends result to /api/agent-score so dashboard updates.
        """
        if not self.profile_built:
            logger.info(
                f"[Agent] Profile not built yet — "
                f"have {len(self.buffer)}/{MIN_KS_TO_BUILD} keystrokes"
            )
            return

        ks = list(self.buffer)
        if len(ks) < MIN_KS_TO_SCORE:
            return

        features = self._extract_features(ks[-WINDOW_SIZE:])
        if features is None:
            return

        # Get current context from session monitor
        context = 'normal_browsing'
        try:
            sm = requests.get(f"{API_BASE}/api/session-activity", timeout=2)
            if sm.status_code == 200:
                context = sm.json().get('current_context', 'normal_browsing')
        except Exception:
            pass

        try:
            resp = requests.post(
                f"{API_BASE}/api/score",
                json={
                    "user_id":  self.user_id,
                    "features": features,
                    "context":  context,
                },
                timeout=5
            )
            if resp.status_code != 200:
                logger.warning(f"[Agent] Score failed: {resp.status_code}")
                return

            result = resp.json()
            risk   = result.get('risk_percent', 0)
            dec    = result.get('decision', 'allow')
            thresh = result.get('adaptive_threshold', 0)

            self.last_score    = risk
            self.last_decision = dec
            self.last_context  = context
            self.n_scored     += 1

            logger.info(
                f"[Agent] Score #{self.n_scored}: "
                f"risk={risk}% | decision={dec} | "
                f"context={context} | threshold={thresh} | "
                f"dwell={features[0]*1000:.0f}ms flight={features[5]*1000:.0f}ms"
            )

            # Send to dedicated agent endpoint so dashboard shows it
            try:
                requests.post(
                    f"{API_BASE}/api/agent-score",
                    json=result,
                    timeout=2
                )
            except Exception:
                pass   # non-fatal

            # Handle step-up auth
            if dec == 'step_up_auth':
                self._trigger_step_up(risk, thresh)

        except requests.exceptions.ConnectionError:
            logger.warning("[Agent] Server not reachable for scoring")
        except Exception as e:
            logger.error(f"[Agent] Scoring error: {e}")

    def _trigger_step_up(self, risk: int, threshold: float):
        """
        Trigger step-up authentication directly.
        Now respects grace period to avoid spamming the user.
        """
        logger.warning(
            f"[Agent] ⚠ STEP-UP AUTH — risk {risk}% > threshold {threshold*100:.0f}%"
        )

        # Check if we are still in grace period from previous successful liveness
        if hasattr(self, 'last_liveness_success') and self.last_liveness_success is not None:
            seconds_since = (datetime.now() - self.last_liveness_success).total_seconds()
            if seconds_since < self.liveness_grace_seconds:
                logger.info(f"[Agent] Liveness grace period active ({int(self.liveness_grace_seconds - seconds_since)}s left) — skipping new check")
                return

        # Try face liveness
        try:
            from face_liveness import run_from_api
            logger.info("[Agent] Opening face liveness check...")
            result = run_from_api(timeout_seconds=18)

            if result.get('passed', False):
                logger.info(f"[Agent] ✓ Liveness PASSED")
                if hasattr(self.risk_engine, 'last_liveness_success'):
                    self.risk_engine.last_liveness_success = datetime.now()
                
                # Report to server with longer timeout
                try:
                    requests.post(
                        f"{API_BASE}/api/liveness-check",
                        json={"passed": True, "reason": result.get('reason','')},
                        timeout=10
                    )
                except Exception as e:
                    logger.warning(f"[Agent] Could not report liveness to server: {e}")
            else:
                logger.warning(f"[Agent] ✗ Liveness FAILED — {result.get('reason','')}")

        except ImportError:
            logger.warning("[Agent] face_liveness.py not found")
        except Exception as e:
            logger.error(f"[Agent] Liveness error: {e}")
    # ─────────────────────────────────────────────────────
    # BACKGROUND LOOPS
    # ─────────────────────────────────────────────────────
    def _scoring_loop(self):
        """Score every SCORE_INTERVAL seconds."""
        time.sleep(15)   # initial wait for keystrokes to accumulate
        while self.running:
            # Build profile if not done yet and enough keystrokes
            if not self.profile_built:
                if len(self.buffer) >= MIN_KS_TO_BUILD:
                    self._build_profile()
                else:
                    logger.info(
                        f"[Agent] Waiting for keystrokes: "
                        f"{len(self.buffer)}/{MIN_KS_TO_BUILD}"
                    )
            else:
                self._score()

            time.sleep(SCORE_INTERVAL)

    def _update_loop(self):
        """
        Continuously update profile every UPDATE_INTERVAL.
        This is what makes the system adaptive:
        Profile evolves with the user's changing behaviour over time.
        Without this, the profile becomes stale and normal behaviour
        would eventually look anomalous.
        """
        time.sleep(UPDATE_INTERVAL)
        while self.running:
            if self.profile_built:
                logger.info("[Agent] Periodic profile update...")
                self._build_profile()
            time.sleep(UPDATE_INTERVAL)

    # ─────────────────────────────────────────────────────
    # START / STOP
    # ─────────────────────────────────────────────────────
    def start(self):
        self.running = True

        # Start pynput keyboard listener
        if PYNPUT_AVAILABLE:
            self._listener = pynput_keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
                suppress=False   # DO NOT suppress — user still types normally
            )
            self._listener.start()
            logger.info("[Agent] pynput keyboard listener started")
        else:
            logger.warning("[Agent] pynput not available — install it first")
            return

        # Scoring thread
        threading.Thread(
            target=self._scoring_loop,
            daemon=True, name="AgentScoring"
        ).start()

        # Profile update thread
        threading.Thread(
            target=self._update_loop,
            daemon=True, name="AgentUpdate"
        ).start()

        logger.info(f"[Agent] Started — user {self.user_id}")
        logger.info(f"[Agent] Type anywhere. Profile builds after {MIN_KS_TO_BUILD} keystrokes.")
        logger.info(f"[Agent] Scores every {SCORE_INTERVAL}s. Profile updates every {UPDATE_INTERVAL}s.")

    def stop(self):
        self.running = False
        if PYNPUT_AVAILABLE and hasattr(self, '_listener'):
            self._listener.stop()
        logger.info("[Agent] Stopped")

    def get_status(self) -> dict:
        uptime = int((datetime.now() - self.session_start).total_seconds())
        return {
            'user_id':       self.user_id,
            'running':       self.running,
            'profile_built': self.profile_built,
            'n_updates':     self.n_updates,
            'n_scored':      self.n_scored,
            'buffer_size':   len(self.buffer),
            'last_score':    self.last_score,
            'last_decision': self.last_decision,
            'last_context':  self.last_context,
            'uptime_s':      uptime,
            'profile_saved': os.path.exists(PROFILE_PATH),
        }


# ── Singleton ──────────────────────────────────────────────
agent = KeystrokeAgent(AGENT_USER_ID)


# ── Run standalone ─────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%H:%M:%S'
    )

    print("\n" + "="*55)
    print("  NeuraShield — Background Keystroke Agent")
    print("="*55)
    print(f"  User ID        : {AGENT_USER_ID}")
    print(f"  API            : {API_BASE}")
    print(f"  Build profile  : after {MIN_KS_TO_BUILD} keystrokes")
    print(f"  Score every    : {SCORE_INTERVAL}s")
    print(f"  Update profile : every {UPDATE_INTERVAL}s")
    print(f"  Profile saved  : {PROFILE_PATH}")
    print("="*55)
    print("\nType anywhere on your computer.")
    print("Profile builds automatically. Press Ctrl+C to stop.\n")

    agent.start()

    try:
        while True:
            time.sleep(10)
            s = agent.get_status()
            print(
                f"  [{datetime.now().strftime('%H:%M:%S')}] "
                f"Buffer:{s['buffer_size']:>3} | "
                f"Profile:{'✓' if s['profile_built'] else '...'} "
                f"(updates:{s['n_updates']}) | "
                f"Scored:{s['n_scored']} | "
                f"Last:{s['last_score']}% {s['last_decision'] or ''}"
            )
    except KeyboardInterrupt:
        print("\nStopping...")
        agent.stop()
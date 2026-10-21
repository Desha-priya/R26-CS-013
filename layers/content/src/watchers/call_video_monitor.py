# =============================================================
# NeuraShield — Layer 4 — Real-Time Call VIDEO Monitor


import os
import sys
import time
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.fusion.multimodal_fusion_v5 import predict_video

CAPTURE_INTERVAL_SECONDS = 2     # grab one frame every 2s
FRAMES_PER_CLIP          = 15    # matches the model's expected 15-frame clip


CALL_WINDOW_TITLES = ["zoom meeting", "microsoft teams", "teams meeting",
                      "google meet", "skype"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_call_window_region():
  
    try:
        import pygetwindow as gw
    except ImportError:
        log("[WARN] pygetwindow not installed — will capture full screen. "
            "Run: pip install pygetwindow")
        return None

    for title_fragment in CALL_WINDOW_TITLES:
        for w in gw.getAllWindows():
            if title_fragment in (w.title or "").lower() and w.width > 100 and w.height > 100:
                return {"left": w.left, "top": w.top,
                       "width": w.width, "height": w.height}
    return None


def capture_frame(sct, region):
    """Grab one screen frame as a BGR numpy array (OpenCV convention)."""
    import cv2
    shot = sct.grab(region)
    frame = np.array(shot)              # BGRA
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def write_clip_and_classify(frames: list) -> dict:
    """
    Write a list of BGR frames to a temp .mp4 and classify it with
    the SAME validated predict_video() used everywhere else in this
    layer — no separate model logic to drift out of sync.
    """
    import cv2

    if not frames:
        return {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}

    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
        tmp_path = t.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  
    writer = cv2.VideoWriter(tmp_path, fourcc, 5.0, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    try:
        result = predict_video(tmp_path, n_frames=min(15, len(frames)))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return result


class CallVideoMonitor:
  
    def __init__(self):
        self.latest_result = {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}
        self._frames = []
        self._region = None
        self._sct = None

    def _ensure_capture_ready(self):
        import mss
        if self._sct is None:
            self._sct = mss.mss()
        if self._region is None:
            found = find_call_window_region()
            if found:
                self._region = found
                log(f"Capturing call window region: {found}")
            else:
                mon = self._sct.monitors[1]  # primary monitor
                self._region = mon
                log(f"No call window found — capturing full primary monitor: {mon}")

    def poll_once(self):
        """Capture one frame; if a full clip is ready, classify it."""
        self._ensure_capture_ready()
        frame = capture_frame(self._sct, self._region)
        self._frames.append(frame)
        if len(self._frames) >= FRAMES_PER_CLIP:
            result = write_clip_and_classify(self._frames)
            self.latest_result = result
            self._frames = []   # non-overlapping windows, like the audio monitor
            tag = "🚨 SUSPECTED DEEPFAKE VIDEO" if result["prediction"] == "fake" else "video looks real"
            log(f"  [video] [{tag}] risk={result.get('risk_score',0):.3f} "
                f"conf={result.get('confidence',0):.3f}")
            return result
        return None

    def run_forever(self):
        log("=" * 60)
        log("  NeuraShield — Real-Time Call VIDEO Monitor")
        log(f"  Capturing every {CAPTURE_INTERVAL_SECONDS}s, "
            f"classifying every {FRAMES_PER_CLIP * CAPTURE_INTERVAL_SECONDS}s")
        log("=" * 60)
        try:
            while True:
                try:
                    self.poll_once()
                except Exception as e:
                    log(f"[ERROR] capture/classify failed: {e}")
                time.sleep(CAPTURE_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log("Stopping video monitor...")


if __name__ == "__main__":
    CallVideoMonitor().run_forever()

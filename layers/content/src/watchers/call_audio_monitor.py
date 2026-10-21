# =============================================================
# NeuraShield — Layer C — Real-Time Call Audio Monitor
 
import os
import sys
import time
import tempfile
import numpy as np
 
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
 
from src.fusion.multimodal_fusion_v5 import predict_audio, fuse_scores, save_alert
 
WINDOW_SECONDS   = 4          # must match training: 16000 * 4 samples
CAPTURE_SR       = 48000      # most WASAPI loopback devices run at 48kHz
TARGET_SR        = 16000      # what the audio model expects
POLL_APPS        = ["teams.exe", "zoom.exe", "phoneexperiencehost.exe",
                    "skype.exe", "discord.exe"]
REQUIRE_CALL_APP = True       # set False to monitor continuously regardless
 
_UNKNOWN = {"prediction": "unknown", "risk_score": 0.5, "confidence": 0.0}
 
 
def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
 
 
def call_app_running() -> bool:
    """Check if a known calling app process is active."""
    try:
        import psutil
    except ImportError:
        log("[WARN] psutil not installed — REQUIRE_CALL_APP check disabled, "
            "monitoring continuously. Run: pip install psutil")
        return True
    names = {p.name().lower() for p in psutil.process_iter(['name'])
            if p.info.get('name')}
    return any(app in names for app in POLL_APPS)
 
 
def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Simple resample; uses scipy if available, else linear interpolation."""
    if orig_sr == TARGET_SR:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig_sr, TARGET_SR)
        return resample_poly(audio, TARGET_SR // g, orig_sr // g)
    except ImportError:
        # fallback: linear interpolation (lower quality but no extra dep)
        duration = len(audio) / orig_sr
        n_target = int(duration * TARGET_SR)
        x_old = np.linspace(0, duration, num=len(audio))
        x_new = np.linspace(0, duration, num=n_target)
        return np.interp(x_new, x_old, audio)
 
 
def capture_and_classify(mic) -> dict:
    """Capture one WINDOW_SECONDS chunk, save to temp WAV, classify it."""
    import soundfile as sf
 
    frames = mic.record(numframes=CAPTURE_SR * WINDOW_SECONDS)
    # mic.record returns (n_samples, n_channels); mix down to mono
    if frames.ndim > 1:
        frames = frames.mean(axis=1)
 
    frames_16k = resample_to_16k(frames.astype(np.float32), CAPTURE_SR)
 
    # Pad/trim to exactly 4s at 16kHz, same convention as training
    target_len = TARGET_SR * WINDOW_SECONDS
    if len(frames_16k) > target_len:
        frames_16k = frames_16k[:target_len]
    elif len(frames_16k) < target_len:
        frames_16k = np.pad(frames_16k, (0, target_len - len(frames_16k)))
 
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        tmp_path = t.name
    sf.write(tmp_path, frames_16k, TARGET_SR)
 
    try:
        result = predict_audio(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return result
 
 
def main():
    try:
        import soundcard as sc
    except ImportError:
        log("[FATAL] soundcard not installed. Run: pip install soundcard")
        return
 
    log("=" * 60)
    log("  NeuraShield — Real-Time Call Audio Monitor")
    log(f"  Window: {WINDOW_SECONDS}s  |  Require call app: {REQUIRE_CALL_APP}")
    log("=" * 60)
 
    default_speaker = sc.default_speaker()
    loopback_mic = sc.get_microphone(id=str(default_speaker.name),
                                     include_loopback=True)
    log(f"Listening on loopback device: {default_speaker.name}")
 
    was_active = False
    try:
        with loopback_mic.recorder(samplerate=CAPTURE_SR) as mic:
            while True:
                active = (not REQUIRE_CALL_APP) or call_app_running()
                if active and not was_active:
                    log("Call app detected — monitoring started")
                if not active and was_active:
                    log("No call app running — monitoring paused")
                was_active = active
 
                if not active:
                    time.sleep(3)
                    continue
 
                try:
                    result = capture_and_classify(mic)
                except Exception as e:
                    log(f"[ERROR] capture/classify failed: {e}")
                    time.sleep(2)
                    continue
 
                alert = fuse_scores(_UNKNOWN, _UNKNOWN, result)
                tag = ("🚨 SUSPECTED CLONED VOICE" if result["prediction"] == "fake"
                      else "human voice — OK")
                log(f"  [{tag}] risk={result['risk_score']:.3f} "
                    f"conf={result['confidence']:.3f}")
 
                if alert["final_decision"] in ("HIGH_RISK", "MEDIUM_RISK"):
                    save_alert(alert)
 
    except KeyboardInterrupt:
        log("Stopping monitor...")
 
 
if __name__ == "__main__":
    main()
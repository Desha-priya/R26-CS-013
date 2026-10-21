
import cv2
import time
import os
import sys
import json
import numpy as np
from datetime import datetime

from ns_paths import data_path
import ns_secure

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EYE_CASCADE  = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

BLINK_THRESHOLD = 1
LIVENESS_GRACE_SECONDS = 60

FACE_TEMPLATE_SIZE = 120
# Written at run time, so it lives in the data folder (see ns_paths.py).
FACE_REF_PATH = data_path("models", "face_reference_v1.npy")
FACE_REF_DIR  = os.path.dirname(FACE_REF_PATH)

# Cross-correlation score (-1..1) required to call it a match.
# Tuned against a same-person-vs-different-person test with
# realistic capture noise (lighting change + a few pixels of
# shift): same-person scored ~0.45, a different face scored
# ~0.03 — a large gap. 0.40 sits below the same-person floor
# with margin while staying well above the different-person
# ceiling. Re-tune per deployment if your webcam/lighting is
# very different — if you see frequent false rejects, lower it;
# if it accepts too easily, raise it and re-test.

FACE_MATCH_THRESHOLD = 0.40


# ================================================================
# FACE TEMPLATE HELPERS


def _extract_face_template(gray_frame, face_rect):
    """
    Crops, resizes, and histogram-equalizes a face region into a
    fixed-size normalized template so two captures taken at
    different times/distances can be compared consistently.
    """

    x, y, w, h = face_rect

    crop = gray_frame[y:y + h, x:x + w]

    resized = cv2.resize(
        crop,
        (FACE_TEMPLATE_SIZE, FACE_TEMPLATE_SIZE)
    )

    return cv2.equalizeHist(resized)


def _face_similarity(template_a, template_b):
    """
    Normalized cross-correlation between two same-size templates.
    Returns a float in roughly [-1, 1]; higher = more similar.
    """

    result = cv2.matchTemplate(
        template_a.astype("float32"),
        template_b.astype("float32"),
        cv2.TM_CCOEFF_NORMED
    )

    return float(result[0][0])


def save_face_reference(template):

    os.makedirs(
        FACE_REF_DIR,
        exist_ok=True
    )

    # Encrypted at rest (Windows DPAPI; see ns_secure.py).
    ns_secure.save_npy(
        FACE_REF_PATH,
        template
    )


def _load_face_reference():

    if not os.path.exists(FACE_REF_PATH):
        return None

    try:

        return ns_secure.load_npy(FACE_REF_PATH)

    except Exception:

        return None


def has_face_reference():

    return os.path.exists(FACE_REF_PATH)


# ================================================================
# LIVENESS + IDENTITY CHALLENGE
# ================================================================

def detect_liveness(timeout_seconds: int = 25, last_success_time=None) -> dict:
    """
    Runs the webcam challenge. Must be called from the main
    thread — use run_from_api() when calling from FastAPI.

    Passing requires ALL of:
      1. Exactly one face visible (ambiguous multi-face frames
         are rejected rather than guessed at).
      2. Liveness confirmed (blink, or SPACE as a manual
         fallback if blink detection is unreliable for the
         person/lighting).
      3. IF a reference face is enrolled: the live face matches
         it above FACE_MATCH_THRESHOLD. If no reference exists
         yet, this step is skipped and flagged in the result so
         it's visible in the audit log rather than silently
         treated as a pass.
    """

    if last_success_time:

        seconds_since_success = (
            datetime.now() - last_success_time
        ).total_seconds()

        if seconds_since_success < LIVENESS_GRACE_SECONDS:

            return {
                'passed': True,
                'reason': f'Grace period active ({int(LIVENESS_GRACE_SECONDS - seconds_since_success)}s remaining)',
                'duration': 0,
                'blink_count': 0,
                'grace_active': True,
                'identity_checked': False,
            }

    reference = _load_face_reference()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {'passed': False, 'reason': 'Webcam not available',
                'duration': 0, 'blink_count': 0, 'identity_checked': False}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    start          = time.time()
    face_detected  = False
    blink_count    = 0
    eyes_prev      = False
    no_eye_frames  = 0
    bad_reads      = 0
    last_template  = None

    print(f"\n[LIVENESS] Starting — {timeout_seconds}s | BLINK or press SPACE\n")
    if reference is None:
        print("[LIVENESS] WARNING: no enrolled face reference — identity check will be skipped.\n")

    while True:
        elapsed = time.time() - start

        if elapsed > timeout_seconds:
            cap.release()
            cv2.destroyAllWindows()
            return {'passed': False,
                    'reason': f'Timeout — no liveness in {timeout_seconds}s',
                    'duration': round(elapsed, 1),
                    'blink_count': blink_count,
                    'identity_checked': False}

        ret, frame = cap.read()
        if not ret:
            bad_reads += 1
            if bad_reads > 20:
                cap.release()
                cv2.destroyAllWindows()
                return {'passed': False, 'reason': 'Camera read failed repeatedly',
                        'duration': round(elapsed, 1), 'blink_count': 0,
                        'identity_checked': False}
            time.sleep(0.05)
            continue
        bad_reads = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w,44), (8,8,20), -1)
        cv2.putText(frame, 'NEURASHIELD  IDENTITY VERIFICATION',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,200,255), 2)
        cv2.putText(frame, f'{int(timeout_seconds-elapsed)}s',
                    (w-60,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.rectangle(frame, (0,h-44), (w,h), (8,8,20), -1)
        cv2.putText(frame, f'Blinks:{blink_count}  BLINK or press SPACE to verify',
                    (10,h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,200), 1)

        eyes_now = False

        if len(faces) == 1:

            face_detected = True

            (fx, fy, fw, fh) = faces[0]

            cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(0,255,80),2)
            cv2.putText(frame,'FACE OK',(fx,fy-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,80),1)

            last_template = _extract_face_template(gray, (fx, fy, fw, fh))

            roi_g = gray[fy:fy+fh, fx:fx+fw]
            roi_c = frame[fy:fy+fh, fx:fx+fw]
            eyes  = EYE_CASCADE.detectMultiScale(
                roi_g, scaleFactor=1.1, minNeighbors=4, minSize=(15,15))
            if len(eyes) >= 2:
                eyes_now = True
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_c,(ex,ey),(ex+ew,ey+eh),(255,100,0),1)

        elif len(faces) > 1:

            # Ambiguous frame — more than one person in view.
            # Do not guess; just show a warning and keep waiting.
            face_detected = False
            cv2.putText(frame, 'MULTIPLE FACES — step out of frame',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,140,255), 2)

        # Blink detection
        if eyes_prev and not eyes_now:
            no_eye_frames += 1
        elif not eyes_prev and eyes_now and no_eye_frames >= BLINK_THRESHOLD:
            blink_count   += 1
            no_eye_frames  = 0
            print(f"[LIVENESS] Blink {blink_count} detected!")
        elif eyes_now:
            no_eye_frames = 0
        eyes_prev = eyes_now

        cv2.imshow('NeuraShield — Identity Verification', frame)
        key = cv2.waitKey(1) & 0xFF

        confirmed = False
        confirm_reason = ""

        if key == ord(' '):
            confirmed = True
            confirm_reason = "Manual verification (SPACE)"

        elif key == 27:
            cap.release(); cv2.destroyAllWindows()
            return {'passed': False, 'reason': 'Cancelled (ESC)',
                    'duration': round(time.time()-start,1), 'blink_count': 0,
                    'identity_checked': False}

        elif face_detected and blink_count >= 1:
            confirmed = True
            confirm_reason = f'Liveness confirmed — {blink_count} blink(s)'

        if confirmed:

            cap.release()
            cv2.destroyAllWindows()

            duration = round(time.time() - start, 1)

            # --------------------------------------------------
            # IDENTITY CHECK
            # --------------------------------------------------

            if reference is None:

                print("[LIVENESS] Passed liveness, but no reference "
                      "enrolled — identity NOT verified.")

                return {
                    'passed': True,
                    'reason': confirm_reason + ' (no face reference enrolled — identity not verified)',
                    'duration': duration,
                    'blink_count': blink_count,
                    'identity_checked': False,
                }

            if last_template is None:

                return {
                    'passed': False,
                    'reason': 'No clear single-face frame captured for identity check',
                    'duration': duration,
                    'blink_count': blink_count,
                    'identity_checked': False,
                }

            similarity = _face_similarity(last_template, reference)

            identity_match = similarity >= FACE_MATCH_THRESHOLD

            print(f"[LIVENESS] Face similarity vs reference: {similarity:.3f} "
                  f"(threshold {FACE_MATCH_THRESHOLD}) -> "
                  f"{'MATCH' if identity_match else 'NO MATCH'}")

            return {
                'passed': bool(identity_match),
                'reason': (
                    f'{confirm_reason}; face match={similarity:.3f}'
                    if identity_match else
                    f'Liveness OK but face does NOT match enrolled reference '
                    f'(similarity={similarity:.3f}, need >= {FACE_MATCH_THRESHOLD})'
                ),
                'duration': duration,
                'blink_count': blink_count,
                'identity_checked': True,
                'face_similarity': round(similarity, 4),
            }
        # loop continues


def enroll_face(timeout_seconds: int = 20) -> dict:
    """
    Interactive capture of the reference face used for later
    identity matching. Must run on the main thread — use
    run_face_enrollment_from_api() from FastAPI.
    """

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {'saved': False, 'reason': 'Webcam not available'}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start = time.time()

    print(f"\n[FACE ENROLL] Look at the camera and press SPACE to save "
          f"your reference face ({timeout_seconds}s)\n")

    while True:

        elapsed = time.time() - start

        if elapsed > timeout_seconds:
            cap.release()
            cv2.destroyAllWindows()
            return {'saved': False, 'reason': f'Timeout after {timeout_seconds}s'}

        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w,44), (8,8,20), -1)
        cv2.putText(frame, 'NEURASHIELD  FACE ENROLLMENT',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,200,255), 2)

        template = None

        if len(faces) == 1:
            (fx, fy, fw, fh) = faces[0]
            cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(0,255,80),2)
            cv2.putText(frame, 'Press SPACE to save', (fx, fy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,80), 1)
            template = _extract_face_template(gray, (fx, fy, fw, fh))

        elif len(faces) > 1:
            cv2.putText(frame, 'MULTIPLE FACES — only you in frame',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,140,255), 2)

        cv2.imshow('NeuraShield — Face Enrollment', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cap.release(); cv2.destroyAllWindows()
            return {'saved': False, 'reason': 'Cancelled (ESC)'}

        if key == ord(' ') and template is not None:
            cap.release()
            cv2.destroyAllWindows()
            save_face_reference(template)
            print("[FACE ENROLL] Reference face saved.")
            return {'saved': True, 'reason': 'Face reference saved'}


_NO_WINDOW = 0x08000000 if sys.platform.startswith("win") else 0


def _child_command(mode: str, timeout_seconds: int, result_file: str):
    """
    Command line for the child process that owns the OpenCV window.

    Running from source: start this file with the Python interpreter.
    Frozen (.exe): sys.executable is the packaged program itself, so the
    child is the same exe started with the mode flag. The launcher
    (neurashield_launcher.py) recognises the flag and calls main_cli()
    before it starts any server code. The original command line
    (interpreter + __file__) would have started a second platform inside
    a packaged build.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, mode, str(timeout_seconds), result_file]
    return [sys.executable, os.path.abspath(__file__), mode,
            str(timeout_seconds), result_file]


def _run_child(mode: str, timeout_seconds: int):
    """
    Runs one challenge in a child process and returns its result dict, or
    None if no result could be read. The result is passed back through a
    temporary file as well as stdout, because a windowed .exe may have no
    usable stdout.
    """
    import subprocess, tempfile

    fd, result_file = tempfile.mkstemp(prefix="ns_", suffix=".json")
    os.close(fd)
    stderr_text = ""
    try:
        proc = subprocess.run(
            _child_command(mode, timeout_seconds, result_file),
            capture_output=True, text=True,
            timeout=timeout_seconds + 10,
            creationflags=_NO_WINDOW,
        )
        stderr_text = (proc.stderr or "")[:200]
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text.startswith("{"):
                return json.loads(text), stderr_text
        except Exception:
            pass
        for line in reversed((proc.stdout or "").strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line), stderr_text
                except Exception:
                    continue
        return None, stderr_text
    finally:
        try:
            os.remove(result_file)
        except OSError:
            pass


def run_from_api(timeout_seconds: int = 25) -> dict:
    """
    Called from FastAPI. Runs the liveness+identity challenge as
    a SUBPROCESS so OpenCV's GUI has its own main thread.
    """
    import subprocess
    try:
        result, err = _run_child("--subprocess", timeout_seconds)
        if result is not None:
            return result
        return {'passed': False,
                'reason': f'Subprocess error: {err}',
                'duration': 0, 'blink_count': 0, 'identity_checked': False}
    except subprocess.TimeoutExpired:
        return {'passed': False, 'reason': 'Liveness check timed out',
                'duration': timeout_seconds, 'blink_count': 0, 'identity_checked': False}
    except Exception as e:
        return {'passed': False, 'reason': f'Error: {str(e)}',
                'duration': 0, 'blink_count': 0, 'identity_checked': False}


def run_face_enrollment_from_api(timeout_seconds: int = 20) -> dict:
    """
    Called from FastAPI to enroll the reference face, as a
    subprocess for the same main-thread-GUI reason as above.
    """
    import subprocess
    try:
        result, err = _run_child("--enroll-face", timeout_seconds)
        if result is not None:
            return result
        return {'saved': False, 'reason': f'Subprocess error: {err}'}
    except subprocess.TimeoutExpired:
        return {'saved': False, 'reason': 'Enrollment timed out'}
    except Exception as e:
        return {'saved': False, 'reason': f'Error: {str(e)}'}


def _emit(result: dict, result_file=None):
    text = json.dumps(result)
    if result_file:
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass
    try:
        print(text)
    except Exception:
        pass


def main_cli(argv):
    """
    Entry point for the child process. argv is sys.argv[1:]:
        --subprocess  <timeout> [result_file]
        --enroll-face <timeout> [result_file]
    Returns True if the arguments were handled.
    """
    if not argv:
        return False
    mode = argv[0]
    result_file = argv[2] if len(argv) >= 3 else None

    if mode == '--subprocess':
        timeout = int(argv[1]) if len(argv) >= 2 else 12
        _emit(detect_liveness(timeout_seconds=timeout), result_file)
        return True

    if mode == '--enroll-face':
        timeout = int(argv[1]) if len(argv) >= 2 else 20
        _emit(enroll_face(timeout_seconds=timeout), result_file)
        return True

    return False


if __name__ == "__main__":

    if not main_cli(sys.argv[1:]):
        result = detect_liveness(timeout_seconds=12)
        print(f"\n{'PASSED' if result['passed'] else 'FAILED'} — {result['reason']}")
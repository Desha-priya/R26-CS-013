# face_liveness.py — FINAL FIXED VERSION
# Two verification methods:
#   1. Blink detection (automatic) — look at camera and blink
#   2. SPACE key (manual fallback) — press SPACE while webcam window is open
# ESC key = cancel and fail

import cv2
import time

FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH  = cv2.data.haarcascades + 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)

BLINK_THRESHOLD = 2   # consecutive frames with no eyes = blink


def detect_liveness(timeout_seconds: int = 12) -> dict:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return {'passed': False, 'reason': 'Webcam not available', 'duration': 0, 'blink_count': 0}

    start_time         = time.time()
    face_detected      = False
    blink_count        = 0
    eyes_visible_prev  = False
    frames_no_eyes     = 0

    print("\n[LIVENESS] Webcam open. Look at camera and BLINK, or press SPACE to verify manually.")
    print(f"[LIVENESS] You have {timeout_seconds} seconds. Press ESC to cancel.\n")

    # ── MAIN LOOP — must stay inside while True ──────────────
    while True:
        elapsed   = time.time() - start_time
        remaining = int(timeout_seconds - elapsed)

        # ── TIMEOUT CHECK ──
        if elapsed > timeout_seconds:
            cap.release()
            cv2.destroyAllWindows()
            return {
                'passed':      False,
                'reason':      f'Timeout — no liveness confirmed in {timeout_seconds}s',
                'duration':    round(elapsed, 1),
                'blink_count': blink_count
            }

        # ── READ FRAME ──
        ret, frame = cap.read()
        if not ret:
            continue   # skip bad frame, keep looping

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── DETECT FACES ──
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        # ── DRAW UI ──
        cv2.putText(frame, 'NEURASHIELD LIVENESS CHECK', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f'Time: {remaining}s', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'Blinks: {blink_count}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
        cv2.putText(frame, 'BLINK naturally or press SPACE', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        eyes_visible_now = False

        # ── PROCESS EACH FACE ──
        for (fx, fy, fw, fh) in faces:
            face_detected = True
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
            cv2.putText(frame, 'FACE OK', (fx, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            face_roi_gray  = gray[fy:fy+fh, fx:fx+fw]
            face_roi_color = frame[fy:fy+fh, fx:fx+fw]

            eyes = eye_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(15, 15)
            )

            if len(eyes) >= 2:
                eyes_visible_now = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_roi_color,
                                  (ex, ey), (ex+ew, ey+eh), (255, 100, 0), 1)

        # ── BLINK DETECTION LOGIC ──
        if eyes_visible_prev and not eyes_visible_now:
            frames_no_eyes += 1
        elif not eyes_visible_prev and eyes_visible_now and frames_no_eyes >= BLINK_THRESHOLD:
            blink_count   += 1
            frames_no_eyes = 0
            print(f"[LIVENESS] Blink {blink_count} detected!")
        elif eyes_visible_now:
            frames_no_eyes = 0

        eyes_visible_prev = eyes_visible_now

        # ── SHOW FRAME ──
        cv2.imshow('NeuraShield — Identity Verification', frame)
        key = cv2.waitKey(1) & 0xFF

        # ── KEY HANDLERS ──
        if key == ord(' '):
            # SPACE = manual verification (second method)
            cap.release()
            cv2.destroyAllWindows()
            print("[LIVENESS] ✓ Manual verification via SPACE key")
            return {
                'passed':      True,
                'reason':      'Manual verification accepted (SPACE key)',
                'duration':    round(time.time() - start_time, 1),
                'blink_count': blink_count
            }

        if key == 27:
            # ESC = cancel
            cap.release()
            cv2.destroyAllWindows()
            print("[LIVENESS] Cancelled by user")
            return {
                'passed':      False,
                'reason':      'Cancelled by user (ESC)',
                'duration':    round(time.time() - start_time, 1),
                'blink_count': 0
            }

        # ── PASS CONDITION: face + blink ──
        if face_detected and blink_count >= 1:
            cap.release()
            cv2.destroyAllWindows()
            print(f"[LIVENESS] ✓ Confirmed — {blink_count} blink(s) in {elapsed:.1f}s")
            return {
                'passed':      True,
                'reason':      f'Liveness confirmed — {blink_count} blink(s) detected',
                'duration':    round(elapsed, 1),
                'blink_count': blink_count
            }

        # ── LOOP CONTINUES (no return here — this was the original bug) ──


def run_liveness_check() -> bool:
    result = detect_liveness(timeout_seconds=12)
    print(f"\n[LIVENESS] {'PASSED' if result['passed'] else 'FAILED'} — {result['reason']}")
    return result['passed']


if __name__ == "__main__":
    passed = run_liveness_check()
    print("\n✓ Session continues" if passed else "\n✗ Session locked")
# face_liveness.py — FINAL VERSION
# Fix: When called from FastAPI (background thread), OpenCV imshow
# crashes because it needs the main thread.
# Solution: run_from_api() launches this script as a SUBPROCESS
# so it gets its own main thread with a proper GUI loop.

import cv2
import time
import os
import sys
import json

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EYE_CASCADE  = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

BLINK_THRESHOLD = 2


def detect_liveness(timeout_seconds: int = 25) -> dict:
    """
    Run face liveness check. Must be called from main thread.
    For FastAPI use run_from_api() instead.
    """
    # Try DSHOW first (Windows DirectShow — avoids MSMF spam)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {'passed': False,
                'reason': 'Webcam not available',
                'duration': 0, 'blink_count': 0}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    start          = time.time()
    face_detected  = False
    blink_count    = 0
    eyes_prev      = False
    no_eye_frames  = 0
    bad_reads      = 0

    print(f"\n[LIVENESS] Starting — {timeout_seconds}s | BLINK or press SPACE\n")

    while True:
        elapsed = time.time() - start

        if elapsed > timeout_seconds:
            cap.release()
            cv2.destroyAllWindows()
            return {'passed': False,
                    'reason': f'Timeout — no liveness in {timeout_seconds}s',
                    'duration': round(elapsed, 1),
                    'blink_count': blink_count}

        ret, frame = cap.read()
        if not ret:
            bad_reads += 1
            if bad_reads > 20:
                cap.release()
                cv2.destroyAllWindows()
                return {'passed': False,
                        'reason': 'Camera read failed repeatedly',
                        'duration': round(elapsed, 1),
                        'blink_count': 0}
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
        for (fx,fy,fw,fh) in faces:
            face_detected = True
            cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(0,255,80),2)
            cv2.putText(frame,'FACE OK',(fx,fy-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,80),1)
            roi_g = gray[fy:fy+fh, fx:fx+fw]
            roi_c = frame[fy:fy+fh, fx:fx+fw]
            eyes  = EYE_CASCADE.detectMultiScale(
                roi_g, scaleFactor=1.1, minNeighbors=4, minSize=(15,15))
            if len(eyes) >= 2:
                eyes_now = True
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_c,(ex,ey),(ex+ew,ey+eh),(255,100,0),1)

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

        if key == ord(' '):
            cap.release(); cv2.destroyAllWindows()
            print("[LIVENESS] ✓ SPACE key verification")
            return {'passed': True, 'reason': 'Manual verification (SPACE)',
                    'duration': round(time.time()-start,1),
                    'blink_count': blink_count}

        if key == 27:
            cap.release(); cv2.destroyAllWindows()
            return {'passed': False, 'reason': 'Cancelled (ESC)',
                    'duration': round(time.time()-start,1), 'blink_count': 0}

        if face_detected and blink_count >= 1:
            cap.release(); cv2.destroyAllWindows()
            print(f"[LIVENESS] ✓ Confirmed — {blink_count} blink(s)")
            return {'passed': True,
                    'reason': f'Liveness confirmed — {blink_count} blink(s)',
                    'duration': round(elapsed,1),
                    'blink_count': blink_count}
        # loop continues


def run_from_api(timeout_seconds: int = 25) -> dict:
    """
    Called from FastAPI endpoint.
    Runs liveness as a SUBPROCESS so it gets its own main thread.
    FastAPI runs in async threads where OpenCV GUI cannot work directly.
    Subprocess gets a fresh main thread — imshow works correctly.
    Result is returned via JSON printed to stdout.
    """
    import subprocess, sys
    try:
        result = subprocess.run(
            [sys.executable, __file__, '--subprocess', str(timeout_seconds)],
            capture_output=True, text=True, timeout=timeout_seconds + 10
        )
        # Parse JSON result from subprocess stdout
        output = result.stdout.strip()
        # Find the last JSON line
        for line in reversed(output.split('\n')):
            line = line.strip()
            if line.startswith('{'):
                try:
                    return json.loads(line)
                except Exception:
                    continue
        # If no JSON found, subprocess failed
        return {'passed': False,
                'reason': f'Subprocess error: {result.stderr[:200]}',
                'duration': 0, 'blink_count': 0}
    except subprocess.TimeoutExpired:
        return {'passed': False, 'reason': 'Liveness check timed out',
                'duration': timeout_seconds, 'blink_count': 0}
    except Exception as e:
        return {'passed': False, 'reason': f'Error: {str(e)}',
                'duration': 0, 'blink_count': 0}


if __name__ == "__main__":
    # When run as subprocess from API: print result as JSON to stdout
    if len(sys.argv) >= 2 and sys.argv[1] == '--subprocess':
        timeout = int(sys.argv[2]) if len(sys.argv) >= 3 else 12
        result  = detect_liveness(timeout_seconds=timeout)
        # Print JSON last so run_from_api can find it
        print(json.dumps(result))
    else:
        # Normal standalone run
        result = detect_liveness(timeout_seconds=12)
        print(f"\n{'PASSED' if result['passed'] else 'FAILED'} — {result['reason']}")

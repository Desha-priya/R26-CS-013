"""
NeuraShield Layer D - install check.   Run:  python check_install.py

Run it from the folder you normally start the platform from. It does not start
anything; it only reports what is missing, with the exact fix.
"""
import importlib
import os
import sys

OK, BAD, WARN = "  [ OK ]", "  [FAIL]", "  [WARN]"
problems = 0


def say(tag, msg):
    global problems
    if tag == BAD:
        problems += 1
    print(f"{tag} {msg}")


print("Python", sys.version.split()[0], "-", sys.platform)
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

print("\n1. Code files (must all be in the same folder)")
for f in ("ns_paths.py", "ns_secure.py", "neurashield_platform_v3.py", "keystroke_agent_v3.py", "risk_engine_v3.py",
          "session_monitor.py", "face_liveness.py"):
    say(OK if os.path.exists(os.path.join(here, f)) else BAD,
        f if os.path.exists(os.path.join(here, f)) else f"{f} is missing - copy it next to this script")

print("\n2. Python packages")
for mod, pip, needed in (("fastapi", "fastapi", True), ("uvicorn", "uvicorn", True), ("pydantic", "pydantic", True),
                          ("numpy", "numpy", True), ("requests", "requests", True), ("torch", "torch", True),
                          ("pynput", "pynput", True), ("cv2", "opencv-python", True), ("psutil", "psutil", True),
                          ("win32gui", "pywin32", sys.platform.startswith("win"))):
    try:
        importlib.import_module(mod)
        say(OK, mod)
    except Exception as exc:
        say(BAD if needed else WARN, f"{mod} not importable ({type(exc).__name__}) - pip install {pip}")

print("\n3. Where files are read from and written to")
try:
    import ns_paths
    say(OK, f"data folder (audit log, PIN database, profiles, api_key.txt): {ns_paths.data_dir()}")
    model = ns_paths.resource_path("processed", "siamese_bilstm_v3", "model_training_v3", "checkpoints",
                                   "best_siamese_bilstm_v3.pt")
    scaler = ns_paths.resource_path("processed", "siamese_bilstm_v3", "reports", "normalization_audit",
                                    "train_only_scaler_v3.npz")
    for label, p in (("model checkpoint", os.environ.get("NEURASHIELD_MODEL_PATH") or model),
                     ("scaler", os.environ.get("NEURASHIELD_SCALER_PATH") or scaler)):
        say(OK if os.path.exists(p) else BAD,
            f"{label}: {p}" if os.path.exists(p) else
            f"{label} not found at {p}\n         keep 'processed\\siamese_bilstm_v3\\...' in the folder you start from, "
            f"or set NEURASHIELD_MODEL_PATH / NEURASHIELD_SCALER_PATH")
    dash = ns_paths.resource_path("templates", "central_dashboard.html")
    if not os.path.exists(dash):
        say(BAD, f"templates\\central_dashboard.html not found (looked at {dash})")
    else:
        html = open(dash, encoding="utf-8", errors="ignore").read()
        say(OK if "recent_events" not in html or "__NEURASHIELD_API_KEY__" in html else WARN, f"dashboard: {dash}")
        if "__NEURASHIELD_API_KEY__" not in html:
            say(WARN, "central_dashboard.html has no __NEURASHIELD_API_KEY__ placeholder - it will not get the API key")
    idx = ns_paths.resource_path("templates", "index.html")
    say(OK if os.path.exists(idx) else WARN, f"Layer 3 dashboard: {idx}" if os.path.exists(idx)
        else "templates\\index.html not found - /layer3 will show 'index.html not found'")
except Exception as exc:
    say(BAD, f"ns_paths failed: {exc}")

print("\n4. Imports of the platform modules")
for mod in ("risk_engine_v3", "session_monitor", "face_liveness", "keystroke_agent_v3"):
    try:
        importlib.import_module(mod)
        say(OK, mod)
    except Exception as exc:
        say(BAD, f"{mod}: {type(exc).__name__}: {exc}")

print("\n5. API key")
try:
    k = ns_paths.get_api_key()
    src = "environment variable" if os.environ.get("NEURASHIELD_API_KEY") else ns_paths.api_key_file()
    say(OK, f"key length {len(k)} from {src}")
    if k == "nue-ras-hie-ld":
        say(WARN, "this is the old development key - fine for a demo, change it before deployment")
    print("         The browser extension and the other layers must send this same key (X-API-Key header).")
except Exception as exc:
    say(BAD, f"API key: {exc}")

print()
print("RESULT:", "all good" if not problems else f"{problems} problem(s) to fix (see [FAIL] lines)")
sys.exit(1 if problems else 0)

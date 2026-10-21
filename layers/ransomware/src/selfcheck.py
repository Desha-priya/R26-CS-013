"""
selfcheck.py
--------------
"Is this machine ready to run Layer 2?"  Run with:   python -m main --check

Prints one line per check:  [ OK ]  [WARN]  [FAIL]
Exit code 0 means no FAIL. Nothing is changed on the machine, except that
the runtime folder is created if it does not exist yet.
"""

import importlib
import os
import socket
import sys
from urllib.parse import urlparse

from src.utils import paths


def _is_admin() -> bool:
    try:
        if os.name == "nt":
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        return os.geteuid() == 0
    except Exception:
        return False


def run_selfcheck() -> int:
    rows = []

    def add(level, name, detail=""):
        rows.append((level, name, detail))

    # ---- environment
    add("OK", "Python", sys.version.split()[0] + (" (packaged exe)" if paths.is_frozen() else ""))
    for module, required in [("watchdog", True), ("psutil", True), ("yaml", True), ("requests", True),
                             ("flask", True), ("waitress", True), ("joblib", True), ("numpy", True),
                             ("sklearn", True), ("shap", False)]:
        try:
            importlib.import_module(module)
            add("OK", f"library {module}")
        except ImportError:
            add("FAIL" if required else "WARN", f"library {module}",
                "not installed" + ("" if required else " (only needed for SHAP explanations)"))

    # ---- config
    cfg = None
    try:
        cfg = paths.load_config()
        add("OK", "config file", paths.config_file())
    except Exception as e:
        add("FAIL", "config file", str(e))

    # ---- runtime folder
    try:
        probe = paths.runtime("selfcheck.tmp")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        add("OK", "runtime folder writable", paths.runtime_root())
    except OSError as e:
        add("FAIL", "runtime folder writable", f"{paths.runtime_root()}: {e}")

    # ---- secrets
    if cfg is not None:
        from src.alerting.hmac_auth import secret_source
        source = secret_source(cfg)
        add("OK" if source != "generated key file" else "WARN", "signing secret", source +
            ("  (other layers cannot verify these alerts unless they share it)" if source == "generated key file" else ""))

    key = bool(os.environ.get("NEURASHIELD_API_KEY"))
    url = (os.environ.get("NEURASHIELD_PLATFORM_URL") or ((cfg or {}).get("alerting", {}) or {}).get("platform_base_url") or "").strip()
    add("OK" if key else "WARN", "platform API key", "set" if key else "NOT set - alerts will not be sent to the platform")
    if url:
        parsed = urlparse(url)
        try:
            with socket.create_connection((parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)), timeout=2):
                add("OK", "platform reachable", url)
        except (OSError, TypeError):
            add("WARN", "platform reachable", f"{url} did not answer (alerts will be retried later)")
    else:
        add("WARN", "platform URL", "not set - alerts stay local")

    # ---- protection scope
    if cfg is not None:
        existing = [p for p in cfg["live_monitor"]["watch_paths"] if os.path.isdir(os.path.expanduser(p))]
        add("OK" if existing else "FAIL", "watch folders",
            f"{len(existing)} of {len(cfg['live_monitor']['watch_paths'])} exist")
        try:
            from src.response.rollback_engine import RollbackEngine
            stats = RollbackEngine().stats()
            add("OK", "backup folder", f"{stats['backup_root']} ({stats['files_tracked']} files tracked)")
        except Exception as e:
            add("FAIL", "backup folder", str(e))

        # ---- models
        try:
            from src.detection.model_integrity import verify_all, load_manifest
            problems = [(n, m) for n, ok, m in verify_all(cfg) if not ok]
            manifest = load_manifest()
            if problems:
                add("FAIL", "model files", "; ".join(f"{n}: {m}" for n, m in problems))
            elif manifest is None:
                add("WARN", "model files", "present, but no MANIFEST.json (integrity not checked)")
            else:
                add("OK", "model files", "all fingerprints match")
                try:
                    import sklearn
                    saved = manifest.get("saved_with_sklearn")
                    if saved and saved.split(".")[:2] != sklearn.__version__.split(".")[:2]:
                        add("WARN", "scikit-learn version", f"models saved with {saved}, installed {sklearn.__version__}")
                except ImportError:
                    pass
        except Exception as e:
            add("WARN", "model files", str(e))

    try:
        from src.consent import ConsentStore
        watch = [os.path.expanduser(p) for p in cfg["live_monitor"]["watch_paths"]] if cfg else []
        if os.environ.get("NEURASHIELD_AUTO_CONSENT") == "1" or (cfg or {}).get("main", {}).get("require_consent", True) is False:
            add("WARN", "protection consent", "automatic consent is ON (the platform must show the consent screen)")
        elif ConsentStore().covers(watch):
            add("OK", "protection consent", "granted - protection starts automatically")
        else:
            add("WARN", "protection consent", "not granted - the service will start in STANDBY (use the dashboard button or --consent)")
    except Exception as e:
        add("WARN", "protection consent", str(e))

    add("OK" if _is_admin() else "WARN", "administrator rights",
        "yes" if _is_admin() else "no - stopping an administrator-run attacker needs admin rights")

    # ---- print
    width = max(len(r[1]) for r in rows) + 2
    print("\nNeuraShield Layer 2 - readiness report\n" + "-" * 60)
    for level, name, detail in rows:
        tag = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[level]
        print(f"{tag} {name:<{width}} {detail}")
    fails = sum(1 for r in rows if r[0] == "FAIL")
    warns = sum(1 for r in rows if r[0] == "WARN")
    print("-" * 60)
    print(f"{fails} failure(s), {warns} warning(s).")
    return 0 if fails == 0 else 1

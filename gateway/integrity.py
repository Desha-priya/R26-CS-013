"""
integrity.py  —  NeuraShield data & code self-protection
--------------------------------------------------------
A security product must be able to tell whether its own files have been
tampered with. This module:

  * builds/refreshes a baseline manifest of SHA-256 hashes for the platform's
    critical files (gateway code, layer entry points, configs, model files);
  * verifies the current files against that baseline and reports any file that
    was added, removed, or modified;
  * signs the baseline with the shared HMAC secret so the manifest itself
    cannot be silently swapped.

The console runs a verification at startup and on demand (self-check), and
surfaces any mismatch as a high-severity alert.
"""

import hashlib
import hmac
import json
import os
import time
from pathlib import Path

MANIFEST_NAME = "integrity_baseline.json"

# Files/globs that matter for integrity (relative to install root). Model files
# are included when present; large data/runtime dirs are excluded.
_WATCH = [
    "gateway/console_app.py", "gateway/supervisor.py", "gateway/secrets_store.py",
    "gateway/notifier.py", "gateway/consent_notice.py", "gateway/integrity.py",
    "neurashield_app.py", "run_neurashield.py",
    "config/layers.json",
    "core/srC/neurashield_platform_v3.py", "core/srC/risk_engine_v3.py",
    "core/srC/ns_secure.py", "core/srC/ns_paths.py",
    "layers/network/monitor-2.py", "layers/network/app.py",
    "layers/ransomware/main.py",
    "layers/content/demo/app.py", "layers/content/src/fusion/multimodal_fusion_v5.py",
]
_MODEL_GLOBS = [
    "core/processed/**/*.pt", "core/processed/**/*.pth",
    "layers/ransomware/models/*.pkl",
    "layers/content/models/**/*.h5", "layers/content/models/**/*.pth",
    "layers/content/models/**/*.safetensors", "layers/content/models/**/*.bin",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _frozen_exes(root: Path) -> list:
    """Per-layer executables named in config/layers.json ("frozen_exe")."""
    try:
        spec = json.loads((root / "config" / "layers.json").read_text(encoding="utf-8"))
        return [v["frozen_exe"] for v in spec.values()
                if isinstance(v, dict) and v.get("frozen_exe")]
    except Exception:
        return []


def _collect(root: Path) -> dict:
    files = {}
    for rel in _WATCH:
        p = root / rel
        if p.exists() and p.is_file():
            files[rel] = _sha256(p)
    # Packaged install: the code is in the executables, not in .py files.
    for rel in ["gateway/NeuraShield.exe"] + _frozen_exes(root):
        p = root / rel
        if p.exists() and p.is_file():
            files[rel] = _sha256(p)
    for pattern in _MODEL_GLOBS:
        for p in root.glob(pattern):
            if p.is_file():
                files[str(p.relative_to(root)).replace("\\", "/")] = _sha256(p)
    return files


def _manifest_path(root: Path) -> Path:
    base = os.environ.get("NEURASHIELD_SECRET_DIR")
    d = Path(base) if base else (root / "data" / "integrity")
    d.mkdir(parents=True, exist_ok=True)
    return d / MANIFEST_NAME


def build_baseline(root: Path, hmac_secret: str) -> dict:
    """Record the current hashes as the trusted baseline (first run / update)."""
    files = _collect(root)
    body = {"created": time.time(), "files": files}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["sig"] = hmac.new(hmac_secret.encode(), raw, hashlib.sha256).hexdigest()
    path = _manifest_path(root)
    path.write_text(json.dumps(body, indent=2))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return body


def verify(root: Path, hmac_secret: str) -> dict:
    """Compare current files to the baseline. Returns a report dict:
       {status: ok|modified|no_baseline|manifest_tampered, modified:[], added:[],
        removed:[], checked:N}"""
    path = _manifest_path(root)
    if not path.exists():
        # First run: establish the baseline so subsequent runs can detect change.
        build_baseline(root, hmac_secret)
        return {"status": "baseline_created", "checked": 0, "modified": [],
                "added": [], "removed": []}
    try:
        body = json.loads(path.read_text())
        sig = body.pop("sig", "")
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        expect = hmac.new(hmac_secret.encode(), raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expect):
            return {"status": "manifest_tampered", "checked": 0,
                    "modified": [], "added": [], "removed": []}
    except Exception:
        return {"status": "manifest_unreadable", "checked": 0,
                "modified": [], "added": [], "removed": []}

    baseline = body.get("files", {})
    current = _collect(root)
    modified, removed = [], []
    for rel, h in baseline.items():
        if rel not in current:
            removed.append(rel)
        elif current[rel] != h:
            modified.append(rel)
    added = [rel for rel in current if rel not in baseline]
    status = "ok" if not (modified or removed) else "modified"
    return {"status": status, "checked": len(baseline),
            "modified": modified, "removed": removed, "added": added}


def sign_text(text: str, hmac_secret: str) -> str:
    """Detached HMAC signature for an exported file (e.g. the audit log)."""
    return hmac.new(hmac_secret.encode(), text.encode("utf-8"), hashlib.sha256).hexdigest()

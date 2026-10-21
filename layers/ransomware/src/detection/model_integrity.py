"""
model_integrity.py
---------------------
Why this exists: the trained models are stored as pickle (.pkl) files, and
loading a pickle can run any code hidden inside it. If someone swapped a
model file for a malicious one, loading it would run their code inside a
security tool - the worst possible place.

So every model file gets a SHA-256 fingerprint stored in models/MANIFEST.json
when the models are built (tools/make_model_manifest.py). Before a model is
loaded, its fingerprint is checked. A file that does not match is REFUSED.

Rules:
  - Manifest present, fingerprint matches   -> load.
  - Manifest present, fingerprint differs   -> refuse (ModelIntegrityError).
  - Manifest present, file not listed       -> refuse.
  - Manifest missing                        -> refuse if NEURASHIELD_REQUIRE_MODEL_MANIFEST=1,
                                               otherwise load with a loud warning
                                               (so a fresh developer checkout still runs).
The final package ships with a manifest, and the self-check reports its state.
"""

import hashlib
import json
import os

from src.utils import paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelIntegrityError(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path() -> str:
    return paths.resource(os.path.join("models", "MANIFEST.json"))


def load_manifest():
    try:
        with open(manifest_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def check_file(path: str) -> tuple:
    """Returns (ok: bool, message: str) for one model file."""
    manifest = load_manifest()
    if not os.path.exists(path):
        return False, f"model file not found: {path}"
    if manifest is None:
        if os.environ.get("NEURASHIELD_REQUIRE_MODEL_MANIFEST") == "1":
            return False, "models/MANIFEST.json is missing and a manifest is required"
        return True, "no manifest (integrity NOT checked)"
    expected = manifest.get("files", {}).get(os.path.basename(path))
    if expected is None:
        return False, f"{os.path.basename(path)} is not listed in the model manifest"
    if sha256_file(path) != expected:
        return False, f"{os.path.basename(path)} does not match its fingerprint - file was changed"
    return True, "fingerprint matches"


def verified_path(path: str) -> str:
    """Returns the path if the file passes the check, otherwise raises."""
    ok, message = check_file(path)
    if not ok:
        logger.error(f"[model_integrity] REFUSING to load {path}: {message}")
        raise ModelIntegrityError(message)
    if message.startswith("no manifest"):
        logger.warning(f"[model_integrity] {os.path.basename(path)}: {message}")
    return path


def verify_all(cfg: dict) -> list:
    """Checks every model file named in the config. Returns a list of (name, ok, message)."""
    names = [
        cfg["isolation_forest"]["model_path"],
        cfg["random_forest"]["model_path"],
        cfg["scaler_path"],
        cfg["scaler_path"].replace(".pkl", "_features.pkl"),
        cfg["scaler_path"].replace(".pkl", "_rf.pkl"),
        cfg["random_forest"]["model_path"].replace(".pkl", "_features.pkl"),
    ]
    results = []
    for rel in names:
        full = paths.resource(rel)
        ok, message = check_file(full)
        results.append((os.path.basename(rel), ok, message))
    return results

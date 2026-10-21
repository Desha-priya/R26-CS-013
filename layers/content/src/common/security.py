
import os
import secrets
from pathlib import Path

_KEY_DIR = Path(os.environ.get("NEURASHIELD_KEY_DIR", ".neurashield"))
_KEY_FILE = _KEY_DIR / "api_key.local"


def get_api_key() -> str:
    """Resolve the ONE NeuraShield platform API key.

    Priority:
      1. NEURASHIELD_API_KEY            (injected by the NeuraShield supervisor)
      2. NEURASHIELD_KEY_FILE           (shared key file the supervisor writes)
      3. existing local key file / freshly generated  - STANDALONE DEV ONLY.

    When the layer is launched by NeuraShield (NEURASHIELD_SUPERVISED=1) step 3 is
    forbidden: inventing a key of our own would silently split the system into
    layers that reject each other (401), so we stop with a clear error instead.
    """
    env_key = (os.environ.get("NEURASHIELD_API_KEY") or "").strip()
    if env_key:
        return env_key

    shared = os.environ.get("NEURASHIELD_KEY_FILE")
    if shared:
        try:
            k = Path(shared).read_text(encoding="utf-8").strip()
            if k:
                return k
        except OSError:
            pass

    if os.environ.get("NEURASHIELD_SUPERVISED") == "1":
        raise RuntimeError(
            "NeuraShield platform API key not available (NEURASHIELD_API_KEY / "
            "NEURASHIELD_KEY_FILE). Refusing to generate a different key.")

    if _KEY_FILE.exists():
        key = _KEY_FILE.read_text().strip()
        if key:
            return key

    _KEY_DIR.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(32)
    _KEY_FILE.write_text(key)
    try:
        os.chmod(_KEY_FILE, 0o600)   # best-effort; Windows mostly ignores this
    except Exception:
        pass
    print(f"[security] No NEURASHIELD_API_KEY set — generated a new local "
          f"dev key at {_KEY_FILE.resolve()}")
    print( "[security] Set NEURASHIELD_API_KEY explicitly before any real "
          "demo/deployment, or if processes run on different machines.")
    return key


def key_fingerprint(key: str) -> str:
    """Short, non-secret fingerprint for logging — confirms two processes
    are using the same key without ever printing the key itself."""
    return f"...{key[-6:]}" if key and len(key) > 6 else "***"


if __name__ == "__main__":
    k = get_api_key()
    print(f"Resolved key fingerprint: {key_fingerprint(k)}")

"""
secrets_store.py  —  NeuraShield shared secret manager (gateway)
----------------------------------------------------------------
One place that resolves the two secrets every layer shares:

  * NEURASHIELD_API_KEY      — the platform API key (X-API-Key on every call)
  * NEURASHIELD_HMAC_SECRET  — signing secret for alerts and rollback commands

Design rules (match the ZTA layer's own pattern):
  * No secret is ever hard-coded or written to a config file.
  * Priority: existing environment variable  >  protected file on disk  >
    freshly generated random secret stored protected.
  * On Windows the file is encrypted with DPAPI (per-user/machine), so another
    user or machine cannot read it. Elsewhere it is a 0600 file (dev fallback).
  * The supervisor calls get_api_key()/get_hmac_secret() ONCE at start and
    injects the values into every layer's environment, so all layers use the
    exact same values without any of them hard-coding a default.

This module is deliberately dependency-free (standard library only, plus an
optional cryptography fallback off Windows) so it can run before any layer's
virtual-environment is active.
"""

import os
import sys
import secrets as _secrets
from pathlib import Path

_IS_WIN = sys.platform == "win32"
_MAGIC = b"NSGW1"
_ENTROPY = b"NeuraShield-Gateway-Secrets-v1"


def _secret_dir() -> Path:
    """Folder that holds the protected secret files."""
    override = os.environ.get("NEURASHIELD_SECRET_DIR")
    if override:
        d = Path(override)
    elif _IS_WIN:
        base = os.environ.get("ProgramData", r"C:\ProgramData")
        d = Path(base) / "NeuraShield" / "secrets"
    else:
        d = Path(os.environ.get("NEURASHIELD_HOME", str(Path.home() / ".neurashield"))) / "secrets"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- DPAPI (Windows)
def _dpapi(data: bytes, encrypt: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD),
                    ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    fn = crypt32.CryptProtectData if encrypt else crypt32.CryptUnprotectData
    fn.argtypes = [ctypes.POINTER(DATA_BLOB), ctypes.c_void_p,
                   ctypes.POINTER(DATA_BLOB), ctypes.c_void_p,
                   ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DATA_BLOB)]
    fn.restype = wintypes.BOOL

    def blob(b):
        buf = ctypes.create_string_buffer(b, len(b))
        return DATA_BLOB(len(b), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf

    b_in, _k1 = blob(data)
    b_ent, _k2 = blob(_ENTROPY)
    b_out = DATA_BLOB()
    if not fn(ctypes.byref(b_in), None, ctypes.byref(b_ent), None, None, 0x1,
              ctypes.byref(b_out)):
        raise OSError("DPAPI operation failed")
    try:
        return ctypes.string_at(b_out.pbData, b_out.cbData)
    finally:
        kernel32.LocalFree(b_out.pbData)


def _fernet():
    from cryptography.fernet import Fernet
    key_path = _secret_dir() / ".gw_local_key"
    if key_path.exists():
        return Fernet(key_path.read_bytes().strip())
    key = Fernet.generate_key()
    key_path.write_bytes(key)
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    return Fernet(key)


def _protect(data: bytes) -> bytes:
    if _IS_WIN:
        return _MAGIC + b"W" + _dpapi(data, True)
    try:
        return _MAGIC + b"F" + _fernet().encrypt(data)
    except Exception:
        return _MAGIC + b"P" + data  # last-resort plain (dev only)


def _unprotect(blob: bytes) -> bytes:
    if not blob.startswith(_MAGIC):
        return blob.strip()  # legacy plain file
    kind = blob[len(_MAGIC):len(_MAGIC) + 1]
    payload = blob[len(_MAGIC) + 1:]
    if kind == b"W":
        return _dpapi(payload, False)
    if kind == b"F":
        return _fernet().decrypt(payload)
    return payload


def _resolve(env_name: str, file_name: str, generator) -> str:
    env_val = os.environ.get(env_name)
    if env_val and env_val.strip():
        return env_val.strip()
    path = _secret_dir() / file_name
    if path.exists():
        try:
            val = _unprotect(path.read_bytes()).decode("utf-8").strip()
            if val:
                return val
        except Exception:
            pass
    val = generator()
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(_protect(val.encode("utf-8")))
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except Exception:
        pass
    return val


def _existing_zta_key():
    """The Zero-Trust core owns the platform key. If it already has one on disk
    (plain api_key.txt in its data folder) and no gateway key exists yet, ADOPT it
    so the key you already use - browser extension, scripts - stays the key."""
    root = os.environ.get("NEURASHIELD_ROOT")
    if not root:
        return None
    try:
        v = (Path(root) / "data" / "core" / "api_key.txt").read_text(encoding="utf-8").strip()
        return v if len(v) >= 16 else None
    except OSError:
        return None


def get_api_key() -> str:
    """Resolve the shared platform API key (stable across restarts)."""
    return _resolve("NEURASHIELD_API_KEY", "api_key.dat",
                    lambda: _existing_zta_key() or _secrets.token_urlsafe(32))


def get_hmac_secret() -> str:
    """Resolve the shared HMAC signing secret (stable across restarts)."""
    return _resolve("NEURASHIELD_HMAC_SECRET", "hmac_secret.dat",
                    lambda: _secrets.token_urlsafe(32))


def fingerprint(value: str) -> str:
    """Non-secret fingerprint for logs — confirms two processes share a value."""
    return f"...{value[-6:]}" if value and len(value) > 6 else "***"


if __name__ == "__main__":
    print("API key fingerprint :", fingerprint(get_api_key()))
    print("HMAC fingerprint    :", fingerprint(get_hmac_secret()))
    print("Secret dir          :", _secret_dir())

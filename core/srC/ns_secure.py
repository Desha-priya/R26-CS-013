
import io
import os
import sys

import numpy as np

from ns_paths import data_path

MAGIC = b"NSENC1"
_ENTROPY = b"NeuraShield-LayerZTA-v3"
STRONG = sys.platform == "win32"


class ProtectionError(Exception):
    pass


# ----------------------------------------------------------- DPAPI
def _dpapi(data: bytes, encrypt: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD),
                    ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    fn = crypt32.CryptProtectData if encrypt else crypt32.CryptUnprotectData
    fn.argtypes = [ctypes.POINTER(DATA_BLOB), ctypes.c_void_p,
                   ctypes.POINTER(DATA_BLOB), ctypes.c_void_p,
                   ctypes.c_void_p, wintypes.DWORD,
                   ctypes.POINTER(DATA_BLOB)]
    fn.restype = wintypes.BOOL

    def blob(b):
        buf = ctypes.create_string_buffer(b, len(b))
        return DATA_BLOB(len(b), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf

    b_in, keep1 = blob(data)
    b_ent, keep2 = blob(_ENTROPY)
    b_out = DATA_BLOB()
    CRYPTPROTECT_UI_FORBIDDEN = 0x1
    ok = fn(ctypes.byref(b_in), None, ctypes.byref(b_ent), None, None,
            CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(b_out))
    if not ok:
        raise ProtectionError(
            f"DPAPI {'encrypt' if encrypt else 'decrypt'} failed "
            f"(Windows error {ctypes.GetLastError()}). The file may belong "
            "to a different Windows user or machine."
        )
    try:
        return ctypes.string_at(b_out.pbData, b_out.cbData)
    finally:
        kernel32.LocalFree(b_out.pbData)


# --------------------------------------------------- Fernet fallback
def _fernet():
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise ProtectionError(
            "Non-Windows fallback needs the 'cryptography' package "
            "(pip install cryptography)."
        ) from exc
    key_path = data_path(".ns_local_key")
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            key = f.read().strip()
    else:
        key = Fernet.generate_key()
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, "wb") as f:
            f.write(key)
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass
    return Fernet(key)


# ------------------------------------------------------------ API
def protect(data: bytes) -> bytes:
    if STRONG:
        return MAGIC + b"W" + _dpapi(data, True)
    return MAGIC + b"F" + _fernet().encrypt(data)


def unprotect(blob: bytes) -> bytes:
    if not blob.startswith(MAGIC):
        raise ProtectionError("Not a protected NeuraShield file.")
    kind = blob[len(MAGIC):len(MAGIC) + 1]
    payload = blob[len(MAGIC) + 1:]
    if kind == b"W":
        if not STRONG:
            raise ProtectionError("File was protected with Windows DPAPI "
                                  "and can only be opened on Windows.")
        return _dpapi(payload, False)
    if kind == b"F":
        try:
            return _fernet().decrypt(payload)
        except Exception as exc:
            raise ProtectionError(f"Fallback decryption failed: {exc}")
    raise ProtectionError("Unknown protection format.")


def is_protected(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def write_protected(path: str, data: bytes) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(protect(data))
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def read_bytes(path: str, migrate: bool = True) -> bytes:
    """Return the plain bytes of a (protected or legacy plain) file."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(MAGIC):
        return unprotect(raw)
    if migrate:
        try:
            write_protected(path, raw)
        except Exception:
            pass
    return raw


# ---------------------------------------------------- numpy helpers
def save_npy(path: str, array) -> None:
    buf = io.BytesIO()
    np.save(buf, np.asarray(array), allow_pickle=False)
    write_protected(path, buf.getvalue())


def load_npy(path: str):
    return np.load(io.BytesIO(read_bytes(path)), allow_pickle=False)


def save_npz(path: str, **arrays) -> None:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    write_protected(path, buf.getvalue())


def load_npz(path: str):
    return np.load(io.BytesIO(read_bytes(path)), allow_pickle=False)


if __name__ == "__main__":
    # Self-test: python ns_secure.py
    backend = "Windows DPAPI" if STRONG else "Fernet fallback (not a boundary)"
    print("Backend:", backend)
    blob = protect(b"neurashield self test")
    assert blob.startswith(MAGIC) and b"self test" not in blob
    assert unprotect(blob) == b"neurashield self test"
    print("Round trip: OK")

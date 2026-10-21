"""
src/common/oauth_store.py
Encrypted local storage for the Gmail OAuth client + refresh token.

Same pattern, same Fernet key file and same limits as credential_store.py
(the encryption key lives next to the data: it protects against casual
exposure such as an accidental commit or a stray backup, NOT against someone
who already controls this user account - an OS keychain is the production
upgrade). This module deliberately reuses credential_store's key helper and
leaves credential_store.py itself untouched.

Stored fields:  client_id, client_secret, refresh_token, email
"""

import json
import os

from src.common import credential_store as _cs

_OAUTH_FILE = _cs._STORE_DIR / "gmail_oauth.enc"


def _fernet():
    from cryptography.fernet import Fernet
    return Fernet(_cs._get_encryption_key())


def load_gmail() -> dict:
    """Decrypted dict ({} if nothing stored or unreadable)."""
    if not _OAUTH_FILE.exists():
        return {}
    try:
        data = json.loads(_fernet().decrypt(_OAUTH_FILE.read_bytes()))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_gmail(**fields) -> None:
    """Merge `fields` into the stored record and re-encrypt."""
    data = load_gmail()
    data.update({k: v for k, v in fields.items() if v is not None})
    _cs._STORE_DIR.mkdir(parents=True, exist_ok=True)
    _OAUTH_FILE.write_bytes(_fernet().encrypt(json.dumps(data).encode()))
    try:
        os.chmod(_OAUTH_FILE, 0o600)
    except Exception:
        pass


def clear_gmail() -> bool:
    existed = _OAUTH_FILE.exists()
    if existed:
        _OAUTH_FILE.unlink()
    return existed


def clear_gmail_token() -> None:
    """Forget the sign-in (refresh token + email) but keep the client config."""
    data = load_gmail()
    for k in ("refresh_token", "email"):
        data.pop(k, None)
    if data:
        _cs._STORE_DIR.mkdir(parents=True, exist_ok=True)
        _OAUTH_FILE.write_bytes(_fernet().encrypt(json.dumps(data).encode()))
    else:
        clear_gmail()


def gmail_summary() -> dict:
    """Non-secret summary for the dashboard - never returns secrets."""
    d = load_gmail()
    return {
        "client_configured": bool(d.get("client_id") and d.get("client_secret")),
        "connected": bool(d.get("refresh_token")),
        "email": d.get("email", ""),
    }

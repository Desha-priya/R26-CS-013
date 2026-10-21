"""
src/common/credential_store.py
Encrypted local storage for watcher credentials (currently: mail).

WHY THIS EXISTS
    The dashboard now lets you type a mailbox host/user/app-password
    directly into a form instead of setting environment variables in
    a terminal. That's much easier to use, but it means this process
    needs somewhere safe to keep that password between requests (and
    across restarts, so you don't have to re-enter it every time).

    Storing it in plain text on disk would undo the whole point of
    using an app-password in the first place. This module encrypts
    it at rest using a locally-generated key (Fernet — AES128 in
    CBC mode with HMAC authentication), following the exact same
    "generate once, persist locally, .gitignore it" pattern already
    established in src/common/security.py for the API key.

WHAT THIS DOES NOT PROTECT AGAINST
    Anyone with the ability to read files AND run code as this same
    user (i.e. anyone who has already compromised this machine) can
    decrypt these credentials, because the decryption key lives next
    to the encrypted data — the same way a house key on a hook by
    the door doesn't protect against someone already inside the
    house. Real protection against that threat model needs an OS
    keychain (Windows Credential Manager, macOS Keychain) or a
    secrets manager (Vault, AWS Secrets Manager) — noted as a
    production upgrade in the GRC documentation, not implemented
    here because it would make the "type it into the dashboard once"
    UX vendor/OS-specific. What THIS protects against: casual
    exposure (accidentally committing the file, someone glancing at
    disk contents, a backup landing somewhere it shouldn't).
"""

import os
import json
from pathlib import Path

_STORE_DIR = Path(os.environ.get("NEURASHIELD_KEY_DIR", ".neurashield"))
_CRED_FILE = _STORE_DIR / "mail_credentials.enc"
_CRED_KEY_FILE = _STORE_DIR / "cred_key.local"


def _get_encryption_key() -> bytes:
    from cryptography.fernet import Fernet
    if _CRED_KEY_FILE.exists():
        return _CRED_KEY_FILE.read_text().strip().encode()
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    _CRED_KEY_FILE.write_text(key.decode())
    try:
        os.chmod(_CRED_KEY_FILE, 0o600)
    except Exception:
        pass
    return key


def save_mail_credentials(host: str, user: str, password: str) -> None:
    """Encrypt and persist mail credentials, overwriting any previous set."""
    from cryptography.fernet import Fernet
    key = _get_encryption_key()
    f = Fernet(key)
    payload = json.dumps({"host": host, "user": user, "password": password}).encode()
    token = f.encrypt(payload)
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    _CRED_FILE.write_bytes(token)
    try:
        os.chmod(_CRED_FILE, 0o600)
    except Exception:
        pass


def load_mail_credentials() -> dict | None:
    """Return {'host','user','password'} or None if nothing is configured."""
    if not _CRED_FILE.exists():
        return None
    try:
        from cryptography.fernet import Fernet
        key = _get_encryption_key()
        f = Fernet(key)
        token = _CRED_FILE.read_bytes()
        payload = f.decrypt(token)
        return json.loads(payload)
    except Exception:
        return None


def clear_mail_credentials() -> bool:
    """Remove stored credentials. Returns True if something was removed."""
    existed = _CRED_FILE.exists()
    if existed:
        _CRED_FILE.unlink()
    return existed


def has_mail_credentials() -> bool:
    return _CRED_FILE.exists()


def mail_credentials_summary() -> dict:
    """Non-secret summary for the dashboard — never returns the password."""
    creds = load_mail_credentials()
    if not creds:
        return {"configured": False}
    return {"configured": True, "host": creds["host"], "user": creds["user"]}

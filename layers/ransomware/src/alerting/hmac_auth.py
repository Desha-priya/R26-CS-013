"""
hmac_auth.py
--------------
HMAC-SHA256 signing and verification for messages passed between the
Ransomware Killer and the Brain component (alerts going out, rollback
commands coming back). This closes a requirement stated explicitly in
the proposal's functional requirements: "The system will authenticate
the commands for rolling back using HMAC authentication."

Why this matters: without it, a malicious process on the same machine
could send a fake "approved" message to trigger an unwanted rollback,
or tamper with an alert before the Brain reads it. HMAC signing means
the receiver can verify BOTH that the message wasn't altered AND that
it came from a party who knows the shared secret.

This is symmetric-key HMAC (both sides share one secret), which fits
a single endpoint device. If NeuraShield layers ever run on different
machines, upgrade to asymmetric signing (e.g. Ed25519).

SECRET KEY - secure by default
  Order used to find the secret:
    1. Environment variable NEURASHIELD_HMAC_SECRET   (use this when
       several layers must share one secret)
    2. security.hmac_secret_key in config.yaml, unless it is empty or
       the old placeholder text
    3. A random 256-bit key generated on first start and stored in
       <runtime folder>/secret.key (readable only by the current user
       where the OS supports it)
  There is no longer any built-in default secret that anyone who reads
  the source code could use to forge a message.
"""

import hmac
import hashlib
import json
import os
import time

from src.utils import paths
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PLACEHOLDERS = {"", "changeme-in-production-use-env-var"}
MIN_SECRET_BYTES = 16


def _canonical_json(payload: dict) -> bytes:
    """Deterministic JSON serialization - same dict always produces the
    same bytes, regardless of key insertion order. Required for HMAC:
    signer and verifier must hash the exact same bytes."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _key_file_path() -> str:
    return paths.runtime("secret.key")


def _load_or_create_key_file() -> bytes:
    """Reads the generated key, creating it the first time. Safe when two
    processes (service + dashboard) start at the same moment."""
    path = _key_file_path()
    for _ in range(50):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                value = f.read().strip()
            if len(value) >= MIN_SECRET_BYTES * 2:
                return value.encode("utf-8")
            time.sleep(0.05)  # another process is still writing it
            continue
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            time.sleep(0.05)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(os.urandom(32).hex())
        logger.info("[hmac] Generated a new random signing key (stored in the runtime folder).")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip().encode("utf-8")


def get_secret_key(config: dict = None) -> bytes:
    """Loads the HMAC secret using the order described at the top of this file."""
    env_key = os.environ.get("NEURASHIELD_HMAC_SECRET")
    if env_key:
        if len(env_key.encode("utf-8")) < MIN_SECRET_BYTES:
            logger.warning("[hmac] NEURASHIELD_HMAC_SECRET is shorter than 16 bytes - "
                           "use a longer random value.")
        return env_key.encode("utf-8")

    config_key = ((config or {}).get("security", {}) or {}).get("hmac_secret_key", "") or ""
    if config_key not in _PLACEHOLDERS:
        return config_key.encode("utf-8")

    return _load_or_create_key_file()


def secret_source(config: dict = None) -> str:
    """Tells which source get_secret_key() will use (for the self-check report)."""
    if os.environ.get("NEURASHIELD_HMAC_SECRET"):
        return "environment variable"
    config_key = ((config or {}).get("security", {}) or {}).get("hmac_secret_key", "") or ""
    if config_key not in _PLACEHOLDERS:
        return "config file"
    return "generated key file"


def sign_payload(payload: dict, secret: bytes) -> str:
    """Returns a hex-encoded HMAC-SHA256 signature over payload's
    canonical JSON form. The payload itself should NOT already contain
    a 'signature' key - add the signature as a separate field after
    calling this."""
    digest = hmac.new(secret, _canonical_json(payload), hashlib.sha256)
    return digest.hexdigest()


def verify_payload(payload: dict, signature: str, secret: bytes) -> bool:
    """
    Recomputes the signature over `payload` and compares it against
    `signature` using a constant-time comparison (hmac.compare_digest)
    to avoid leaking timing information that could help an attacker
    guess a valid signature byte-by-byte.
    """
    if not isinstance(signature, str):
        return False
    try:
        expected = sign_payload(payload, secret)
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(expected, signature)


if __name__ == "__main__":
    secret = b"test-secret-for-smoke-test-only"

    message = {"command": "rollback", "target_pid": 4821, "confidence": 0.97}
    signature = sign_payload(message, secret)
    print("Signature:", signature)
    print("Verify (unmodified):", verify_payload(message, signature, secret))

    tampered = dict(message)
    tampered["target_pid"] = 9999
    print("Verify (tampered):", verify_payload(tampered, signature, secret))
    print("Verify (wrong secret):", verify_payload(message, signature, b"wrong-secret"))

"""
src/common/google_oauth.py
Google OAuth 2.0 for an installed/desktop app: authorization-code flow with
PKCE and a loopback redirect. No third-party Google libraries - just `requests`.

WHY THIS EXISTS
    Google no longer accepts a normal password for IMAP, and "app passwords"
    are only offered when 2-Step Verification is enabled on the account.
    OAuth is the supported way for a third-party app to read a mailbox.

LEAST PRIVILEGE
    We request ONLY  gmail.readonly  (view email). NeuraShield never needs to
    send, delete or modify mail, so it never asks for the broad
    https://mail.google.com/ scope that IMAP would require.

HONEST LIMITATIONS (also shown in the dashboard)
    - gmail.readonly is a Google "restricted" scope. While the Google Cloud
      project is in "Testing" status (fine for a research prototype / demo):
        * only accounts you add as "test users" can sign in,
        * Google shows an "unverified app" warning (Advanced -> continue),
        * the refresh token expires after 7 days -> sign in again weekly.
      Removing these needs Google OAuth verification (and, for restricted
      scopes, a third-party security assessment) - a commercialization step,
      not something a student prototype can do.
    - The client secret of a *Desktop* OAuth client is not confidential
      (Google documents this); it is still stored encrypted, never logged.

The authorization `state` nonce protects the callback against CSRF.
"""

import base64
import hashlib
import secrets
import threading
import time
from urllib.parse import urlencode

SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"

_STATE_TTL_SECONDS = 600
_HTTP_TIMEOUT = 15


class OAuthError(Exception):
    """Safe-to-display OAuth failure (never contains secrets)."""


class InvalidGrantError(OAuthError):
    """Refresh token expired/revoked - the user must sign in again."""


_pending = {}                 # state -> {verifier, redirect_uri, created}
_pending_lock = threading.Lock()


def _http_post(url, data):
    import requests
    return requests.post(url, data=data, timeout=_HTTP_TIMEOUT)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _purge_expired(now=None):
    now = now or time.time()
    for s in [s for s, v in _pending.items()
              if now - v["created"] > _STATE_TTL_SECONDS]:
        _pending.pop(s, None)


def begin_authorization(client_id: str, redirect_uri: str,
                        login_hint: str = None) -> str:
    """Create PKCE material + state and return the Google consent URL."""
    if not client_id:
        raise OAuthError("Google client ID is not configured.")
    verifier = secrets.token_urlsafe(64)
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    state = secrets.token_urlsafe(24)
    with _pending_lock:
        _purge_expired()
        _pending[state] = {"verifier": verifier,
                           "redirect_uri": redirect_uri,
                           "created": time.time()}
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",     # needed to receive a refresh token
        "prompt": "consent",          # re-issue refresh token on re-sign-in
    }
    if login_hint:
        params["login_hint"] = login_hint
    return AUTH_URL + "?" + urlencode(params)


def pending_state_valid(state: str) -> bool:
    with _pending_lock:
        _purge_expired()
        return state in _pending


def complete_authorization(state: str, code: str, client_id: str,
                           client_secret: str, http_post=None) -> dict:
    """
    Exchange the authorization code for tokens. The state is single-use:
    it is removed whether or not the exchange succeeds.
    Returns {"refresh_token", "access_token", "expires_at"}.
    """
    with _pending_lock:
        _purge_expired()
        entry = _pending.pop(state, None)
    if entry is None:
        raise OAuthError("Sign-in session expired or was not started here. "
                         "Click 'Sign in with Google' again.")
    post = http_post or _http_post
    try:
        resp = post(TOKEN_URL, {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "code_verifier": entry["verifier"],
            "grant_type": "authorization_code",
            "redirect_uri": entry["redirect_uri"],
        })
    except Exception as e:
        raise OAuthError(f"Could not reach Google to finish sign-in: "
                         f"{type(e).__name__}")
    body = _json(resp)
    if resp.status_code != 200:
        raise OAuthError("Google rejected the sign-in "
                         f"({body.get('error', resp.status_code)}). Check the "
                         "client ID/secret and that your account is a test user.")
    if not body.get("refresh_token"):
        raise OAuthError("Google did not return a refresh token. Remove "
                         "NeuraShield at myaccount.google.com/permissions and "
                         "sign in again.")
    return {
        "refresh_token": body["refresh_token"],
        "access_token": body.get("access_token", ""),
        "expires_at": time.time() + int(body.get("expires_in", 3600)),
    }


def refresh_access_token(client_id: str, client_secret: str,
                         refresh_token: str, http_post=None) -> dict:
    """Returns {"access_token", "expires_at"}; raises InvalidGrantError."""
    post = http_post or _http_post
    try:
        resp = post(TOKEN_URL, {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        })
    except Exception as e:
        raise OAuthError(f"Could not reach Google: {type(e).__name__}")
    body = _json(resp)
    if resp.status_code != 200:
        err = body.get("error", str(resp.status_code))
        if err == "invalid_grant":
            raise InvalidGrantError(
                "Google sign-in expired or was revoked. Click 'Sign in with "
                "Google' again (test-mode sign-ins expire after 7 days).")
        raise OAuthError(f"Google token refresh failed ({err}).")
    return {"access_token": body["access_token"],
            "expires_at": time.time() + int(body.get("expires_in", 3600))}


def revoke(token: str, http_post=None) -> bool:
    """Best-effort revoke at Google. Never raises."""
    if not token:
        return False
    post = http_post or _http_post
    try:
        return post(REVOKE_URL, {"token": token}).status_code == 200
    except Exception:
        return False


def _json(resp) -> dict:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

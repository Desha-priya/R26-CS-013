# =============================================================
# NeuraShield — Layer 4 — Extra Email Source API routes
# File   : src/watchers/email_routes.py
#
# A self-contained FastAPI APIRouter, so app.py needs only TWO new lines:
#
#     from src.watchers.email_routes import router as email_sources_router
#     app.include_router(email_sources_router)
#
# Auth: same X-API-Key (src/common/security.get_api_key) with constant-time
# comparison, same as the rest of the API, PLUS an Origin check: a browser
# request coming from any non-loopback website is refused, so a random web
# page cannot drive these endpoints even if CORS is permissive elsewhere.
#
# Google sign-in uses the documented "loopback" desktop flow: connect starts a
# one-shot listener on http://127.0.0.1:<free port>/ that receives Google's
# redirect (protected by the single-use random `state` nonce + PKCE), shows a
# confirmation page, and shuts itself down. No extra route is exposed on the
# main API for the callback.
#
#   GET    /api/sources/status
#   POST   /api/sources/gmail/config          {client_id, client_secret}
#   POST   /api/sources/gmail/connect         -> {auth_url}
#   POST   /api/sources/gmail/start | /stop
#   DELETE /api/sources/gmail                 sign out (?forget_client=true)
#   POST   /api/sources/folder/start          {path?}
#   POST   /api/sources/folder/stop
#   POST   /api/sources/notifications/start | /stop
#   POST   /api/sources/eml/upload            multipart .eml file
# =============================================================

import hmac
import html
import os
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.common import google_oauth, oauth_store, sources_config
from src.common.security import get_api_key
from src.watchers import folder_watcher, gmail_watcher, source_manager

router = APIRouter()

_API_KEY = get_api_key()
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_MAX_EML_BYTES = folder_watcher.MAX_BYTES
_LISTEN_TIMEOUT_SECONDS = 300          # how long the sign-in listener waits


def require_api_key(request: Request, x_api_key: str = Header(default=None)):
    # A browser attaches Origin to cross-site requests; refuse any that is not
    # this machine (non-browser callers send no Origin and are unaffected).
    origin = request.headers.get("origin")
    if origin and origin != "null":
        host = (urlparse(origin).hostname or "").lower()
        if host not in _LOOPBACK_HOSTS:
            raise HTTPException(status_code=403, detail="Cross-site request refused.")
    elif origin == "null":
        raise HTTPException(status_code=403, detail="Cross-site request refused.")
    if not x_api_key or not hmac.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return True


_RATE_LIMIT = int(os.environ.get("NEURASHIELD_RATE_LIMIT_PER_MIN", "30"))
_buckets = defaultdict(deque)


def rate_limit():
    now = time.time()
    b = _buckets["eml"]
    while b and now - b[0] > 60:
        b.popleft()
    if len(b) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    b.append(now)


# ── status ───────────────────────────────────────────────────────────
@router.get("/api/sources/status", tags=["Email sources"])
def sources_status(_auth=Depends(require_api_key)):
    return {
        "sources": source_manager.get_status(),
        "gmail": oauth_store.gmail_summary(),
        "folder": {"path": str(folder_watcher.resolve_folder())},
    }


# ── Gmail (OAuth) ────────────────────────────────────────────────────
class GmailConfig(BaseModel):
    client_id: str
    client_secret: str


@router.post("/api/sources/gmail/config", tags=["Email sources"])
def gmail_config(body: GmailConfig, _auth=Depends(require_api_key)):
    cid, sec = body.client_id.strip(), body.client_secret.strip()
    if not cid or not sec:
        raise HTTPException(status_code=400, detail="Client ID and client secret are required.")
    if not cid.endswith(".apps.googleusercontent.com"):
        raise HTTPException(status_code=400, detail=
                            "That doesn't look like a Google OAuth client ID "
                            "(it should end in .apps.googleusercontent.com).")
    oauth_store.save_gmail(client_id=cid, client_secret=sec)
    return {"ok": True, **oauth_store.gmail_summary()}


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuraShield</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;max-width:460px;margin:12vh auto;
padding:0 20px;color:#1c2330;background:#fff}}h1{{font-size:20px}}
p{{line-height:1.5;color:#4a5568}}@media(prefers-color-scheme:dark){{
body{{background:#10141c;color:#e6e9ef}}p{{color:#a7b0c0}}}}</style></head>
<body><h1>{title}</h1><p>{msg}</p></body></html>"""


def _page_html(title, msg) -> bytes:
    return _PAGE.format(title=html.escape(title), msg=html.escape(msg)).encode("utf-8")


def finish_google_signin(code: str, state: str, error: str = ""):
    """Complete a Google sign-in. Returns (title, message, http_status).
    Messages are fixed/safe strings - request input is never echoed."""
    if error:
        return ("Sign-in cancelled", "Google sign-in was not completed. You can "
                "close this tab and try again from NeuraShield.", 400)
    if not code or not state or not google_oauth.pending_state_valid(state):
        return ("Sign-in expired", "This sign-in link is no longer valid. Go back "
                "to NeuraShield and click 'Sign in with Google' again.", 400)
    d = oauth_store.load_gmail()
    try:
        tok = google_oauth.complete_authorization(
            state, code, d.get("client_id", ""), d.get("client_secret", ""))
    except google_oauth.OAuthError as e:
        return ("Sign-in failed", str(e), 400)
    addr = gmail_watcher.fetch_profile_email(tok["access_token"])
    oauth_store.save_gmail(refresh_token=tok["refresh_token"], email=addr)
    started = source_manager.start("gmail_watcher")
    note = ("NeuraShield is now watching new inbox mail (read-only)."
            if started["ok"] else
            "Signed in, but the watcher did not start: " + started["detail"])
    return ("Google account connected", note + " You can close this tab.", 200)


# One-shot loopback listener for Google's redirect (Google's documented
# desktop-app flow). Only one can be active; a new sign-in replaces the old.
_listener = {"server": None, "timer": None}
_listener_lock = threading.Lock()


def _stop_listener():
    with _listener_lock:
        srv, tmr = _listener["server"], _listener["timer"]
        _listener["server"] = _listener["timer"] = None
    if tmr:
        tmr.cancel()
    if srv:
        # shutdown() must not run on the handler's own thread (it would wait
        # on itself), so stop + close on a helper thread, in that order.
        def _close(s=srv):
            try:
                s.shutdown()
                s.server_close()
            except Exception:
                pass
        threading.Thread(target=_close, daemon=True).start()


class _CallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):          # keep codes/states out of the console
        pass

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        first = lambda k: (q.get(k) or [""])[0]
        if urlparse(self.path).path != "/" or not (first("code") or first("error")):
            self.send_response(404)
            self.end_headers()
            return
        title, msg, status = finish_google_signin(
            first("code"), first("state"), first("error"))
        body = _page_html(title, msg)
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)
        if status == 200 or first("error"):
            _stop_listener()            # done (a bad state keeps listening until timeout)


def _start_listener() -> int:
    """Start the loopback listener; returns its port."""
    _stop_listener()
    srv = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
    threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.1),
                     daemon=True, name="neurashield-oauth-callback").start()
    tmr = threading.Timer(_LISTEN_TIMEOUT_SECONDS, _stop_listener)
    tmr.daemon = True
    tmr.start()
    with _listener_lock:
        _listener["server"], _listener["timer"] = srv, tmr
    return srv.server_port


@router.post("/api/sources/gmail/connect", tags=["Email sources"])
def gmail_connect(_auth=Depends(require_api_key)):
    d = oauth_store.load_gmail()
    if not (d.get("client_id") and d.get("client_secret")):
        raise HTTPException(status_code=400, detail="Save your Google client ID and secret first.")
    port = _start_listener()
    try:
        url = google_oauth.begin_authorization(d["client_id"], f"http://127.0.0.1:{port}/")
    except google_oauth.OAuthError as e:
        _stop_listener()
        raise HTTPException(status_code=400, detail=str(e))
    return {"auth_url": url}


@router.post("/api/sources/gmail/start", tags=["Email sources"])
def gmail_start(_auth=Depends(require_api_key)):
    r = source_manager.start("gmail_watcher")
    if not r["ok"]:
        raise HTTPException(status_code=400, detail=r["detail"])
    return r


@router.post("/api/sources/gmail/stop", tags=["Email sources"])
def gmail_stop(_auth=Depends(require_api_key)):
    return source_manager.stop("gmail_watcher")


@router.delete("/api/sources/gmail", tags=["Email sources"])
def gmail_signout(forget_client: bool = False, _auth=Depends(require_api_key)):
    source_manager.stop("gmail_watcher")
    d = oauth_store.load_gmail()
    revoked = google_oauth.revoke(d.get("refresh_token", ""))
    if forget_client:
        oauth_store.clear_gmail()
    else:
        oauth_store.clear_gmail_token()
    gmail_watcher.reset_state()
    return {"ok": True, "revoked_at_google": revoked, "client_forgotten": forget_client}


# ── drop folder ──────────────────────────────────────────────────────
class FolderStart(BaseModel):
    path: str = ""


@router.post("/api/sources/folder/start", tags=["Email sources"])
def folder_start(body: FolderStart = None, _auth=Depends(require_api_key)):
    path = (body.path.strip() if body and body.path else "") or str(folder_watcher.resolve_folder())
    ok, detail = folder_watcher.validate_folder(path)
    if not ok:
        raise HTTPException(status_code=400, detail=detail)
    sources_config.set_enabled("folder", True, path=path)
    r = source_manager.start("folder_watcher")
    if not r["ok"]:
        raise HTTPException(status_code=400, detail=r["detail"])
    return {**r, "path": path}


@router.post("/api/sources/folder/stop", tags=["Email sources"])
def folder_stop(_auth=Depends(require_api_key)):
    return source_manager.stop("folder_watcher")


# ── notifications ────────────────────────────────────────────────────
@router.post("/api/sources/notifications/start", tags=["Email sources"])
def notifications_start(_auth=Depends(require_api_key)):
    r = source_manager.start("notification_watcher")
    if not r["ok"]:
        raise HTTPException(status_code=400, detail=r["detail"])
    return r


@router.post("/api/sources/notifications/stop", tags=["Email sources"])
def notifications_stop(_auth=Depends(require_api_key)):
    return source_manager.stop("notification_watcher")


# ── manual .eml upload ───────────────────────────────────────────────
@router.post("/api/sources/eml/upload", tags=["Email sources"])
def eml_upload(eml_file: UploadFile = File(...), _auth=Depends(require_api_key),
               _rl=Depends(rate_limit)):
    name = (eml_file.filename or "").lower()
    if not name.endswith(".eml"):
        raise HTTPException(status_code=400, detail="Please upload an .eml file.")
    raw = eml_file.file.read(_MAX_EML_BYTES + 1)
    if len(raw) > _MAX_EML_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 5 MB limit.")
    if not raw.strip():
        raise HTTPException(status_code=400, detail="File is empty.")
    res = folder_watcher.scan_eml_bytes(raw, intake="eml_upload")
    if not res["scanned"]:
        raise HTTPException(status_code=422, detail=
                            "Could not score this email (" + res["reason"] + ").")
    return {"modality": "text", "intake": "eml_upload",
            "phishing": res["phishing"], "risk_score": res["risk_score"],
            "final_decision": res["decision"]}

"""
console_app.py  —  NeuraShield unified operator console (gateway)
-----------------------------------------------------------------
ONE dashboard for the whole product. It is the only surface the operator opens
(http://127.0.0.1:9000). It:

  * requires an admin login (argon2id password, lockout, signed session cookie,
    CSRF token, strict security headers) — the layer dashboards are never
    exposed directly;
  * shows every layer's live state, the unified alert timeline (pulled from the
    ZTA platform with the shared key), consent state and the audit log;
  * lets the operator enable / disable each optional layer (calls the
    supervisor); turning one off requires the password again and is audited;
  * proxies read calls to the platform so the API key never reaches the browser.

It holds the shared API key server-side (from secrets_store) and attaches it to
every upstream call itself. The browser never sees the key.

Run:  python -m gateway.console_app
"""

import os
import sys
import json
import time
import hmac
import secrets
import hashlib
import threading
from pathlib import Path
from urllib import request as _request, error as _urlerror

import re
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import secrets_store                     # noqa: E402
from supervisor import Supervisor, _restrict_to_owner  # noqa: E402
from consent_notice import CONSENT_HTML, NOTICE_VERSION  # noqa: E402
from notifier import Notifier  # noqa: E402
import integrity  # noqa: E402
import autostart  # noqa: E402
from reset_service import ResetService, spawn_restart  # noqa: E402
from alert_store import AlertStore, AlertCollector  # noqa: E402

ROOT = Path(os.environ.get("NEURASHIELD_ROOT", Path(__file__).resolve().parent.parent))


def _find_console_dir() -> Path:
    """Source run: gateway/console next to this file. Frozen run: the folder
    PyInstaller unpacked the bundled data into."""
    candidates = [Path(__file__).resolve().parent / "console"]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "gateway" / "console")
    for c in candidates:
        if (c / "login.html").exists():
            return c
    return candidates[0]


CONSOLE_DIR = _find_console_dir()

# --------------------------------------------------------------------------- state
sup = Supervisor(str(ROOT))
API_KEY = sup.api_key
PLATFORM_URL = sup.platform_url
PRODUCT = sup.product

_admin_store = ROOT / "data" / "console" / "admin.json"
_audit_file = ROOT / "data" / "console" / "console_audit.log"
_sessions = {}            # token -> {created, last_seen, long}
_SESSION_TTL = 60 * 30    # 30 minutes idle
_SESSION_TTL_LONG = 60 * 60 * 24 * 30   # "stay signed in" = 30 days
_login_fails = {}         # ip -> [timestamps]
_LOCKOUT_MAX = 5
_LOCKOUT_WINDOW = 300

# Per-run token that lets the NeuraShield desktop app open the console WITHOUT
# a login prompt and lets the tray ask for a full shutdown. The app process
# reads it from a file only the current user can open; a plain browser has no
# way to get it, so browser access still requires login. It is regenerated on
# every console start and stays valid for as long as this console process lives
# (the desktop app may be opened hours after the background service started).
_APP_TOKEN = secrets.token_urlsafe(32)


def _is_app_token(candidate: str) -> bool:
    return bool(candidate) and hmac.compare_digest(candidate, _APP_TOKEN)


# The app token itself never appears in a URL (URLs end up in browser history).
# The app trades it, over a header, for a ticket that signs ONE window in and
# expires after a minute.
_TICKET_TTL = 60
_tickets = {}             # ticket -> expiry


def _issue_ticket() -> str:
    now = time.time()
    for k in [k for k, exp in _tickets.items() if exp < now]:
        _tickets.pop(k, None)
    t = secrets.token_urlsafe(24)
    _tickets[t] = now + _TICKET_TTL
    return t


def _consume_ticket(candidate: str) -> bool:
    exp = _tickets.pop(candidate or "", None)
    return exp is not None and exp >= time.time()


def _write_app_token():
    """Write the app token to a protected file the desktop shell reads."""
    p = ROOT / "data" / "console" / "app_token"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(_APP_TOKEN)
        _restrict_to_owner(p)
    except OSError:
        pass

app = FastAPI(title="NeuraShield Console", docs_url=None, redoc_url=None, openapi_url=None)


# --------------------------------------------------------------------------- argon2 / password
def _hash_password(password: str) -> str:
    """Prefer argon2id; fall back to PBKDF2-HMAC-SHA256 if argon2 is absent."""
    try:
        from argon2 import PasswordHasher
        return "argon2$" + PasswordHasher().hash(password)
    except Exception:
        salt = secrets.token_bytes(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
        return "pbkdf2$" + salt.hex() + "$" + dk.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        if stored.startswith("argon2$"):
            from argon2 import PasswordHasher
            return PasswordHasher().verify(stored[len("argon2$"):], password)
        if stored.startswith("pbkdf2$"):
            _, salt_hex, dk_hex = stored.split("$")
            dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                     bytes.fromhex(salt_hex), 200_000)
            return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False
    return False


def _load_admin():
    """Read the admin record. The file is DPAPI/Fernet-protected at rest; the
    password itself is additionally stored only as an argon2id hash."""
    if not _admin_store.exists():
        return None
    try:
        raw = _admin_store.read_bytes()
        try:
            plain = secrets_store._unprotect(raw)   # protected file
        except Exception:
            plain = raw                               # legacy plain file
        return json.loads(plain.decode("utf-8"))
    except Exception:
        return None


def _norm_answer(a: str) -> str:
    return " ".join((a or "").strip().lower().split())


def _save_admin(username: str, password: str, security=None):
    """security: optional list of {q, a} — the answers are hashed, never stored
    in the clear, so the recovery questions cannot be read off disk."""
    _admin_store.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "username": username,
        "password_hash": _hash_password(password),
        "created": time.time(),
    }
    if security:
        rec["security"] = [
            {"q": s["q"], "a_hash": _hash_password(_norm_answer(s["a"]))}
            for s in security if s.get("q") and s.get("a")
        ]
    record = json.dumps(rec).encode("utf-8")
    try:
        data = secrets_store._protect(record)   # DPAPI on Windows, Fernet elsewhere
    except Exception:
        data = record
    _admin_store.write_bytes(data)
    try:
        os.chmod(_admin_store, 0o600)
    except OSError:
        pass


def _update_password(new_password: str):
    admin = _load_admin()
    if not admin:
        return
    admin["password_hash"] = _hash_password(new_password)
    record = json.dumps(admin).encode("utf-8")
    try:
        data = secrets_store._protect(record)
    except Exception:
        data = record
    _admin_store.write_bytes(data)
    try:
        os.chmod(_admin_store, 0o600)
    except OSError:
        pass


def audit(event: str, detail: str = ""):
    _audit_file.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": time.time(), "event": event, "detail": detail})
    with open(_audit_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- sessions
def _new_session(long: bool = False) -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = {"created": time.time(), "last_seen": time.time(),
                        "csrf": secrets.token_urlsafe(16), "long": bool(long)}
    return token


def _valid_session(request: Request):
    token = request.cookies.get("ns_session")
    if not token or token not in _sessions:
        return None
    s = _sessions[token]
    ttl = _SESSION_TTL_LONG if s.get("long") else _SESSION_TTL
    if time.time() - s["last_seen"] > ttl:
        _sessions.pop(token, None)
        return None
    s["last_seen"] = time.time()
    return {"token": token, **s}


def _set_session_cookie(resp, token: str, long: bool):
    resp.set_cookie("ns_session", token, httponly=True, samesite="strict",
                    secure=False, max_age=(_SESSION_TTL_LONG if long else _SESSION_TTL))


def require_login(request: Request):
    s = _valid_session(request)
    if not s:
        raise HTTPException(status_code=401, detail="Login required")
    return s


def _locked_out(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _login_fails.get(ip, []) if now - t < _LOCKOUT_WINDOW]
    _login_fails[ip] = hits
    return len(hits) >= _LOCKOUT_MAX


# --------------------------------------------------------------------------- request guard
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def _hostname(host_header: str) -> str:
    h = (host_header or "").strip().lower()
    if h.startswith("["):
        return h.split("]")[0] + "]"
    return h.rsplit(":", 1)[0] if ":" in h else h


@app.middleware("http")
async def request_guard(request: Request, call_next):
    """Browser-borne attacks on a loopback service:
      * DNS rebinding — a hostile site re-points its name at 127.0.0.1; the Host
        header then still carries that name, so anything but a loopback name is
        refused;
      * cross-site request forgery — a hostile page can fire a POST at
        127.0.0.1:9000 (including /setup before an admin exists, or the tray
        quit URL). Browsers always attach Origin to such requests, and it must
        match the Host we were reached on. Non-browser clients send no Origin."""
    host = request.headers.get("host", "")
    if _hostname(host) not in _LOCAL_HOSTS:
        return JSONResponse({"detail": "Host not allowed"}, status_code=400)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        # Modern browsers state who is calling in Sec-Fetch-Site; that is the
        # reliable signal (Origin can legitimately be the string "null" on a
        # same-site form post when a no-referrer policy is in force).
        site = (request.headers.get("sec-fetch-site") or "").lower()
        if site:
            if site not in ("same-origin", "none"):
                return JSONResponse({"detail": "Cross-origin request refused"}, status_code=403)
        else:
            origin = request.headers.get("origin")
            if origin is not None and urlparse(origin).netloc.lower() != host.strip().lower():
                return JSONResponse({"detail": "Cross-origin request refused"}, status_code=403)
    return await call_next(request)


# --------------------------------------------------------------------------- security headers
@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["Cache-Control"] = "no-store"
    path = request.url.path
    # The embedded layer dashboards are served to an iframe INSIDE the console
    # (same origin). Those routes must therefore allow same-origin framing and a
    # relaxed CSP (the layer pages use inline scripts/styles and Google Fonts).
    # Every other console route stays locked down: no framing by anyone.
    if path.startswith("/embed/") or path.startswith("/proxy/"):
        resp.headers["X-Frame-Options"] = "SAMEORIGIN"
        resp.headers["Content-Security-Policy"] = "frame-ancestors 'self'"
    else:
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; img-src 'self' data:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
    return resp


# --------------------------------------------------------------------------- upstream helpers
def _platform_get(path: str):
    url = PLATFORM_URL.rstrip("/") + path
    req = _request.Request(url, headers={"X-API-Key": API_KEY}, method="GET")
    try:
        with _request.urlopen(req, timeout=4) as r:
            return json.loads(r.read().decode("utf-8")), r.status
    except _urlerror.HTTPError as e:
        return {"error": f"platform HTTP {e.code}"}, e.code
    except Exception as e:
        return {"error": f"platform unreachable: {e}"}, 503


def _platform_post(path: str, body: dict = None):
    url = PLATFORM_URL.rstrip("/") + path
    data = json.dumps(body or {}).encode("utf-8")
    req = _request.Request(url, data=data, method="POST",
                           headers={"X-API-Key": API_KEY,
                                    "Content-Type": "application/json"})
    try:
        with _request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode("utf-8")), r.status
    except _urlerror.HTTPError as e:
        return {"error": f"platform HTTP {e.code}"}, e.code
    except Exception as e:
        return {"error": f"platform unreachable: {e}"}, 503


def _save_product():
    (ROOT / "config" / "product.json").write_text(json.dumps(PRODUCT, indent=2))
    sup.reload_product()


def _consent_accepted() -> bool:
    return bool(PRODUCT.get("consent", {}).get("accepted"))


# --------------------------------------------------------------------------- notifier
_notifier = Notifier(
    platform_get=_platform_get,
    settings_cb=lambda: PRODUCT.get("notifications", {}),
    console_url=f"http://{PRODUCT.get('console', {}).get('host', '127.0.0.1')}:"
                f"{PRODUCT.get('console', {}).get('port', 9000)}",
)


# --------------------------------------------------------------------------- alert history
_alert_store = AlertStore(ROOT / "data" / "console" / "alert_history.jsonl")


def _fetch_json(url):
    """GET a layer URL with the platform key; parsed JSON or None."""
    try:
        req = _request.Request(url, headers={"X-API-Key": API_KEY}, method="GET")
        with _request.urlopen(req, timeout=4) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


_collector = AlertCollector(_alert_store, _fetch_json, on_new=_notifier.notify_record)
_reset = ResetService(ROOT, secrets_store._secret_dir())
_notifier.external_alerts = True


def _ensure_autostart_default():
    """First consent -> protect across restarts. The user can switch it off in
    Settings; once they have chosen, their choice is never overridden."""
    ui = PRODUCT.setdefault("ui", {})
    if "autostart" in ui:
        return
    try:
        if autostart.supported():
            autostart.enable(ROOT)
            audit("autostart_enabled", "default after consent")
        ui["autostart"] = True
        _save_product()
    except Exception as e:
        print(f"[console] autostart not enabled: {e}")


_activate_lock = threading.Lock()


def _activate_async():
    """Bring the platform up after consent: start the core and every enabled
    layer, wait for the hub, switch ZTA capture on, start notifications. Runs in
    a background thread because starting the core (model load) takes a while and
    must never block the console's web server."""
    def _run():
        with _activate_lock:
            try:
                sup.start_enabled()          # returns once the core answers
                if _consent_accepted() and PRODUCT.get("enabled_layers", {}).get("zta", True):
                    for _ in range(10):
                        _, code = _platform_post("/api/privacy/accept")
                        if code == 200:
                            break
                        time.sleep(2)
                if _consent_accepted():
                    _notifier.start()
                    _collector.start()
                # Heavy layers (ML models) take a while to come up; once they do,
                # prove they all run on the ONE platform key.
                for lid, lp in list(sup.procs.items()):
                    if lp.is_running() and not lp.spec.get("always_on"):
                        sup.wait_healthy(lid, timeout=240)
                rep = sup.verify_keys()
                bad = [k for k, v in rep.items() if v["ok"] is False]
                audit("key_check", "all layers share the platform key" if not bad
                      else "MISMATCH: " + ", ".join(bad))
                print("[console] key check: " + ("OK - every layer uses the platform key" if not bad
                                                  else f"MISMATCH in {bad}"))
                if bad:
                    try:
                        _notifier._toast("NeuraShield - key mismatch",
                                         "These layers are not using the platform key: " + ", ".join(bad), "high")
                    except Exception:
                        pass
            except Exception as e:
                print(f"[console] activation error: {e}")
    threading.Thread(target=_run, daemon=True, name="activate").start()


# --------------------------------------------------------------------------- routes: setup / auth
@app.get("/setup", response_class=HTMLResponse)
async def setup_page():
    if _load_admin():
        return RedirectResponse("/login", status_code=303)
    html = (CONSOLE_DIR / "setup.html").read_text(encoding="utf-8")
    return html.replace("<!--CONSENT_NOTICE-->", CONSENT_HTML)


@app.post("/setup")
async def setup_submit(request: Request,
                       username: str = Form(...), password: str = Form(...),
                       consent: str = Form(default=""),
                       q1: str = Form(default=""), a1: str = Form(default=""),
                       q2: str = Form(default=""), a2: str = Form(default=""),
                       L_zta: str = Form(default=""), L_network: str = Form(default=""),
                       L_ransomware: str = Form(default=""), L_content: str = Form(default="")):
    if _load_admin():
        raise HTTPException(status_code=400, detail="Admin already configured")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not (q1 and a1.strip() and q2 and a2.strip()):
        raise HTTPException(status_code=400, detail="Both security questions must be answered")
    if q1 == q2:
        raise HTTPException(status_code=400, detail="Choose two different security questions")
    security = [{"q": q1, "a": a1}, {"q": q2, "a": a2}]
    _save_admin(username.strip(), password, security=security)
    audit("admin_created", username.strip())
    # The layers the user chose at setup.
    chosen = {
        "zta": bool(L_zta), "network": bool(L_network),
        "ransomware": bool(L_ransomware), "content": bool(L_content),
    }
    PRODUCT["enabled_layers"] = chosen
    _save_product()
    audit("layers_selected", ",".join(k for k, v in chosen.items() if v) or "none")
    # Consent is captured at account creation. Declining still creates the
    # account, but the platform stays in standby (nothing monitors).
    if consent == "1":
        PRODUCT.setdefault("consent", {})
        PRODUCT["consent"]["accepted"] = True
        PRODUCT["consent"]["accepted_at"] = time.time()
        PRODUCT["consent"]["notice_version"] = NOTICE_VERSION
        _save_product()
        audit("consent_accepted", "at setup")
        _ensure_autostart_default()
        _activate_async()
    else:
        audit("consent_declined", "at setup")
    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    if not _load_admin():
        return RedirectResponse("/setup", status_code=303)
    return (CONSOLE_DIR / "login.html").read_text(encoding="utf-8")


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...),
                       remember: str = Form(default="")):
    ip = request.client.host if request.client else "unknown"
    if _locked_out(ip):
        audit("login_lockout", ip)
        raise HTTPException(status_code=429, detail="Too many attempts. Try again later.")
    admin = _load_admin()
    if not admin or username.strip() != admin["username"] or not _verify_password(password, admin["password_hash"]):
        _login_fails.setdefault(ip, []).append(time.time())
        audit("login_failed", username.strip())
        raise HTTPException(status_code=401, detail="Invalid username or password")
    _login_fails[ip] = []
    long = (remember == "1")
    token = _new_session(long=long)
    audit("login_success", username.strip())
    resp = RedirectResponse("/", status_code=303)
    _set_session_cookie(resp, token, long)
    return resp


# ---- account recovery via security questions --------------------------------
@app.get("/recover", response_class=HTMLResponse)
async def recover_page():
    if not _load_admin():
        return RedirectResponse("/setup", status_code=303)
    return (CONSOLE_DIR / "recover.html").read_text(encoding="utf-8")


@app.get("/api/recover/questions")
async def recover_questions():
    admin = _load_admin()
    sec = (admin or {}).get("security") or []
    if len(sec) < 2:
        raise HTTPException(status_code=400, detail="No security questions are set for this account.")
    return {"questions": [s["q"] for s in sec]}


@app.post("/api/recover/reset")
async def recover_reset(request: Request,
                        a1: str = Form(...), a2: str = Form(...),
                        new_password: str = Form(...)):
    ip = request.client.host if request.client else "unknown"
    if _locked_out(ip):
        raise HTTPException(status_code=429, detail="Too many attempts. Try again later.")
    admin = _load_admin()
    sec = (admin or {}).get("security") or []
    if len(sec) < 2:
        raise HTTPException(status_code=400, detail="No security questions are set.")
    ok1 = _verify_password(_norm_answer(a1), sec[0]["a_hash"])
    ok2 = _verify_password(_norm_answer(a2), sec[1]["a_hash"])
    if not (ok1 and ok2):
        _login_fails.setdefault(ip, []).append(time.time())
        audit("recover_failed")
        raise HTTPException(status_code=401, detail="One or more answers are incorrect.")
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    _login_fails[ip] = []
    _update_password(new_password)
    audit("password_reset", "via security questions")
    return {"ok": True}


@app.post("/logout")
async def logout(request: Request, session=Depends(require_login)):
    _sessions.pop(session["token"], None)
    audit("logout")
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("ns_session")
    return resp


def _check_csrf(request: Request, session: dict):
    sent = request.headers.get("X-CSRF-Token", "")
    if not sent or not hmac.compare_digest(sent, session.get("csrf", "")):
        raise HTTPException(status_code=403, detail="Bad CSRF token")


# --------------------------------------------------------------------------- routes: app shell
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not _load_admin():
        # First run: the desktop app may pass the app token so setup opens
        # without any prompt. Browser callers simply see setup.
        return RedirectResponse("/setup", status_code=303)
    # Desktop-app trusted entry: a valid one-time app token signs the window in
    # with no password. The token is single-use; the browser cannot obtain it.
    if _consume_ticket(request.query_params.get("app_token")):
        token = _new_session(long=True)
        audit("app_login", "desktop shell")
        resp = RedirectResponse("/", status_code=303)
        _set_session_cookie(resp, token, True)
        return resp
    if not _valid_session(request):
        return RedirectResponse("/login", status_code=303)
    return (CONSOLE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/me")
async def me(request: Request, session=Depends(require_login)):
    admin = _load_admin()
    return {"username": admin["username"], "csrf": session["csrf"],
            "product": PRODUCT.get("product_name"), "version": PRODUCT.get("version"),
            "consent_html": CONSENT_HTML,
            "tour_seen": bool(PRODUCT.get("ui", {}).get("tour_seen"))}


@app.post("/api/app/ticket")
async def app_ticket(request: Request):
    """Desktop shell / tray: trade the per-run app token for a one-time sign-in
    ticket (used as /?app_token=<ticket>)."""
    if not _is_app_token(request.headers.get("X-App-Token", "")):
        raise HTTPException(status_code=403, detail="App token required")
    return {"ticket": _issue_ticket()}


@app.post("/api/app/quit")
async def app_quit(request: Request):
    """Called by the tray 'Quit' action. Stops every layer and the console.
    Authenticated with the per-run app token (readable only by the signed-in
    user), so a web page or another local process cannot shut protection down."""
    if not _is_app_token(request.headers.get("X-App-Token", "")):
        raise HTTPException(status_code=403, detail="App token required")
    audit("app_quit", "tray")
    import threading as _t
    def _bye():
        time.sleep(0.3)
        _shutdown_everything()
        os._exit(0)
    _t.Thread(target=_bye, daemon=True).start()
    return {"ok": True}


@app.post("/api/platform-key")
async def platform_key(request: Request, session=Depends(require_login)):
    """Reveal the shared platform key to the signed-in operator (POST + CSRF),
    e.g. to paste into the browser-extension popup. Audited."""
    _check_csrf(request, session)
    audit("platform_key_viewed", "operator")
    return {"key": API_KEY}


@app.post("/api/tour-seen")
async def tour_seen(request: Request, session=Depends(require_login)):
    PRODUCT.setdefault("ui", {})["tour_seen"] = True
    _save_product()
    return {"ok": True}


# ---- alert history (persistent: what happened, also while the console was closed) ----
@app.get("/api/history")
def alert_history(limit: int = 50, layer: str = "", severity: str = "", unacked: int = 0,
                  session=Depends(require_login)):
    limit = max(1, min(int(limit), 500))
    rows = _alert_store.recent(limit, layer=layer or None, min_severity=severity or None,
                               unacked_only=bool(unacked))
    return {"alerts": rows, "counts": _alert_store.counts(), "now": time.time()}


@app.post("/api/history/{rid:path}/ack")
def alert_ack(rid: str, request: Request, session=Depends(require_login)):
    _check_csrf(request, session)
    ok = _alert_store.ack(rid)
    if ok:
        audit("alert_acknowledged", rid[:120])
    return {"ok": ok, "counts": _alert_store.counts()}


# ---- your data: where it is, clear it, delete the account ---------------------------
def _reauth(request: Request, password: str):
    """Destructive actions need the password again, even inside a signed-in session."""
    ip = request.client.host if request.client else "unknown"
    if _locked_out(ip):
        raise HTTPException(status_code=429, detail="Too many attempts. Try again later.")
    admin = _load_admin()
    if not admin or not _verify_password(password or "", admin["password_hash"]):
        _login_fails.setdefault(ip, []).append(time.time())
        audit("reauth_failed", "data/account action")
        raise HTTPException(status_code=401, detail="That password is not correct.")
    _login_fails[ip] = []


@app.get("/api/data/info")
def data_info(session=Depends(require_login)):
    return _reset.info()


def _do_clear(alerts: bool, audit_log: bool, layers: bool, backups: bool) -> dict:
    done = {}
    if layers:
        # layers hold their files open: stop them, wipe, then bring them back
        _platform_post("/api/privacy/withdraw")
        sup.stop_all(include_core=True)
        done["layer_data"] = _reset.clear_layer_data()
    if alerts:
        _alert_store.clear()
        done["alerts"] = _reset.clear_alerts()
    if audit_log:
        done["audit"] = _reset.clear_audit()
    if backups:
        done["backups"] = _reset.delete_backups()
    audit("data_cleared", ", ".join(k for k in done) or "nothing selected")
    if layers and _consent_accepted():
        _activate_async()
    return done


@app.post("/api/data/clear")
async def data_clear(request: Request, session=Depends(require_login)):
    _check_csrf(request, session)
    body = await request.json()
    _reauth(request, body.get("password"))
    done = await run_in_threadpool(_do_clear, bool(body.get("alerts")), bool(body.get("audit")),
                                   bool(body.get("layers")), bool(body.get("backups")))
    return {"ok": True, "cleared": done}


def _reset_everything(backups: bool):
    """Stop everything, erase the account and all stored data, return the platform to
    its first-run state, then start a fresh copy of the app."""
    time.sleep(0.8)                       # let the HTTP response reach the browser first
    try:
        _notifier.stop()
        _collector.stop()
        _platform_post("/api/privacy/withdraw")
        sup.stop_all(include_core=True)
        try:
            autostart.disable()
        except Exception:
            pass
        _reset.wipe_everything()
        if backups:
            _reset.delete_backups()
        fresh = _reset.reset_product(PRODUCT)
        _sessions.clear()
        _tickets.clear()
        _login_fails.clear()
        PRODUCT.clear()
        PRODUCT.update(fresh)
        sup.reload_product()
        _alert_store.clear()
    except Exception as e:
        print(f"[console] reset error: {e}")
    if os.environ.get("NEURASHIELD_NO_RESTART") == "1":     # tests only
        print("[console] reset complete (restart suppressed)")
        return
    print("[console] reset complete - restarting")
    _shutdown_everything()
    try:
        spawn_restart(3)
    finally:
        os._exit(0)


@app.post("/api/account/delete")
async def account_delete(request: Request, session=Depends(require_login)):
    _check_csrf(request, session)
    body = await request.json()
    _reauth(request, body.get("password"))
    if str(body.get("confirm", "")).strip() != "DELETE":
        raise HTTPException(status_code=400, detail='Type DELETE to confirm.')
    threading.Thread(target=_reset_everything, args=(bool(body.get("backups")),),
                     daemon=True, name="reset").start()
    resp = JSONResponse({"ok": True, "restarting": True})
    resp.delete_cookie("ns_session")
    return resp


# ---- start with Windows -----------------------------------------------------------
@app.get("/api/autostart")
def get_autostart(session=Depends(require_login)):
    st = autostart.status()
    st["command"] = autostart.command(ROOT) if st.get("supported") else ""
    return st


@app.post("/api/autostart")
async def set_autostart(request: Request, session=Depends(require_login)):
    _check_csrf(request, session)
    body = await request.json()
    want = bool(body.get("enabled"))
    if not autostart.supported():
        raise HTTPException(status_code=400, detail="Start with Windows is only available on Windows")
    if want:
        autostart.enable(ROOT)
    else:
        autostart.disable()
    PRODUCT.setdefault("ui", {})["autostart"] = want
    _save_product()
    audit("autostart_" + ("enabled" if want else "disabled"), "by operator")
    return autostart.status()


# ---- notification settings --------------------------------------------------
@app.get("/api/notifications")
async def get_notifications(request: Request, session=Depends(require_login)):
    return PRODUCT.get("notifications", {
        "enabled": True, "min_severity": "medium", "stepup": True,
        "layers": {"zta": True, "network": True, "ransomware": True, "content": True}})


@app.post("/api/notifications")
async def set_notifications(request: Request, session=Depends(require_login)):
    _check_csrf(request, session)
    body = await request.json()
    n = PRODUCT.setdefault("notifications", {})
    if "enabled" in body:      n["enabled"] = bool(body["enabled"])
    if "min_severity" in body: n["min_severity"] = str(body["min_severity"])
    if "stepup" in body:       n["stepup"] = bool(body["stepup"])
    if "layers" in body and isinstance(body["layers"], dict):
        n.setdefault("layers", {})
        for k in ("zta", "network", "ransomware", "content"):
            if k in body["layers"]:
                n["layers"][k] = bool(body["layers"][k])
    _save_product()
    audit("notifications_updated")
    return {"ok": True, "notifications": n}


@app.post("/api/notifications/test")
async def test_notification(request: Request, session=Depends(require_login)):
    _check_csrf(request, session)
    _notifier._toast("NeuraShield", "Notifications are working. You'll be alerted here when something needs attention.",
                     "low")
    return {"ok": True}


# --------------------------------------------------------------------------- routes: data (login-gated)
@app.get("/api/overview")
def overview(request: Request, session=Depends(require_login)):
    status = sup.status()
    platform, code = _platform_get("/api/platform/status")
    # Present ZTA as its own toggizable monitoring layer, distinct from the
    # always-on platform core it runs inside.
    platform_up = code == 200
    zta_enabled = bool(PRODUCT.get("enabled_layers", {}).get("zta", True))
    zta_on = zta_enabled and platform_up
    zta_layer = {
        "name": "Zero-Trust Authentication", "layer_number": 3,
        "enabled": zta_enabled,
        "running": bool(zta_on),
        "healthy": bool(zta_on),
        "port": 8000, "always_on": False, "is_zta": True,
        "pid": None, "restarts": 0,
    }
    # Relabel the raw 'core' entry as the platform core (plumbing).
    if "core" in status:
        status["core"]["name"] = "Platform Core"
        status["core"]["is_platform"] = True
    ordered = {"zta": zta_layer}
    for k, v in status.items():
        ordered[k] = v

    # --- ZTA live risk (from the platform's latest agent score) ----------
    agent = (platform or {}).get("latest_agent_score", {}) if code == 200 else {}
    risk = agent.get("overall_risk")
    try:
        risk = float(risk) if risk is not None else None
    except (TypeError, ValueError):
        risk = None
    threshold = agent.get("adaptive_threshold")
    try:
        threshold = float(threshold) if threshold is not None else None
    except (TypeError, ValueError):
        threshold = None

    risk_block = {
        "score": risk,                                   # 0..1 or None
        "percent": agent.get("risk_percent"),
        "threshold": threshold,
        "level": agent.get("risk_level"),
        "decision": agent.get("decision") or agent.get("risk_decision"),
        "behavioral": agent.get("behavioral_risk"),
        "alert": agent.get("alert_risk"),
        "context": agent.get("context_risk"),
        "available": bool(zta_on and risk is not None),
    }

    # alert counts per layer source (network/ransomware/content)
    active = (platform or {}).get("active_alerts", {}) if code == 200 else {}
    layer_alerts = {"1": 0, "2": 0, "4": 0}
    for v in active.values():
        ln = str(v.get("layer", ""))
        if ln in layer_alerts:
            layer_alerts[ln] += 1

    return {"layers": ordered, "platform": platform, "platform_ok": code == 200,
            "consent": {"accepted": _consent_accepted()},
            "risk": risk_block, "layer_alerts": layer_alerts,
            "n_active_alerts": (platform or {}).get("n_active_alerts", 0) if code == 200 else 0,
            "threat_level": (platform or {}).get("platform_threat_level", "low") if code == 200 else "low"}


@app.get("/api/alerts")
def alerts(request: Request, session=Depends(require_login)):
    data, code = _platform_get("/api/platform/status")
    active = (data or {}).get("active_alerts", {}) if code == 200 else {}
    events = (data or {}).get("recent_events", []) if code == 200 else []
    return {"active_alerts": active, "recent_events": events,
            "threat_level": (data or {}).get("platform_threat_level", "unknown")}


# ---- operator quick actions --------------------------------------------
@app.post("/api/pause-all")
def pause_all(request: Request, session=Depends(require_login)):
    """Pause monitoring across all layers without withdrawing consent — a
    quick 'stand down' the operator can reverse. Stops optional layer
    processes and ZTA capture; the platform and console keep running."""
    _check_csrf(request, session)
    _platform_post("/api/privacy/withdraw")
    sup.stop_all(include_core=False)
    audit("pause_all")
    return {"ok": True}


@app.post("/api/resume-all")
def resume_all(request: Request, session=Depends(require_login)):
    _check_csrf(request, session)
    if not _consent_accepted():
        raise HTTPException(status_code=403, detail="Consent required first")
    _activate_async()
    audit("resume_all")
    return {"ok": True}


@app.get("/api/audit/export")
async def audit_export(request: Request, session=Depends(require_login)):
    """Download the console audit log as a file, with a detached HMAC signature
    appended so the exported record is tamper-evident."""
    content = _audit_file.read_text(encoding="utf-8") if _audit_file.exists() else ""
    try:
        sig = integrity.sign_text(content, sup.hmac_secret)
        content = content + f"\n# NeuraShield signed export · HMAC-SHA256: {sig}\n"
    except Exception:
        pass
    fn = time.strftime("neurashield_audit_%Y%m%d_%H%M%S.log")
    return Response(content=content, media_type="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{fn}"'})


# ---- incident timeline + response ------------------------------------------
_incident_state_file = ROOT / "data" / "console" / "incident_state.json"


def _load_incident_state():
    try:
        return json.loads(_incident_state_file.read_text())
    except Exception:
        return {}


def _save_incident_state(st):
    _incident_state_file.parent.mkdir(parents=True, exist_ok=True)
    _incident_state_file.write_text(json.dumps(st, indent=2))


@app.get("/api/incidents")
def incidents(request: Request, session=Depends(require_login)):
    """One incident timeline across every layer. Pulls the platform's active
    alerts + recent events and the ransomware layer's own incident store,
    normalises them, and merges in the operator's acknowledge/resolve/notes."""
    state = _load_incident_state()
    items = []

    pdata, pcode = _platform_get("/api/platform/status")
    if pcode == 200 and isinstance(pdata, dict):
        for src, a in (pdata.get("active_alerts") or {}).items():
            ln = a.get("layer")
            items.append({
                "id": f"plat:{src}:{a.get('timestamp','')}",
                "layer": {1: "Network Guardian", 2: "Ransomware Killer",
                          3: "Zero-Trust", 4: "Content Threat"}.get(ln, "Platform"),
                "severity": (a.get("severity") or "medium").lower(),
                "title": a.get("message", "Security event"),
                "action": "Alert raised",
                "ts": a.get("timestamp"),
            })

    # Ransomware layer keeps detailed incidents (what was killed, files restored)
    def _g(url, needs_key=False):
        headers = {"X-API-Key": API_KEY} if needs_key else {}
        try:
            req = _request.Request(url, headers=headers, method="GET")
            with _request.urlopen(req, timeout=4) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None

    rinc = _g("http://127.0.0.1:5050/api/incidents")
    rlist = (rinc.get("incidents") if isinstance(rinc, dict) else rinc) or []
    for it in (rlist if isinstance(rlist, list) else [])[:25]:
        acted = it.get("action_taken", {}) or {}
        parts = []
        if acted.get("process_killed"):
            parts.append("process terminated")
        if acted.get("files_restored"):
            parts.append(f"{acted.get('files_restored')} file(s) restored")
        items.append({
            "id": f"ransom:{it.get('incident_id') or it.get('id') or it.get('detection_time','')}",
            "layer": "Ransomware Killer",
            "severity": (it.get("severity") or "high").lower(),
            "title": it.get("trigger") or it.get("summary") or "Ransomware incident",
            "action": ", ".join(parts) or "contained",
            "ts": it.get("detection_time") or it.get("timestamp"),
        })

    # merge operator state (ack/resolve/notes)
    for it in items:
        s = state.get(it["id"], {})
        it["status"] = s.get("status", "open")      # open | acknowledged | resolved
        it["note"] = s.get("note", "")

    # newest first
    def _k(x):
        return str(x.get("ts") or "")
    items.sort(key=_k, reverse=True)
    counts = {"open": 0, "acknowledged": 0, "resolved": 0}
    for it in items:
        counts[it["status"]] = counts.get(it["status"], 0) + 1
    return {"incidents": items[:60], "counts": counts}


@app.post("/api/incidents/{incident_id}/status")
async def set_incident_status(incident_id: str, request: Request, session=Depends(require_login)):
    _check_csrf(request, session)
    body = await request.json()
    status = body.get("status", "acknowledged")
    note = body.get("note", "")
    st = _load_incident_state()
    st[incident_id] = {"status": status, "note": note, "at": time.time()}
    _save_incident_state(st)
    audit(f"incident_{status}", incident_id)
    return {"ok": True}


@app.get("/api/integrity")
async def integrity_check(request: Request, session=Depends(require_login)):
    """Verify the platform's own files against the signed baseline."""
    try:
        rep = integrity.verify(ROOT, sup.hmac_secret)
    except Exception as e:
        rep = {"status": "error", "detail": str(e)}
    return rep


@app.post("/api/integrity/baseline")
async def integrity_rebaseline(request: Request, session=Depends(require_login)):
    """Re-establish the integrity baseline (after a legitimate update)."""
    _check_csrf(request, session)
    integrity.build_baseline(ROOT, sup.hmac_secret)
    audit("integrity_rebaselined")
    return {"ok": True}


@app.get("/api/selfcheck")
def selfcheck(request: Request, session=Depends(require_login)):
    """A quick health report the operator can run on demand."""
    checks = []
    status = sup.status()
    _, pcode = _platform_get("/api/platform/status")
    checks.append({"name": "Platform core", "ok": pcode == 200,
                   "detail": "reachable on :8000" if pcode == 200 else "not responding"})
    checks.append({"name": "Consent", "ok": _consent_accepted(),
                   "detail": "given" if _consent_accepted() else "not given — monitoring is off"})
    for lid, L in status.items():
        if L.get("always_on"):
            continue
        if PRODUCT.get("enabled_layers", {}).get(lid):
            checks.append({"name": L.get("name", lid), "ok": L.get("running") and L.get("healthy"),
                           "detail": ("running" if L.get("running") else "enabled but not running")})
    keyrep = sup.verify_keys()
    if keyrep:
        bad = [k for k, v in keyrep.items() if v["ok"] is False]
        checks.append({"name": "Shared API key", "ok": not bad,
                       "detail": ("every layer uses the platform key" if not bad else
                                  "mismatch: " + ", ".join(f"{k} ({keyrep[k]['detail']})" for k in bad))})
    verify, vcode = _platform_get("/api/platform/audit-verify")
    if vcode == 200 and isinstance(verify, dict) and "valid" in verify:
        checks.append({"name": "Audit integrity", "ok": bool(verify.get("valid")),
                       "detail": "chain intact" if verify.get("valid") else "chain broken"})
    # File / model self-protection
    try:
        rep = integrity.verify(ROOT, sup.hmac_secret)
        ok = rep["status"] in ("ok", "baseline_created")
        n = len(rep.get("modified", [])) + len(rep.get("removed", []))
        detail = ("all files verified" if rep["status"] == "ok"
                  else "baseline recorded" if rep["status"] == "baseline_created"
                  else f"{n} file(s) changed" if rep["status"] == "modified"
                  else rep["status"])
        checks.append({"name": "File integrity", "ok": ok, "detail": detail})
    except Exception:
        pass
    audit("selfcheck_run")
    return {"checks": checks, "ok": all(c["ok"] for c in checks)}


# ---- consent (global gate) ----------------------------------------------
@app.get("/api/consent")
async def consent_status(request: Request, session=Depends(require_login)):
    c = PRODUCT.get("consent", {})
    return {"accepted": bool(c.get("accepted")),
            "accepted_at": c.get("accepted_at"),
            "notice_version": c.get("notice_version", "1.0")}


@app.post("/api/consent/accept")
def consent_accept(request: Request, session=Depends(require_login)):
    _check_csrf(request, session)
    PRODUCT.setdefault("consent", {})
    PRODUCT["consent"]["accepted"] = True
    PRODUCT["consent"]["accepted_at"] = time.time()
    PRODUCT["consent"]["notice_version"] = NOTICE_VERSION
    _save_product()
    audit("consent_accepted")
    _ensure_autostart_default()
    # Bring up the core and the layers the user has enabled, and (unless ZTA was
    # turned off) start Zero-Trust monitoring via the platform's consent endpoint.
    _activate_async()
    return {"ok": True}


@app.post("/api/consent/withdraw")
def consent_withdraw(request: Request, session=Depends(require_login)):
    _check_csrf(request, session)
    PRODUCT.setdefault("consent", {})
    PRODUCT["consent"]["accepted"] = False
    PRODUCT["consent"]["accepted_at"] = None
    _save_product()
    audit("consent_withdrawn")
    # Withdrawing stops EVERYTHING: switch ZTA capture off while the platform can
    # still hear us, then stop every layer including the platform core. Only the
    # console web server stays up, so the user can consent again later.
    _notifier.stop()
    _collector.stop()
    _platform_post("/api/privacy/withdraw")
    sup.stop_all(include_core=True)
    sup.revoke_layer_consents()
    return {"ok": True}


@app.post("/api/layer/{layer_id}/{action}")
def control_layer(layer_id: str, action: str, request: Request,
                  session=Depends(require_login)):
    _check_csrf(request, session)
    if not _consent_accepted():
        raise HTTPException(status_code=403, detail="Consent required first")
    enable = (action == "start")

    # ZTA is a monitoring layer that lives inside the always-on platform
    # process, so it is toggled through the platform's privacy/capture
    # endpoints rather than by starting/stopping a process.
    if layer_id == "zta":
        if action not in ("start", "stop"):
            raise HTTPException(status_code=400, detail="Unknown action")
        resp, code = _platform_post(
            "/api/privacy/accept" if enable else "/api/privacy/withdraw")
        PRODUCT.setdefault("enabled_layers", {})["zta"] = enable
        _save_product()
        audit(f"zta_{action}")
        return {"ok": code == 200, "layer": "zta", "action": action}

    if layer_id not in sup.procs:
        raise HTTPException(status_code=404, detail="Unknown layer")
    if sup.procs[layer_id].spec.get("always_on") and action in ("stop", "disable"):
        raise HTTPException(status_code=400, detail="The platform core cannot be stopped")
    try:
        if action == "start":
            sup.start(layer_id)
        elif action == "stop":
            sup.stop(layer_id)
        elif action == "restart":
            sup.restart(layer_id)
        else:
            raise HTTPException(status_code=400, detail="Unknown action")
    except (RuntimeError, FileNotFoundError, OSError) as e:
        audit(f"layer_{action}_failed", f"{layer_id}: {e}"[:200])
        raise HTTPException(status_code=409, detail=str(e))
    PRODUCT.setdefault("enabled_layers", {})[layer_id] = (action != "stop")
    _save_product()
    audit(f"layer_{action}", layer_id)
    return {"ok": True, "layer": layer_id, "action": action}


_ABS_ATTR = re.compile(r'''(\s(?:src|href|action)\s*=\s*["'])/(?!/)''', re.I)
_ABS_CSS_URL = re.compile(r'''(url\(\s*["']?)/(?!/)''', re.I)


def _rewrite_layer_html(layer_id: str, html: str) -> str:
    """Make a layer's own page work inside the console, on the console's origin.

    1. Remove the platform key if the layer baked it into the page. The proxy adds
       the key server-side, so the browser never needs (or sees) it.
    2. Point root-relative asset/link URLs (src="/static/x.js", href="/layer3",
       url(/x.png)) at /proxy/<layer>/. <base href> does NOT do this for URLs that
       start with "/", so without it the layer's CSS/JS 404 against the console.
    3. Inject a fetch/XHR shim ahead of the page scripts so runtime-built
       "/api/..." calls are routed through the same proxy."""
    base = f"/proxy/{layer_id}/"
    for variant in {API_KEY, json.dumps(API_KEY)[1:-1]}:
        if variant:
            html = html.replace(variant, "")
    html = html.replace("__NEURASHIELD_API_KEY__", "")
    html = _ABS_ATTR.sub(lambda m: m.group(1) + base, html)
    html = _ABS_CSS_URL.sub(lambda m: m.group(1) + base, html)
    shim = (
        f'<base href="{base}">'
        '<script>(function(){var B="' + base + '";'
        'function fix(u){try{if(typeof u!=="string")return u;'
        'if(/^https?:/i.test(u)||u.indexOf(B)===0)return u;'
        'if(u.charAt(0)==="/")return B+u.slice(1);return u;}catch(e){return u;}}'
        'var f=window.fetch;if(f)window.fetch=function(u,o){return f(fix(u),o);};'
        'var X=window.XMLHttpRequest;if(X){var op=X.prototype.open;'
        'X.prototype.open=function(m,u){return op.apply(this,[m,fix(u)].concat([].slice.call(arguments,2)));};}'
        '})();</script>'
    )
    if re.search(r"<head[\s>]", html, re.I):
        return re.sub(r"(<head[^>]*>)", lambda m: m.group(1) + shim, html, count=1, flags=re.I)
    return shim + html


@app.get("/embed/{layer_id}", response_class=HTMLResponse)
def embed_layer(layer_id: str, request: Request):
    """Serve a layer's own dashboard HTML inside the console, so the operator
    sees it in the same UI without the API key ever reaching the browser.
    The console fetches the layer page server-side (adding the key where the
    layer requires it) and rewrites relative asset/API calls back through this
    proxy. Login-gated like every console page."""
    if not _valid_session(request):
        return RedirectResponse("/login", status_code=303)
    # Each layer's own landing page. We always proxy (never iframe the layer
    # origin directly) so framing works even for layers that send
    # X-Frame-Options: DENY (the ransomware dashboard does), and so the key is
    # added server-side by the proxy instead of living in the framed page.
    pages = {
        "network":    "http://127.0.0.1:5000/",
        "ransomware": "http://127.0.0.1:5050/",
        "content":    "http://127.0.0.1:8001/",
        "zta":        "http://127.0.0.1:8000/",
    }
    if layer_id not in pages:
        raise HTTPException(status_code=404, detail="Unknown layer")
    # Always fetch the layer page WITH the key (harmless if the layer's "/" is
    # open; required if it isn't), so the embedded dashboard always loads.
    headers = {"X-API-Key": API_KEY}
    try:
        req = _request.Request(pages[layer_id], headers=headers, method="GET")
        with _request.urlopen(req, timeout=5) as r:
            html = r.read().decode("utf-8", "replace")
    except Exception as e:
        return HTMLResponse(_embed_starting(layer_id, str(e)))
    return HTMLResponse(_rewrite_layer_html(layer_id, html))


# Every layer's own dashboard protects at least some of its API routes with the
# shared key (content's /api/analyse, ransomware's actions, etc.). The console
# proxy therefore adds the key to EVERY forwarded request — harmless for the few
# routes that don't require it, and it is what makes the embedded dashboards
# (e.g. content phishing analysis) work without exposing the key in the page.
_LAYER_NEEDS_KEY = {"network": True, "ransomware": True, "content": True, "zta": True}


def _embed_starting(layer_id: str, detail: str = "") -> str:
    name = {"zta": "Zero-Trust Authentication", "network": "Network Guardian",
            "ransomware": "Ransomware Killer", "content": "Content Threat Detection"}.get(layer_id, layer_id)
    return (
        "<div style=\"font-family:system-ui,Segoe UI,sans-serif;padding:48px;color:#56605c;text-align:center\">"
        "<div style=\"font-size:15px;font-weight:600;color:#1b2220;margin-bottom:8px\">"
        f"{esc_html(name)} is starting…</div>"
        "<p style=\"font-size:13.5px;max-width:420px;margin:0 auto\">This layer's dashboard will appear here once the "
        "service is up. If it was just enabled, give it a few seconds and this view refreshes automatically.</p>"
        "</div>")


@app.api_route("/proxy/{layer_id}/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_layer(layer_id: str, path: str, request: Request):
    """Reverse-proxy the embedded layer's own API/asset calls, adding the key."""
    if not _valid_session(request):
        raise HTTPException(status_code=401, detail="Login required")
    ports = {"network": 5000, "ransomware": 5050, "content": 8001, "zta": 8000}
    if layer_id not in ports:
        raise HTTPException(status_code=404)
    qs = ("?" + request.url.query) if request.url.query else ""
    url = f"http://127.0.0.1:{ports[layer_id]}/{path}{qs}"
    headers = {}
    if _LAYER_NEEDS_KEY.get(layer_id):
        headers["X-API-Key"] = API_KEY
    # Forward the few request headers a layer's own guards look for. The
    # ransomware dashboard rejects every POST that lacks X-Requested-With
    # (its CSRF check); the browser's script sends it, so it must survive here.
    for h in ("X-Requested-With", "Accept", "Content-Type"):
        v = request.headers.get(h)
        if v:
            headers[h] = v
    body = await request.body() if request.method in ("POST", "PUT", "PATCH") else None
    if body and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"
    # Content-layer analyses (video/audio models) can run for minutes; plain page
    # and status reads must still fail fast.
    timeout = 300 if request.method != "GET" else 20

    def _fetch():
        try:
            req = _request.Request(url, data=body, headers=headers, method=request.method)
            with _request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read(), r.headers
        except _urlerror.HTTPError as e:
            return e.code, e.read(), e.headers

    # urllib blocks; run it off the event loop so one slow layer call cannot
    # freeze the whole console.
    try:
        status, data, rh = await run_in_threadpool(_fetch)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    ctype = rh.get("Content-Type", "application/octet-stream")
    if ctype.lower().startswith("text/html") and status < 400:
        # Pages navigated to inside a layer get the same rewrite as /embed/.
        data = _rewrite_layer_html(layer_id, data.decode("utf-8", "replace")).encode("utf-8")
        ctype = "text/html; charset=utf-8"
    out_headers = {}
    if rh.get("Content-Disposition"):
        out_headers["Content-Disposition"] = rh["Content-Disposition"]
    return Response(content=data, status_code=status, media_type=ctype, headers=out_headers)


def esc_html(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


@app.get("/api/layer/{layer_id}/detail")
def layer_detail(layer_id: str, request: Request, session=Depends(require_login)):
    """Normalise each layer's own JSON into one shape the console renders in its
    own UI — so every layer looks like part of one platform, not a separate app
    with its own template. Data is fetched from the layer through the proxy
    (key added server-side)."""
    ports = {"zta": 8000, "network": 5000, "ransomware": 5050, "content": 8001}
    if layer_id not in ports:
        raise HTTPException(status_code=404, detail="Unknown layer")
    running = sup.procs.get("core" if layer_id == "zta" else layer_id)
    is_up = bool(running and running.is_running()) or (
        layer_id == "zta" and _platform_get("/api/platform/status")[1] == 200)

    def g(path, needs_key=True):
        url = f"http://127.0.0.1:{ports[layer_id]}{path}"
        # Always send the key — open routes ignore it, protected routes need it.
        headers = {"X-API-Key": API_KEY}
        try:
            req = _request.Request(url, headers=headers, method="GET")
            with _request.urlopen(req, timeout=4) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None

    out = {"id": layer_id, "running": is_up, "stats": [], "alerts": [], "meta": {}}

    if layer_id == "zta":
        d = g("/api/platform/status", True) or {}
        ag = d.get("latest_agent_score", {})
        out["stats"] = [
            {"label": "Session risk", "value": (f"{ag.get('risk_percent')}%" if ag.get("risk_percent") is not None else "—")},
            {"label": "Decision", "value": (ag.get("decision") or "—")},
            {"label": "Model", "value": d.get("behavioral_model", "siamese_bilstm_v3")},
            {"label": "Active alerts", "value": d.get("n_active_alerts", 0)},
        ]
        for e in (d.get("recent_events") or [])[-8:][::-1]:
            out["alerts"].append({"sev": e.get("severity", "info"), "msg": e.get("message", ""), "ts": e.get("timestamp")})

    elif layer_id == "ransomware":
        s = g("/api/status", False) or {}
        st = g("/api/stats", False) or {}
        al = g("/api/alerts", False) or {}
        out["stats"] = [
            {"label": "State", "value": s.get("state", "offline")},
            {"label": "Protected", "value": "Yes" if s.get("protected") else "No"},
            {"label": "Backups", "value": (f"{s.get('backup_percent')}%" if s.get("backup_percent") is not None else "—")},
            {"label": "CPU / Mem", "value": f"{s.get('cpu_percent','—')}% / {s.get('memory_mb','—')}MB"},
        ]
        items = al.get("alerts", al) if isinstance(al, (dict, list)) else []
        for a in (items if isinstance(items, list) else [])[:8]:
            out["alerts"].append({"sev": (a.get("threat", {}).get("severity") or a.get("severity") or "medium").lower(),
                                  "msg": a.get("message") or (a.get("threat", {}) or {}).get("type", "ransomware event"),
                                  "ts": a.get("timestamp")})

    elif layer_id == "content":
        st = g("/api/stats", False) or {}
        al = g("/api/alerts?limit=8", False) or {}
        bym = st.get("by_modality", {})
        out["stats"] = [
            {"label": "Analyses", "value": st.get("total_analyses", 0)},
            {"label": "Threats", "value": st.get("threats_detected", 0)},
            {"label": "High risk", "value": st.get("high_risk", 0)},
            {"label": "Text / Video / Audio", "value": f"{bym.get('text',0)} / {bym.get('video',0)} / {bym.get('audio',0)}"},
        ]
        items = al.get("alerts", al) if isinstance(al, dict) else al
        for a in (items if isinstance(items, list) else [])[:8]:
            out["alerts"].append({"sev": (a.get("risk") or a.get("severity") or "low").lower(),
                                  "msg": a.get("summary") or a.get("message") or a.get("source", "content event"),
                                  "ts": a.get("timestamp")})

    elif layer_id == "network":
        d = g("/api/data", True) or {}
        s = d.get("stats", {})
        out["stats"] = [
            {"label": "Flows", "value": s.get("total", 0)},
            {"label": "Alerts", "value": s.get("alert", 0)},
            {"label": "Warnings", "value": s.get("warning", 0)},
            {"label": "ML flags", "value": s.get("ml_flags", 0)},
        ]
        for a in (d.get("alerts") or [])[:8]:
            out["alerts"].append({"sev": (a.get("severity") or "medium").lower(),
                                  "msg": a.get("message", "network event"), "ts": a.get("timestamp")})
    return out


@app.get("/api/audit")
def audit_tail(request: Request, session=Depends(require_login)):
    lines = []
    if _audit_file.exists():
        for ln in _audit_file.read_text(encoding="utf-8").splitlines()[-200:]:
            try:
                lines.append(json.loads(ln))
            except Exception:
                pass
    # Also pull the platform's own tamper-evident audit verification.
    verify, code = _platform_get("/api/platform/audit-verify")
    return {"console_audit": list(reversed(lines)),
            "platform_audit": verify if code == 200 else None}


app.mount("/static", StaticFiles(directory=str(CONSOLE_DIR / "static")), name="static")


def _shutdown_everything(*_a):
    """Stop every layer and free every port. Runs when the platform entry point
    exits for ANY reason (Ctrl+C, SIGTERM, window closed, process killed)."""
    try:
        _notifier.stop()
        _collector.stop()
    except Exception:
        pass
    try:
        sup.shutdown()
    except Exception as e:
        print(f"[console] shutdown error: {e}")


def main():
    import uvicorn, atexit, signal as _signal
    host = PRODUCT.get("console", {}).get("host", "127.0.0.1")
    port = int(PRODUCT.get("console", {}).get("port", 9000))
    print(f"[console] NeuraShield console on http://{host}:{port}")
    print(f"[console] api key {secrets_store.fingerprint(API_KEY)}")
    _write_app_token()   # let the desktop shell open the window without a prompt

    # One platform: when this entry point dies, EVERYTHING it started dies too.
    atexit.register(_shutdown_everything)
    for _s in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(_signal, _s, None)
        if sig is not None:
            try:
                _signal.signal(sig, lambda *a: (_shutdown_everything(), os._exit(0)))
            except (ValueError, OSError):
                pass

    # IMPORTANT: no layer and not even the platform core starts here. The
    # console web server is the only thing running until the admin signs in
    # AND consent is on record (start happens in /setup or /api/consent/accept,
    # and start_enabled() itself refuses without consent). This is why nothing
    # runs in the background before consent.
    # Verify our own files on startup (self-protection). First run records the
    # baseline; later runs detect tampering and raise a notification.
    try:
        rep = integrity.verify(ROOT, sup.hmac_secret)
        if rep["status"] == "modified":
            msg = f"{len(rep['modified'])+len(rep['removed'])} platform file(s) changed since baseline"
            print(f"[integrity] WARNING: {msg}: {rep['modified'] + rep['removed']}")
            audit("integrity_mismatch", ", ".join(rep["modified"] + rep["removed"])[:300])
            try:
                _notifier._toast("NeuraShield — integrity warning",
                                 msg + ". Open the console to review.", "high")
            except Exception:
                pass
        elif rep["status"] == "manifest_tampered":
            print("[integrity] WARNING: integrity baseline signature invalid")
            audit("integrity_manifest_tampered")
        else:
            print(f"[integrity] {rep['status']} ({rep.get('checked',0)} files)")
    except Exception as e:
        print(f"[integrity] check skipped: {e}")

    if os.environ.get("NEURASHIELD_CONSOLE_SUPERVISE", "1") == "1":
        sup.start_watchdog()      # watchdog only restarts layers that ARE running
        if sup.consent_accepted():
            _activate_async()     # resuming an already-consented install
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()

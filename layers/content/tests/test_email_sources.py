"""
tests/test_email_sources.py
Offline tests for the extra email sources (Gmail OAuth, drop-folder,
notifications, .eml upload) plus regression checks on the original IMAP path.

Runs WITHOUT models, GPU, network or a Google account:
  - the fusion module is replaced by a tiny stub (keyword "phishing" model)
  - Google is replaced by an in-process fake HTTP server

Run from the project root:
    python -m unittest tests.test_email_sources -v
Optional: set NEURASHIELD_ORIG_ROOT=<path to an untouched copy of the project>
to also run the byte-for-byte regression check of the original IMAP parser.
"""

import base64
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import types
import unittest
from email.message import EmailMessage
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request as UrlRequest, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── isolated environment (must be set BEFORE the modules are imported) ──
_TMP = tempfile.mkdtemp(prefix="ns_test_")
os.chdir(_TMP)
os.environ["NEURASHIELD_KEY_DIR"] = str(Path(_TMP) / ".neurashield")
os.environ["NEURASHIELD_API_KEY"] = "test-key-123"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NEURASHIELD_RATE_LIMIT_PER_MIN"] = "1000"

# ── stub fusion module ──
SAVED, POSTED = [], []
_stub = types.ModuleType("src.fusion.multimodal_fusion_v5")


def _predict_text(t):
    low = (t or "").lower()
    ph = any(k in low for k in ("verify your account", "urgent", "password"))
    return {"prediction": "phishing" if ph else "normal",
            "risk_score": 0.95 if ph else 0.05, "confidence": 0.9}


def _fuse(t, v, a, weights=None):
    r = t["risk_score"]
    dec = "HIGH_RISK" if r >= 0.65 else "LOW_RISK"
    if dec == "HIGH_RISK":
        POSTED.append(dec)
    return {"final_risk_score": r, "final_decision": dec,
            "modality_scores": {"text": t, "video": v, "audio": a}}


def _save(alert, filename=None):
    SAVED.append(dict(alert))
    return "alert.json"


_stub.predict_text, _stub.fuse_scores, _stub.save_alert = _predict_text, _fuse, _save
sys.modules["src.fusion.multimodal_fusion_v5"] = _stub

from src.common import (credential_store, email_parse, google_oauth,   # noqa: E402
                        oauth_store, sources_config)
from src.watchers import (email_intake, folder_watcher, gmail_watcher,  # noqa: E402
                          mail_watcher, notification_watcher, source_manager,
                          watcher_manager)

PHISH = ("URGENT: verify your account password now or it will be closed. "
         "Click the link below to keep access to your mailbox today.")
SAFE = "Hi team, attached are the minutes from Tuesday's planning meeting. Thanks!"


def mk_eml(subject, plain=None, html=None):
    m = EmailMessage()
    m["Subject"], m["From"], m["To"] = subject, "a@example.com", "b@example.com"
    if plain is not None and html is not None:
        m.set_content(plain)
        m.add_alternative(html, subtype="html")
    elif html is not None:
        m.set_content(html, subtype="html")
    else:
        m.set_content(plain or "")
    return m


def wipe():
    shutil.rmtree(os.environ["NEURASHIELD_KEY_DIR"], ignore_errors=True)
    SAVED.clear()
    POSTED.clear()
    for s in (gmail_watcher.STATS, folder_watcher.STATS, notification_watcher.STATS):
        for k, v in list(s.items()):
            if isinstance(v, int):
                s[k] = 0
            elif k in ("last_check", "last_error"):
                s[k] = None


# ═════════════════════════════════════════════════════════════
class TestEmailParse(unittest.TestCase):
    def test_plain_only(self):
        s, f, b = email_parse.extract_email(mk_eml("Hello", plain="Body text"))
        self.assertEqual((s, b.strip()), ("Hello", "Body text"))

    def test_html_only_is_not_empty(self):
        html = ("<html><head><style>p{color:red}</style><script>evil()</script></head>"
                "<body><p>Verify your <b>account</b> now</p><div>Thanks&nbsp;&amp; bye</div></body></html>")
        _, _, b = email_parse.extract_email(mk_eml("x", html=html))
        self.assertIn("Verify your account now", b)
        self.assertIn("& bye", b)
        self.assertNotIn("evil", b)
        self.assertNotIn("color:red", b)
        self.assertNotIn("<", b)

    def test_plain_preferred_over_html(self):
        _, _, b = email_parse.extract_email(
            mk_eml("x", plain="plain wins", html="<p>html loses</p>"))
        self.assertIn("plain wins", b)
        self.assertNotIn("html loses", b)

    def test_rfc2047_subject(self):
        m = mk_eml("x", plain="b")
        del m["Subject"]
        m["Subject"] = "=?utf-8?B?VXJnZW50OiB2ZXJpZnk=?="
        self.assertEqual(email_parse.extract_email(m)[0], "Urgent: verify")

    def test_malformed_and_huge_html_never_raise_and_are_capped(self):
        self.assertIsInstance(email_parse.html_to_text("<div><p>unclosed <b>"), str)
        self.assertEqual(email_parse.html_to_text(""), "")
        big = "<p>" + ("word " * 200000) + "</p>"
        self.assertLessEqual(len(email_parse.html_to_text(big)), email_parse.MAX_BODY_CHARS)


# ═════════════════════════════════════════════════════════════
class TestOriginalImapPathUnchanged(unittest.TestCase):
    """The old IMAP watcher must behave exactly as before wherever it used to work."""

    def _cases(self):
        multi = EmailMessage()
        multi["Subject"] = "m"
        multi.set_content("plain part")
        multi.add_alternative("<p>html part</p>", subtype="html")
        multi.add_attachment(b"zz", maintype="application", subtype="octet-stream",
                             filename="a.bin")
        return [mk_eml("a", plain="just plain"), multi,
                mk_eml("h", html="<p>only html</p>")]

    def test_html_only_now_scored_not_skipped(self):
        body = mail_watcher._extract_body(mk_eml("h", html="<p>Verify your account</p>"))
        self.assertIn("Verify your account", body)
        self.assertNotIn("<p>", body)

    def test_plain_emails_identical_to_original(self):
        orig_root = os.environ.get("NEURASHIELD_ORIG_ROOT")
        if not orig_root:
            self.skipTest("set NEURASHIELD_ORIG_ROOT to compare with the untouched project")
        spec = importlib.util.spec_from_file_location(
            "orig_mail_watcher", Path(orig_root) / "src/watchers/mail_watcher.py")
        orig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(orig)
        for msg in self._cases()[:2]:                      # emails that HAD a plain part
            self.assertEqual(mail_watcher._extract_body(msg), orig._extract_body(msg))


# ═════════════════════════════════════════════════════════════
class TestGoogleOAuth(unittest.TestCase):
    def setUp(self):
        wipe()
        google_oauth._pending.clear()

    @staticmethod
    def _resp(status, body):
        return types.SimpleNamespace(status_code=status, json=lambda: body)

    def test_auth_url_has_pkce_state_and_least_privilege_scope(self):
        url = google_oauth.begin_authorization("cid.apps.googleusercontent.com",
                                               "http://127.0.0.1:8001/oauth/google/callback")
        q = parse_qs(urlparse(url).query)
        self.assertEqual(q["scope"], ["https://www.googleapis.com/auth/gmail.readonly"])
        self.assertEqual(q["code_challenge_method"], ["S256"])
        state = q["state"][0]
        verifier = google_oauth._pending[state]["verifier"]
        want = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        self.assertEqual(q["code_challenge"][0], want)
        self.assertNotIn("client_secret", url)

    def test_exchange_success_and_state_is_single_use(self):
        url = google_oauth.begin_authorization("cid", "http://127.0.0.1:1/cb")
        state = parse_qs(urlparse(url).query)["state"][0]
        sent = {}

        def post(u, data):
            sent.update(data)
            return self._resp(200, {"refresh_token": "RT", "access_token": "AT", "expires_in": 3600})
        tok = google_oauth.complete_authorization(state, "CODE", "cid", "sec", http_post=post)
        self.assertEqual(tok["refresh_token"], "RT")
        self.assertEqual(sent["grant_type"], "authorization_code")
        self.assertTrue(sent["code_verifier"])
        with self.assertRaises(google_oauth.OAuthError):          # replay
            google_oauth.complete_authorization(state, "CODE", "cid", "sec", http_post=post)

    def test_unknown_state_rejected(self):
        with self.assertRaises(google_oauth.OAuthError):
            google_oauth.complete_authorization("nope", "c", "cid", "sec",
                                                http_post=lambda *a: self._resp(200, {}))

    def test_expired_state_rejected(self):
        url = google_oauth.begin_authorization("cid", "http://127.0.0.1:1/cb")
        state = parse_qs(urlparse(url).query)["state"][0]
        google_oauth._pending[state]["created"] -= google_oauth._STATE_TTL_SECONDS + 5
        self.assertFalse(google_oauth.pending_state_valid(state))

    def test_error_messages_never_leak_secrets(self):
        url = google_oauth.begin_authorization("cid", "http://127.0.0.1:1/cb")
        state = parse_qs(urlparse(url).query)["state"][0]
        with self.assertRaises(google_oauth.OAuthError) as cm:
            google_oauth.complete_authorization(
                state, "CODE", "cid", "TOP-SECRET",
                http_post=lambda u, d: self._resp(400, {"error": "invalid_client"}))
        self.assertNotIn("TOP-SECRET", str(cm.exception))

    def test_missing_refresh_token(self):
        url = google_oauth.begin_authorization("cid", "http://127.0.0.1:1/cb")
        state = parse_qs(urlparse(url).query)["state"][0]
        with self.assertRaises(google_oauth.OAuthError):
            google_oauth.complete_authorization(
                state, "c", "cid", "s",
                http_post=lambda u, d: self._resp(200, {"access_token": "AT"}))

    def test_refresh_invalid_grant(self):
        with self.assertRaises(google_oauth.InvalidGrantError):
            google_oauth.refresh_access_token(
                "cid", "s", "RT",
                http_post=lambda u, d: self._resp(400, {"error": "invalid_grant"}))

    def test_network_failure_is_oautherror_not_crash(self):
        def boom(u, d):
            raise ConnectionError("down")
        with self.assertRaises(google_oauth.OAuthError):
            google_oauth.refresh_access_token("cid", "s", "RT", http_post=boom)
        self.assertFalse(google_oauth.revoke("RT", http_post=boom))


# ═════════════════════════════════════════════════════════════
class TestOAuthStore(unittest.TestCase):
    def setUp(self):
        wipe()

    def test_encrypted_at_rest_and_merge(self):
        oauth_store.save_gmail(client_id="CID.apps.googleusercontent.com", client_secret="SUPERSECRET")
        oauth_store.save_gmail(refresh_token="REFRESH-XYZ", email="me@gmail.com")
        blob = oauth_store._OAUTH_FILE.read_bytes()
        for s in (b"SUPERSECRET", b"REFRESH-XYZ", b"me@gmail.com"):
            self.assertNotIn(s, blob)
        d = oauth_store.load_gmail()
        self.assertEqual(d["client_secret"], "SUPERSECRET")
        self.assertEqual(d["refresh_token"], "REFRESH-XYZ")

    def test_summary_has_no_secrets_and_signout_keeps_client(self):
        oauth_store.save_gmail(client_id="c", client_secret="s", refresh_token="r", email="e@x.com")
        summ = json.dumps(oauth_store.gmail_summary())
        for s in ('"s"', '"r"', "secret", "refresh"):
            self.assertNotIn(s, summ.replace("client_configured", "").replace("connected", ""))
        oauth_store.clear_gmail_token()
        after = oauth_store.gmail_summary()
        self.assertTrue(after["client_configured"])
        self.assertFalse(after["connected"])
        oauth_store.clear_gmail()
        self.assertFalse(oauth_store.gmail_summary()["client_configured"])

    def test_corrupt_file_is_treated_as_empty(self):
        oauth_store._OAUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        oauth_store._OAUTH_FILE.write_bytes(b"garbage")
        self.assertEqual(oauth_store.load_gmail(), {})


# ═════════════════════════════════════════════════════════════
class FakeGoogle:
    """Fake token + Gmail endpoints served over real HTTP on localhost, so the
    production requests.get / requests.post code paths are exercised."""

    def __init__(self):
        self.messages = {}          # id -> raw bytes
        self.list_ids = []
        self.queries = []
        self.token_calls = 0
        self.fail_list_status = None
        self.token_error = None
        self.reject_first_api_call = False
        self._rejected = False
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                self.rfile.read(n)
                if self.path.startswith("/token"):
                    outer.token_calls += 1
                    if outer.token_error:
                        return self._send(400, {"error": outer.token_error})
                    return self._send(200, {"access_token": f"AT{outer.token_calls}", "expires_in": 3600})
                self._send(200, {})

            def do_GET(self):
                u = urlparse(self.path)
                if not self.headers.get("Authorization", "").startswith("Bearer AT"):
                    return self._send(401, {})
                if outer.reject_first_api_call and not outer._rejected:
                    outer._rejected = True
                    return self._send(401, {})
                if u.path.endswith("/messages"):
                    outer.queries.append(parse_qs(u.query).get("q", [""])[0])
                    if outer.fail_list_status:
                        return self._send(outer.fail_list_status, {})
                    return self._send(200, {"messages": [{"id": i} for i in outer.list_ids]})
                if "/messages/" in u.path:
                    mid = u.path.rsplit("/", 1)[1]
                    if mid not in outer.messages:
                        return self._send(404, {})
                    raw = base64.urlsafe_b64encode(outer.messages[mid]).rstrip(b"=").decode()
                    return self._send(200, {"raw": raw})
                if u.path.endswith("/profile"):
                    return self._send(200, {"emailAddress": "me@gmail.com"})
                self._send(404, {})

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


class TestGmailWatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = FakeGoogle()
        cls._orig = (google_oauth.TOKEN_URL, google_oauth.GMAIL_API)
        google_oauth.TOKEN_URL = cls.g.url + "/token"
        google_oauth.GMAIL_API = cls.g.url + "/gmail/v1/users/me"

    @classmethod
    def tearDownClass(cls):
        google_oauth.TOKEN_URL, google_oauth.GMAIL_API = cls._orig
        cls.g.close()

    def setUp(self):
        wipe()
        g = self.g
        g.messages, g.list_ids, g.queries = {}, [], []
        g.token_calls, g.fail_list_status, g.token_error = 0, None, None
        g.reject_first_api_call, g._rejected = False, False
        oauth_store.save_gmail(client_id="cid", client_secret="sec",
                               refresh_token="RT", email="me@gmail.com")
        g.messages = {
            "m1": mk_eml("Meeting", plain=SAFE).as_bytes(),
            "m2": mk_eml("Act now", plain=PHISH).as_bytes(),
            "m3": mk_eml("HTML only", html="<p>URGENT: verify your account password</p>").as_bytes(),
        }
        g.list_ids = ["m3", "m2", "m1"]            # Gmail returns newest first

    def test_scans_new_mail_dedupes_and_stores_hashes_only(self):
        sess = gmail_watcher._Session()
        state = gmail_watcher._load_state("me@gmail.com")
        n = gmail_watcher.poll_once(sess, state)
        self.assertEqual(n, 3)
        self.assertEqual(gmail_watcher.STATS["checked"], 3)
        self.assertEqual(gmail_watcher.STATS["phishing_found"], 2)       # plain phish + HTML-only phish
        self.assertTrue(all(a["intake"] == "gmail_api" for a in SAVED))
        self.assertIn("in:inbox after:", self.g.queries[0])
        # oldest first
        self.assertEqual([a["modality_scores"]["text"]["prediction"] for a in SAVED],
                         ["normal", "phishing", "phishing"])
        # second pass: nothing re-scanned
        self.assertEqual(gmail_watcher.poll_once(sess, state), 0)
        self.assertEqual(len(SAVED), 3)
        # persisted state holds hashes only - never ids or content
        text = gmail_watcher._STATE_FILE.read_text()
        for secret in ("m1", "m2", "m3", "URGENT", "Meeting"):
            self.assertNotIn(f'"{secret}"', text)
        self.assertEqual(len(json.loads(text)["seen"]), 3)

    def test_restart_does_not_rescan(self):
        sess = gmail_watcher._Session()
        gmail_watcher.poll_once(sess, gmail_watcher._load_state("me@gmail.com"))
        SAVED.clear()
        gmail_watcher.poll_once(gmail_watcher._Session(), gmail_watcher._load_state("me@gmail.com"))
        self.assertEqual(SAVED, [])

    def test_new_account_resets_baseline(self):
        gmail_watcher.poll_once(gmail_watcher._Session(), gmail_watcher._load_state("me@gmail.com"))
        st = gmail_watcher._load_state("other@gmail.com")
        self.assertEqual(st["seen"], [])

    def test_401_triggers_one_token_refresh_then_succeeds(self):
        self.g.reject_first_api_call = True
        sess = gmail_watcher._Session()
        n = gmail_watcher.poll_once(sess, gmail_watcher._load_state("me@gmail.com"))
        self.assertEqual(n, 3)
        self.assertGreaterEqual(self.g.token_calls, 2)

    def test_missing_message_404_is_skipped(self):
        self.g.list_ids = ["gone", "m1"]
        n = gmail_watcher.poll_once(gmail_watcher._Session(), gmail_watcher._load_state("me@gmail.com"))
        self.assertEqual(n, 1)

    def test_403_stops_with_clear_message(self):
        self.g.fail_list_status = 403
        ev = threading.Event()
        with self.assertRaises(RuntimeError) as cm:
            gmail_watcher.main(ev)
        self.assertIn("Gmail API", str(cm.exception))

    def test_revoked_token_stops_with_signin_message(self):
        self.g.token_error = "invalid_grant"
        with self.assertRaises(RuntimeError) as cm:
            gmail_watcher.main(threading.Event())
        self.assertIn("sign in", str(cm.exception).lower())

    def test_not_signed_in(self):
        oauth_store.clear_gmail()
        with self.assertRaises(RuntimeError):
            gmail_watcher.main(threading.Event())

    def test_transient_500_keeps_running_and_stops_on_event(self):
        self.g.fail_list_status = 500
        old = gmail_watcher.POLL_SECONDS
        gmail_watcher.POLL_SECONDS = 0.2
        ev = threading.Event()
        t = threading.Thread(target=gmail_watcher.main, args=(ev,), daemon=True)
        t.start()
        time.sleep(0.8)
        self.assertTrue(t.is_alive())                       # did not crash on 500
        self.assertTrue(gmail_watcher.STATS["last_error"])
        self.g.fail_list_status = None
        time.sleep(0.6)
        self.assertGreaterEqual(gmail_watcher.STATS["checked"], 3)   # recovered
        ev.set()
        t.join(3)
        self.assertFalse(t.is_alive())
        gmail_watcher.POLL_SECONDS = old


# ═════════════════════════════════════════════════════════════
class TestFolderWatcher(unittest.TestCase):
    def setUp(self):
        wipe()
        self.dir = Path(tempfile.mkdtemp(prefix="ns_drop_"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _drop(self, name, data):
        p = self.dir / name
        p.write_bytes(data if isinstance(data, bytes) else data.encode())
        old = time.time() - 60
        os.utime(p, (old, old))
        return p

    def test_scans_allowed_types_only_and_dedupes(self):
        self._drop("a.eml", mk_eml("x", plain=PHISH).as_bytes())
        self._drop("b.eml", mk_eml("y", plain=SAFE).as_bytes())
        self._drop("c.txt", PHISH)
        self._drop("d.html", "<p>URGENT verify your account</p>")
        self._drop("e.exe", b"MZ" + b"\0" * 50)
        self._drop("f.eml", b"")                                    # empty
        (self.dir / "sub").mkdir()
        seen = []
        n = folder_watcher.scan_folder_once(self.dir, seen)
        self.assertEqual(n, 4)
        self.assertEqual(folder_watcher.STATS["phishing_found"], 3)
        self.assertTrue(all(a["intake"] == "eml_folder" for a in SAVED))
        self.assertEqual(folder_watcher.scan_folder_once(self.dir, seen), 0)   # dedupe
        self.assertTrue(all(len(h) == 32 for h in seen))

    def test_user_files_never_modified_or_deleted(self):
        p = self._drop("a.eml", mk_eml("x", plain=PHISH).as_bytes())
        before = (p.read_bytes(), p.stat().st_mtime)
        folder_watcher.scan_folder_once(self.dir, [])
        self.assertTrue(p.exists())
        self.assertEqual((p.read_bytes(), p.stat().st_mtime), before)

    def test_file_still_being_written_is_deferred(self):
        p = self.dir / "new.eml"
        p.write_bytes(mk_eml("x", plain=PHISH).as_bytes())          # mtime = now
        seen = []
        self.assertEqual(folder_watcher.scan_folder_once(self.dir, seen), 0)
        self.assertEqual(folder_watcher.scan_folder_once(self.dir, seen, now=time.time() + 10), 1)

    def test_oversize_and_symlink_ignored(self):
        big = self.dir / "big.eml"
        with open(big, "wb") as f:
            f.truncate(folder_watcher.MAX_BYTES + 1)
        os.utime(big, (time.time() - 60,) * 2)
        target = self._drop("t.eml", mk_eml("x", plain=PHISH).as_bytes())
        link = self.dir / "link.eml"
        try:
            link.symlink_to(target)
        except OSError:
            link = None
        seen = []
        n = folder_watcher.scan_folder_once(self.dir, seen)
        self.assertEqual(n, 1)                                      # only the real file

    def test_corrupt_eml_does_not_break_the_pass(self):
        self._drop("bad.eml", b"\xff\xfe\x00garbage\x00\x01")
        self._drop("ok.eml", mk_eml("x", plain=PHISH).as_bytes())
        folder_watcher.scan_folder_once(self.dir, [])
        self.assertGreaterEqual(folder_watcher.STATS["phishing_found"], 1)

    def test_validate_folder(self):
        self.assertTrue(folder_watcher.validate_folder(str(self.dir))[0])
        self.assertFalse(folder_watcher.validate_folder(str(self.dir / "nope"))[0])
        f = self._drop("file.txt", "x")
        self.assertFalse(folder_watcher.validate_folder(str(f))[0])

    def test_main_loop_picks_up_file_dropped_while_running(self):
        sources_config.set_enabled("folder", True, path=str(self.dir))
        old = folder_watcher.POLL_SECONDS
        folder_watcher.POLL_SECONDS = 0.2
        ev = threading.Event()
        t = threading.Thread(target=folder_watcher.main, args=(ev,), daemon=True)
        t.start()
        time.sleep(0.5)
        self._drop("late.eml", mk_eml("x", plain=PHISH).as_bytes())
        time.sleep(1.0)
        ev.set()
        t.join(3)
        folder_watcher.POLL_SECONDS = old
        self.assertEqual(folder_watcher.STATS["phishing_found"], 1)
        self.assertTrue(folder_watcher._SEEN_FILE.exists())


# ═════════════════════════════════════════════════════════════
def toast_xml(*texts):
    inner = "".join(f"<text>{t}</text>" for t in texts)
    return (f'<toast><visual><binding template="ToastGeneric">{inner}</binding></visual></toast>')


def make_wpn_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE NotificationHandler (RecordId INTEGER PRIMARY KEY, PrimaryId TEXT)")
    con.execute("CREATE TABLE Notification (Id INTEGER PRIMARY KEY, HandlerId INTEGER, Type TEXT, Payload BLOB)")
    handlers = {}
    for _id, app, payload in rows:
        if app not in handlers:
            handlers[app] = len(handlers) + 1
            con.execute("INSERT INTO NotificationHandler VALUES (?,?)", (handlers[app], app))
        con.execute("INSERT INTO Notification VALUES (?,?,?,?)", (_id, handlers[app], "toast", payload))
    con.commit()
    con.close()


class TestNotificationWatcher(unittest.TestCase):
    def setUp(self):
        wipe()
        self.db = Path(tempfile.mkdtemp(prefix="ns_wpn_")) / "wpndatabase.db"

    def tearDown(self):
        shutil.rmtree(self.db.parent, ignore_errors=True)

    def test_only_new_mail_toasts_are_scanned_and_others_never_parsed(self):
        make_wpn_db(self.db, [(1, "Microsoft.Office.OUTLOOK.EXE.15", toast_xml("old", PHISH))])
        base = notification_watcher.max_notification_id(self.db)
        self.assertEqual(base, 1)
        con = sqlite3.connect(self.db)
        con.execute("INSERT INTO NotificationHandler VALUES (50,'WhatsApp')")
        con.execute("INSERT INTO NotificationHandler VALUES (51,'Chrome')")
        con.execute("INSERT INTO NotificationHandler VALUES (52,'Microsoft.Office.OUTLOOK.EXE.15')")
        new = [
            (2, 50, toast_xml("Mum", "URGENT verify your account password please")),   # other app
            (3, 51, toast_xml("mail.google.com", "Boss", "URGENT: verify your account password")),
            (4, 51, toast_xml("news.example.com", "URGENT: verify your account password")),  # chrome, not webmail
            (5, 52, toast_xml("Bank", "URGENT: verify your account password today please")),
            (6, 52, toast_xml("Hi")),                                                  # too short
            (7, 52, toast_xml("Team", SAFE)),
        ]
        for nid, hid, payload in new:
            con.execute("INSERT INTO Notification VALUES (?,?,?,?)", (nid, hid, "toast", payload))
        con.commit()
        con.close()

        parsed_for = []
        real = notification_watcher.toast_texts
        notification_watcher.toast_texts = lambda p: (parsed_for.append(p), real(p))[1]
        try:
            last = notification_watcher.scan_new_notifications(self.db, base)
        finally:
            notification_watcher.toast_texts = real
        self.assertEqual(last, 7)
        s = notification_watcher.STATS
        self.assertEqual(s["checked"], 3)                 # gmail-web phish, outlook phish, outlook safe
        self.assertEqual(s["phishing_found"], 2)
        self.assertEqual(s["ignored_other_apps"], 2)      # WhatsApp + non-webmail Chrome
        # privacy allow-list: the WhatsApp payload must never even be parsed
        self.assertEqual(len(parsed_for), 5)              # ids 3,4 (browser) + 5,6,7 (Outlook)
        self.assertFalse(any(b"Mum" in (p if isinstance(p, bytes) else str(p).encode()) for p in parsed_for))
        self.assertTrue(all(a["intake"] == "mail_notification" for a in SAVED))
        # site attribution line is not part of the scanned text
        self.assertEqual(notification_watcher.scan_new_notifications(self.db, last), last)   # nothing new

    def test_utf16_payload_and_namespaced_xml(self):
        xml = ('<toast xmlns="urn:x"><visual><binding><text>Sender</text><text>Subject line here that is long enough</text>'
               '</binding></visual></toast>')
        self.assertEqual(notification_watcher.toast_texts(xml.encode("utf-16")),
                         ["Sender", "Subject line here that is long enough"])
        self.assertEqual(notification_watcher.toast_texts(b"\xff\x00 not xml"), [])
        self.assertEqual(notification_watcher.toast_texts(None), [])

    def test_unknown_schema_is_a_clear_error_not_a_guess(self):
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE Notification (foo INTEGER)")
        con.execute("CREATE TABLE NotificationHandler (bar INTEGER)")
        con.commit()
        con.close()
        with self.assertRaises(RuntimeError) as cm:
            notification_watcher.main(threading.Event(), db_path=self.db)
        self.assertIn("Unrecognised", str(cm.exception))

    def test_missing_db_is_clear_error(self):
        with self.assertRaises(RuntimeError):
            notification_watcher.main(threading.Event(), db_path=self.db.parent / "none.db")

    def test_main_loop_baseline_and_new_toast(self):
        make_wpn_db(self.db, [(1, "Microsoft.Office.OUTLOOK.EXE.15", toast_xml("old", PHISH))])
        old = notification_watcher.POLL_SECONDS
        notification_watcher.POLL_SECONDS = 0.2
        ev = threading.Event()
        t = threading.Thread(target=notification_watcher.main, args=(ev, self.db), daemon=True)
        t.start()
        time.sleep(0.5)
        con = sqlite3.connect(self.db)
        con.execute("INSERT INTO Notification VALUES (9,1,'toast',?)",
                    (toast_xml("Bank", "URGENT: verify your account password right now"),))
        con.commit()
        con.close()
        time.sleep(0.8)
        ev.set()
        t.join(3)
        notification_watcher.POLL_SECONDS = old
        self.assertEqual(notification_watcher.STATS["checked"], 1)       # old row (id 1) ignored
        self.assertEqual(notification_watcher.STATS["phishing_found"], 1)

    def test_not_windows_without_db_override(self):
        if sys.platform == "win32":
            self.skipTest("windows")
        with self.assertRaises(RuntimeError):
            notification_watcher.main(threading.Event())


# ═════════════════════════════════════════════════════════════
class TestManagerAndRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.watchers import email_routes
        cls.routes = email_routes
        app = FastAPI()
        app.include_router(email_routes.router)
        cls.client = TestClient(app, base_url="http://127.0.0.1:8001")
        cls.H = {"X-API-Key": "test-key-123"}

    def setUp(self):
        wipe()
        google_oauth._pending.clear()
        for n in source_manager.SOURCES:
            source_manager.stop(n, persist=False)
        self._orig_main = gmail_watcher.main
        gmail_watcher.main = lambda ev, *a, **k: ev.wait()          # idle stub watcher

    def tearDown(self):
        gmail_watcher.main = self._orig_main
        for n in source_manager.SOURCES:
            source_manager.stop(n, persist=False)

    # auth
    def test_every_api_route_requires_key(self):
        c = self.client
        calls = [("get", "/api/sources/status"), ("post", "/api/sources/gmail/connect"),
                 ("post", "/api/sources/gmail/start"), ("post", "/api/sources/gmail/stop"),
                 ("delete", "/api/sources/gmail"), ("post", "/api/sources/folder/start"),
                 ("post", "/api/sources/folder/stop"), ("post", "/api/sources/notifications/start"),
                 ("post", "/api/sources/notifications/stop")]
        for m, u in calls:
            self.assertEqual(getattr(c, m)(u).status_code, 401, u)
            self.assertEqual(getattr(c, m)(u, headers={"X-API-Key": "wrong"}).status_code, 401, u)
        r = c.post("/api/sources/gmail/config", json={"client_id": "a", "client_secret": "b"})
        self.assertEqual(r.status_code, 401)
        r = c.post("/api/sources/eml/upload", files={"eml_file": ("a.eml", b"x")})
        self.assertEqual(r.status_code, 401)

    def test_status_shape(self):
        d = self.client.get("/api/sources/status", headers=self.H).json()
        self.assertEqual(set(d["sources"]), {"gmail_watcher", "folder_watcher", "notification_watcher"})
        self.assertFalse(d["gmail"]["connected"])

    # gmail config + oauth flow
    def test_config_validation(self):
        r = self.client.post("/api/sources/gmail/config", headers=self.H,
                             json={"client_id": "not-a-google-id", "client_secret": "x"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/sources/gmail/config", headers=self.H,
                             json={"client_id": " ", "client_secret": ""})
        self.assertEqual(r.status_code, 400)

    def test_connect_requires_client(self):
        self.assertEqual(self.client.post("/api/sources/gmail/connect", headers=self.H).status_code, 400)

    def test_cross_site_requests_are_refused_but_local_ones_work(self):
        evil = {**self.H, "Origin": "https://evil.example"}
        for m, u in (("get", "/api/sources/status"), ("post", "/api/sources/gmail/stop"),
                     ("post", "/api/sources/folder/stop"), ("delete", "/api/sources/gmail")):
            self.assertEqual(getattr(self.client, m)(u, headers=evil).status_code, 403, u)
        self.assertEqual(self.client.get("/api/sources/status",
                                         headers={**self.H, "Origin": "null"}).status_code, 403)
        for ok_origin in ("http://127.0.0.1:8001", "http://localhost:8000"):
            self.assertEqual(self.client.get("/api/sources/status",
                                             headers={**self.H, "Origin": ok_origin}).status_code, 200)
        # wrong key from a local origin is still 401, not 403
        self.assertEqual(self.client.get("/api/sources/status",
                                         headers={"X-API-Key": "bad", "Origin": "http://127.0.0.1:1"}).status_code, 401)

    def _full_signin(self):
        self.client.post("/api/sources/gmail/config", headers=self.H,
                         json={"client_id": "x.apps.googleusercontent.com", "client_secret": "SECRET-S"})
        url = self.client.post("/api/sources/gmail/connect", headers=self.H).json()["auth_url"]
        q = parse_qs(urlparse(url).query)
        ru = urlparse(q["redirect_uri"][0])
        self.assertEqual((ru.scheme, ru.hostname, ru.path), ("http", "127.0.0.1", "/"))   # Google loopback flow
        self.assertGreater(ru.port, 1024)
        self.redirect = q["redirect_uri"][0]
        return q["state"][0]

    def _cb(self, params, path=""):
        """Real HTTP GET to the one-shot loopback listener (what Google's redirect does)."""
        from urllib.parse import urlencode
        url = self.redirect.rstrip("/") + (path or "/") + ("?" + urlencode(params) if params else "")
        try:
            r = urlopen(UrlRequest(url), timeout=5)
            return r.status, r.read().decode(), dict(r.headers)
        except HTTPError as e:
            return e.code, e.read().decode(), dict(e.headers)

    def _listener_up(self):
        import socket
        port = urlparse(self.redirect).port
        s = socket.socket(); s.settimeout(1)
        try:
            s.connect(("127.0.0.1", port)); return True
        except OSError:
            return False
        finally:
            s.close()

    def test_callback_happy_path_stores_encrypted_token_and_starts_watcher(self):
        state = self._full_signin()
        orig_post = google_oauth._http_post
        revoked = []

        def fake_post(url, data):
            if url == google_oauth.TOKEN_URL:
                return types.SimpleNamespace(status_code=200, json=lambda: {
                    "refresh_token": "RT-1", "access_token": "AT-1", "expires_in": 3600})
            revoked.append(data)
            return types.SimpleNamespace(status_code=200, json=lambda: {})
        google_oauth._http_post = fake_post
        orig_prof = gmail_watcher.fetch_profile_email
        gmail_watcher.fetch_profile_email = lambda tok, http_get=None: "me@gmail.com"
        try:
            self.assertTrue(self._listener_up())
            status, body, headers = self._cb({"code": "C", "state": state})
            self.assertEqual(status, 200)
            self.assertIn("connected", body.lower())
            self.assertEqual(headers.get("Cache-Control"), "no-store")
            d = oauth_store.load_gmail()
            self.assertEqual((d["refresh_token"], d["email"]), ("RT-1", "me@gmail.com"))
            time.sleep(0.4)
            self.assertFalse(self._listener_up())                 # listener closed itself
            st = self.client.get("/api/sources/status", headers=self.H).json()
            self.assertTrue(st["gmail"]["connected"])
            self.assertIn(st["sources"]["gmail_watcher"]["state"], ("starting", "running"))
            self.assertTrue(sources_config.is_enabled("gmail"))
            # sign out: revoke + forget token, keep client
            r3 = self.client.delete("/api/sources/gmail", headers=self.H)
            self.assertTrue(r3.json()["revoked_at_google"])
            self.assertEqual(revoked[0]["token"], "RT-1")
            s = oauth_store.gmail_summary()
            self.assertTrue(s["client_configured"])
            self.assertFalse(s["connected"])
            self.assertFalse(sources_config.is_enabled("gmail"))
            self.client.delete("/api/sources/gmail", headers=self.H, params={"forget_client": "true"})
            self.assertFalse(oauth_store.gmail_summary()["client_configured"])
        finally:
            google_oauth._http_post = orig_post
            gmail_watcher.fetch_profile_email = orig_prof
            self.routes._stop_listener()

    def test_callback_rejects_bad_state_error_param_and_does_not_reflect_input(self):
        self._full_signin()
        try:
            status, body, _ = self._cb({"code": "c", "state": "<script>alert(1)</script>"})
            self.assertEqual(status, 400)
            self.assertNotIn("<script>", body)
            self.assertTrue(self._listener_up())                  # bad state does NOT close it (attacker can't DoS sign-in)
            self.assertFalse(oauth_store.gmail_summary()["connected"])
            self.assertEqual(self._cb({}, path="/")[0], 404)       # no code/error -> 404
            self.assertEqual(self._cb({"code": "c"}, path="/other")[0], 404)
            status, body, _ = self._cb({"error": "access_denied<b>x"})
            self.assertEqual(status, 400)
            self.assertNotIn("<b>x", body)
            time.sleep(0.4)
            self.assertFalse(self._listener_up())                 # user cancelled -> closed
        finally:
            self.routes._stop_listener()

    def test_state_is_single_use_across_the_listener(self):
        state = self._full_signin()
        orig_post = google_oauth._http_post
        google_oauth._http_post = lambda url, data: types.SimpleNamespace(
            status_code=400, json=lambda: {"error": "invalid_grant"})
        try:
            s1 = self._cb({"code": "C", "state": state})[0]
            s2 = self._cb({"code": "C", "state": state})[0]       # replay with the consumed state
            self.assertEqual((s1, s2), (400, 400))
            self.assertFalse(oauth_store.gmail_summary()["connected"])
        finally:
            google_oauth._http_post = orig_post
            self.routes._stop_listener()

    def test_new_signin_replaces_old_listener_and_listener_times_out(self):
        self._full_signin()
        first = self.redirect
        self._full_signin()
        self.assertNotEqual(first, self.redirect)
        time.sleep(0.6)
        import socket
        s = socket.socket(); s.settimeout(1)
        with self.assertRaises(OSError):
            s.connect(("127.0.0.1", urlparse(first).port))        # old one is gone
        s.close()
        old = self.routes._LISTEN_TIMEOUT_SECONDS
        self.routes._LISTEN_TIMEOUT_SECONDS = 0.5
        try:
            self._full_signin()
            time.sleep(1.0)
            self.assertFalse(self._listener_up())                 # timed out by itself
        finally:
            self.routes._LISTEN_TIMEOUT_SECONDS = old
            self.routes._stop_listener()

    def test_start_gmail_without_signin_is_400(self):
        r = self.client.post("/api/sources/gmail/start", headers=self.H)
        self.assertEqual(r.status_code, 400)

    def test_no_secret_ever_in_any_api_response(self):
        self._full_signin()
        oauth_store.save_gmail(refresh_token="RT-LEAK-CHECK", email="e@x.com")
        bodies = [self.client.get("/api/sources/status", headers=self.H).text,
                  self.client.post("/api/sources/gmail/config", headers=self.H,
                                   json={"client_id": "x.apps.googleusercontent.com",
                                         "client_secret": "SECRET-S"}).text]
        for b in bodies:
            self.assertNotIn("SECRET-S", b)
            self.assertNotIn("RT-LEAK-CHECK", b)

    # folder
    def test_folder_start_stop_and_validation(self):
        d = tempfile.mkdtemp(prefix="ns_drop_")
        try:
            self.assertEqual(self.client.post("/api/sources/folder/start", headers=self.H,
                                              json={"path": d + "/missing"}).status_code, 400)
            r = self.client.post("/api/sources/folder/start", headers=self.H, json={"path": d})
            self.assertEqual(r.status_code, 200)
            time.sleep(0.5)
            st = self.client.get("/api/sources/status", headers=self.H).json()
            self.assertIn(st["sources"]["folder_watcher"]["state"], ("starting", "running"))
            self.assertEqual(sources_config.folder_path(), d)
            self.assertEqual(self.client.post("/api/sources/folder/start", headers=self.H,
                                              json={"path": d}).status_code, 400)       # already running
            self.assertTrue(self.client.post("/api/sources/folder/stop", headers=self.H).json()["ok"])
            self.assertFalse(sources_config.is_enabled("folder"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    # notifications
    def test_notifications_start_fails_cleanly_off_windows(self):
        if sys.platform == "win32":
            self.skipTest("windows")
        r = self.client.post("/api/sources/notifications/start", headers=self.H)
        self.assertEqual(r.status_code, 200)             # accepted, then reports error as status
        time.sleep(0.5)
        st = self.client.get("/api/sources/status", headers=self.H).json()["sources"]["notification_watcher"]
        self.assertEqual(st["state"], "error")
        self.assertIn("Windows", st["detail"])

    # eml upload
    def test_eml_upload(self):
        c, H = self.client, self.H
        ok = c.post("/api/sources/eml/upload", headers=H,
                    files={"eml_file": ("p.eml", mk_eml("x", plain=PHISH).as_bytes())})
        self.assertEqual(ok.status_code, 200)
        self.assertTrue(ok.json()["phishing"])
        self.assertEqual(ok.json()["final_decision"], "HIGH_RISK")
        safe = c.post("/api/sources/eml/upload", headers=H,
                      files={"eml_file": ("s.eml", mk_eml("x", plain=SAFE).as_bytes())})
        self.assertFalse(safe.json()["phishing"])
        self.assertEqual(c.post("/api/sources/eml/upload", headers=H,
                                files={"eml_file": ("x.exe", b"abc")}).status_code, 400)
        self.assertEqual(c.post("/api/sources/eml/upload", headers=H,
                                files={"eml_file": ("x.eml", b"   ")}).status_code, 400)
        big = b"A" * (folder_watcher.MAX_BYTES + 10)
        self.assertEqual(c.post("/api/sources/eml/upload", headers=H,
                                files={"eml_file": ("big.eml", big)}).status_code, 400)
        self.assertTrue(all(a.get("intake") == "eml_upload" for a in SAVED))

    # manager integration with the ORIGINAL watcher_manager
    def test_watcher_manager_status_is_superset_of_original(self):
        st = watcher_manager.get_watcher_status()
        self.assertIn("mail_watcher", st)
        self.assertIn("call_monitor", st)
        for k in ("gmail_watcher", "folder_watcher", "notification_watcher"):
            self.assertIn(k, st)
        self.assertEqual(st["mail_watcher"]["state"], "disabled")

    def test_autostart_only_starts_what_user_enabled(self):
        source_manager.autostart()
        self.assertTrue(all(v["state"] == "disabled" for v in source_manager.get_status().values()))
        oauth_store.save_gmail(client_id="c", client_secret="s", refresh_token="r", email="e@x.com")
        sources_config.set_enabled("gmail", True)
        source_manager.autostart()
        time.sleep(0.3)
        self.assertIn(source_manager.get_status()["gmail_watcher"]["state"], ("starting", "running"))

    def test_start_all_watchers_never_raises_and_does_not_autostart_original_mail(self):
        watcher_manager.start_all_watchers()
        st = watcher_manager.get_watcher_status()
        self.assertEqual(st["mail_watcher"]["state"], "disabled")
        self.assertEqual(st["call_monitor"]["state"], "disabled")


class TestIntake(unittest.TestCase):
    def setUp(self):
        wipe()

    def test_never_raises_on_model_failure_and_skips_empty(self):
        self.assertFalse(email_intake.scan_email("", "", "t")["scanned"])
        real = _stub.predict_text
        _stub.predict_text = lambda t: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            stats = {}
            r = email_intake.scan_email("s", "body text here", "t", stats=stats)
            self.assertFalse(r["scanned"])
            self.assertEqual(stats["errors"], 1)
        finally:
            _stub.predict_text = real

    def test_no_sender_or_subject_in_console(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            email_intake.scan_email("SECRET-SUBJECT-LINE", PHISH, "gmail_api")
        self.assertNotIn("SECRET-SUBJECT-LINE", buf.getvalue())
        self.assertNotIn("URGENT", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)

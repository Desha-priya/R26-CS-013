# session_monitor.py
# Windows session activity monitor for NeuraShield
# Runs as background thread — detects foreground app and maps to context mode
# Answers panel question: "how do you know what the user is doing?"
#
# Uses: psutil (process info) + pywin32 (Windows active window API)
# Install: pip install psutil pywin32

import re
import time
import threading
import logging
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Context mapping rules ─────────────────────────────────
# Maps detected app/window title keywords to context modes
# Order matters — first match wins

CONTEXT_RULES = [
    # Financial context — highest sensitivity
    {
        'context':  'financial',
        'keywords': [
            'banking', 'bank', 'payment', 'paypal', 'stripe',
            'transaction', 'transfer', 'checkout', 'invoice',
            'quickbooks', 'xero', 'sage', 'wallet', 'crypto',
            'bitcoin', 'trading', 'brokerage', 'natwest', 'barclays',
            'hsbc', 'lloyds', 'monzo', 'revolut', 'wise'
        ],
        'processes': ['quickbooks', 'xero', 'sage50']
    },

    # Sensitive access — high sensitivity
    {
        'context':  'sensitive_access',
        'keywords': [
            'admin', 'administrator', 'control panel', 'regedit',
            'registry', 'task manager', 'system32', 'group policy',
            'active directory', 'server manager', 'confidential',
            'sensitive', 'private', 'classified', 'restricted',
            'vpn', 'ssh', 'putty', 'remote desktop', 'rdp',
            'password', 'credentials', 'certificate', 'keystore',
            'database', 'sql', 'pgadmin', 'mysql', 'mongodb',
            'cmd', 'powershell', 'terminal', 'command prompt'
        ],
        'processes': [
            'regedit.exe', 'taskmgr.exe', 'mmc.exe',
            'ssms.exe', 'pgadmin4.exe', 'putty.exe',
            'mstsc.exe', 'cmd.exe', 'powershell.exe',
            'WindowsTerminal.exe'
        ]
    },

    # Normal browsing — relaxed
    {
        'context':  'normal_browsing',
        'keywords': [
            'google', 'youtube', 'chrome', 'firefox', 'edge',
            'safari', 'browser', 'news', 'wikipedia',
            'stackoverflow', 'github', 'reddit', 'twitter',
            'facebook', 'instagram', 'linkedin', 'outlook',
            'word', 'excel', 'powerpoint', 'notepad', 'vscode',
            'visual studio', 'pycharm', 'intellij'
        ],
        'processes': [
            'chrome.exe', 'firefox.exe', 'msedge.exe',
            'WINWORD.EXE', 'EXCEL.EXE', 'POWERPNT.EXE',
            'notepad.exe', 'code.exe', 'pycharm64.exe'
        ]
    },
]

DEFAULT_CONTEXT = 'normal_browsing'

# Browsers whose foreground window can carry a context supplied by the
# browser extension (see SessionMonitor._url_context_for).
BROWSER_PROCESSES = {
    'chrome.exe', 'msedge.exe', 'firefox.exe',
    'brave.exe', 'opera.exe', 'vivaldi.exe',
}

# If the extension has not re-confirmed a URL context, keep using it for
# this long while the same browser tab stays in front.
URL_CONTEXT_TTL_SECONDS = 30

_KW_CACHE = {}


def _kw_match(text: str, keyword: str) -> bool:
    """
    Whole-word match (an optional plural 's'/'es' is allowed).

    The original code used plain substring tests, so 'sage' matched
    'message' and 'usage', 'wise' matched 'likewise', and so on, which
    pushed ordinary windows into the financial context. A keyword now
    has to start and end on a word boundary.
    """
    pattern = _KW_CACHE.get(keyword)
    if pattern is None:
        pattern = re.compile(
            r'(?<![a-z0-9])' + re.escape(keyword.lower()) + r'(?:es|s)?(?![a-z0-9])'
        )
        _KW_CACHE[keyword] = pattern
    return pattern.search(text) is not None


def _split_url(url: str):
    """Return (host_with_port, path, tokens_of_the_first_path_segment)."""
    u = url.strip().lower()
    if '://' not in u:
        u = 'http://' + u
    parsed = urlparse(u)
    host = parsed.netloc.split('@')[-1]
    first_segment = parsed.path.strip('/').split('/')[0]
    tokens = [t for t in re.split(r'[^a-z0-9]+', first_segment) if t]
    return host, parsed.path, tokens


def _url_matches(url_parts, item: str) -> bool:
    """
    Match one entry of a domain list against the host and the start of
    the path only, not the whole URL string. Previously 'payment' matched
    anywhere in a URL, including question titles on a programming site.
    """
    host, path, early_tokens = url_parts
    if '/' in item:                       # e.g. 'google.com/pay'
        return item in (host + path)
    if ':' in item:                       # e.g. 'localhost:8000'
        return item in host
    host_name = host.split(':')[0]
    if len(item) <= 4:                    # short brand names: whole label
        return item in re.split(r'[.\-]', host_name)
    return item in host_name or item in early_tokens


class SessionMonitor:
    """
    Background thread that monitors the active Windows window
    and automatically determines the session context mode.

    How it works:
    1. Every 3 seconds, calls Windows API to get foreground window title
    2. Checks window title + process name against CONTEXT_RULES
    3. Updates current_context which risk engine reads before scoring
    4. If NeuraShield receives a cross-layer alert, context switches to
       'under_attack' and overrides all other detection until cleared
    """

    def __init__(self):
        self.current_context    = DEFAULT_CONTEXT
        self.current_window     = "Unknown"
        self.current_process    = "Unknown"
        self.last_updated       = datetime.now()
        self.running            = False
        self.alert_override     = False   # True when under_attack alert active
        self._thread            = None
        self._lock              = threading.Lock()
        self.recent_sensitive = []          # list of recent sensitive detections
        self.SENSITIVE_MEMORY_SECONDS = 20 # remember for 20 seconds
        # Context last supplied by the browser extension (see update_from_url)
        self._url_context       = None
        self._url_title         = ""
        self._url_time          = None

    def start(self):
        """Start background monitoring thread."""
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop,
                                        daemon=True, name="SessionMonitor")
        self._thread.start()
        logger.info("[SessionMonitor] Started — checking active window every 3s")

    def stop(self):
        """Stop background monitoring thread."""
        self.running = False
        logger.info("[SessionMonitor] Stopped")

    def set_alert_override(self, active: bool):
        """
        Called when cross-layer alert received.
        Overrides context to 'under_attack' while active.
        """
        with self._lock:
            self.alert_override = active
            if active:
                self.current_context = 'under_attack'
                logger.info("[SessionMonitor] Alert override — context: under_attack")

    def get_status(self) -> dict:
        """Return current monitoring status for API."""
        with self._lock:
            return {
                'current_context':  self.current_context,
                'current_window':   self.current_window,
                'current_process':  self.current_process,
                'last_updated':     self.last_updated.isoformat(),
                'alert_override':   self.alert_override,
                'monitoring_active': self.running,
                'background_sensitive': self.get_background_sensitive(),
            }

    def _monitor_loop(self):
        """Main monitoring loop — runs every 3 seconds."""
        while self.running:
            try:
                window_title, process_name = self._get_active_window()
                context = self._determine_context(window_title, process_name)

                with self._lock:
                    self.current_window  = window_title
                    self.current_process = process_name
                    self.last_updated    = datetime.now()
                    # A context supplied by the browser extension stays in
                    # force while the same browser tab is in front. Before,
                    # this loop overwrote it with a window-title guess every
                    # 3 seconds.
                    url_context = self._url_context_for(window_title, process_name)
                    if url_context is not None:
                        context = url_context
                    if self.alert_override:
                        # Defensive: the override must survive until it is
                        # cleared explicitly.
                        self.current_context = 'under_attack'
                    else:
                        if context != self.current_context:
                            logger.info(
                                f"[SessionMonitor] Context: {self.current_context} "
                                f"→ {context}"
                            )
                        self.current_context = context

            except Exception as e:
                logger.debug(f"[SessionMonitor] Error: {e}")

            time.sleep(3)

    def _get_active_window(self) -> tuple:
        """
        Get the title and process name of the currently active window.
        Uses pywin32 on Windows.
        Falls back to 'Unknown' if pywin32 not available.
        """
        try:
            import win32gui
            import win32process
            import psutil

            hwnd         = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(hwnd).lower()

            # Get process name from window handle
            _, pid       = win32process.GetWindowThreadProcessId(hwnd)
            try:
                proc     = psutil.Process(pid)
                process_name = proc.name().lower()
            except Exception:
                process_name = "unknown"

            return window_title, process_name

        except ImportError:
            # pywin32 not installed — return placeholder
            return "pywin32 not installed", "unknown"
        except Exception:
            return "unknown", "unknown"

    def _determine_context(self, window_title: str, process_name: str) -> str:
        """
        Match window title and process name against context rules.
        Returns the appropriate context mode string.
        """
        title   = window_title.lower()
        process = process_name.lower()

        for rule in CONTEXT_RULES:
            # Check keyword match in window title (whole words only)
            if any(_kw_match(title, kw) for kw in rule['keywords']):
                return rule['context']
            # Check process name match
            if any(p.lower() in process for p in rule.get('processes', [])):
                return rule['context']

        return DEFAULT_CONTEXT

    def update_from_url(self, url: str, title: str = ""):
        """
        Called by browser extension.
        Uses full URL for better context detection.
        """
        url_lower = url.lower()
        title_lower = title.lower()
        url_parts = _split_url(url_lower)

        # High priority domain-based detection
        financial_domains = [
            "hdfcbank", "sbi", "onlinesbi", "icici", "axisbank", "kotak", "paypal", "stripe",
            "razorpay", "paytm", "phonepe", "google.com/pay", "wise.com", "revolut",
            "netbanking", "onlinebanking", "banking", "payment", "checkout"
        ]

        sensitive_domains = [
            "admin", "regedit", "localhost:8000", "dashboard", "portal", "console"
        ]

        new_context = "normal_browsing"

        if any(_url_matches(url_parts, d) for d in financial_domains):
            new_context = "financial"
        elif any(_url_matches(url_parts, d) for d in sensitive_domains):
            new_context = "sensitive_access"
        else:
            # Fallback to the window-title keyword rules
            new_context = self._determine_context(title_lower, "")

        if new_context in ['financial', 'sensitive_access']:
            self.add_sensitive_detection(
                source=new_context,
                detail=f"{title} | {url[:70]}"
            )

        with self._lock:
            # Remember what the extension said, so the 3-second window poll
            # does not overwrite it while the same tab is in front.
            self._url_context = new_context
            self._url_title = title_lower.strip()
            self._url_time = datetime.now()

            if self.alert_override:
                # An active cross-layer alert wins over the extension.
                # Before this check, a page report arriving during an alert
                # replaced 'under_attack' with the page's own context.
                self.current_window = title
                self.last_updated = datetime.now()
                return

            if new_context != self.current_context:
                print(f"[SessionMonitor] Context changed via Extension → {new_context}")
            self.current_context = new_context
            self.current_window = title
            self.last_updated = datetime.now()

    def _url_context_for(self, window_title: str, process_name: str):
        """
        Context the browser extension reported, if it still applies to the
        window that is in front now; otherwise None. Called with the lock
        held.
        """
        if not self._url_context or self._url_time is None:
            return None
        if process_name not in BROWSER_PROCESSES:
            return None
        age = (datetime.now() - self._url_time).total_seconds()
        same_tab = bool(self._url_title) and self._url_title in window_title
        if same_tab or age < URL_CONTEXT_TTL_SECONDS:
            return self._url_context
        return None

    def add_sensitive_detection(self, source: str, detail: str):
        """Record a sensitive detection. Short memory only."""
        now = datetime.now()
        # Remove old entries first
        self.recent_sensitive = [
            item for item in self.recent_sensitive
            if (now - item['time']).total_seconds() < self.SENSITIVE_MEMORY_SECONDS
        ]
        # Add new one
        self.recent_sensitive.append({
            'source': source,
            'detail': detail,
            'time': now
        })

    def get_background_sensitive(self) -> list:
        """Return only recent (still likely open) sensitive activities."""
        now = datetime.now()
        return [
            {
                'source': item['source'],
                'detail': item['detail'],
                'seconds_ago': int((now - item['time']).total_seconds())
            }
            for item in self.recent_sensitive
            if (now - item['time']).total_seconds() < self.SENSITIVE_MEMORY_SECONDS
        ]

                


# ── Singleton instance — shared across the app ────────────
monitor = SessionMonitor()


# ── Standalone test ───────────────────────────────────────
if __name__ == "__main__":
    print("Session Monitor — standalone test")
    print("Monitoring active window for 30 seconds...\n")

    logging.basicConfig(level=logging.INFO)
    monitor.start()

    try:
        for i in range(10):
            time.sleep(3)
            status = monitor.get_status()
            print(f"[{i+1:02d}] Context: {status['current_context']:<20} "
                  f"Window: {status['current_window'][:40]:<40} "
                  f"Process: {status['current_process']}")
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        print("\nMonitor stopped.")
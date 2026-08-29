# session_monitor.py
# Windows session activity monitor for NeuraShield
# Runs as background thread — detects foreground app and maps to context mode
# Answers panel question: "how do you know what the user is doing?"
#
# Uses: psutil (process info) + pywin32 (Windows active window API)
# Install: pip install psutil pywin32

import time
import threading
import logging
from datetime import datetime

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
                    # Only update context if no alert override
                    if not self.alert_override:
                        if context != self.current_context:
                            logger.info(
                                f"[SessionMonitor] Context: {self.current_context} "
                                f"→ {context} | Window: {window_title[:50]}"
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
            # Check keyword match in window title
            if any(kw in title for kw in rule['keywords']):
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

        # High priority domain-based detection
        financial_domains = [
            "hdfcbank", "sbi", "icici", "axisbank", "kotak", "paypal", "stripe",
            "razorpay", "paytm", "phonepe", "google.com/pay", "wise.com", "revolut",
            "netbanking", "onlinebanking", "banking", "payment", "checkout"
        ]

        sensitive_domains = [
            "admin", "regedit", "localhost:8000", "dashboard", "portal", "console"
        ]

        new_context = "normal_browsing"

        if any(domain in url_lower for domain in financial_domains):
            new_context = "financial"
        elif any(domain in url_lower for domain in sensitive_domains):
            new_context = "sensitive_access"
        else:
            # Fallback to old keyword method
            new_context = self._determine_context(title_lower, "")

        with self._lock:
            if new_context != self.current_context:
                print(f"[SessionMonitor] Context changed via Extension → {new_context} | URL: {url[:60]}")
            self.current_context = new_context
            self.current_window = title
            self.last_updated = datetime.now()


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

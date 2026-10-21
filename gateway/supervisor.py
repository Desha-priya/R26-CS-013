"""
supervisor.py  —  NeuraShield process supervisor
-------------------------------------------------
Starts, stops, restarts and health-checks each enabled layer. It is the only
place that knows the shared secrets: it resolves the API key and HMAC secret
ONCE (secrets_store) and injects them, plus per-layer ports and data dirs,
into each layer's environment. No layer hard-codes a secret; every layer
receives the same values here.

It does NOT modify any layer's code. It only launches each layer's own entry
point (from config/layers.json) in that layer's own folder, with its own
Python runtime if one is provisioned under runtimes/<id>/.

Process hygiene (rule: stopping the entry point leaves nothing behind):
  * every layer runs in its own process group AND, on Windows, inside one job
    object with KILL_ON_JOB_CLOSE, so if the console dies for ANY reason
    (crash, taskkill, window closed) the OS kills every layer and every child;
  * stop() kills the whole tree; shutdown() additionally frees our ports;
  * each layer's stdout/stderr go to data/logs/<id>.log (UTF-8) instead of
    vanishing, so a layer that fails to start can actually be diagnosed;
  * the watchdog restarts a crashed layer with backoff and gives up after
    repeated immediate failures instead of respawning forever.

Public API (used by the console):
    sup = Supervisor(install_root)
    sup.start_enabled()          # start core + every enabled optional layer
    sup.start(layer_id)          # start one layer
    sup.stop(layer_id)           # stop one layer
    sup.restart(layer_id)
    sup.status()                 # {id: {running, pid, healthy, ...}}
    sup.shutdown()               # stop everything

Run standalone:  python -m gateway.supervisor  (foreground, Ctrl+C to stop)
"""

import json
import os
import sys
import time
import socket
import signal
import threading
import subprocess
from pathlib import Path
from urllib import request as _request, error as _urlerror

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import secrets_store  # noqa: E402

_FROZEN = bool(getattr(sys, "frozen", False))

# A layer that dies sooner than this after launch counts as a failed start.
_QUICK_FAIL_SECONDS = 30
_MAX_QUICK_FAILS = 6
_LOG_ROTATE_BYTES = 5 * 1024 * 1024


def _port_busy(port: int, host: str = "127.0.0.1") -> bool:
    """True if something is already LISTENING on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, int(port))) == 0


def _listener_pids(port: int) -> set:
    pids = set()
    try:
        if sys.platform == "win32":
            out = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                                 capture_output=True, text=True).stdout
            for line in out.splitlines():
                parts = line.split()
                # Proto Local Foreign State PID
                if (len(parts) >= 5 and parts[3] == "LISTENING"
                        and parts[1].rsplit(":", 1)[-1] == str(port)):
                    if parts[-1].isdigit() and parts[-1] != "0":
                        pids.add(int(parts[-1]))
        else:
            out = subprocess.run(["bash", "-c", f"lsof -ti tcp:{port} -sTCP:LISTEN"],
                                 capture_output=True, text=True).stdout
            pids = {int(x) for x in out.split() if x.isdigit()}
    except Exception:
        pass
    return pids


def _kill_listener_on_port(port: int):
    """Last-resort: terminate whatever process is still listening on a port a
    layer of ours was started on. Used during shutdown so no layer survives the
    platform stopping. Never kills this process."""
    me = os.getpid()
    for pid in _listener_pids(port):
        if pid == me:
            continue
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True)
            else:
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def _restrict_to_owner(path: Path):
    """Best-effort: make a secret file readable only by the current user.
    os.chmod does nothing useful on Windows, so use icacls there."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    if sys.platform == "win32":
        try:
            user = os.environ.get("USERNAME")
            if user:
                subprocess.run(
                    ["icacls", str(path), "/inheritance:r", "/grant:r",
                     f"{user}:(R,W)", "SYSTEM:(F)", "*S-1-5-32-544:(F)"],
                    capture_output=True, timeout=10,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            pass


# ---------------------------------------------------------------- job object
_JOB = None
_JOB_LOCK = threading.Lock()


def _job_handle():
    """One Windows job object, created lazily, that kills all its processes
    when the last handle to it closes (i.e. when this process exits)."""
    global _JOB
    if sys.platform != "win32":
        return None
    with _JOB_LOCK:
        if _JOB is not None:
            return _JOB or None
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            k32.SetInformationJobObject.restype = wintypes.BOOL

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class BASIC(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                            ("PerJobUserTimeLimit", ctypes.c_longlong),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class EXTENDED(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", BASIC),
                            ("IoInfo", IO_COUNTERS),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            job = k32.CreateJobObjectW(None, None)
            if not job:
                _JOB = 0
                return None
            info = EXTENDED()
            info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(job, 9, ctypes.byref(info),
                                               ctypes.sizeof(info)):
                _JOB = 0
                return None
            _JOB = job
            return _JOB
        except Exception:
            _JOB = 0
            return None


def _assign_to_job(proc) -> bool:
    job = _job_handle()
    if not job:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        return bool(k32.AssignProcessToJobObject(job, int(proc._handle)))
    except Exception:
        return False


class LayerProcess:
    def __init__(self, spec: dict, root: Path):
        self.spec = spec
        self.root = root
        self.id = spec["id"]
        self.name = spec.get("name", self.id)
        self.proc = None
        self.started_at = None
        self.last_exit = None
        self.restarts = 0
        self.enabled = False
        # watchdog bookkeeping
        self.fail_streak = 0
        self.next_retry = None
        self.gave_up = False
        self.last_error = ""
        self._log_fh = None

    # ---- launch environment -------------------------------------------------
    def _python(self) -> str:
        """The product runs every layer on ONE environment (runtimes/all, from
        requirements-all.txt). An older per-layer runtimes/<id> is still honoured."""
        sub = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        for name in ("all", self.id):
            cand = self.root / "runtimes" / name / sub
            if cand.exists():
                return str(cand)
        if not _FROZEN:
            print(f"[supervisor] note: no runtimes/all environment "
                  f"(see requirements-all.txt); using {sys.executable}")
        return sys.executable

    def _build_env(self, api_key: str, hmac_secret: str, platform_url: str) -> dict:
        env = os.environ.copy()
        # The console may itself run inside a venv; never leak its interpreter
        # configuration into a layer that has its own runtime.
        for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "__PYVENV_LAUNCHER__"):
            env.pop(k, None)
        # UTF-8 everywhere: layers print non-ASCII (box drawing, emoji) and their
        # output is redirected to a file, where the default would be cp1252 and
        # raise UnicodeEncodeError (killing the layer at its first banner).
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["NEURASHIELD_ROOT"] = str(self.root)
        # Tells every layer it is part of the platform: it must use THIS key and
        # must never fall back to generating a key of its own.
        env["NEURASHIELD_SUPERVISED"] = "1"
        console = self.__dict__.get("console_url") or "http://127.0.0.1:9000"
        env["NEURASHIELD_CONSOLE_URL"] = console
        env["NEURASHIELD_API_KEY"] = api_key
        env["NEURASHIELD_HMAC_SECRET"] = hmac_secret
        env["NEURASHIELD_PLATFORM_URL"] = platform_url
        # Per-layer env from the contract, with relative paths made absolute.
        for k, v in (self.spec.get("env") or {}).items():
            if v and ("DIR" in k or "HOME" in k) and not os.path.isabs(v):
                p = self.root / v
                p.mkdir(parents=True, exist_ok=True)
                env[k] = str(p)
            else:
                env[k] = v
        # A packaged layer carries its read-only files (templates, models) inside
        # its own bundle; pointing it at the source-tree folder would hide them.
        if self._frozen_exe() is not None:
            env.pop("NEURASHIELD_RESOURCE_DIR", None)
        return env

    def _frozen_exe(self):
        """Packaged install: every layer is THIS executable started in layer mode
        (NeuraShield.exe --run-layer <id>), so there is nothing else to locate."""
        return Path(sys.executable) if _FROZEN else None

    def _cwd(self) -> Path:
        if _FROZEN:
            d = self.root / "data" / (self.spec.get("data_subdir") or self.id)
            d.mkdir(parents=True, exist_ok=True)
            return d
        return self.root / self.spec["dir"]

    def _preflight(self):
        """Source runs only: make sure the interpreter we are about to use really
        has this layer's packages, instead of letting the layer die with a
        traceback in a log file nobody is reading."""
        if _FROZEN:
            return
        mods = self.spec.get("requires") or []
        if not mods:
            return
        py = self._python()
        code = ("import importlib.util as u;"
                f"print(','.join(m for m in {mods!r} if u.find_spec(m) is None))")
        try:
            r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=60,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            missing = [m for m in r.stdout.strip().split(",") if m]
        except Exception:
            return          # cannot probe: let the real start report any problem
        if missing:
            msg = (f"{self.name} cannot start: Python packages missing ({', '.join(missing)}) in {py}. "
                   "Create the product environment once with:  python scripts/bootstrap_envs.py")
            self.last_error = msg
            raise RuntimeError(msg)

    def _command(self) -> list:
        if _FROZEN:
            return [sys.executable, "--run-layer", self.id]
        return [self._python()] + list(self.spec["entry"])

    def _data_dir(self, env: dict) -> Path:
        return Path(env.get("NEURASHIELD_DATA_DIR")
                    or env.get("NEURASHIELD_HOME")
                    or self._cwd())

    def _reconcile_key_files(self, env: dict, api_key: str):
        """Write the ONE shared key into every location this layer might read a
        key from, so env, key files and the platform can never disagree. This is
        what guarantees inter-layer alerts authenticate (no more 401). Env still
        takes precedence in every layer, but the files are made consistent too so
        a layer started standalone later also uses the same key."""
        candidates = []
        layer_dir = self._cwd()
        # One shared key file for every layer (fallback if env is not inherited).
        shared = Path(env.get("NEURASHIELD_SECRET_DIR")
                      or (Path(env.get("NEURASHIELD_ROOT", ".")) / "data" / "secrets")
                      ) / "shared_api_key"
        env["NEURASHIELD_KEY_FILE"] = str(shared)
        candidates.append(shared)
        data_dir = self._data_dir(env)
        if self.id == "core":
            # ns_paths: api_key.txt / api_key.dat in the data dir. The platform
            # prefers env, but we also remove any STALE DPAPI api_key.dat and
            # write a matching api_key.txt so there is exactly one key on disk.
            for stale in (data_dir / "api_key.dat", layer_dir / "api_key.dat"):
                try:
                    if stale.exists():
                        stale.unlink()
                except OSError:
                    pass
            candidates += [data_dir / "api_key.txt", layer_dir / "api_key.txt"]
        elif self.id == "content":
            kd = Path(env.get("NEURASHIELD_KEY_DIR") or (data_dir / ".neurashield"))
            candidates += [kd / "api_key.local"]
            candidates += [layer_dir / ".neurashield" / "api_key.local"]
            candidates += [layer_dir / "demo" / ".neurashield" / "api_key.local"]
        # network + ransomware take the key only from env (no key file), so
        # nothing to reconcile for them.
        for path in candidates:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(api_key, encoding="utf-8")
                _restrict_to_owner(path)
            except Exception:
                pass

    # ---- logging ------------------------------------------------------------
    def log_path(self) -> Path:
        return self.root / "data" / "logs" / f"{self.id}.log"

    def _open_log(self):
        path = self.log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.exists() and path.stat().st_size > _LOG_ROTATE_BYTES:
                old = path.with_suffix(".log.1")
                if old.exists():
                    old.unlink()
                path.rename(old)
        except OSError:
            pass
        fh = open(path, "ab", buffering=0)
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} starting "
                 f"{self.id} =====\n".encode("utf-8"))
        return fh

    def _close_log(self):
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None

    # ---- lifecycle ----------------------------------------------------------
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, api_key, hmac_secret, platform_url):
        if self.is_running():
            return
        cwd = self._cwd()
        if not cwd.exists():
            raise FileNotFoundError(f"[{self.id}] layer folder missing: {cwd}")
        port = self.spec.get("port")
        if port and _port_busy(port, self.spec.get("host", "127.0.0.1")):
            msg = (f"port {port} is already in use by another process — "
                   f"cannot start {self.name}")
            self.last_error = msg
            raise RuntimeError(f"[{self.id}] {msg}")
        self._preflight()
        env = self._build_env(api_key, hmac_secret, platform_url)
        # Make env + any key files agree on ONE key before the layer starts, so
        # inter-layer alerts always authenticate (fixes 401 Unauthorized).
        self._reconcile_key_files(env, api_key)
        command = self._command()
        self._close_log()
        self._log_fh = self._open_log()
        popen_kw = dict(cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                        stdout=self._log_fh, stderr=subprocess.STDOUT)
        # Launch each layer in its OWN process group / session so that when we
        # stop it we can kill the layer AND every child it spawned (the network
        # layer starts monitor-2.py, etc.). Otherwise those children are
        # orphaned and keep running after the platform stops.
        if sys.platform == "win32":
            popen_kw["creationflags"] = (subprocess.CREATE_NO_WINDOW
                                         | subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            popen_kw["start_new_session"] = True   # setsid: new process group
        try:
            self.proc = subprocess.Popen(command, **popen_kw)
        except Exception as e:
            self.last_error = str(e)
            self._close_log()
            raise
        self.last_error = ""
        self.started_at = time.time()
        self.next_retry = None
        if sys.platform == "win32":
            # Tie the layer's lifetime to ours (children inherit membership).
            _assign_to_job(self.proc)
        print(f"[supervisor] started {self.id} (pid {self.proc.pid}) "
              f"port {self.spec.get('port')} · log {self.log_path()}")

    def stop(self, timeout=8):
        if not self.is_running():
            self._close_log()
            return
        p = self.proc
        pid = p.pid
        try:
            if sys.platform == "win32":
                # Kill the whole process tree (/T) forcefully (/F): parent +
                # every child it spawned.
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True)
            else:
                try:
                    pgid = os.getpgid(pid)
                except Exception:
                    pgid = pid
                try:
                    os.killpg(pgid, signal.SIGTERM)   # ask the whole group to exit
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pgid, signal.SIGKILL)  # force the group
                    except ProcessLookupError:
                        pass
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
        finally:
            self.last_exit = p.poll()
            self._close_log()
            print(f"[supervisor] stopped {self.id} (pid {pid})")

    # ---- health -------------------------------------------------------------
    def health(self, api_key: str) -> bool:
        url = self.spec.get("health_url")
        if not url:
            return self.is_running()
        headers = {}
        if self.spec.get("health_needs_key"):
            headers["X-API-Key"] = api_key
        try:
            req = _request.Request(url, headers=headers, method="GET")
            with _request.urlopen(req, timeout=3) as resp:
                return 200 <= resp.status < 500  # 401 still means "alive"
        except _urlerror.HTTPError as e:
            return e.code < 500
        except Exception:
            return False


class Supervisor:
    def __init__(self, install_root: str = None):
        self.root = Path(install_root or os.environ.get(
            "NEURASHIELD_ROOT", Path(__file__).resolve().parent.parent))
        self.layers_spec = json.loads(
            (self.root / "config" / "layers.json").read_text(encoding="utf-8"))
        self.product = json.loads(
            (self.root / "config" / "product.json").read_text(encoding="utf-8"))
        self.api_key = secrets_store.get_api_key()
        self.hmac_secret = secrets_store.get_hmac_secret()
        self.platform_url = self.product.get("platform_url", "http://127.0.0.1:8000")
        self.procs = {
            lid: LayerProcess(spec, self.root)
            for lid, spec in self.layers_spec.items()
            if not lid.startswith("_")
        }
        self._monitor_thread = None
        self._stop_monitor = threading.Event()
        self._lock = threading.RLock()

    # -- consent --------------------------------------------------------------
    def consent_accepted(self) -> bool:
        return bool(self.product.get("consent", {}).get("accepted"))

    def reload_product(self):
        """Re-read product.json (consent / enabled layers may have changed)."""
        self.product = json.loads(
            (self.root / "config" / "product.json").read_text(encoding="utf-8"))

    # -- which layers are on --------------------------------------------------
    def _is_enabled(self, lid: str) -> bool:
        spec = self.layers_spec.get(lid, {})
        # No layer runs until the user has consented — NOT EVEN the platform
        # core. Before consent only the console web server is up.
        if not self.consent_accepted():
            return False
        # The platform core (hub) always runs once consent is on record; it is
        # the plumbing, not a monitoring layer.
        if spec.get("always_on"):
            return True
        return bool(self.product.get("enabled_layers", {}).get(lid, False))

    # -- control --------------------------------------------------------------
    def start(self, lid: str):
        with self._lock:
            lp = self.procs[lid]
            lp.enabled = True
            lp.gave_up = False
            lp.fail_streak = 0
            lp.next_retry = None
            lp.start(self.api_key, self.hmac_secret, self.platform_url)

    def stop(self, lid: str):
        with self._lock:
            self.procs[lid].enabled = False
            self.procs[lid].stop()

    def restart(self, lid: str):
        self.stop(lid)
        time.sleep(1.0)
        self.start(lid)

    def start_enabled(self):
        # NOTHING starts until consent is on record — not even the platform
        # core. Before consent the console web server is the only thing running,
        # so the machine is untouched until the user agrees.
        if not self.consent_accepted():
            print("[supervisor] consent not given — no services started")
            return
        # Core first (owns key + hub), then the optional layers.
        order = ["core"] + [l for l in self.procs if l != "core"]
        for lid in order:
            if lid in self.procs and self._is_enabled(lid):
                try:
                    self.start(lid)
                    if lid == "core":
                        self.wait_healthy("core", timeout=45)  # hub up before layers call it
                except Exception as e:
                    print(f"[supervisor] could not start {lid}: {e}")

    def wait_healthy(self, lid: str, timeout: float = 45.0) -> bool:
        """Block until the layer answers its health URL (or it exits / times out)."""
        lp = self.procs.get(lid)
        end = time.time() + timeout
        while time.time() < end:
            if lp is None or not lp.is_running():
                return False
            if lp.health(self.api_key):
                return True
            time.sleep(0.5)
        return False

    # Authenticated GET used to prove a layer enforces THE shared key. Ransomware
    # only sends alerts (its dashboard has no key check), so it cannot be probed.
    _KEY_PROBES = {
        "core": "http://127.0.0.1:{port}/api/platform/status",
        "network": "http://127.0.0.1:{port}/api/alerts",
        "content": "http://127.0.0.1:{port}/api/sources/status",
    }

    def verify_keys(self) -> dict:
        """Prove every running key-enforcing layer uses the one platform key:
        it must answer 200 to the shared key and 401 to a wrong one.
        Returns {layer_id: {"ok": True|False|None, "detail": str}}."""
        out = {}
        wrong = "wrong-" + self.api_key[::-1]

        def code(url, key):
            try:
                req = _request.Request(url, headers={"X-API-Key": key}, method="GET")
                with _request.urlopen(req, timeout=5) as r:
                    return r.status
            except _urlerror.HTTPError as e:
                return e.code
            except Exception:
                return None

        for lid, lp in self.procs.items():
            if not lp.is_running():
                continue
            tmpl = self._KEY_PROBES.get(lid)
            if not tmpl:
                out[lid] = {"ok": None, "detail": "sends alerts only; key injected via environment"}
                continue
            url = tmpl.format(port=lp.spec.get("port"))
            good, bad = code(url, self.api_key), code(url, wrong)
            if good == 200 and bad == 401:
                out[lid] = {"ok": True, "detail": "accepts the shared key, rejects others"}
            elif good == 401:
                out[lid] = {"ok": False, "detail": "REJECTS the shared key - this layer is using a different key"}
            elif good is None:
                out[lid] = {"ok": None, "detail": "not answering yet"}
            else:
                out[lid] = {"ok": False, "detail": f"unexpected answer (shared key -> {good}, wrong key -> {bad})"}
        return out

    def stop_all(self, include_core: bool = False):
        """Stop every optional layer (and the core too when include_core)."""
        for lid in reversed(list(self.procs)):
            lp = self.procs[lid]
            if lp.spec.get("always_on") and not include_core:
                continue
            try:
                self.stop(lid)
            except Exception as e:
                print(f"[supervisor] error stopping {lid}: {e}")

    def revoke_layer_consents(self):
        """Layers keep their own consent record (ransomware: per watched folder).
        When the user withdraws consent in the console, drop those records too, so
        a layer started later does not resume watching on a stale approval."""
        home = self.layers_spec.get("ransomware", {}).get("env", {}).get("NEURASHIELD_HOME")
        if home:
            base = Path(home) if os.path.isabs(home) else self.root / home
            try:
                (base / "data" / "consent.json").unlink()
            except OSError:
                pass

    def shutdown(self):
        """Stop every layer and free every port the platform uses. After this
        returns, nothing the supervisor launched is left running."""
        self._stop_monitor.set()
        for lid in reversed(list(self.procs)):
            try:
                self.procs[lid].enabled = False
                self.procs[lid].stop()
            except Exception as e:
                print(f"[supervisor] error stopping {lid}: {e}")
        # Belt-and-suspenders: free the ports of layers WE started that a stray
        # child the tree-kill missed may still hold. Layers we never launched
        # are left alone — their ports may belong to someone else's program.
        self._free_ports()
        print("[supervisor] all services stopped; ports released")

    def _free_ports(self):
        ports = set()
        for lid, lp in self.procs.items():
            if lp.started_at is not None and lp.spec.get("port"):
                ports.add(int(lp.spec["port"]))
        for port in ports:
            _kill_listener_on_port(port)

    # -- status ---------------------------------------------------------------
    def status(self) -> dict:
        out = {}
        for lid, lp in self.procs.items():
            running = lp.is_running()
            out[lid] = {
                "name": lp.name,
                "layer_number": lp.spec.get("layer_number"),
                "enabled": self._is_enabled(lid),
                "running": running,
                "pid": lp.proc.pid if running else None,
                "port": lp.spec.get("port"),
                "healthy": lp.health(self.api_key) if running else False,
                "restarts": lp.restarts,
                "always_on": bool(lp.spec.get("always_on")),
                "gave_up": lp.gave_up,
                "last_error": lp.last_error,
            }
        return out

    # -- auto-restart watchdog ------------------------------------------------
    def _monitor_loop(self):
        while not self._stop_monitor.wait(3.0):
            now = time.time()
            for lid, lp in self.procs.items():
                if not lp.enabled or lp.gave_up or lp.is_running():
                    continue
                if lp.next_retry is None:
                    # First time we notice this exit: classify it and back off.
                    lp.last_exit = lp.proc.poll() if lp.proc else None
                    uptime = now - (lp.started_at or now)
                    lp.fail_streak = lp.fail_streak + 1 if uptime < _QUICK_FAIL_SECONDS else 0
                    if lp.fail_streak >= _MAX_QUICK_FAILS:
                        lp.gave_up = True
                        lp.last_error = (f"exited immediately {lp.fail_streak} times "
                                         f"(code {lp.last_exit}); see {lp.log_path()}")
                        print(f"[supervisor] {lid}: giving up — {lp.last_error}")
                        continue
                    lp.next_retry = now + min(60, 3 * 2 ** lp.fail_streak)
                    print(f"[supervisor] {lid} exited (code {lp.last_exit}); "
                          f"restart in {lp.next_retry - now:.0f}s")
                if now >= lp.next_retry:
                    lp.restarts += 1
                    lp.next_retry = None
                    try:
                        with self._lock:
                            lp.start(self.api_key, self.hmac_secret, self.platform_url)
                    except Exception as e:
                        lp.next_retry = now + 30
                        print(f"[supervisor] restart of {lid} failed: {e}")

    def start_watchdog(self):
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()


def main():
    sup = Supervisor()
    print(f"[supervisor] root={sup.root}")
    print(f"[supervisor] api key {secrets_store.fingerprint(sup.api_key)} | "
          f"hmac {secrets_store.fingerprint(sup.hmac_secret)}")
    sup.start_enabled()
    sup.start_watchdog()
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n[supervisor] shutting down...")
        sup.shutdown()


if __name__ == "__main__":
    main()

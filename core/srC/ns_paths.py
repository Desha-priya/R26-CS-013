
import os
import secrets
import sys

FROZEN = bool(getattr(sys, "frozen", False))


_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Files that mean "this folder already holds run-time data from an earlier run".
_DATA_MARKERS = ("neurashield_auth.db", "audit_log.csv", "models")


def _source_roots():
    """Folders searched for read-only files when running from source."""
    roots = []
    for r in (os.getcwd(), _CODE_DIR, os.path.dirname(_CODE_DIR)):
        if r not in roots:
            roots.append(r)
    return roots


def resource_dir() -> str:
    override = os.environ.get("NEURASHIELD_RESOURCE_DIR")
    if override:
        return override
    if FROZEN:
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return _CODE_DIR


def data_dir() -> str:
    override = os.environ.get("NEURASHIELD_DATA_DIR")
    if override:
        path = override
    elif FROZEN:
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        path = os.path.join(base, "NeuraShield")
    else:
        path = _CODE_DIR
        for candidate in (os.getcwd(), _CODE_DIR):
            if any(os.path.exists(os.path.join(candidate, m)) for m in _DATA_MARKERS):
                path = candidate
                break
    os.makedirs(path, exist_ok=True)
    return path


def resource_path(*parts: str) -> str:
    """Path of a read-only file shipped with the program.

    From source, the first existing match in the search folders wins; if
    none exists the default location is returned so the error message names
    a sensible place to put the file.
    """
    if os.environ.get("NEURASHIELD_RESOURCE_DIR") or FROZEN:
        return os.path.join(resource_dir(), *parts)
    for root in _source_roots():
        candidate = os.path.join(root, *parts)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_CODE_DIR, *parts)


def data_path(*parts: str) -> str:
    path = os.path.join(data_dir(), *parts)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    return path


# ------------------------------------------------------------------
# API key shared by the platform, the agent and the other layers.
#                  ****
# Order: environment variable NEURASHIELD_API_KEY, then the key file in
# the data directory, otherwise a random key is generated once and
# stored in that file. There is no built-in default key.

API_KEY_FILE_NAME = "api_key.txt"


def _strong_protection() -> bool:
    try:
        import ns_secure
        return ns_secure.STRONG
    except Exception:
        return False


def api_key_file() -> str:
    """Windows: DPAPI-protected api_key.dat. Elsewhere: api_key.txt."""
    if _strong_protection():
        return os.path.join(data_dir(), "api_key.dat")
    return os.path.join(data_dir(), API_KEY_FILE_NAME)


def get_api_key() -> str:
    env = os.environ.get("NEURASHIELD_API_KEY")
    if env and env.strip():
        return env.strip()
    # The shared key file the NeuraShield supervisor writes (same key every
    # layer receives) - used if the environment variable was not inherited.
    shared = os.environ.get("NEURASHIELD_KEY_FILE")
    if shared:
        try:
            with open(shared, "r", encoding="utf-8") as f:
                k = f.read().strip()
            if k:
                return k
        except OSError:
            pass
    if os.environ.get("NEURASHIELD_SUPERVISED") == "1":
        # Launched by NeuraShield: never invent a key of our own, that would
        # split the system into layers that reject each other.
        raise RuntimeError("NeuraShield platform API key not available "
                           "(NEURASHIELD_API_KEY / NEURASHIELD_KEY_FILE)")
    if _strong_protection():
        return _get_protected_api_key()
    path = api_key_file()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                key = f.read().strip()
            if key:
                return key
    except OSError:
        pass
    key = secrets.token_urlsafe(24)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(key)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError:
        pass
    return key


def _get_protected_api_key() -> str:
    """Windows: the key lives in api_key.dat, encrypted with DPAPI.
    An old plain api_key.txt is read once, protected, then deleted."""
    import ns_secure
    dat = os.path.join(data_dir(), "api_key.dat")
    legacy = os.path.join(data_dir(), API_KEY_FILE_NAME)
    try:
        if os.path.exists(dat):
            key = ns_secure.read_bytes(dat).decode("utf-8").strip()
            if key:
                return key
    except Exception:
        pass
    key = ""
    try:
        if os.path.exists(legacy):
            with open(legacy, "r", encoding="utf-8") as f:
                key = f.read().strip()
    except OSError:
        pass
    key = key or secrets.token_urlsafe(24)
    try:
        ns_secure.write_protected(dat, key.encode("utf-8"))
        if os.path.exists(legacy):
            os.remove(legacy)
    except Exception:
        pass
    return key


if __name__ == "__main__":
    # python ns_paths.py --show-key   (to configure the extension / layers)
    if "--show-key" in sys.argv:
        print(get_api_key())
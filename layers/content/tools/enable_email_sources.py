"""
NeuraShield Layer 4 - one-step enabler for the extra email sources.

Adds the two lines that register the email-source routes to your app.py,
right after the `app = FastAPI(...)` block. Safe to run more than once
(does nothing if already present). Makes a backup first: app.py.bak_email

Usage (from the project root):
    python tools/enable_email_sources.py            # finds demo/app.py or app.py
    python tools/enable_email_sources.py path\\to\\app.py
"""
import re, shutil, sys
from pathlib import Path

IMPORT = "from src.watchers.email_routes import router as email_sources_router"
INCLUDE = "app.include_router(email_sources_router)"


def find_app(arg=None):
    if arg:
        return Path(arg)
    for c in ("demo/app.py", "app.py"):
        if Path(c).is_file():
            return Path(c)
    return None


def patch_text(text):
    """Return (new_text, changed). Raises ValueError if app = FastAPI( not found."""
    if INCLUDE in text and "email_sources_router" in text.replace(INCLUDE, ""):
        return text, False
    nl = "\r\n" if "\r\n" in text else "\n"
    m = re.search(r"^app\s*=\s*FastAPI\(", text, re.M)
    if not m:
        raise ValueError("could not find the line 'app = FastAPI(' in this file")
    i, depth, instr = m.end(), 1, None
    while i < len(text) and depth:
        ch = text[i]
        if instr:
            if ch == "\\":
                i += 1
            elif text.startswith(instr, i):
                i += len(instr) - 1
                instr = None
        elif ch == "#":
            i = text.find("\n", i) - 1 if text.find("\n", i) != -1 else len(text)
        elif text.startswith('"""', i) or text.startswith("'''", i):
            instr = text[i:i + 3]; i += 2
        elif ch in "\"'":
            instr = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    if depth:
        raise ValueError("could not find the end of the FastAPI(...) call")
    eol = text.find("\n", i)
    eol = len(text) if eol == -1 else eol + 1
    block = (nl + "# -- Extra email sources (Gmail sign-in, drop folder, notifications) --" + nl
             + IMPORT + nl + INCLUDE + nl + nl)
    return text[:eol] + block + text[eol:], True


def main():
    p = find_app(sys.argv[1] if len(sys.argv) > 1 else None)
    if not p or not p.is_file():
        print("Could not find app.py. Run this from the project root, or pass the path.")
        return 1
    raw = p.read_bytes().decode("utf-8")
    try:
        new, changed = patch_text(raw)
    except ValueError as e:
        print("Not changed:", e); return 1
    if not changed:
        print(f"{p}: already enabled. Nothing to do. (If the card still shows 404, restart the app.)")
        return 0
    shutil.copy2(p, str(p) + ".bak_email")
    p.write_bytes(new.encode("utf-8"))
    compile(new, str(p), "exec")
    print(f"{p}: email sources enabled (backup: {p}.bak_email).")
    print("Now RESTART the app, then hard-refresh the dashboard (Ctrl+F5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
layer_host.py  —  run any NeuraShield layer from inside the ONE NeuraShield.exe
------------------------------------------------------------------------------
The product ships as a single executable that contains every layer. The
supervisor starts each layer as a separate OS process (so a crash in one layer
cannot take down the others, and the process-tree / job-object cleanup keeps
working) by launching this same exe in layer mode:

    NeuraShield.exe --run-layer core|network|ransomware|content
    NeuraShield.exe --monitor                  (network packet-capture child)
    NeuraShield.exe --subprocess|--enroll-face (core face-liveness child)

dispatch() is called first thing in the exe's entry point. It returns False when
argv is a normal app start, otherwise it runs the requested layer (blocking) and
returns True. The layers' own source is bundled as plain files and executed with
runpy exactly like `python <entry>` would - no layer code is modified for this.

Works from source too (python neurashield_app.py --run-layer content), which is
how it is tested.
"""

import os
import runpy
import sys
from pathlib import Path

# layer id -> (folder holding the entry script, entry script, extra sys.path
#              folders, default args, env var that tells the layer where its
#              read-only bundled files are)
_LAYERS = {
    "core":       ("core/srC",        "neurashield_launcher.py", [],                 [],              ("NEURASHIELD_RESOURCE_DIR", "core")),
    "network":    ("layers/network",  "app.py",                  [],                 [],              None),
    "ransomware": ("layers/ransomware", "main.py",               [],                 ["--dashboard"], ("NEURASHIELD_BUNDLE_ROOT", "layers/ransomware")),
    "content":    ("layers/content/demo", "app.py",              ["layers/content"], [],              None),
}


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def _exec(root: Path, folder: str, script: str, extra_paths, argv, env_hint=None):
    layer_dir = root / folder
    if env_hint:
        # point the layer at its read-only files unless the caller already did
        os.environ.setdefault(env_hint[0], str(root / env_hint[1]))
    for p in [str(layer_dir)] + [str(root / e) for e in extra_paths]:
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.argv = [str(layer_dir / script)] + list(argv)
    runpy.run_path(str(layer_dir / script), run_name="__main__")


def dispatch(argv) -> bool:
    """argv = sys.argv[1:]. Returns True if a layer mode was run."""
    if not argv:
        return False
    # Layers print symbols (check marks, box drawing, emoji). Their output goes to
    # a log file, where Windows defaults to cp1252 and a single such print raises
    # UnicodeEncodeError and kills the layer. Make that impossible for every layer.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    root = _bundle_root()
    head = argv[0]

    if head in ("--subprocess", "--enroll-face"):
        folder, script, extra, _d, env_hint = _LAYERS["core"]
        _exec(root, folder, script, extra, argv, env_hint)
        return True

    if head == "--monitor":
        # network packet-capture child (see layers/network/app.py start_monitor)
        os.chdir(root / "layers" / "network")
        sys.path.insert(0, str(root / "layers" / "network"))
        sys.argv = [str(root / "layers" / "network" / "monitor-2.py")]
        runpy.run_path(str(root / "layers" / "network" / "monitor-2.py"), run_name="__main__")
        return True

    if head == "--run-layer" and len(argv) >= 2:
        lid = argv[1]
        if lid not in _LAYERS:
            sys.exit(f"unknown layer: {lid}")
        folder, script, extra, default_args, env_hint = _LAYERS[lid]
        _exec(root, folder, script, extra, default_args + list(argv[2:]), env_hint)
        return True

    return False

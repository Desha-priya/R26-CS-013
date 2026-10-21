r"""
build_all.py  —  freeze NeuraShield into ONE executable (Windows)
------------------------------------------------------------------
    python installer\build_all.py

Produces   dist\gateway\NeuraShield\NeuraShield.exe   (+ its _internal folder)

That single program is the whole product: the desktop app, the console, the
supervisor AND every layer (core/ZTA, network, ransomware, content). The
supervisor starts a layer by launching this same exe as `--run-layer <id>` (see
gateway/layer_host.py), so the layers still run as separate, isolated processes.

All layers therefore share ONE Python environment - requirements-all.txt. Create it with
    python -m venv runtimes\all  &&  runtimes\all\Scripts\pip install -r requirements-all.txt
(or let installer\build_release.py do it). It is a one-folder build on purpose: a
one-file exe would unpack ~4 GB of ML libraries on every start.

Set NEURASHIELD_BUILD_OUT to build somewhere other than the repo root, and
NEURASHIELD_PY to build with a specific interpreter.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("NEURASHIELD_BUILD_OUT") or ROOT)
IS_WIN = sys.platform == "win32"
EXE_NAME = "NeuraShield"

# (repo-relative source, destination inside the bundle). Layer sources travel as
# plain files and are executed by layer_host.py; read-only resources sit beside them.
DATA = [
    ("gateway/console", "gateway/console"),
    ("config/layers.json", "config"),     # defaults for a run outside the installed layout
    # core / Zero-Trust (ns_paths resolves templates + processed under core/)
    ("core/srC", "core/srC"),
    ("core/templates", "core/templates"),
    ("core/processed", "core/processed"),
    ("core/browser_extension", "core/browser_extension"),
    # network guardian (monitor-2.py loads its .pkl/.pth models from this folder)
    ("layers/network", "layers/network"),
    # ransomware killer
    ("layers/ransomware/main.py", "layers/ransomware"),
    ("layers/ransomware/src", "layers/ransomware/src"),
    ("layers/ransomware/config", "layers/ransomware/config"),
    ("layers/ransomware/models", "layers/ransomware/models"),
    # content threat: only the models actually loaded (models/text/*.pkl is a
    # 1.2 GB legacy classifier that the running layer never touches)
    ("layers/content/demo", "layers/content/demo"),
    ("layers/content/src", "layers/content/src"),
    ("layers/content/template", "layers/content/template"),
    # Only the files from_pretrained() reads. The folder also holds runs/ - 1.5 GB of
    # training checkpoints and optimizer state that the product never loads.
    *[(f"layers/content/models/text/distilbert_phishing/{n}", "layers/content/models/text/distilbert_phishing")
      for n in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                "accuracy.txt", "training_summary.json")],
    ("layers/content/models/image", "layers/content/models/image"),
    ("layers/content/models/audio", "layers/content/models/audio"),
]

PYI_ARGS = [
    "--noconsole", "--paths", str(ROOT / "gateway"),
    "--collect-submodules", "uvicorn", "--collect-submodules", "pystray",
    "--collect-submodules", "scapy", "--collect-submodules", "webview",
    # the web stacks import many modules by name at run time
    "--collect-submodules", "fastapi", "--collect-submodules", "starlette",
    "--collect-submodules", "anyio", "--collect-submodules", "flask",
    "--collect-submodules", "werkzeug", "--collect-submodules", "jinja2",
    "--collect-submodules", "waitress", "--collect-submodules", "watchdog",
    "--collect-submodules", "pydantic", "--collect-submodules", "multipart",
    "--collect-submodules", "python_multipart", "--collect-submodules", "h11",
    "--collect-submodules", "psutil", "--collect-submodules", "yaml",
    "--collect-all", "tf_keras", "--collect-all", "transformers.models.distilbert",
    "--collect-all", "soundfile", "--collect-data", "cv2", "--collect-data", "librosa",
    "--collect-data", "shap", "--copy-metadata", "tqdm", "--copy-metadata", "transformers",
    "--copy-metadata", "huggingface-hub", "--copy-metadata", "safetensors",
    "--copy-metadata", "tokenizers", "--copy-metadata", "requests", "--copy-metadata", "packaging",
    "--copy-metadata", "numpy", "--copy-metadata", "regex", "--copy-metadata", "filelock",
]


def python() -> str:
    override = os.environ.get("NEURASHIELD_PY")
    if override:
        return override
    p = ROOT / "runtimes" / "all" / ("Scripts/python.exe" if IS_WIN else "bin/python")
    return str(p) if p.exists() else sys.executable


def data_args():
    sep = ";" if IS_WIN else ":"
    args = []
    for rel, dest in DATA:
        src = ROOT / rel
        if src.exists():
            args += ["--add-data", f"{src}{sep}{dest}"]
        else:
            print(f"  ! not bundled (missing): {rel}")
    return args


def main():
    py = python()
    print(f"=== building {EXE_NAME}.exe with {py} ===")
    subprocess.run([py, "-m", "pip", "install", "-q", "pyinstaller"], check=False)
    print("--- scanning the layers' imports")
    subprocess.run([py, str(ROOT / "installer" / "gen_bundle_imports.py")], check=True)
    out = OUT / "dist" / "gateway"
    cmd = [py, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", EXE_NAME,
           "--distpath", str(out), "--workpath", str(OUT / "build"),
           "--specpath", str(OUT / "build")] + PYI_ARGS + data_args() + [str(ROOT / "neurashield_app.py")]
    rc = subprocess.run(cmd).returncode
    exe = out / EXE_NAME / (EXE_NAME + (".exe" if IS_WIN else ""))
    ok = rc == 0 and exe.exists()
    print(f"\n{'OK' if ok else 'FAILED'}: {exe}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

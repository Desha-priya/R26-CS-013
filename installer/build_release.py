r"""
build_release.py  -  ONE command from source to NeuraShield-Setup-<version>.exe
--------------------------------------------------------------------------------
    python installer\build_release.py             # env + the exe + installer
    python installer\build_release.py --skip-env  # runtimes\all already exists
    python installer\build_release.py --no-installer

1) preflight (Windows, Python 3.11, models present)  2) the single Python
environment (requirements-all.txt -> runtimes\all)  3) freeze NeuraShield.exe -
the one executable containing every layer  4) sanity check  5) Inno Setup.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "runtimes" / "all"
ENV_PY = ENV / "Scripts" / "python.exe"

REQUIRED_MODELS = {
    "ZTA Siamese checkpoint": "core/processed/siamese_bilstm_v3/model_training_v3/checkpoints/best_siamese_bilstm_v3.pt",
    "ZTA feature scaler": "core/processed/siamese_bilstm_v3/reports/normalization_audit/train_only_scaler_v3.npz",
    "Content text model": "layers/content/models/text/distilbert_phishing/model.safetensors",
    "Content video model": "layers/content/models/image/deepfake_cnn_model.h5",
    "Content audio model": "layers/content/models/audio/spoof_resnet18.pth",
    "Ransomware classifier": "layers/ransomware/models/classifier.pkl",
}
EXE = ROOT / "dist" / "gateway" / "NeuraShield" / "NeuraShield.exe"


def step(msg):
    print(f"\n{'=' * 70}\n  {msg}\n{'=' * 70}")


def find_iscc():
    found = shutil.which("iscc")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            cand = Path(base) / "Inno Setup 6" / "ISCC.exe"
            if cand.exists():
                return str(cand)
    return None


def main():
    skip_env = "--skip-env" in sys.argv
    no_installer = "--no-installer" in sys.argv

    step("1/5 preflight")
    if sys.platform != "win32":
        sys.exit("The executable is a Windows binary - run this on Windows.")
    if sys.version_info[:2] != (3, 11):
        print(f"  ! running on Python {sys.version_info.major}.{sys.version_info.minor}; the pinned wheels "
              "(torch 2.6, tensorflow 2.16) are for 3.11 x64.")
    missing = [n for n, p in REQUIRED_MODELS.items() if not (ROOT / p).exists()]
    for n in missing:
        print(f"  ! missing model: {n}")
    if missing:
        sys.exit("Put the model files in place first (see installer/BUILD_WINDOWS.md).")
    print("  ok - all models present")

    if not skip_env:
        step("2/5 the single Python environment (downloads torch/tensorflow: slow the first time)")
        if not ENV_PY.exists():
            subprocess.run([sys.executable, "-m", "venv", str(ENV)], check=True)
        rc = subprocess.run([str(ENV_PY), "-m", "pip", "install", "-r", str(ROOT / "requirements-all.txt")]).returncode
        if rc != 0:
            sys.exit("pip install failed - fix the errors above, then rerun.")

    step("3/5 freezing NeuraShield.exe (every layer inside) - this takes a while")
    if subprocess.run([str(ENV_PY), str(ROOT / "installer" / "build_all.py")]).returncode != 0:
        sys.exit("build failed - see the messages above.")

    step("4/5 checking the executable")
    if not EXE.exists():
        sys.exit(f"missing: {EXE}")
    print(f"  ok - {EXE}")

    if no_installer:
        return
    step("5/5 compiling the installer")
    iscc = find_iscc()
    if not iscc:
        sys.exit("Inno Setup 6 not found - install it from jrsoftware.org, then run:\n"
                 "    iscc installer\neurashield.iss")
    if subprocess.run([iscc, str(ROOT / "installer" / "neurashield.iss")]).returncode != 0:
        sys.exit("Inno Setup failed.")
    outs = sorted((ROOT / "installer" / "Output").glob("NeuraShield-Setup-*.exe"))
    shown = outs[-1] if outs else "installer/Output"
    print(f"\nDONE  ->  {shown}")


if __name__ == "__main__":
    main()

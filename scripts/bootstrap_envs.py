"""
bootstrap_envs.py  -  create the ONE Python environment the whole product runs on
-----------------------------------------------------------------------------
NeuraShield ships as a single executable containing every layer, so every layer
runs on the same package versions (requirements-all.txt; the content layer's
TensorFlow 2.16 sets the ceiling: numpy 1.26, protobuf 4.25).

    python scripts/bootstrap_envs.py        # -> runtimes/all  (Python 3.11 x64)

The supervisor uses runtimes/all for a source run; the packaged exe needs none.
"""
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "runtimes" / "all"


def main():
    py = ENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not py.exists():
        print(f"creating {ENV}")
        venv.EnvBuilder(with_pip=True).create(str(ENV))
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=False)
    rc = subprocess.run([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements-all.txt")]).returncode
    print("OK - runtimes/all ready" if rc == 0 else "FAILED - see pip output above")
    sys.exit(rc)


if __name__ == "__main__":
    main()

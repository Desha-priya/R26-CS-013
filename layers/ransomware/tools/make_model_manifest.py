"""
make_model_manifest.py
--------------------------
Writes models/MANIFEST.json: a SHA-256 fingerprint for every model file.
Run this ONCE after training (or retraining) the models:

    python tools/make_model_manifest.py

It also records the scikit-learn version the models were saved with, found
by loading them and reading scikit-learn's own version warning, so the
self-check can tell you if the machine has a different version.
"""

import os
import re
import sys
import json
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import joblib
import yaml

from src.detection.model_integrity import sha256_file


def main():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rel_paths = [
        cfg["isolation_forest"]["model_path"],
        cfg["random_forest"]["model_path"],
        cfg["scaler_path"],
        cfg["scaler_path"].replace(".pkl", "_features.pkl"),
        cfg["scaler_path"].replace(".pkl", "_rf.pkl"),
        cfg["random_forest"]["model_path"].replace(".pkl", "_features.pkl"),
    ]

    files, versions = {}, set()
    for rel in rel_paths:
        if not os.path.exists(rel):
            sys.exit(f"Missing model file: {rel}  (train the models first)")
        files[os.path.basename(rel)] = sha256_file(rel)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            joblib.load(rel)
        for w in caught:
            m = re.search(r"from version ([\d.]+)", str(w.message))
            if m:
                versions.add(m.group(1))

    import sklearn
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "saved_with_sklearn": sorted(versions)[0] if versions else sklearn.__version__,
        "files": files,
    }
    with open(os.path.join("models", "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote models/MANIFEST.json")
    print(f"  models saved with scikit-learn {manifest['saved_with_sklearn']} "
          f"(this machine has {sklearn.__version__})")
    for name, digest in files.items():
        print(f"  {name}  {digest[:16]}...")


if __name__ == "__main__":
    main()

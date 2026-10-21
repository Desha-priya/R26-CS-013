import numpy as np
from pathlib import Path

DATA_DIR = Path(r"processed\sequence_dataset\normalized")

print("=" * 70)
print("NORMALIZED DATASET STRUCTURE AUDIT")
print("=" * 70)

files = sorted(DATA_DIR.glob("*.npz"))

print(f"\nFound {len(files)} NPZ files:")

for path in files:
    print(f"\n{'-' * 70}")
    print(f"FILE: {path.name}")

    data = np.load(path, allow_pickle=True)

    print("Arrays stored:")

    for key in data.files:
        arr = data[key]

        print(f"\n  {key}")
        print(f"      Shape : {arr.shape}")
        print(f"      Dtype : {arr.dtype}")

        if np.issubdtype(arr.dtype, np.number):
            print(f"      NaN   : {np.isnan(arr).sum()}")
            print(f"      Inf   : {np.isinf(arr).sum()}")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
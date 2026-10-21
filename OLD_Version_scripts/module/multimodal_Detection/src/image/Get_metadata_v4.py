# this ran after the copyoing 300 video to the data folder raw.

import os
import pandas as pd
from pathlib import Path

# ================== PATH ==================
raw_root = Path("data/video/raw")

print("Creating metadata from folders...")

data = []

# Process real videos
real_dir = raw_root / "original"
for file in real_dir.glob("*.mp4"):
    data.append({
        "file_path": str(file),
        "file": file.name,
        "label": "real",
        "modality": "video"
    })

# Process fake videos
fake_dir = raw_root / "manipulated"
for file in fake_dir.glob("*.mp4"):
    data.append({
        "file_path": str(file),
        "file": file.name,
        "label": "fake",
        "modality": "video"
    })

meta_df = pd.DataFrame(data)
meta_df.to_csv("data/video/processed/metadata_v2.csv", index=False)

print(f"Metadata created successfully!")
print(f"Total videos: {len(meta_df)}")
print(meta_df['label'].value_counts())
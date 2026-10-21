import os
import pandas as pd
from pathlib import Path

# ================== STEP 1: SET YOUR DATASET PATH ==================
# Change this to where you extracted FaceForensics++
dataset_root = r"C:\Users\Nethma Sankalpa\Documents\SLIIT campus\CS-Research(PC)\B datasets\archive\faceFo"   # ←←← CHANGE THIS LINE

# ================== STEP 2: EXPLORE THE FOLDER STRUCTURE ==================
print("=== Exploring FaceForensics++ Dataset Structure ===")

media_files = []
for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.mp4', '.avi')):
            full_path = os.path.join(root, file)

            # Simple label guessing from folder name
            label = "real" if "real" in root.lower() or "original" in root.lower() else "fake"
            
            modality = "video" if file.lower().endswith(('.mp4', '.avi')) else "image"
            
            media_files.append({
                "file_path": full_path,
                "filename": file,
                "label": label,
                "modality": modality,
                "folder": root
            })

# Create a simple metadata table
meta_df = pd.DataFrame(media_files)
print(f"Total media files found: {len(meta_df)}")
print(f"Real samples: {len(meta_df[meta_df['label'] == 'real'])}")
print(f"Fake samples: {len(meta_df[meta_df['label'] == 'fake'])}")
print("\nFirst 5 samples:")
print(meta_df.head())

# Save metadata for later use
meta_df.to_csv("faceforensics_metadata.csv", index=False)
print("\nMetadata saved to: faceforensics_metadata.csv")
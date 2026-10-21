# feature_extraction_v3.py
# Extract features from 600 videos (300 real + 300 fake)
# use metadata_v2.csv to get file paths and labels
# save features to improved_visual_features_v3.csv

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import time

start_time = time.time()

# ================== CONFIG ==================
RAW_ROOT = Path("data/video/raw")
OUTPUT_CSV = "data/video/processed/improved_visual_features_v3.csv"

print("Starting feature extraction from 600 videos...")

data = []

# Process Real videos
real_dir = RAW_ROOT / "original"
real_files = list(real_dir.glob("*.mp4"))
print(f"Found {len(real_files)} real videos")

# Process Fake videos
fake_dir = RAW_ROOT / "manipulated"
fake_files = list(fake_dir.glob("*.mp4"))
print(f"Found {len(fake_files)} fake videos")

# Combine
all_videos = [("real", f) for f in real_files] + [("fake", f) for f in fake_files]

processed = 0

for label, video_path in all_videos:
    processed += 1
    print(f"Processing {processed}/{len(all_videos)}: {video_path.name} ({label})")
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"   Could not open video")
            continue
            
        frame_count = 0
        mean_r_list, mean_g_list, mean_b_list = [], [], []
        edge_ratios = []
        
        # Extract from first 30 frames (good balance between speed and quality)
        while frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
                
            height, width = frame.shape[:2]
            
            # Color features
            mean_color = cv2.mean(frame)[:3]
            mean_r_list.append(mean_color[2])
            mean_g_list.append(mean_color[1])
            mean_b_list.append(mean_color[0])
            
            # Edge features (helps detect deepfake artifacts)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = np.sum(edges > 0) / (height * width)
            edge_ratios.append(edge_ratio)
            
            frame_count += 1
        
        cap.release()
        
        if frame_count == 0:
            continue
            
        features = {
            "file": video_path.name,
            "label": label,
            "modality": "video",
            "width": width,
            "height": height,
            "mean_r": np.mean(mean_r_list),
            "mean_g": np.mean(mean_g_list),
            "mean_b": np.mean(mean_b_list),
            "std_color": np.std(mean_r_list + mean_g_list + mean_b_list),
            "edge_ratio": np.mean(edge_ratios),
            "frames_used": frame_count
        }
        data.append(features)
        
    except Exception as e:
        print(f"   Error processing {video_path.name}: {e}")

# Save features
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*60)
print("FEATURE EXTRACTION COMPLETE!")
print("="*60)
print(f"Successfully processed {len(df)} videos")
print(f"Real : {len(df[df['label']=='real'])}")
print(f"Fake : {len(df[df['label']=='fake'])}")
print(f"Saved to: {OUTPUT_CSV}")
print(f"Time taken: {time.time() - start_time:.1f} seconds")
print("\nFirst 5 rows:")
print(df.head())
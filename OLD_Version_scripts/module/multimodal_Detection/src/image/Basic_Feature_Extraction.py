import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# Load metadata
meta_df = pd.read_csv("faceforensics_metadata.csv")

features_list = []

for idx, row in meta_df.iterrows():
    file_path = row['file_path']
    label = row['label']
    modality = row['modality']
    
    print(f"Processing {idx+1}/{len(meta_df)}: {Path(file_path).name} ({label})")
    
    try:
        if modality == "image":
            img = cv2.imread(file_path)
            if img is None:
                continue
            height, width = img.shape[:2]
            mean_color = cv2.mean(img)[:3]
            std_color = np.std(img, axis=(0,1))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = np.sum(edges > 0) / (height * width)
            
            features = {
                "file": Path(file_path).name,
                "label": label,
                "modality": "image",
                "width": width,
                "height": height,
                "mean_r": mean_color[2],
                "mean_g": mean_color[1],
                "mean_b": mean_color[0],
                "std_color": np.mean(std_color),
                "edge_ratio": edge_ratio
            }
            features_list.append(features)
            
        elif modality == "video":
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                continue
                
            frame_count = 0
            mean_r_list, mean_g_list, mean_b_list = [], [], []
            edge_ratios = []
            
            while frame_count < 10:  # Take only first 10 frames for speed
                ret, frame = cap.read()
                if not ret:
                    break
                    
                height, width = frame.shape[:2]
                mean_color = cv2.mean(frame)[:3]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_ratio = np.sum(edges > 0) / (height * width)
                
                mean_r_list.append(mean_color[2])
                mean_g_list.append(mean_color[1])
                mean_b_list.append(mean_color[0])
                edge_ratios.append(edge_ratio)
                
                frame_count += 1
            
            cap.release()
            
            if frame_count == 0:
                continue
                
            features = {
                "file": Path(file_path).name,
                "label": label,
                "modality": "video",
                "width": width,
                "height": height,
                "mean_r": np.mean(mean_r_list),
                "mean_g": np.mean(mean_g_list),
                "mean_b": np.mean(mean_b_list),
                "std_color": np.std(mean_r_list + mean_g_list + mean_b_list),
                "edge_ratio": np.mean(edge_ratios)
            }
            features_list.append(features)
            
    except Exception as e:
        print(f"   Error on {file_path}: {e}")

# Save improved features
feature_df = pd.DataFrame(features_list)
feature_df.to_csv("improved_visual_features.csv", index=False)

print(f"\nSuccessfully processed {len(feature_df)} samples with improved features")
print("Saved to: improved_visual_features.csv")
print("\nFirst 5 rows:")
print(feature_df.head())
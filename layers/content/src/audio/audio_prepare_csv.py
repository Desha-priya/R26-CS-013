import pandas as pd
import os

PROTOCOL_FILE = "data/audio/raw/ASVspoof2019_LA_train/ASVspoof2019.LA.cm.train.trn.txt"
FLAC_DIR = "data/audio/raw/ASVspoof2019_LA_train/flac"

rows = []

with open(PROTOCOL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()

        file_name = parts[1]
        label = parts[-1]

        path = os.path.join(FLAC_DIR, file_name + ".flac")

        label_num = 0 if label == "bonafide" else 1

        rows.append([path, label_num])

df = pd.DataFrame(rows, columns=["path", "label"])

print(df.head())

df.to_csv("train.csv", index=False)

print("Saved train.csv")
"""import pandas as pd
import os

# Change this to your actual path where you unzipped BB-MAS
dataset_path = "BB-MAS_Dataset"   # ←←← CHANGE THIS

# List all files
files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
print(f"Total files: {len(files)}")
print("First 5 files:", files[:5])

# Load one sample file to see the structure
sample_file = os.path.join(dataset_path, files[0])
df = pd.read_csv(sample_file)
print("\nSample data shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())  """

import pandas as pd
import os

dataset_path = "BB-MAS_Dataset"
# List all CSV files
csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
print("All CSV files found:")
for f in csv_files:
    print(f" - {f}")

print("\n=== Exploring Desktop_Freetext.csv (most important for keystroke) ===")
desktop_file = os.path.join(dataset_path, "Desktop_Freetext.csv")
df_desktop = pd.read_csv(desktop_file, nrows=10000)   # Load first 10,000 rows to see structure quickly

print("Shape:", df_desktop.shape)
print("Columns:", df_desktop.columns.tolist())
print("\nFirst 5 rows:\n", df_desktop.head())

# Show unique users
print("\nNumber of unique users:", df_desktop['User ID'].nunique() if 'User ID' in df_desktop.columns else "No User ID column")

# Show some example keystroke columns
keystroke_cols = [col for col in df_desktop.columns if any(x in col.lower() for x in ['dwell', 'flight', 'key', 'press', 'release', 'typing'])]
print("\nPossible keystroke columns:", keystroke_cols[:15])
import pandas as pd
import os
import glob
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT_DIR.parent / "BB-MAS_Dataset"
base_path = DEFAULT_DATASET if DEFAULT_DATASET.exists() else Path("BB-MAS_Dataset")  # same path as your keyboard script

user_folders = sorted([f for f in os.listdir(base_path) if f.isdigit()])
if not user_folders:
    print("No user folders found!")
else:
    user_id = user_folders[0]
    user_folder = os.path.join(base_path, user_id)
    print(f"Checking user folder: {user_folder}\n")

    for mouse_type in ["mouse_button", "mouse_move", "mouse_wheel"]:
        # Try common naming patterns in BB-MAS
        patterns = [
            f"*{mouse_type}*.csv",
            f"*_Mouse_Button.csv",
            f"*_Mouse_Move.csv",
            f"*_Mouse_Wheel.csv",
            f"*Mouse*.csv",
        ]
        found = []
        for pat in patterns:
            found += glob.glob(os.path.join(user_folder, pat), recursive=False)
        found = list(set(found))  # deduplicate

        print(f"=== {mouse_type.upper()} ===")
        if found:
            f = found[0]
            print(f"  File: {os.path.basename(f)}")
            df = pd.read_csv(f)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  First 5 rows:\n{df.head()}\n")
        else:
            # List ALL files in the folder so we can find the right name
            all_files = os.listdir(user_folder)
            print(f"  NOT FOUND. All files in folder:")
            for fn in all_files:
                print(f"    {fn}")
            print()
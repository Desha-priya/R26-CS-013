

#this is for identify columns and data inside the keystrok csv. aftyer run this  need to do extract_keystroke_features to calculate 


import pandas as pd
import os
import glob

base_path = r"BB-MAS_Dataset"  

# Find the first user folder (e.g. folder "1")
user_folders = [f for f in os.listdir(base_path) if f.isdigit()]
if not user_folders:
    print("No user folders found!")
else:
    user_id = user_folders[0]   # Start with first user
    user_folder = os.path.join(base_path, user_id)
    
    # Find keyboard files (Desktop, Phone, Tablet)
    keyboard_files = glob.glob(os.path.join(user_folder, "*Keyboard.csv"))
    
    if keyboard_files:
        sample_file = keyboard_files[0]   # Take the first keyboard file
        print(f" Found keyboard file: {os.path.basename(sample_file)}")
        
        df = pd.read_csv(sample_file)
        print(f"Shape: {df.shape} rows x {df.shape[1]} columns")
        print("Columns:", df.columns.tolist())
        print("\nFirst 10 rows:\n", df.head(10))
        
        # Show unique keys
        print("\nUnique keys:", df['key'].unique()[:20])
    else:
        print("No *Keyboard.csv files found in user folder", user_id)
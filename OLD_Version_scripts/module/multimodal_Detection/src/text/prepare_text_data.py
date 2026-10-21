# prepare_text_data.py
import pandas as pd
from pathlib import Path

print("=== Preparing Text Phishing Dataset (Nazario) ===")


nazario_path = Path("data/text/raw") 

# Get all CSV files
csv_files = list(nazario_path.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")

phishing_dfs = []
normal_dfs = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        print(f"Loaded {file.name} - Shape: {df.shape} - Columns: {list(df.columns)}")
        
        # Try to find email content column (common names)
        text_column = None
        for col in df.columns:
            if col.lower() in ['email', 'text', 'body', 'message', 'content']:
                text_column = col
                break
        if text_column is None and len(df.columns) > 0:
            text_column = df.columns[0]   # take first column as fallback
        
        df = df[[text_column]].copy()
        df.rename(columns={text_column: 'email_text'}, inplace=True)
        df = df.dropna(subset=['email_text'])
        
        # Classify based on filename
        filename_lower = file.name.lower()
        if any(word in filename_lower for word in ['phish', 'nazario', 'nigerian', 'fraud', 'spam']):
            df['label'] = 'phishing'
            phishing_dfs.append(df)
            print(f"  → Marked as PHISHING")
        else:
            df['label'] = 'normal'
            normal_dfs.append(df)
            print(f"  → Marked as NORMAL")
            
    except Exception as e:
        print(f"Error processing {file.name}: {e}")

# Combine all
final_df = pd.concat(phishing_dfs + normal_dfs, ignore_index=True)

final_df.to_csv("data/text/processed/phishing_emails_prepared.csv", index=False)

print("\n" + "="*60)
print("TEXT DATASET PREPARATION COMPLETE!")
print("="*60)
print(f"Total emails: {len(final_df)}")
print(final_df['label'].value_counts())
print(f"\nSaved to: data/text/processed/phishing_emails_prepared.csv")
print("Ready for training!")
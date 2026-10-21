import pandas as pd
import re
from pathlib import Path

SRC = Path(r"data/text/processed/phishing_emails_prepared.csv")
OUT = Path(r"data/text/processed/phishing_distilbert_clean.csv")

df = pd.read_csv(SRC)
print("Raw rows:", len(df))
print("Columns:", df.columns.tolist())
print(df["label"].value_counts(dropna=False))

# normalize labels
df["label"] = df["label"].astype(str).str.strip().str.lower()
# keep only clear binary labels
df = df[df["label"].isin(["phishing", "normal", "ham", "legitimate", "legit", "0", "1"])]

def map_label(x):
    if x in ["phishing", "1"]:
        return 1
    return 0

df["y"] = df["label"].map(map_label)

def clean_email(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    # remove very common header noise lines
    t = re.sub(r"(?im)^(from|to|subject|date|return-path|received|message-id|mime-version|content-type|x-[\w-]+):.*$", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)          # html tags
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["text"] = df["email_text"].apply(clean_email)
df = df[df["text"].str.len() >= 30]

print("After clean:", len(df))
print(df["y"].value_counts())

# balanced subset for practical CPU/GPU training
N_EACH = 15000  # increase later if GPU works
pos = df[df["y"] == 1]
neg = df[df["y"] == 0]
n = min(N_EACH, len(pos), len(neg))
subset = pd.concat([
    pos.sample(n, random_state=42),
    neg.sample(n, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

subset[["text", "y"]].to_csv(OUT, index=False)
print("Saved:", OUT)
print(subset["y"].value_counts())
print(subset.head(2))
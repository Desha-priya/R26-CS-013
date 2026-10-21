import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

# ================== CONFIG ==================
DATA_FILE = Path(r"data/text/processed/phishing_distilbert_clean.csv")
OUT_DIR   = Path(r"models/text/distilbert_phishing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 256
BATCH_SIZE = 8          # reduce to 4 if CPU RAM is low
EPOCHS = 3
LR = 2e-5
SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ================== DATA ==================
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["text", "y"]).copy()
df["text"] = df["text"].astype(str)
df["y"] = df["y"].astype(int)

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df["y"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["y"]
)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class EmailDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_ds = EmailDataset(train_df["text"], train_df["y"])
val_ds   = EmailDataset(val_df["text"], val_df["y"])
test_ds  = EmailDataset(test_df["text"], test_df["y"])

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

args = TrainingArguments(
    output_dir=str(OUT_DIR / "runs"),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=2,
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

print("\nTraining DistilBERT...")
trainer.train()

# ================== TEST ==================
preds_output = trainer.predict(test_ds)
pred_labels = np.argmax(preds_output.predictions, axis=-1)
true_labels = np.array(test_df["y"])

print("\n===== TEST RESULTS =====")
print(classification_report(
    true_labels, pred_labels,
    target_names=["normal", "phishing"], digits=4
))
print("Confusion matrix:")
print(confusion_matrix(true_labels, pred_labels))

# ================== SAVE ==================
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))

summary = {
    "model": "distilbert-base-uncased",
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "max_len": MAX_LEN,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "device": str(device),
    "test_accuracy": float(accuracy_score(true_labels, pred_labels)),
}
with open(OUT_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved model to:", OUT_DIR)
print("Done.")
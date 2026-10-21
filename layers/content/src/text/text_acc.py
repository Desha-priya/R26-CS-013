# test_text_accuracy.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

os.makedirs("outputs/results", exist_ok=True)

model      = joblib.load("models/text/phishing_text_classifier_v3.pkl")
vectorizer = joblib.load("models/text/tfidf_vectorizer_v3.pkl")

df = pd.read_csv("data/text/processed/phishing_emails_prepared.csv")
df = df.dropna(subset=['email_text', 'label'])
df['email_text'] = df['email_text'].astype(str)

X = vectorizer.transform(df['email_text'])
y_true = df['label']
y_pred = model.predict(X)

acc    = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=['Normal', 'Phishing'])
cm     = confusion_matrix(y_true, y_pred)

lines = [
    "=" * 60,
    "  NeuraShield — Text Model Accuracy Report",
    "=" * 60,
    f"  Model     : TF-IDF + Random Forest",
    f"  Dataset   : Nazario + Enron + SpamAssasin (7 datasets)",
    f"  Samples   : {len(df)}",
    f"  Accuracy  : {acc*100:.2f}%",
    "",
    "Classification Report:",
    report,
    "Confusion Matrix:",
    f"  TN={cm[0][0]}  FP={cm[0][1]}",
    f"  FN={cm[1][0]}  TP={cm[1][1]}",
    "=" * 60
]

output = "\n".join(lines)
print(output)

with open("outputs/results/text_model_accuracy.txt", "w") as f:
    f.write(output)

print("\nSaved to: outputs/results/text_model_accuracy.txt")
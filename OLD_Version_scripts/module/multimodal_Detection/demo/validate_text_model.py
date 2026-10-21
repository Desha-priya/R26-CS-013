# validate_text_model.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime

print("="*70)
print("NeuraShield - Text Phishing Model Validation")
print("="*70)

# ================== CONFIG ==================
MODEL_PATH = "models/text/phishing_text_classifier_v3.pkl"
VECTORIZER_PATH = "models/text/tfidf_vectorizer_v3.pkl"
TEST_DATA_PATH = "data/text/processed/phishing_emails_prepared.csv"

# ================== LOG FILE SETUP ==================
Path("outputs/results").mkdir(parents=True, exist_ok=True)

log_file = f"outputs/results/text_model_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log = open(log_file, "w", encoding="utf-8")

# Redirect print to both terminal + file
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log)

# ================== MODEL LOAD ==================
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print(f"Loaded model: {MODEL_PATH}")

# ================== DATA LOAD ==================
df = pd.read_csv(TEST_DATA_PATH)
print(f"Loaded {len(df)} emails for validation")

text_column = 'email_text'
X = vectorizer.transform(df[text_column].astype(str))
y_true = df['label']

# ================== PREDICTION ==================
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1] if len(model.classes_) == 2 else None

# ================== METRICS ==================
accuracy = accuracy_score(y_true, y_pred)

print("\n" + "="*60)
print("MODEL VALIDATION RESULTS")
print("="*60)
print(f"Accuracy          : {accuracy:.4f} ({accuracy*100:.2f}%)")

if y_prob is not None:
    auc = roc_auc_score(y_true.map({'normal':0, 'phishing':1}), y_prob)
    print(f"AUC Score         : {auc:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred))

# ================== CONFUSION MATRIX ==================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Phishing'],
            yticklabels=['Normal', 'Phishing'])
plt.title('Text Phishing Detection - Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

Path("outputs/results").mkdir(parents=True, exist_ok=True)
plt.savefig('outputs/results/text_validation_confusion_matrix.png')

print("\nConfusion matrix saved to outputs/results/")

# ================== SAMPLE PREDICTIONS ==================
print("\nSample Predictions:")
sample_df = df.sample(5, random_state=42).copy()
sample_df['predicted'] = model.predict(vectorizer.transform(sample_df[text_column].astype(str)))
print(sample_df[['email_text', 'label', 'predicted']].head())

# ================== RESTORE OUTPUT ==================
sys.stdout = sys.__stdout__
log.close()

print(f"\n Full log saved to: {log_file}")
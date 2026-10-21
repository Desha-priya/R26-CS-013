# train_text_phishing_final.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Final Improved Text Phishing Training with Validation ===")

# Load dataset
df = pd.read_csv("data/text/processed/phishing_emails_prepared.csv")

text_column = 'email_text'
df = df.dropna(subset=[text_column]).copy()
df[text_column] = df[text_column].astype(str)

print(f"Total samples: {len(df)}")
print(df['label'].value_counts())

# === Dataset Validation & Analysis ===
print("\n=== Dataset Validation ===")
df['text_length'] = df[text_column].str.len()
print("Average email length:", df['text_length'].mean().round(1))
print("Phishing avg length:", df[df['label']=='phishing']['text_length'].mean().round(1))
print("Normal avg length:", df[df['label']=='normal']['text_length'].mean().round(1))

# Keep URLs (important for phishing)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()   # only clean extra spaces
    return text.lower()

df[text_column] = df[text_column].apply(clean_text)

# === Feature Extraction ===
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),      # unigrams + bigrams
    min_df=2,
    max_df=0.85
)

X = vectorizer.fit_transform(df[text_column])
y = df['label']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# === Model Training ===
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# === Model Validation ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("FINAL MODEL EVALUATION")
print("="*70)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Phishing'], 
            yticklabels=['Normal', 'Phishing'])
plt.title('Text Phishing Detection Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('outputs/results/text_confusion_matrix.png')
print("Confusion matrix saved!")

# Save models
joblib.dump(model, "models/text/phishing_text_classifier_v3.pkl")
joblib.dump(vectorizer, "models/text/tfidf_vectorizer_v3.pkl")

print("\n✅ Final models saved as v3")
print("Ready for fusion!")
# train_text_phishing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

print("Loading prepared phishing email dataset...")

#root_dir= Path(__file__).resolve().parent

df = pd.read_csv("data\\text\\processed\\phishing_emails_prepared.csv")

# Use the email text column
text_column = 'email_text' if 'email_text' in df.columns else 'text_combined' if 'text_combined' in df.columns else df.columns[0]

print(f"Using column '{text_column}' as email text")
print(f"Total samples: {len(df)}")
print(df['label'].value_counts())

# Clean and prepare
df = df.dropna(subset=[text_column])
df[text_column] = df[text_column].astype(str)

# Convert text to features
print("Converting text to numerical features using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df[text_column])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Train model
print("Training Random Forest classifier on text data...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print("TEXT PHISHING CLASSIFIER TRAINING COMPLETE!")
print("="*60)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "models/text/phishing_text_classifier.pkl")
joblib.dump(vectorizer, "models/text/tfidf_vectorizer.pkl")

print("\nModels saved successfully!")
print("   - models/text/phishing_text_classifier.pkl")
print("   - models/text/tfidf_vectorizer.pkl")
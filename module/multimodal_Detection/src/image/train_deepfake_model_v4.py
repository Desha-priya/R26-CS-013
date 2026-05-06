# train_deepfake_model_v4.py

# Train model using 600 videos with improved features

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading improved features from 600 videos...")

df = pd.read_csv("data/video/processed/improved_visual_features_v3.csv")

print(f"Loaded {len(df)} samples")
print(df['label'].value_counts())

# Prepare features and labels
X = df.drop(columns=['file', 'label', 'modality'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train stronger model
model = RandomForestClassifier(
    n_estimators=400,          # More trees = better accuracy
    random_state=42,
    class_weight='balanced',   # Handle slight imbalance
    n_jobs=-1                  # Use all CPU cores
)

print("Training model... (this may take 10-30 seconds)")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)
print(f"Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/image/deepfake_model_v4.pkl")
print("\nModel saved as: models/image/deepfake_model_v4.pkl")

# Optional: Save confusion matrix image
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'], 
            yticklabels=['Fake', 'Real'])
plt.title('Deepfake Detection Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('outputs/results/confusion_matrix_deepfake.png')
print("Confusion matrix saved as: outputs/results/confusion_matrix_deepfake.png")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the new improved features
df = r"src\image\improved_visual_features_v2.csv"
df = pd.read_csv(df)


print(f"Loaded {len(df)} samples")
print("Label distribution:")
print(df['label'].value_counts())

# Prepare data
X = df.drop(columns=['file', 'label', 'modality'])
y = df['label']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/image/improved_deepfake_model.pkl")
print("\nModel saved as: improved_deepfake_model.pkl")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the new improved features
df = r"src\image\improved_visual_features_v2.csv"
df = pd.read_csv(df)

print(f"Loaded {len(df)} samples")
print(df['label'].value_counts())

X = df.drop(columns=['file', 'label', 'modality'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y   # changed to 25% test
)

print(f"Training: {len(X_train)}, Test: {len(X_test)}")

# Better model with more trees + class balancing
model = RandomForestClassifier(
    n_estimators=300, 
    random_state=42,
    class_weight='balanced'   # Helps with imbalance
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nNew Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/image/improved_deepfake_model_v3.pkl")
print("Model saved as: improved_deepfake_model_v3.pkl")
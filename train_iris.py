# train_iris.py
# Training script for Iris classification using Random Forest

import joblib
import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("📊 Data loaded successfully!")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=5
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model trained!")
print(f"   Accuracy: {accuracy:.4f}")

# Log with MLflow
mlflow.log_params({
    "n_estimators": 100,
    "random_state": 42,
    "max_depth": 5,
    "test_size": 0.2
})

mlflow.log_metrics({
    "accuracy": accuracy,
    "train_samples": len(X_train),
    "test_samples": len(X_test)
})

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")

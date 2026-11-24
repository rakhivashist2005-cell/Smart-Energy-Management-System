
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("Loading training and testing datasets...")

X_train = pd.read_csv("data/processed/X_train_theft.csv")
X_test = pd.read_csv("data/processed/X_test_theft.csv")

y_train = pd.read_csv("data/processed/y_train_theft.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test_theft.csv").squeeze()

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel training completed!")

print("\nMaking predictions...")

y_pred = model.predict(X_test)

print("\nModel Evaluation Results")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "models/electricity_theft_model.pkl")

print("\nModel saved successfully!")

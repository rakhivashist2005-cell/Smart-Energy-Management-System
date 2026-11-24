
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

print("Loading datasets...")

X_train = pd.read_csv("data/processed/X_train_theft.csv")
X_test = pd.read_csv("data/processed/X_test_theft.csv")

y_train = pd.read_csv("data/processed/y_train_theft.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test_theft.csv").squeeze()

print("\nTraining final XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# Final threshold
threshold = 0.50

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

print("\nFINAL MODEL RESULTS")

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, zero_division=0))
print("Recall:", recall_score(y_test, predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, predictions, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, probabilities))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "models/final_electricity_theft_model.pkl")

print("\nFinal XGBoost model saved successfully!")

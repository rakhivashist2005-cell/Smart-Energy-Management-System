
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from xgboost import XGBClassifier

print("Loading training dataset...")

X = pd.read_csv("data/processed/X_train_theft.csv")
y = pd.read_csv("data/processed/y_train_theft.csv").squeeze()

# Create validation data from training data only
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining XGBoost model...")

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

# Get theft probabilities
probabilities = model.predict_proba(X_val)[:, 1]

print("\nTHRESHOLD COMPARISON")
print("-" * 65)
print(f"{'Threshold':<15}{'Precision':<15}{'Recall':<15}{'F1 Score':<15}")

best_threshold = 0
best_f1 = 0

for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    print(
        f"{threshold:<15.2f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{f1:<15.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest threshold based on validation F1:", best_threshold)
print("Best validation F1 score:", best_f1)


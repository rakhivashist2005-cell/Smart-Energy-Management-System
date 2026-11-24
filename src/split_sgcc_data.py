
import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading feature dataset...")

df = pd.read_csv("data/processed/sgcc_features.csv")

# Separate input features and target
X = df.drop(columns=["CONS_NO", "FLAG"])
y = df["FLAG"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset splitting completed!")

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

# Save split datasets
X_train.to_csv("data/processed/X_train_theft.csv", index=False)
X_test.to_csv("data/processed/X_test_theft.csv", index=False)

y_train.to_csv("data/processed/y_train_theft.csv", index=False)
y_test.to_csv("data/processed/y_test_theft.csv", index=False)

print("\nAll train-test files saved successfully!")

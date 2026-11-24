
import pandas as pd

# Load updated dataset
df = pd.read_csv(
    "data/processed/prediction_data.csv",
    index_col="Datetime",
    parse_dates=True
)

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Chronological 80-20 split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Save datasets
X_train.to_csv("data/processed/X_train.csv")
X_test.to_csv("data/processed/X_test.csv")

y_train.to_csv("data/processed/y_train.csv")
y_test.to_csv("data/processed/y_test.csv")

print("Updated data splitting completed!")

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("\nTraining Target:", y_train.shape)
print("Testing Target:", y_test.shape)

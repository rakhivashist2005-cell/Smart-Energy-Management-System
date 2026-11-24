
import pandas as pd
import numpy as np

print("Loading cleaned SGCC dataset...")

df = pd.read_csv("data/processed/sgcc_cleaned.csv")

# Separate labels and consumption readings
labels = df[["CONS_NO", "FLAG"]].copy()
consumption = df.drop(columns=["CONS_NO", "FLAG"])

print("Creating statistical features...")

features = pd.DataFrame()

# Basic statistical features
features["average_consumption"] = consumption.mean(axis=1)
features["total_consumption"] = consumption.sum(axis=1)
features["std_consumption"] = consumption.std(axis=1)
features["min_consumption"] = consumption.min(axis=1)
features["max_consumption"] = consumption.max(axis=1)
features["median_consumption"] = consumption.median(axis=1)

# Consumption behavior
features["zero_consumption_days"] = (consumption == 0).sum(axis=1)

features["high_consumption_days"] = (
    consumption > consumption.mean(axis=1).values[:, None]
).sum(axis=1)

# Coefficient of variation
features["consumption_variation"] = (
    features["std_consumption"] /
    features["average_consumption"].replace(0, np.nan)
)

features["consumption_variation"] = (
    features["consumption_variation"].fillna(0)
)

# Combine features and target
final_data = pd.concat(
    [labels, features],
    axis=1
)

# Save feature dataset
final_data.to_csv(
    "data/processed/sgcc_features.csv",
    index=False
)

print("\nFeature engineering completed!")
print("Final dataset shape:", final_data.shape)

print("\nGenerated features:")
print(final_data.columns.tolist())

print("\nDataset preview:")
print(final_data.head())

print("\nSaved: data/processed/sgcc_features.csv")

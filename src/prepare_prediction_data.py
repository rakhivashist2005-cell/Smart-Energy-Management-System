
import pandas as pd
import numpy as np

# Load processed dataset
df = pd.read_csv(
    "data/processed/energy_data.csv",
    index_col="Datetime",
    parse_dates=True
)

# Convert minute-level data into hourly averages
hourly = df.resample("h").mean()

power = hourly["Global_active_power"]

# Create feature dataset
data = pd.DataFrame(index=hourly.index)

# Consumption features
data["current_power"] = power
data["lag_1"] = power.shift(1)
data["lag_2"] = power.shift(2)
data["lag_3"] = power.shift(3)
data["lag_24"] = power.shift(24)

# Rolling average features
data["rolling_3h"] = power.rolling(3).mean()
data["rolling_24h"] = power.rolling(24).mean()
data["rolling_7d"] = power.rolling(168).mean()

# Time-based features
data["hour"] = data.index.hour
data["day_of_week"] = data.index.dayofweek
data["month"] = data.index.month

# Cyclical time features
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

# Predict next hour's consumption
data["target"] = power.shift(-1)

# Remove missing values
data = data.dropna()

# Save improved dataset
data.to_csv("data/processed/prediction_data.csv")

print("Feature engineering completed!")
print("\nNew Dataset Shape:", data.shape)
print("\nFeatures:")
print(data.columns.tolist())

print("\nFirst 5 Rows:")
print(data.head())

print("\nSaved successfully!")

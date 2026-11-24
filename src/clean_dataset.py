
import pandas as pd
import os

file_path = "data/raw/household_power_consumption.txt"

# Load original dataset
df = pd.read_csv(
    file_path,
    sep=";",
    na_values=["?"],
    low_memory=False
)

print("Original dataset shape:", df.shape)

# Select electricity measurement columns
measurement_columns = df.columns[2:]

# Remove rows where all measurement values are missing
df_clean = df.dropna(
    subset=measurement_columns,
    how="all"
)

print("Cleaned dataset shape:", df_clean.shape)

# Save cleaned dataset
os.makedirs("data/processed", exist_ok=True)

df_clean.to_csv(
    "data/processed/cleaned_energy_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")


import pandas as pd

file_path = "data/raw/household_power_consumption.txt"

# Load dataset
df = pd.read_csv(
    file_path,
    sep=";",
    na_values=["?"],
    low_memory=False
)

print("Original dataset shape:", df.shape)

# Remove rows where all 7 measurement values are missing
measurement_columns = df.columns[2:]

df_clean = df.dropna(
    subset=measurement_columns,
    how="all"
)

print("Cleaned dataset shape:", df_clean.shape)

# Save cleaned dataset
df_clean.to_csv(
    "data/processed/cleaned_energy_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

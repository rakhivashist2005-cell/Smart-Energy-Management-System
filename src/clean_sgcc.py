
import pandas as pd

# Load original dataset
file_path = "data/raw/sgcc/sgcc_data.csv"

print("Loading SGCC dataset...")
df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# Separate consumer information and consumption readings
cons_info = df[["CONS_NO", "FLAG"]].copy()
consumption = df.drop(columns=["CONS_NO", "FLAG"])

# Remove consumers having no readings at all
missing_per_row = consumption.isnull().sum(axis=1)
valid_rows = missing_per_row < len(consumption.columns)

cons_info = cons_info.loc[valid_rows].reset_index(drop=True)
consumption = consumption.loc[valid_rows].reset_index(drop=True)

print("Shape after removing completely missing consumers:",
      len(consumption))

# Fill missing values using each consumer's median
print("\nFilling missing values...")
row_medians = consumption.median(axis=1)

consumption = consumption.T.fillna(row_medians).T

# Combine consumer information and readings
cleaned_df = pd.concat([cons_info, consumption], axis=1)

# Save cleaned dataset
output_path = "data/processed/sgcc_cleaned.csv"
cleaned_df.to_csv(output_path, index=False)

print("\nFinal shape:", cleaned_df.shape)
print("Remaining missing values:", cleaned_df.isnull().sum().sum())

print("\nUpdated theft label distribution:")
print(cleaned_df["FLAG"].value_counts())

print("\nCleaned SGCC dataset saved successfully!")


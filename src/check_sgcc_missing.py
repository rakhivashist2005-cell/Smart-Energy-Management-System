
import pandas as pd

file_path = "data/raw/sgcc/sgcc_data.csv"

print("Loading SGCC dataset...")
df = pd.read_csv(file_path)

# Separate consumption columns
consumption_data = df.drop(columns=["CONS_NO", "FLAG"])

# Missing values per consumer
missing_per_row = consumption_data.isnull().sum(axis=1)

print("\n1. Missing values per consumer:")
print(missing_per_row.describe())

print("\n2. Consumers with completely missing readings:")
print((missing_per_row == len(consumption_data.columns)).sum())

print("\n3. Consumers with no missing readings:")
print((missing_per_row == 0).sum())

print("\n4. Missing values by theft category:")
print(
    df.assign(missing_count=missing_per_row)
    .groupby("FLAG")["missing_count"]
    .mean()
)

print("\n5. Overall missing percentage:")
print(
    (consumption_data.isnull().sum().sum()
     / consumption_data.size) * 100
)

print("\nMissing value analysis completed!")

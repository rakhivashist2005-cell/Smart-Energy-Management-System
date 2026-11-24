
import pandas as pd

# Dataset path
file_path = "data/raw/sgcc/sgcc_data.csv"

print("Loading SGCC dataset...")

# Read first 5 rows
df = pd.read_csv(file_path, nrows=5)

print("\nDataset Preview:")
print(df.head())

print("\nNumber of Columns:", len(df.columns))

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nSGCC dataset inspection completed!")

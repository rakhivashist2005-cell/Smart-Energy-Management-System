
import pandas as pd

file_path = "data/raw/sgcc/sgcc_data.csv"

print("Loading complete SGCC dataset...")
df = pd.read_csv(file_path)

print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. Total Consumers:")
print(len(df))

print("\n3. Theft Label Distribution:")
print(df["FLAG"].value_counts())

print("\n4. Theft Percentage:")
print(df["FLAG"].value_counts(normalize=True) * 100)

print("\n5. Missing Values:")
print("Total missing values:", df.isnull().sum().sum())

print("\n6. Duplicate Consumer IDs:")
print(df["CONS_NO"].duplicated().sum())

print("\nSGCC analysis completed!")

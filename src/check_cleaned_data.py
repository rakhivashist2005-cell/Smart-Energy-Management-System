import pandas as pd

file_path = "data/processed/cleaned_energy_data.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())
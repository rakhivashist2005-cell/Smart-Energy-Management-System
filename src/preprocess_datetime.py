import pandas as pd

file_path = "data/processed/cleaned_energy_data.csv"

df = pd.read_csv(file_path)

# Combine Date and Time
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True
)

# Sort data by datetime
df = df.sort_values("Datetime")

# Remove old Date and Time columns
df = df.drop(columns=["Date", "Time"])

# Set Datetime as index
df = df.set_index("Datetime")

# Save processed dataset
output_path = "data/processed/energy_data.csv"
df.to_csv(output_path)

print("Datetime preprocessing completed!")
print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset saved successfully at:", output_path)
import pandas as pd

df = pd.read_csv(
    "data/processed/energy_data.csv",
    index_col="Datetime",
    parse_dates=True
)

print("Electricity Consumption Statistical Analysis")

print("\nGlobal Active Power:")
print(df["Global_active_power"].describe())

print("\nMedian Consumption:")
print(df["Global_active_power"].median())

print("\nMaximum Consumption:")
print(df["Global_active_power"].max())

print("\nMinimum Consumption:")
print(df["Global_active_power"].min())
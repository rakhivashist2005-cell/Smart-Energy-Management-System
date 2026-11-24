import pandas as pd
import matplotlib.pyplot as plt

# Load processed dataset
df = pd.read_csv(
    "data/processed/energy_data.csv",
    index_col="Datetime",
    parse_dates=True
)

# Calculate hourly average consumption
hourly_consumption = df["Global_active_power"].resample("h").mean()

# Plot graph
plt.figure(figsize=(12, 5))

plt.plot(hourly_consumption.index, hourly_consumption)

plt.title("Hourly Electricity Consumption")
plt.xlabel("Date")
plt.ylabel("Global Active Power (kW)")
plt.grid(True)

plt.tight_layout()
plt.savefig("reports/hourly_consumption.png")

plt.show()

print("Hourly consumption graph saved successfully!")

# Calculate daily average consumption
daily_consumption = df["Global_active_power"].resample("D").mean()

# Plot daily consumption
plt.figure(figsize=(12, 5))

plt.plot(daily_consumption.index, daily_consumption)

plt.title("Daily Average Electricity Consumption")
plt.xlabel("Date")
plt.ylabel("Global Active Power (kW)")
plt.grid(True)

plt.tight_layout()
plt.savefig("reports/daily_consumption.png")

plt.show()

print("Daily consumption graph saved successfully!")


# Calculate monthly average consumption
monthly_consumption = df["Global_active_power"].resample("ME").mean()

# Plot monthly consumption
plt.figure(figsize=(12, 5))

plt.plot(monthly_consumption.index, monthly_consumption, marker="o")

plt.title("Monthly Average Electricity Consumption")
plt.xlabel("Month")
plt.ylabel("Global Active Power (kW)")
plt.grid(True)

plt.tight_layout()
plt.savefig("reports/monthly_consumption.png")

plt.show()

print("Monthly consumption graph saved successfully!")
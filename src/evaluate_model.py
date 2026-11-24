
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load testing data
X_test = pd.read_csv(
    "data/processed/X_test.csv",
    index_col="Datetime",
    parse_dates=True
)

y_test = pd.read_csv(
    "data/processed/y_test.csv",
    index_col="Datetime",
    parse_dates=True
).squeeze()

# Load model
model = joblib.load("models/improved_energy_prediction_model.pkl")

# Predict
predictions = model.predict(X_test)

# Plot first 200 observations
plt.figure(figsize=(12, 5))

plt.plot(y_test.index[:200], y_test.iloc[:200],
        label="Actual Consumption")

plt.plot(y_test.index[:200], predictions[:200],
        label="Predicted Consumption")

plt.title("Actual vs Predicted Electricity Consumption")
plt.xlabel("Date")
plt.ylabel("Global Active Power (kW)")

# Format date axis
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
    ax.xaxis.get_major_locator()
))

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("reports/actual_vs_predicted.png")
plt.show()

print("Evaluation graph saved successfully!")

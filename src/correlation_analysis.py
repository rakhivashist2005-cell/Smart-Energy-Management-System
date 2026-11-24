import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/processed/energy_data.csv",
    index_col="Datetime",
    parse_dates=True
)

# Calculate correlation
correlation = df.corr()

# Create heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Electricity Consumption Feature Correlation")

plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png")
plt.show()

print("Correlation analysis completed!")
print(correlation["Global_active_power"].sort_values(ascending=False))
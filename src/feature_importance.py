
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load(
    "models/improved_energy_prediction_model.pkl"
)

# Load feature names
X_train = pd.read_csv(
    "data/processed/X_train.csv",
    index_col="Datetime"
)

# Get feature importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

# Sort by importance
importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("Feature Importance:")
print(importance)

# Plot graph
plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("reports/feature_importance.png")
plt.show()

print("\nFeature importance analysis completed!")


import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Load updated training data
X_train = pd.read_csv(
    "data/processed/X_train.csv",
    index_col="Datetime"
)

X_test = pd.read_csv(
    "data/processed/X_test.csv",
    index_col="Datetime"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv",
    index_col="Datetime"
).squeeze()

y_test = pd.read_csv(
    "data/processed/y_test.csv",
    index_col="Datetime"
).squeeze()

# Create improved Random Forest model
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Train model
print("Training improved model...")

model.fit(X_train, y_train)

print("Training completed!")

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nImproved Model Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save improved model separately
joblib.dump(
    model,
    "models/improved_energy_prediction_model.pkl"
)

print("\nImproved model saved successfully!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ---------- Step 1: Load dataset ----------
df = pd.read_csv("augmented_outpatient_data.csv", parse_dates=["Visit_Date"])

# ---------- Step 2: Clean & Feature Engineering ----------
# Temperature average
temp_cols = [f"Temp_Day_-{i}" for i in range(1, 8)]
df["Avg_Temp"] = df[temp_cols].mean(axis=1)

# Daily-level aggregation
daily_df = df.groupby("Visit_Date").agg(
    Outpatient_Count=("Age", "count"),
    Avg_Temp=("Avg_Temp", "mean"),
    avg_Cost=("Cost", "mean"),
    avg_Age=("Age", "mean"),
    location_diversity=("Location", lambda x: x.nunique())
).reset_index()

# Add temporal features
daily_df["is_weekend"] = daily_df["Visit_Date"].dt.weekday >= 5
daily_df["dayofweek"] = daily_df["Visit_Date"].dt.dayofweek

# Prepare for model
daily_df = daily_df.rename(columns={"Visit_Date": "ds", "Outpatient_Count": "y"})

# ---------- Step 3: Select Features ----------
features = ["Avg_Temp", "avg_Cost", "avg_Age", "location_diversity", "is_weekend", "dayofweek"]
X = daily_df[features]
y = daily_df["y"]

# ---------- Step 4: Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ---------- Step 5: Train Multiple Models ----------
models = {
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "LightGBM": LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=5,
        random_state=42
    ),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
}

results = {}

# ---------- Step 6: Train and Evaluate ----------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

    print(f"\n🔧 {name} Model Performance:")
    print(f"🔹 RMSE: {rmse:.2f}")
    print(f"🔹 MAE : {mae:.2f}")
    print(f"🔹 R²  : {r2:.3f}")

# ---------- Step 7: Feature Importance (best model) ----------
best_model_name = min(results, key=lambda k: results[k]["RMSE"])
best_model = models[best_model_name]

print(f"\n📌 Feature importances from best model ({best_model_name}):")
importances = best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else None
if importances is not None:
    for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"{feat}: {imp:.4f}")
else:
    print("This model does not support feature_importances_.")

# ---------- Step 8: Visual RMSE Comparison ----------
fig, ax = plt.subplots(figsize=(10, 5))
model_names = list(results.keys())
rmse_vals = [results[m]["RMSE"] for m in model_names]

ax.bar(model_names, rmse_vals, color="skyblue")
ax.set_title("Model Comparison - RMSE (Lower is Better)")
ax.set_ylabel("RMSE")
plt.tight_layout()
plt.show()

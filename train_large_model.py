# In train_large_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

print("--- Training New Outpatient Model on Large Dataset ---")

# --- Step 1: Load the Large Dataset ---
try:
    df = pd.read_csv('generated_large_outpatient_data.csv', parse_dates=['Visit_Date'])
    print(f"Successfully loaded large dataset with {len(df)} rows.\n")
except FileNotFoundError:
    print("Error: 'generated_large_outpatient_data.csv' not found. Please run generate_data.py first.")
    exit()

# --- Step 2: Aggregate by Day to Get Daily Counts ---
daily_df = df.groupby('Visit_Date').agg(Outpatient_Count=('Patient_ID', 'count'))
print("Aggregated data to get daily visit counts.\n")


# --- Step 3: Feature Engineering ---
print("Creating time-based features...\n")
daily_df['dayofweek'] = daily_df.index.dayofweek
daily_df['month'] = daily_df.index.month
daily_df['is_weekend'] = (daily_df.index.dayofweek >= 5).astype(int)


# --- Step 4: Train the New Model ---
features = ['dayofweek', 'month', 'is_weekend']
target = 'Outpatient_Count'

X = daily_df[features]
y = daily_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)

print("--- Training new model... ---")
model.fit(X_train, y_train)
print("Training complete.\n")

# --- Step 5: Evaluate the New Model ---
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("--- New Model Performance ---")
print(f"Mean Absolute Error: {mae:.2f} patients\n")


# --- Step 6: Save the NEW and IMPROVED Model ---
new_model_filename = 'large_outpatient_model.joblib'
joblib.dump(model, new_model_filename)
print(f"--- POWERFUL Model Saved ---")
print(f"New model saved as '{new_model_filename}'")
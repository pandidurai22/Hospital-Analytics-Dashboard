import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("--- Starting ULTIMATE Outpatient Model Training ---")

# --- Step 1: Load BOTH datasets ---
try:
    visits_df = pd.read_csv('generated_large_outpatient_data.csv', parse_dates=['Visit_Date'])
    patients_df = pd.read_csv('augmented_patients.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# --- Step 2: Standardize all column names ---
visits_df.columns = visits_df.columns.str.strip().str.lower()
patients_df.columns = patients_df.columns.str.strip().str.lower()
print("Standardized all column names to lowercase.")

# --- Step 3: Standardize Patient IDs for merging ---
visits_df['patient_id'] = visits_df['patient_id'].astype(str).str.replace('p', '')
patients_df['patient_id'] = patients_df['patient_id'].astype(str)
print("Standardized 'patient_id' columns for merging.\n")

# --- Step 4: Merge the datasets ---
merged_df = pd.merge(visits_df, patients_df, on='patient_id', how='left')
print(f"Successfully merged data. New dataset has {len(merged_df)} rows.\n")

# --- Step 5: Aggregate by Day ---
print("Aggregating merged data by day...")
daily_df = merged_df.groupby('visit_date').agg(
    outpatient_count=('patient_id', 'count'),
    avg_age=('age', 'mean'),
    avg_cost=('cost', 'mean'),
    avg_satisfaction=('satisfaction_x', 'mean') # Using the correct '_x' column
)
print("Daily aggregation complete.")

# --- THIS IS THE FINAL FIX ---
# Instead of dropping rows with missing data, we fill the missing values.
# We fill any missing average with the overall average of that column.
daily_df['avg_age'].fillna(daily_df['avg_age'].mean(), inplace=True)
daily_df['avg_cost'].fillna(daily_df['avg_cost'].mean(), inplace=True)
daily_df['avg_satisfaction'].fillna(daily_df['avg_satisfaction'].mean(), inplace=True)
print("Handled missing values after aggregation. DataFrame is NOT empty.\n")


# --- Step 6: Feature Engineering & Training ---
daily_df['dayofweek'] = daily_df.index.dayofweek
daily_df['month'] = daily_df.index.month
daily_df['is_weekend'] = (daily_df['dayofweek'] >= 5).astype(int)

features = ['dayofweek', 'month', 'is_weekend', 'avg_age', 'avg_cost', 'avg_satisfaction']
target = 'outpatient_count'

X = daily_df[features]
y = daily_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("--- Training the ultimate model... ---")
model.fit(X_train, y_train)
print("Training complete.\n")

# --- Step 7: Evaluate and Save ---
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("--- Ultimate Model Performance ---")
print(f"Mean Absolute Error: {mae:.2f} patients\n")

ultimate_model_filename = 'ultimate_outpatient_model.joblib'
joblib.dump(model, ultimate_model_filename)
print(f"--- ULTIMATE Model Saved as '{ultimate_model_filename}' ---")
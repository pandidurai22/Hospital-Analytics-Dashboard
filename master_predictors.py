import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. LOAD ALL DATASETS ---
print("Step 1: Loading all datasets...")
patient_df = pd.read_csv("augmented_outpatient_data.csv", parse_dates=["Visit_Date"])
outbreaks_df = pd.read_csv("Outbreak_events.csv", parse_dates=["EVENT_DATE"], date_format="%d-%b-%y")
resources_df = pd.read_csv("Hospital_resources.csv", parse_dates=["RESOURCE_DATE"], date_format="%d-%b-%y")
holidays_df = pd.read_csv("Public_Holiday.csv", parse_dates=["HOLIDAY_DATE"], date_format="%d-%b-%y")

# --- 2. PREPARE THE BASE DAILY DATASET ---
print("Step 2: Creating base daily dataset...")
temp_cols = [f"Temp_Day_-{i}" for i in range(1, 8)]
patient_df["Avg_Temp"] = patient_df[temp_cols].mean(axis=1)

daily_df = patient_df.groupby('Visit_Date').agg(
    patient_count=('Patient_ID', 'count'),
    avg_temp_at_visit=('Avg_Temp', 'mean')
).reset_index().sort_values('Visit_Date') # <-- ENSURE IT'S SORTED HERE

# --- 3. FEATURE ENGINEERING ---
print("Step 3: Engineering features...")

# A. Holiday Features
holidays_df['is_holiday'] = 1
daily_df = pd.merge(daily_df, holidays_df[['HOLIDAY_DATE', 'is_holiday']], left_on='Visit_Date', right_on='HOLIDAY_DATE', how='left')
daily_df['is_holiday'] = daily_df['is_holiday'].fillna(0).astype(int)
daily_df.drop(columns=['HOLIDAY_DATE'], inplace=True)

# B. Outbreak Features
# B. Outbreak Features
OUTBREAK_DURATION_DAYS = 30
daily_df['outbreak_severity_score'] = 0
severity_mapping = {'Moderate': 1, 'Severe': 2}
for _, outbreak in outbreaks_df.iterrows():
    start_date = outbreak['EVENT_DATE']
    end_date = start_date + pd.Timedelta(days=OUTBREAK_DURATION_DAYS)
    severity_score = severity_mapping.get(outbreak['SEVERITY_LEVEL'], 0)
    mask = (daily_df['Visit_Date'] >= start_date) & (daily_df['Visit_Date'] <= end_date)
    # This is the corrected line - no more max() function
    daily_df.loc[mask, 'outbreak_severity_score'] = severity_score

# C. Hospital Resource Features (Corrected Logic)
resources_df['occupancy_rate'] = (resources_df['TOTAL_OCCUPIED'] / resources_df['TOTAL_AVAILABLE']) * 100
resources_pivot = resources_df.pivot_table(
    index='RESOURCE_DATE',
    columns='RESOURCE_TYPE',
    values='occupancy_rate'
).add_suffix('_occupancy_rate').reset_index().sort_values('RESOURCE_DATE') # <-- ENSURE IT'S SORTED HERE

# Use merge_asof, which REQUIRES sorted dataframes
daily_df = pd.merge_asof(
    daily_df,
    resources_pivot,
    left_on='Visit_Date',
    right_on='RESOURCE_DATE',
    direction='backward'
)
daily_df.drop(columns=['RESOURCE_DATE'], inplace=True)

# D. Time-Based Features
daily_df['dayofweek'] = daily_df['Visit_Date'].dt.dayofweek
daily_df['is_weekend'] = (daily_df['dayofweek'] >= 5).astype(int)
daily_df['weekofyear'] = daily_df['Visit_Date'].dt.isocalendar().week.astype(int)
daily_df['month'] = daily_df['Visit_Date'].dt.month

# --- 4. FINAL DATA PREPARATION ---
print("Step 4: Finalizing data...")
daily_df.fillna(method='bfill', inplace=True) # Backfill for the very first rows
daily_df.fillna(0, inplace=True)

target = 'patient_count'
features = [
    'avg_temp_at_visit', 'is_holiday', 'outbreak_severity_score', 
    'bed_occupancy_rate', 'icu_bed_occupancy_rate', 'ventilator_occupancy_rate',
    'dayofweek', 'is_weekend', 'weekofyear', 'month'
]
X = daily_df[features]
y = daily_df[target]

# --- 5. TIME-BASED TRAIN-TEST SPLIT ---
print("Step 5: Splitting data into training and testing sets...")
train_size = int(len(daily_df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# --- 6. TRAIN THE LIGHTGBM MODEL ---
print("Step 6: Training the LightGBM model...")
model = lgb.LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1, num_leaves=20)
model.fit(X_train, y_train)

# --- 7. EVALUATE THE MODEL ---
print("Step 7: Evaluating model performance...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\n--- MODEL PERFORMANCE ---")
print(f"RMSE: {rmse:.2f} patients")
print(f"R-squared (R²): {r2:.2f}")
print("-------------------------\n")

# --- 8. ANALYZE FEATURE IMPORTANCE ---
print("Step 8: Analyzing feature importances...")
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)
print("Top 5 Most Important Features:")
print(importance_df.head())

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Feature Importance for Predicting Patient Load')
plt.tight_layout()
plt.show()

# --- 9. VISUALIZE PREDICTIONS VS ACTUALS ---
print("Step 9: Visualizing predictions...")
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.index = daily_df['Visit_Date'].iloc[train_size:]
plt.figure(figsize=(15, 7))
results_df.plot(style=['-', '--'])
plt.title('Actual vs. Predicted Daily Patient Load')
plt.ylabel('Number of Patients')
plt.legend()
plt.show()
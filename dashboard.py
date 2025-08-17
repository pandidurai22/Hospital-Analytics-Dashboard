import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Predictive Patient Load Dashboard",
    page_icon="🏥",
    layout="wide"
)

# --- Caching Functions for Performance ---
@st.cache_data
def load_data():
    """Loads and preprocesses all necessary data files."""
    patient_df = pd.read_csv("augmented_outpatient_data.csv", parse_dates=["Visit_Date"])
    outbreaks_df = pd.read_csv("Outbreak_events.csv", parse_dates=["EVENT_DATE"], date_format="%d-%b-%y")
    resources_df = pd.read_csv("Hospital_resources.csv", parse_dates=["RESOURCE_DATE"], date_format="%d-%b-%y")
    holidays_df = pd.read_csv("Public_Holiday.csv", parse_dates=["HOLIDAY_DATE"], date_format="%d-%b-%y")
    return patient_df, outbreaks_df, resources_df, holidays_df

@st.cache_resource
def train_model(patient_df, outbreaks_df, resources_df, holidays_df):
    """Trains and returns the LightGBM model, feature list, and last day's data."""
    # --- Feature Engineering ---
    temp_cols = [f"Temp_Day_-{i}" for i in range(1, 8)]
    patient_df["Avg_Temp"] = patient_df[temp_cols].mean(axis=1)

    daily_df = patient_df.groupby('Visit_Date').agg(
        patient_count=('Patient_ID', 'count'),
        avg_temp_at_visit=('Avg_Temp', 'mean')
    ).reset_index().sort_values('Visit_Date')

    holidays_df['is_holiday'] = 1
    daily_df = pd.merge(daily_df, holidays_df[['HOLIDAY_DATE', 'is_holiday']], left_on='Visit_Date', right_on='HOLIDAY_DATE', how='left')
    daily_df['is_holiday'] = daily_df['is_holiday'].fillna(0).astype(int)
    daily_df.drop(columns=['HOLIDAY_DATE'], inplace=True)

    OUTBREAK_DURATION_DAYS = 30
    daily_df['outbreak_severity_score'] = 0
    severity_mapping = {'Moderate': 1, 'Severe': 2}
    for _, outbreak in outbreaks_df.iterrows():
        start_date = outbreak['EVENT_DATE']
        end_date = start_date + pd.Timedelta(days=OUTBREAK_DURATION_DAYS)
        severity_score = severity_mapping.get(outbreak['SEVERITY_LEVEL'], 0)
        mask = (daily_df['Visit_Date'] >= start_date) & (daily_df['Visit_Date'] <= end_date)
        daily_df.loc[mask, 'outbreak_severity_score'] = severity_score

    resources_df['occupancy_rate'] = (resources_df['TOTAL_OCCUPIED'] / resources_df['TOTAL_AVAILABLE']) * 100
    resources_pivot = resources_df.pivot_table(index='RESOURCE_DATE', columns='RESOURCE_TYPE', values='occupancy_rate').add_suffix('_occupancy_rate').reset_index().sort_values('RESOURCE_DATE')
    
    daily_df = pd.merge_asof(daily_df, resources_pivot, left_on='Visit_Date', right_on='RESOURCE_DATE', direction='backward')
    daily_df.drop(columns=['RESOURCE_DATE'], inplace=True)

    daily_df['dayofweek'] = daily_df['Visit_Date'].dt.dayofweek
    daily_df['is_weekend'] = (daily_df['dayofweek'] >= 5).astype(int)
    daily_df['weekofyear'] = daily_df['Visit_Date'].dt.isocalendar().week.astype(int)
    daily_df['month'] = daily_df['Visit_Date'].dt.month
    
    daily_df.fillna(method='bfill', inplace=True)
    daily_df.fillna(0, inplace=True)

    # --- Model Training ---
    features = ['avg_temp_at_visit', 'is_holiday', 'outbreak_severity_score', 'bed_occupancy_rate', 'icu_bed_occupancy_rate', 'ventilator_occupancy_rate', 'dayofweek', 'is_weekend', 'weekofyear', 'month']
    X = daily_df[features]
    y = daily_df['patient_count']
    
    model = lgb.LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1, num_leaves=20)
    model.fit(X, y)
    
    return model, features, daily_df.iloc[-1]

# --- Main App ---
st.title("Future Patient Load Predictor")
st.markdown("This tool uses a machine learning model to forecast the number of outpatient visits.")

# Load data and train model
patient_df, outbreaks_df, resources_df, holidays_df = load_data()
model, features, last_day_data = train_model(patient_df, outbreaks_df, resources_df, holidays_df)

st.sidebar.header("Forecast Settings")
forecast_date = st.sidebar.date_input(
    "Select a Date to Forecast",
    value=datetime.today() + timedelta(days=1),
    min_value=datetime.today()
)

# --- Create Features for the Forecast Date ---
st.header(f"Forecast for: {forecast_date.strftime('%A, %B %d, %Y')}")

forecast_features = {
    'avg_temp_at_visit': last_day_data['avg_temp_at_visit'],
    'is_holiday': 1 if forecast_date in holidays_df['HOLIDAY_DATE'].dt.date else 0,
    'outbreak_severity_score': 0,
    'bed_occupancy_rate': last_day_data['bed_occupancy_rate'],
    'icu_bed_occupancy_rate': last_day_data['icu_bed_occupancy_rate'],
    'ventilator_occupancy_rate': last_day_data['ventilator_occupancy_rate'],
    'dayofweek': forecast_date.weekday(),
    'is_weekend': 1 if forecast_date.weekday() >= 5 else 0,
    'weekofyear': pd.to_datetime(forecast_date).isocalendar().week,
    'month': forecast_date.month
}

forecast_features['avg_temp_at_visit'] = st.sidebar.slider(
    "Expected Average Temperature (°C)", 25.0, 40.0, forecast_features['avg_temp_at_visit'], 0.1
)
forecast_features['outbreak_severity_score'] = st.sidebar.select_slider(
    "Assumed Outbreak Severity", options=[0, 1, 2], value=0, format_func=lambda x: {0: "None", 1: "Moderate", 2: "Severe"}[x]
)

# --- Make Prediction ---
forecast_df = pd.DataFrame([forecast_features], columns=features)
prediction = model.predict(forecast_df)[0]
predicted_count = int(round(prediction))

# --- Display Prediction ---
col1, col2 = st.columns(2)
col1.metric("Predicted Patient Load", f"{predicted_count} Patients")

if predicted_count > 40:
    col1.warning("High patient load expected. Consider allocating extra resources.")
elif predicted_count < 10:
    col1.success("Low patient load expected.")
else:
    col1.info("Normal patient load expected.")

# --- Explain the Prediction ---
with col2:
    st.subheader("Key Factors Influencing Forecast:")
    
    day_type = 'Weekend' if forecast_features['is_weekend'] else 'Weekday'
    st.markdown(f"- **Day of the Week:** {forecast_date.strftime('%A')} ({day_type})")

    st.markdown(f"- **Time of Year:** Week {forecast_features['weekofyear']} (Month: {forecast_features['month']})")
    
    st.markdown(f"- **Expected Temperature:** {forecast_features['avg_temp_at_visit']:.1f}°C")

    holiday_status = 'Yes' if forecast_features['is_holiday'] else 'No'
    st.markdown(f"- **Holiday:** {holiday_status}")

    outbreak_map = {0: "None", 1: "Moderate", 2: "Severe"}
    outbreak_status = outbreak_map[forecast_features['outbreak_severity_score']]
    st.markdown(f"- **Outbreak Status:** {outbreak_status}")

st.info("Note: The model uses the most recent day's hospital occupancy rates as a baseline for the forecast.")
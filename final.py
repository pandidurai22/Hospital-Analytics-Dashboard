# op_predictor_app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

st.title("🩺 OP Visit Predictor")

# --- Load and cache dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("augmented_outpatient_data.csv", parse_dates=["Visit_Date"])
    return df

df = load_data()

# --- Preprocessing ---
df["Day"] = df["Visit_Date"].dt.date

# Average temp over past 7 days
temp_cols = [f"Temp_Day_-{i}" for i in range(1, 8)]
df["Avg_Temp"] = df[temp_cols].mean(axis=1)

# Extract date-based features
df["DayOfWeek"] = df["Visit_Date"].dt.dayofweek
df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
df["Month"] = df["Visit_Date"].dt.month

# --- Group by date ---
daily = df.groupby("Visit_Date").agg({
    "Avg_Temp": "mean",
    "DayOfWeek": "first",
    "Is_Weekend": "first",
    "Month": "first",
    "Location": lambda x: x.mode()[0],
    "Condition": lambda x: x.mode()[0],
    "Procedure": lambda x: x.mode()[0],
    "Gender": lambda x: x.mode()[0],
    "Age": "mean",
    "Patient_ID": "count"
}).reset_index()

daily.rename(columns={"Patient_ID": "Num_OPs"}, inplace=True)

# --- One-hot encode categorical ---
cat_cols = ["Location", "Condition", "Procedure", "Gender"]
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(daily[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

# Final dataset
X = pd.concat([daily[["Avg_Temp", "DayOfWeek", "Is_Weekend", "Month", "Age"]], encoded_df], axis=1)
y = daily["Num_OPs"]

# --- Train model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Accuracy Metrics ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"📊 **Model Accuracy**")
st.write(f"• RMSE: {rmse:.2f}")
st.write(f"• R² Score: {r2:.2f}")

# --- User Input for Prediction ---
st.subheader("📅 Predict OP Count for a New Date")

with st.form("prediction_form"):
    input_date = st.date_input("Select Date", datetime.today())
    input_temp = st.slider("Avg Temperature (past 7 days)", 30.0, 40.0, 36.5)
    input_age = st.slider("Avg Age", 20, 80, 50)
    input_dow = input_date.weekday()
    input_month = input_date.month
    is_weekend = int(input_dow in [5, 6])
    
    input_location = st.selectbox("Location", df["Location"].unique())
    input_condition = st.selectbox("Condition", df["Condition"].unique())
    input_procedure = st.selectbox("Procedure", df["Procedure"].unique())
    input_gender = st.selectbox("Gender", ["Male", "Female"])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # One-hot encode inputs
    input_dict = {
        "Avg_Temp": [input_temp],
        "DayOfWeek": [input_dow],
        "Is_Weekend": [is_weekend],
        "Month": [input_month],
        "Age": [input_age]
    }
    input_df = pd.DataFrame(input_dict)

    # Encode categorical manually
    cat_input = pd.DataFrame([[input_location, input_condition, input_procedure, input_gender]],
                             columns=cat_cols)
    cat_encoded = encoder.transform(cat_input)
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))

    input_final = pd.concat([input_df, cat_encoded_df], axis=1)

    # Align columns (some encodings may not be present in input)
    input_final = input_final.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_final)[0]
    st.success(f"🔮 Predicted OP Count: **{int(round(prediction))} patients**")

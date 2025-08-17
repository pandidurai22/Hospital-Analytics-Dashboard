import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

st.title("🩺 OP Predictor: Daily Patient Volume Estimator")

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("augmented_outpatient_data.csv", parse_dates=["Visit_Date"])
    return df

df = load_data()

# --- Feature Engineering ---
df["Avg_Temp"] = df[[f"Temp_Day_-{i}" for i in range(1, 8)]].mean(axis=1)
df["DayOfWeek"] = df["Visit_Date"].dt.dayofweek
df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
df["Month"] = df["Visit_Date"].dt.month
df["Day"] = df["Visit_Date"].dt.date

# --- Group by Visit_Date ---
grouped = df.groupby("Visit_Date").agg({
    "Avg_Temp": "mean",
    "DayOfWeek": "first",
    "Is_Weekend": "first",
    "Month": "first",
    "Age": "mean",
    "Gender": lambda x: x.mode()[0],
    "Condition": lambda x: x.mode()[0],
    "Procedure": lambda x: x.mode()[0],
    "Location": lambda x: x.mode()[0],
    "Patient_ID": "count"
}).reset_index()

grouped.rename(columns={"Patient_ID": "Num_OPs"}, inplace=True)

# --- One-hot encode categoricals ---
cat_cols = ["Gender", "Condition", "Procedure", "Location"]
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(grouped[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

# Final training data
X = pd.concat([grouped[["Avg_Temp", "DayOfWeek", "Is_Weekend", "Month", "Age"]], encoded_df], axis=1)
y = grouped["Num_OPs"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# --- Show model performance ---
st.markdown("### 📊 Model Performance")
st.write(f"**RMSE**: `{rmse:.2f}` patients")
st.write(f"**R² Score**: `{r2:.2f}`")

# --- Predict only by date ---
st.markdown("### 🧠 Predict OP Visits from Date")

selected_date = st.date_input("📅 Choose a Date", datetime.today())

if selected_date:
    dow = selected_date.weekday()
    weekend = int(dow in [5, 6])
    month = selected_date.month

    # Use average values from training data
    avg_temp = df["Avg_Temp"].mean()
    avg_age = df["Age"].mean()
    common_gender = df["Gender"].mode()[0]
    common_condition = df["Condition"].mode()[0]
    common_procedure = df["Procedure"].mode()[0]
    common_location = df["Location"].mode()[0]

    # Build input row
    input_data = pd.DataFrame([{
        "Avg_Temp": avg_temp,
        "DayOfWeek": dow,
        "Is_Weekend": weekend,
        "Month": month,
        "Age": avg_age,
        "Gender": common_gender,
        "Condition": common_condition,
        "Procedure": common_procedure,
        "Location": common_location
    }])

    # One-hot encode input
    input_encoded = encoder.transform(input_data[cat_cols])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(cat_cols))

    # Final input for prediction
    input_final = pd.concat([input_data[["Avg_Temp", "DayOfWeek", "Is_Weekend", "Month", "Age"]], input_encoded_df], axis=1)
    input_final = input_final.reindex(columns=X.columns, fill_value=0)

    pred = model.predict(input_final)[0]

    # Display prediction
    st.success(f"🔮 Predicted OP Count on {selected_date.strftime('%Y-%m-%d')}: **{int(round(pred))} patients**")

    with st.expander("📌 Why this prediction?"):
        st.write(f"- Day: {selected_date.strftime('%A')} ({'Weekend' if weekend else 'Weekday'})")
        st.write(f"- Month: {month}")
        st.write(f"- Avg Temp (historical mean): {avg_temp:.1f}°C")
        st.write(f"- Avg Age: {avg_age:.1f}")
        st.write(f"- Most common Gender: {common_gender}")
        st.write(f"- Most common Condition: {common_condition}")
        st.write(f"- Most common Procedure: {common_procedure}")
        st.write(f"- Most common Location: {common_location}")

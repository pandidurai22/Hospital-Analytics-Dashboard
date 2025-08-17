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

X = pd.concat([grouped[["Avg_Temp", "DayOfWeek", "Is_Weekend", "Month", "Age"]], encoded_df], axis=1)
y = grouped["Num_OPs"]

# --- Train/test split and model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Accuracy ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.markdown("### 📊 Model Performance")
st.write(f"**RMSE** (Lower is better): `{rmse:.2f}`")
st.write(f"**R² Score** (Closer to 1 is better): `{r2:.2f}`")

# --- Predict on user input ---
st.markdown("### 🧠 Predict OP Visits for a Date")

with st.form("predict_form"):
    selected_date = st.date_input("Select Date", datetime.today())
    avg_temp = st.slider("Expected Avg Temperature", 30.0, 40.0, 36.0)
    avg_age = st.slider("Average Age", 20, 80, 45)
    gender = st.selectbox("Dominant Gender", df["Gender"].unique())
    condition = st.selectbox("Most Common Condition", df["Condition"].unique())
    procedure = st.selectbox("Most Common Procedure", df["Procedure"].unique())
    location = st.selectbox("Location", df["Location"].unique())
    
    submitted = st.form_submit_button("Predict OP Count")

if submitted:
    dow = selected_date.weekday()
    weekend = int(dow in [5, 6])
    month = selected_date.month

    input_data = pd.DataFrame([{
        "Avg_Temp": avg_temp,
        "DayOfWeek": dow,
        "Is_Weekend": weekend,
        "Month": month,
        "Age": avg_age,
        "Gender": gender,
        "Condition": condition,
        "Procedure": procedure,
        "Location": location
    }])

    # One-hot encode input
    input_encoded = encoder.transform(input_data[cat_cols])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(cat_cols))

    input_final = pd.concat([input_data[["Avg_Temp", "DayOfWeek", "Is_Weekend", "Month", "Age"]], input_encoded_df], axis=1)
    input_final = input_final.reindex(columns=X.columns, fill_value=0)

    pred = model.predict(input_final)[0]
    st.success(f"🔮 Predicted OP Count: **{int(round(pred))} patients**")

    st.markdown("#### 📌 Why this prediction?")
    st.write(f"- **Weather** (Avg Temp): {avg_temp}°C")
    st.write(f"- **Day**: {'Weekend' if weekend else 'Weekday'} ({selected_date.strftime('%A')})")
    st.write(f"- **Month**: {month}")
    st.write(f"- **Condition**: {condition}")
    st.write(f"- **Procedure**: {procedure}")
    st.write(f"- **Location**: {location}")
    st.write(f"- **Gender**: {gender}, **Age**: {avg_age}")

    st.caption("Prediction based on historical patterns with similar features.")

import streamlit as st
import pandas as pd
import joblib
import datetime

# --- LOAD THE ULTIMATE OUTPATIENT MODEL ---
# Use st.cache_resource to load the model only once and save memory
@st.cache_resource
def load_model():
    """Loads the trained model from the file."""
    try:
        model = joblib.load('ultimate_outpatient_model.joblib')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- WEB PAGE INTERFACE ---
st.set_page_config(page_title="Outpatient Visit Predictor", layout="wide")

st.title("📅 Daily Outpatient Visit Predictor (Ultimate Model)")
st.write("""
This tool predicts the total number of outpatient visits for a given day. 
This powerful model uses historical data including visit dates, patient age, procedure cost, and satisfaction. 
Simply select a date to get a prediction!
""")
st.markdown("---")


# --- USER INPUTS ---
st.header("Select a Date for Prediction")
selected_date = st.date_input(
    "Date", 
    value=datetime.date.today() + datetime.timedelta(days=1), # Default to tomorrow
    min_value=datetime.date.today()
)

# --- PREDICTION LOGIC ---
if model is not None:
    if st.button("📈 Predict Number of Visits", type="primary"):
        # 1. Create the time-based features from the user's input
        dayofweek = selected_date.weekday() # Monday=0, Sunday=6
        month = selected_date.month
        is_weekend = 1 if dayofweek >= 5 else 0

        # 2. Create dummy values for the historical average features
        # We use reasonable placeholder averages. The model will rely more heavily on the time-based features.
        avg_age_placeholder = 45.0
        avg_cost_placeholder = 500.0
        avg_satisfaction_placeholder = 3.5

        # 3. Organize features into a pandas DataFrame
        # The columns MUST be in the exact same order as they were during training
        features_for_prediction = ['dayofweek', 'month', 'is_weekend', 'avg_age', 'avg_cost', 'avg_satisfaction']
        
        prediction_input = pd.DataFrame({
            'dayofweek': [dayofweek],
            'month': [month],
            'is_weekend': [is_weekend],
            'avg_age': [avg_age_placeholder],
            'avg_cost': [avg_cost_placeholder],
            'avg_satisfaction': [avg_satisfaction_placeholder]
        })
        
        # Ensure the order is correct
        prediction_input = prediction_input[features_for_prediction]

        # 4. Make the prediction
        prediction = model.predict(prediction_input)
        predicted_visits = int(round(prediction[0]))

        # 5. Display the result
        st.subheader(f"Prediction for {selected_date.strftime('%A, %B %d, %Y')}:")
        st.metric(label="Predicted Outpatient Visits", value=predicted_visits)
        st.info("Please note: This is an estimate based on historical trends and should be used for planning purposes.", icon="💡")
else:
    st.error("Model file not found. Please make sure 'ultimate_outpatient_model.joblib' is in the root directory.", icon="🚨")
# In pages/1_Readmission.py

import streamlit as st
import pandas as pd
import joblib

# Load the Readmission model
@st.cache_resource
def load_model():
    model = joblib.load('readmission_predictor_model.joblib')
    return model
model = load_model()

# Page setup
st.set_page_config(page_title="Readmission Predictor", layout="wide")
st.title('🏥 Patient Readmission Risk Predictor')
st.write("This tool predicts the likelihood of a patient being readmitted.")

# Input columns
col1, col2, col3 = st.columns(3)
with col1:
    st.header("Patient Profile")
    age = st.slider("Age", 1, 100, 50)
    gender = st.selectbox("Gender", ('Male', 'Female'))
    location = st.selectbox("Location", ('Thiruvannamalai', 'Chennai', 'Porur', 'Velachery', 'Anna Nagar', 'Kodambakkam'))
with col2:
    st.header("Medical Details")
    condition = st.selectbox("Condition", ('Diabetes', 'Cancer', 'Heart Attack', 'Hypertension', 'Arthritis', 'Appendicitis', 'Stroke', 'Kidney Stones', 'Childbirth'))
    procedure = st.selectbox("Procedure", ('Insulin Therapy', 'Chemotherapy', 'Angioplasty', 'Medication and Counseling', 'Physical Therapy', 'Appendectomy', 'X-Ray and Splint', 'Lithotripsy', 'Childbirth Delivery and Postnatal Care'))
    cost = st.number_input("Cost of Stay ($)", min_value=0, value=5000)
with col3:
    st.header("Hospital Stay")
    los = st.slider("Length of Stay (days)", 1, 30, 5)
    outcome = st.selectbox("Outcome", ('Recovered', 'Stable', 'Needs Follow-up'))
    satisfaction = st.slider("Patient Satisfaction (1-5)", 1, 5, 4)

# Prediction logic
if st.button('Predict Readmission Risk', type="primary"):
    prediction_input = pd.DataFrame({
        'Age': [age], 'Gender': [gender], 'Condition': [condition], 'Procedure': [procedure],
        'Cost': [cost], 'Length_of_Stay': [los], 'Outcome': [outcome], 'Satisfaction': [satisfaction], 'Location': [location]
    })
    prediction = model.predict(prediction_input)
    prediction_proba = model.predict_proba(prediction_input)

    st.header("Prediction Result")
    if prediction[0] == 'Yes':
        st.error(f"High Risk: The model predicts this patient WILL BE READMITTED.", icon="🚨")
    else:
        st.success(f"Low Risk: The model predicts this patient WILL NOT BE READMITTED.", icon="✅")
    st.write(f"Confidence: **{prediction_proba.max()*100:.0f}%**")
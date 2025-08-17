# In Patient_Predictor_app.py (your main file)

import streamlit as st

st.set_page_config(page_title="Hospital Analytics Dashboard", page_icon="🏥", layout="wide")
st.title("Welcome to the Hospital Analytics Dashboard!")
st.sidebar.success("Select a predictor above.")

st.write("""
This is a multi-page application showcasing two different machine learning models for hospital operations.
**👈 Please select a predictor from the sidebar to get started!**
""")
st.info("""
- **Readmission Predictor:** Predicts the risk of an individual patient being readmitted.
- **Outpatient Predictor:** Forecasts the total number of outpatient visits for a given day.
""")
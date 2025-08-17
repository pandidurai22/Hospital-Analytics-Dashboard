# In pages/7_About.py

import streamlit as st

st.set_page_config(page_title="About")
st.title("About This Project")

st.info("""
This multi-page dashboard is an end-to-end data science project demonstrating the development
of predictive models for hospital operations and their deployment in an interactive web application.
""")

st.subheader("Technologies Used")
st.markdown("""
- **Python:** The core programming language.
- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** For building and training machine learning models.
- **Streamlit:** For creating and serving the interactive web dashboard.
- **Plotly Express:** For generating rich, interactive data visualizations.
""")

st.subheader("Models")
st.markdown("""
- **Readmission Predictor:** A `RandomForestClassifier` trained on patient demographics and stay details to predict the likelihood of readmission.
- **Outpatient Predictor:** A `RandomForestRegressor` trained on temporal data and outbreak events to forecast the number of daily outpatient visits.
""")

st.subheader("Author")
st.write("This project was built by [PANDIDURAI S].")
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, exc

# --- "SMART" DATA LOADER ---
@st.cache_data
def load_performance_data():
    # --- Database Connection Details ---
    DB_PASSWORD = "password"
    DB_NAME = "hospital_db"
    # ... (rest of DB details) ...
    DATABASE_URL = f"postgresql+psycopg2://postgres:{DB_PASSWORD}@localhost:5432/{DB_NAME}"
    
    try:
        engine = create_engine(DATABASE_URL, connect_args={'connect_timeout': 5})
        query = """
        SELECT v.satisfaction, p.cost, ph.physician_name
        FROM visits v
        JOIN patients p ON v.patient_id = p.patient_id
        JOIN physicians ph ON v.physician_id = ph.physician_id;
        """
        df = pd.read_sql(query, engine)
        st.sidebar.success("Live DB Connection", icon="🌐")
        return df
    except (exc.OperationalError, Exception):
        st.sidebar.warning("DB connection failed. Falling back to CSVs.", icon="📁")
        # --- Fallback logic: recreate the join using pandas ---
        visits_df = pd.read_csv("visits_with_physicians.csv")
        patients_df = pd.read_csv("augmented_patients.csv")
        physicians_df = pd.read_csv("physicians.csv")
        
        # Clean columns
        visits_df.columns = [c.lower() for c in visits_df.columns]
        patients_df.columns = [c.lower() for c in patients_df.columns]
        physicians_df.columns = [c.lower() for c in physicians_df.columns]

        # Merge
        merged = pd.merge(visits_df, patients_df, on='patient_id')
        final_df = pd.merge(merged, physicians_df, on='physician_id')
        return final_df

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide")
st.title("👨‍⚕️ Physician Performance Dashboard")
df = load_performance_data()

# The rest of your page code remains the same...
if not df.empty:
    all_physicians = sorted(df['physician_name'].unique())
    selected_physicians = st.sidebar.multiselect('Select Physicians:', options=all_physicians, default=all_physicians)
    df_selection = df[df['physician_name'].isin(selected_physicians)] if selected_physicians else df

    if not df_selection.empty:
        performance_summary = df_selection.groupby('physician_name').agg(
            total_patients=('satisfaction', 'count'),
            avg_patient_satisfaction=('satisfaction', 'mean'),
            avg_cost_of_care=('cost', 'mean')
        ).reset_index()
        st.dataframe(performance_summary)
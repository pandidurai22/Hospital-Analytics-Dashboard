import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, exc
from datetime import datetime, date

st.set_page_config(layout="wide")
st.title("🔬 Patient Visit Analytics Deep Dive (Live Database)")
st.markdown("This page joins data from three database tables for comparative analysis.")

# --- Database Connection and Data Loading (same as before) ---
@st.cache_resource
def get_db_engine():
    DB_PASSWORD = "password" # !!! IMPORTANT: Change to your password !!!
    DATABASE_URL = f"postgresql+psycopg2://postgres:{DB_PASSWORD}@localhost:5432/hospital_db"
    try:
        return create_engine(DATABASE_URL)
    except Exception as e:
        return None

@st.cache_data
def load_and_merge_data_from_db():
    engine = get_db_engine()
    if engine:
        try:
            query = """
            SELECT v.visit_date, p.age, p.gender, p.location, p.condition
            FROM visits v JOIN patients p ON v.patient_id = p.patient_id;
            """
            df = pd.read_sql(query, engine)
            df['visit_date'] = pd.to_datetime(df['visit_date'])
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

df = load_and_merge_data_from_db()

if not df.empty:
    st.sidebar.header("Filter by Time Period:")
    
    # --- NEW: COMPARATIVE TIME PERIOD FILTERS ---
    today = date.today()
    # Current Period Selector
    st.sidebar.subheader("Select Current Period")
    current_start_date = st.sidebar.date_input("Start Date (Current)", today.replace(day=1))
    current_end_date = st.sidebar.date_input("End Date (Current)", today)

    # Comparison Period Selector
    st.sidebar.subheader("Select Comparison Period")
    compare_start_date = st.sidebar.date_input("Start Date (Comparison)", (today.replace(day=1) - pd.DateOffset(months=1)).date())
    compare_end_date = st.sidebar.date_input("End Date (Comparison)", (today - pd.DateOffset(months=1)).date())

    # Filter data for each period
    df_current = df[df['visit_date'].dt.date.between(current_start_date, current_end_date)]
    df_compare = df[df['visit_date'].dt.date.between(compare_start_date, compare_end_date)]

    st.markdown("---")
    st.subheader("Comparative Performance Metrics")

    # --- CALCULATE AND DISPLAY METRICS ---
    # Metric 1: Total Visits
    current_visits = df_current.shape[0]
    compare_visits = df_compare.shape[0]
    # Calculate percentage change, handling division by zero
    delta_visits = ((current_visits - compare_visits) / compare_visits * 100) if compare_visits > 0 else 0

    # Metric 2: Average Patient Age
    current_avg_age = df_current['age'].mean() if not df_current.empty else 0
    compare_avg_age = df_compare['age'].mean() if not df_compare.empty else 0
    delta_age = ((current_avg_age - compare_avg_age) / compare_avg_age * 100) if compare_avg_age > 0 else 0

    # Display in columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label=f"Total Visits ({current_start_date} to {current_end_date})",
            value=f"{current_visits:,}",
            delta=f"{delta_visits:.2f}% vs. Comparison Period"
        )
    with col2:
        st.metric(
            label=f"Average Patient Age ({current_start_date} to {current_end_date})",
            value=f"{current_avg_age:.1f}",
            delta=f"{delta_age:.2f}% vs. Comparison Period"
        )
        
    st.markdown("---")
    st.subheader("Breakdown for Current Period")

    # --- CHARTS FOR THE CURRENT PERIOD ---
    if df_current.empty:
        st.warning("No data available for the selected Current Period.")
    else:
        col3, col4 = st.columns(2)
        with col3:
            visits_over_time = df_current.set_index('visit_date').resample('D').size().reset_index(name='count')
            fig_trend = px.line(visits_over_time, x='visit_date', y='count', title='Daily Visit Trend (Current Period)')
            st.plotly_chart(fig_trend, use_container_width=True)
        with col4:
            fig_conditions = px.pie(df_current, names='condition', title='Medical Conditions (Current Period)', hole=0.4)
            st.plotly_chart(fig_conditions, use_container_width=True)
else:
    st.error("Could not load data from the database.")
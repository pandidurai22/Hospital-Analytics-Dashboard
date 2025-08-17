import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime

# --- Database Connection ---
@st.cache_resource
def get_db_engine():
    # !!! IMPORTANT: Change this to your actual PostgreSQL password !!!
    DB_PASSWORD = "password" 
    DB_NAME = "hospital_db"
    DB_USER = "postgres"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        st.error(f"DB connection failed: {e}")
        return None

engine = get_db_engine()

# --- Load and Merge Data from DATABASE using a 3-TABLE JOIN ---
@st.cache_data
def load_and_merge_data_from_db():
    if engine is not None:
        try:
            # This is our most advanced query. It joins visits, patients, and physicians.
            query = """
            SELECT 
                v.visit_date,
                v.satisfaction,
                p.age,
                p.gender,
                p.location,
                p.condition,
                p.cost,
                ph.physician_name,
                ph.specialty
            FROM visits v
            JOIN patients p ON v.patient_id = p.patient_id
            JOIN physicians ph ON v.physician_id = ph.physician_id;
            """
            df = pd.read_sql(query, engine)
            df['visit_date'] = pd.to_datetime(df['visit_date'])
            return df
        except Exception as e:
            st.error(f"Could not read and join tables from the database: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide")
st.title("🔬 Patient Visit Analytics Deep Dive (Live Database)")
st.markdown("This page joins data from three database tables to provide a complete analytical view.")

df = load_and_merge_data_from_db()

# Check if the required columns for filtering exist
required_columns = ['visit_date', 'age', 'location', 'gender']
if not df.empty and all(col in df.columns for col in required_columns):
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Combined Filters:")
    min_date = df['visit_date'].min().date()
    max_date = df['visit_date'].max().date()
    selected_date_range = st.sidebar.date_input("Select Visit Date Range:", value=(min_date, max_date), min_value=min_date, max_value=max_date, format="YYYY-MM-DD")
    
    min_age_val = int(df['age'].min())
    max_age_val = int(df['age'].max())
    age_range = st.sidebar.slider('Select Age Range:', min_value=min_age_val, max_value=max_age_val, value=(min_age_val, max_age_val))

    all_locations = sorted(df['location'].unique())
    selected_locations = st.sidebar.multiselect('Select Locations:', options=all_locations, default=all_locations)
    
    all_genders = sorted(df['gender'].unique())
    selected_gender = st.sidebar.selectbox('Select Gender:', options=['All'] + list(all_genders), index=0)

    # --- FILTERING ---
    df_selection = df.copy()
    if len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        df_selection = df_selection[df_selection['visit_date'].dt.date.between(start_date, end_date)]
    if not df_selection.empty:
        df_selection = df_selection[df_selection['age'].between(age_range[0], age_range[1])]
    if not df_selection.empty and selected_locations:
        df_selection = df_selection[df_selection['location'].isin(selected_locations)]
    if not df_selection.empty and selected_gender != 'All':
        df_selection = df_selection[df_selection['gender'] == selected_gender]

    # --- DISPLAY ---
    st.markdown("---")
    if df_selection.empty:
        st.warning("No patient visits match the current filter criteria.")
    else:
        st.metric(label="Total Visits Matching Filters", value=f"{df_selection.shape[0]:,}")
        st.markdown("---")
        st.subheader("Analytics for Filtered Patient Visits")
        col1, col2 = st.columns(2)
        with col1:
            visits_over_time = df_selection.set_index('visit_date').resample('D').size().reset_index(name='count')
            fig_trend = px.line(visits_over_time, x='visit_date', y='count', title='Daily Visit Trend for Filtered Group')
            st.plotly_chart(fig_trend, use_container_width=True)
        with col2:
            fig_conditions = px.pie(df_selection, names='condition', title='Breakdown of Medical Conditions for Filtered Group', hole=0.4)
            st.plotly_chart(fig_conditions, use_container_width=True)
else:
    st.error("Could not load and process data from the database. Please ensure all tables (visits, patients, physicians) exist and the 'data_to_db.py' script ran successfully.")
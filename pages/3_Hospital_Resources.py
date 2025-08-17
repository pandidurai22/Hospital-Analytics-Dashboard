import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, exc
from datetime import datetime

# --- "SMART" DATA LOADER ---
# It tries the database first, then falls back to CSV.
@st.cache_data
def load_data():
    # --- Database Connection Details ---
    # !!! IMPORTANT: Change this to your actual PostgreSQL password !!!
    DB_PASSWORD = "password" 
    DB_NAME = "hospital_db"
    DB_USER = "postgres"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    try:
        # --- Try to connect to the database FIRST ---
        engine = create_engine(DATABASE_URL, connect_args={'connect_timeout': 5})
        query = "SELECT * FROM resources;"
        df = pd.read_sql(query, engine)
        df['resource_date'] = pd.to_datetime(df['resource_date'])
        # st.sidebar.success("Live DB Connection", icon="🌐") # Optional: uncomment for local debugging
        return df
    except (exc.OperationalError, Exception):
        # --- If DB connection fails, FALL BACK to CSV ---
        # st.sidebar.warning("DB connection failed. Falling back to CSV.", icon="📁") # Optional: uncomment for local debugging
        try:
            df = pd.read_csv("Hospital_resources.csv")
            # Ensure date format is correct for the CSV file
            df['RESOURCE_DATE'] = pd.to_datetime(df['RESOURCE_DATE'].str.upper(), format='%d-%b-%y', errors='coerce')
            # Ensure column names match the database version (lowercase)
            df.columns = [col.strip().lower() for col in df.columns]
            return df
        except FileNotFoundError:
            st.error("Fallback file 'Hospital_resources.csv' not found.")
            return pd.DataFrame()

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide")
st.title("🏥 Hospital Resource Dashboard")
st.markdown("This dashboard provides a real-time or file-based overview of hospital resources.")

df = load_data()
df.dropna(subset=['resource_date'], inplace=True)

if not df.empty:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter Resources Here:")
    min_date = df['resource_date'].min().date()
    max_date = df['resource_date'].max().date()
    selected_date_range = st.sidebar.date_input("Select Date Range:", value=(min_date, max_date), min_value=min_date, max_value=max_date, format="YYYY-MM-DD")
    
    all_wards = sorted(df['ward'].unique())
    selected_wards = st.sidebar.multiselect("Select Ward(s):", options=all_wards, default=all_wards)
    
    all_resources = sorted(df['resource_type'].unique())
    selected_resources = st.sidebar.multiselect("Select Resource Type(s):", options=all_resources, default=all_resources)
    
    # --- FILTERING ---
    df_selection = df.copy()
    if len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        df_selection = df_selection[df_selection['resource_date'].dt.date.between(start_date, end_date)]
    if selected_wards:
        df_selection = df_selection[df_selection['ward'].isin(selected_wards)]
    if selected_resources:
        df_selection = df_selection[df_selection['resource_type'].isin(selected_resources)]
    
    # --- DISPLAY ---
    st.markdown("---")
    if df_selection.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.markdown("### Resource Status for Selected Filters")
        df_display = df_selection.copy()
        df_display['resource_date'] = df_display['resource_date'].dt.strftime('%Y-%m-%d')
        st.dataframe(df_display[['resource_date', 'hospital_id', 'ward', 'resource_type', 'total_available', 'total_occupied']])

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Resource Availability by Type")
            availability_fig = px.bar(
                df_selection.groupby(['ward', 'resource_type'])['total_available'].sum().reset_index(),
                x='ward', y='total_available', color='resource_type', title='Quantity of Resources by Ward')
            st.plotly_chart(availability_fig, use_container_width=True)
        with col2:
            st.markdown("### Resource Utilization")
            df_selection['utilization'] = (df_selection['total_occupied'] / df_selection['total_available'].replace(0, pd.NA)) * 100
            utilization_fig = px.pie(df_selection, names='resource_type', values='utilization', title='Resource Utilization Rate (%)', hole=0.4)
            st.plotly_chart(utilization_fig, use_container_width=True)
else:
    st.error("Could not load resource data from database or CSV file.")
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime

# --- Database Connection ---
@st.cache_resource
def get_db_engine():
    DB_PASSWORD = "password" # !!! IMPORTANT: Change to your password !!!
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

# --- Load Data from DATABASE ---
@st.cache_data
def load_data_from_db():
    if engine is not None:
        try:
            query = "SELECT * FROM resources;"
            df = pd.read_sql(query, engine)
            df['resource_date'] = pd.to_datetime(df['resource_date'])
            return df
        except Exception as e:
            st.error(f"Could not read from 'resources' table: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide")
st.title("🏥 Hospital Resource Dashboard (Live Database)")

df = load_data_from_db()
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
    
    # --- NEW: ADVANCED CHART CONTROLS in sidebar ---
    st.sidebar.header("Advanced Chart Controls:")
    group_by_option = st.sidebar.selectbox(
        "Group Analysis By:",
        ('Ward', 'Resource Type')
    )
    metric_to_analyze = st.sidebar.selectbox(
        "Metric to Analyze:",
        ('Total Available', 'Total Occupied')
    )
    
    # Map the selection to the actual column names
    group_by_col = 'ward' if group_by_option == 'Ward' else 'resource_type'
    metric_col = 'total_available' if metric_to_analyze == 'Total Available' else 'total_occupied'

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
        st.dataframe(df_selection[['resource_date', 'hospital_id', 'ward', 'resource_type', 'total_available', 'total_occupied']])

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### Analysis of **{metric_to_analyze}** by **{group_by_option}**")
            
            # The bar chart is now fully dynamic based on the advanced controls!
            fig_bar = px.bar(
                df_selection.groupby(group_by_col)[metric_col].sum().reset_index(),
                x=group_by_col,
                y=metric_col,
                title=f'{metric_to_analyze} by {group_by_option}'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            st.markdown("### Overall Resource Utilization")
            df_selection['utilization'] = (df_selection['total_occupied'] / df_selection['total_available'].replace(0, pd.NA)) * 100
            fig_pie = px.pie(df_selection, names='resource_type', values='utilization', title='Utilization Rate (%) by Resource Type', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.error("Could not load resource data from the database.")
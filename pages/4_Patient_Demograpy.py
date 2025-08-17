import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, exc

# --- "SMART" DATA LOADER ---
@st.cache_data
def load_data():
    # --- Database Connection Details ---
    DB_PASSWORD = "password" # Your local password
    DB_NAME = "hospital_db"
    DB_USER = "postgres"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    try:
        # --- Try to connect to the database FIRST ---
        engine = create_engine(DATABASE_URL, connect_args={'connect_timeout': 5})
        query = "SELECT * FROM patients;"
        df = pd.read_sql(query, engine)
        st.sidebar.success("Live DB Connection", icon="🌐")
        return df
    except (exc.OperationalError, Exception):
        # --- If DB connection fails, FALL BACK to CSV ---
        st.sidebar.warning("DB connection failed. Falling back to CSV data.", icon="📁")
        df = pd.read_csv("augmented_patients.csv")
        return df

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide")
st.title("👥 Patient Demographics")

df = load_data()

# The rest of your page code remains the same...
if not df.empty and 'age' in df.columns and 'location' in df.columns:
    st.sidebar.header("Filter Patients Here:")
    # ... (all your filter and chart code goes here)
    age_range = st.sidebar.slider('Select Age Range:', min_value=int(df['age'].min()), max_value=int(df['age'].max()), value=(int(df['age'].min()), int(df['age'].max())))
    all_locations = sorted(df['location'].unique())
    selected_locations = st.sidebar.multiselect('Select Locations:', options=all_locations, default=all_locations)
    df_selection = df[df['age'].between(age_range[0], age_range[1])]
    if selected_locations:
        df_selection = df_selection[df_selection['location'].isin(selected_locations)]
    
    st.markdown("---")
    if df_selection.empty:
        st.warning("No patients match filters.")
    else:
        st.metric(label="Total Patients Matching Filters", value=f"{df_selection.shape[0]:,}")
        st.markdown("---")
        fig_location = px.bar(df_selection['location'].value_counts().reset_index(), x='location', y='count')
        st.plotly_chart(fig_location)
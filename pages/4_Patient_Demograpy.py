import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

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
            query = "SELECT * FROM patients;"
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            st.error(f"Could not read from 'patients' table: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# --- Initialize Session State for storing the list of selected locations ---
if 'db_demographics_locations' not in st.session_state:
    st.session_state.db_demographics_locations = []

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide")
st.title("👥 Patient Demographics (Live Database)")

df = load_data_from_db()

if not df.empty and 'age' in df.columns and 'location' in df.columns and 'gender' in df.columns:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter Patients Here:")

    age_range = st.sidebar.slider('Select Age Range:', min_value=int(df['age'].min()), max_value=int(df['age'].max()), value=(int(df['age'].min()), int(df['age'].max())))

    # --- NEW: Location Search and Add Functionality ---
    st.sidebar.subheader("Filter by Location")
    all_locations = sorted(df['location'].unique())
    location_to_add = st.sidebar.selectbox('Search or select a location to add:', options=all_locations, index=None, placeholder="Type or select a location...")
    if st.sidebar.button('Add Location', use_container_width=True):
        if location_to_add and location_to_add not in st.session_state.db_demographics_locations:
            st.session_state.db_demographics_locations.append(location_to_add)
    
    selected_locations = st.sidebar.multiselect('Currently selected locations (click x to remove):', options=all_locations, default=st.session_state.db_demographics_locations)
    st.session_state.db_demographics_locations = selected_locations

    # Gender Select Box
    st.sidebar.subheader("Filter by Gender")
    all_genders = sorted(df['gender'].unique())
    selected_gender = st.sidebar.selectbox('Select Gender:', options=['All'] + list(all_genders), index=0)

    # --- FILTERING THE DATAFRAME ---
    df_selection = df.copy()
    df_selection = df_selection[df_selection['age'].between(age_range[0], age_range[1])]
    if selected_locations:
        df_selection = df_selection[df_selection['location'].isin(selected_locations)]
    if selected_gender != 'All':
        df_selection = df_selection[df_selection['gender'] == selected_gender]

    # --- DISPLAY METRICS AND CHARTS ---
    st.markdown("---")
    if df_selection.empty:
        st.warning("No patients match the current filter criteria.")
    else:
        st.metric(label="Total Patients Matching Filters", value=f"{df_selection.shape[0]:,}")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Age Distribution")
            fig_age = px.histogram(df_selection, x='age', nbins=20, title="Distribution of Patient Ages")
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            st.subheader("Patient Count by Location")
            location_counts = df_selection['location'].value_counts().reset_index()
            location_counts.columns = ['location', 'count']
            fig_location = px.bar(location_counts, x='location', y='count', title="Number of Patients from Each Location")
            st.plotly_chart(fig_location, use_container_width=True)

        st.markdown("---")
        st.subheader("Condition by Gender")
        fig_condition = px.sunburst(df_selection, path=['gender', 'condition'], title="Breakdown of Medical Conditions by Gender")
        st.plotly_chart(fig_condition, use_container_width=True)
else:
    st.error("Could not load data from the database. Please ensure the 'data_to_db.py' script ran successfully.")
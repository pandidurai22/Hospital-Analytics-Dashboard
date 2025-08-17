import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

# --- Database Connection (reusable) ---
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

# --- Load and Merge Data from DATABASE ---
@st.cache_data
def load_performance_data():
    if engine is not None:
        try:
            query = """
            SELECT v.visit_date, p.age, p.gender, p.location, p.condition, p.cost, v.satisfaction, ph.physician_name, ph.specialty
            FROM visits v
            JOIN patients p ON v.patient_id = p.patient_id
            JOIN physicians ph ON v.physician_id = ph.physician_id;
            """
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            st.error(f"Could not read and join tables from the database: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# --- Initialize Session State for storing the list of selected physicians ---
if 'performance_physicians_list' not in st.session_state:
    st.session_state.performance_physicians_list = []

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide")
st.title("👨‍⚕️ Physician Performance Dashboard")
st.markdown("This module analyzes performance metrics for each physician, based on live database data.")

df = load_performance_data()

if not df.empty:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter by Physician:")
    
    all_physicians = sorted(df['physician_name'].unique())
    
    # --- NEW: Physician Search and Add Functionality ---
    physician_to_add = st.sidebar.selectbox(
        'Search or select a physician to add:',
        options=all_physicians,
        index=None,
        placeholder="Type or select a name..."
    )
    if st.sidebar.button('Add Physician', use_container_width=True):
        if physician_to_add and physician_to_add not in st.session_state.performance_physicians_list:
            st.session_state.performance_physicians_list.append(physician_to_add)

    selected_physicians = st.sidebar.multiselect(
        'Currently selected physicians (click x to remove):',
        options=all_physicians,
        default=st.session_state.performance_physicians_list
    )
    st.session_state.performance_physicians_list = selected_physicians
    
    
    # Filter the dataframe based on selection
    # If no physicians are selected, show all. Otherwise, filter to the selected ones.
    if not selected_physicians:
        df_selection = df.copy() # Show all if the list is empty
    else:
        df_selection = df[df['physician_name'].isin(selected_physicians)]
    
    st.markdown("---")
    
    if df_selection.empty:
        st.warning("No data available for the selected physician(s).")
    else:
        # --- METRICS AND CHARTS ---
        st.subheader("Performance Overview for Selected Physicians")
        
        performance_summary = df_selection.groupby('physician_name').agg(
            total_patients=('visit_date', 'count'),
            avg_patient_satisfaction=('satisfaction', 'mean'),
            avg_cost_of_care=('cost', 'mean')
        ).reset_index()
        
        performance_summary.rename(columns={'physician_name': 'Physician Name', 'total_patients': 'Total Patients Seen', 'avg_patient_satisfaction': 'Avg. Satisfaction (1-5)', 'avg_cost_of_care': 'Avg. Cost of Care ($)'}, inplace=True)
        st.dataframe(performance_summary.round(2))
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_patients = px.bar(performance_summary, x='Physician Name', y='Total Patients Seen', title='Total Patient Load')
            st.plotly_chart(fig_patients, use_container_width=True)
        with col2:
            fig_satisfaction = px.bar(performance_summary, x='Physician Name', y='Avg. Satisfaction (1-5)', title='Average Patient Satisfaction', range_y=[1, 5])
            fig_satisfaction.update_traces(marker_color='mediumseagreen')
            st.plotly_chart(fig_satisfaction, use_container_width=True)
        with col3:
            fig_cost = px.bar(performance_summary, x='Physician Name', y='Avg. Cost of Care ($)', title='Average Cost of Care per Patient')
            fig_cost.update_traces(marker_color='indianred')
            st.plotly_chart(fig_cost, use_container_width=True)
else:
    st.error("Could not load performance data from the database. Please ensure the 'data_to_db.py' script ran successfully.")
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, exc
from datetime import datetime, date

st.set_page_config(layout="wide")
st.title("👨‍⚕️ Physician Performance Dashboard (Live Database)")
st.markdown("This module analyzes and compares physician performance metrics across different time periods.")

# --- Database Connection and Data Loading ---
@st.cache_resource
def get_db_engine():
    DB_PASSWORD = "password" # !!! IMPORTANT: Change to your password !!!
    DATABASE_URL = f"postgresql+psycopg2://postgres:{DB_PASSWORD}@localhost:5432/hospital_db"
    try:
        return create_engine(DATABASE_URL)
    except Exception as e:
        return None

@st.cache_data
def load_performance_data():
    engine = get_db_engine()
    if engine:
        try:
            query = """
            SELECT v.visit_date, v.satisfaction, p.cost, ph.physician_name
            FROM visits v
            JOIN patients p ON v.patient_id = p.patient_id
            JOIN physicians ph ON v.physician_id = ph.physician_id;
            """
            df = pd.read_sql(query, engine)
            df['visit_date'] = pd.to_datetime(df['visit_date'])
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# --- Initialize Session State ---
if 'perf_physicians_list' not in st.session_state:
    st.session_state.perf_physicians_list = []

df = load_performance_data()

if not df.empty:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter by Time Period:")
    today = date.today()
    current_start_date = st.sidebar.date_input("Start Date (Current)", today.replace(day=1), key='perf_current_start')
    current_end_date = st.sidebar.date_input("End Date (Current)", today, key='perf_current_end')
    compare_start_date = st.sidebar.date_input("Start Date (Comparison)", (today.replace(day=1) - pd.DateOffset(months=1)).date(), key='perf_compare_start')
    compare_end_date = st.sidebar.date_input("End Date (Comparison)", (today - pd.DateOffset(months=1)).date(), key='perf_compare_end')

    st.sidebar.header("Filter by Physician:")
    all_physicians = sorted(df['physician_name'].unique())
    physician_to_add = st.sidebar.selectbox('Search or select a physician to add:', options=all_physicians, index=None, placeholder="Type or select a name...")
    if st.sidebar.button('Add Physician', use_container_width=True):
        if physician_to_add and physician_to_add not in st.session_state.perf_physicians_list:
            st.session_state.perf_physicians_list.append(physician_to_add)
    
    selected_physicians = st.sidebar.multiselect('Currently selected physicians (click x to remove):', options=all_physicians, default=st.session_state.perf_physicians_list)
    st.session_state.perf_physicians_list = selected_physicians

    # Filter data
    df_current = df[df['visit_date'].dt.date.between(current_start_date, current_end_date)]
    df_compare = df[df['visit_date'].dt.date.between(compare_start_date, compare_end_date)]
    
    if selected_physicians:
        df_current = df_current[df_current['physician_name'].isin(selected_physicians)]
        df_compare = df_compare[df_compare['physician_name'].isin(selected_physicians)]
    
    st.markdown("---")
    st.subheader("Comparative Performance Overview")
    
    if df_current.empty:
        st.warning("No data available for the selected filters in the Current Period.")
    else:
        # --- CALCULATE AND DISPLAY METRICS ---
        summary_current = df_current.groupby('physician_name').agg(total_patients=('visit_date', 'count'), avg_satisfaction=('satisfaction', 'mean'), avg_cost=('cost', 'mean')).reset_index()
        summary_compare = df_compare.groupby('physician_name').agg(total_patients_comp=('visit_date', 'count'), avg_satisfaction_comp=('satisfaction', 'mean'), avg_cost_comp=('cost', 'mean')).reset_index()

        performance_df = pd.merge(summary_current, summary_compare, on='physician_name', how='left').fillna(0)
        
        if 'avg_satisfaction_comp' in performance_df.columns and performance_df['avg_satisfaction_comp'].sum() > 0:
            performance_df['satisfaction_delta'] = ((performance_df['avg_satisfaction'] - performance_df['avg_satisfaction_comp']) / performance_df['avg_satisfaction_comp'] * 100).round(2)
        else:
            performance_df['satisfaction_delta'] = 0.0

        if 'avg_cost_comp' in performance_df.columns and performance_df['avg_cost_comp'].sum() > 0:
            performance_df['cost_delta'] = ((performance_df['avg_cost'] - performance_df['avg_cost_comp']) / performance_df['avg_cost_comp'] * 100).round(2)
        else:
            performance_df['cost_delta'] = 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown("**Physician Name**")
        col2.markdown("**Total Patients Seen**")
        col3.markdown("**Avg. Satisfaction**")
        col4.markdown("**Avg. Cost of Care ($)**")

        for index, row in performance_df.iterrows():
            col1, col2, col3, col4 = st.columns(4)
            col1.write(row['physician_name'])
            col2.metric(label="", value=int(row['total_patients']), delta=int(row['total_patients'] - row['total_patients_comp']))
            col3.metric(label="", value=f"{row['avg_satisfaction']:.2f}", delta=f"{row['satisfaction_delta']}%")
            col4.metric(label="", value=f"${row['avg_cost']:,.2f}", delta=f"{row['cost_delta']}%", delta_color="inverse")

        # --- THIS IS THE FIX: ADDING ALL THREE CHARTS ---
        st.markdown("---")
        st.subheader("Visual Breakdown for Current Period")

        chart_col1, chart_col2, chart_col3 = st.columns(3)
        with chart_col1:
            fig_patients = px.bar(performance_df, 
                                  x='physician_name', 
                                  y='total_patients', 
                                  title='Total Patient Load')
            st.plotly_chart(fig_patients, use_container_width=True)

        with chart_col2:
            fig_satisfaction = px.bar(performance_df, 
                                      x='physician_name', 
                                      y='avg_satisfaction', 
                                      title='Average Patient Satisfaction', 
                                      range_y=[1, 5])
            fig_satisfaction.update_traces(marker_color='mediumseagreen')
            st.plotly_chart(fig_satisfaction, use_container_width=True)
            
        with chart_col3:
            fig_cost = px.bar(performance_df, 
                              x='physician_name', 
                              y='avg_cost', 
                              title='Average Cost of Care per Patient')
            fig_cost.update_traces(marker_color='indianred')
            st.plotly_chart(fig_cost, use_container_width=True)
else:
    st.error("Could not load performance data from the database.")
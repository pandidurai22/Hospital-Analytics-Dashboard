import streamlit as st
import pandas as pd

# --- Page Configuration (Good Practice) ---
st.set_page_config(layout="wide", page_title="Outbreak Impact Analysis")

st.title("🏥 Patient Load and Outbreak Analysis")

# --- Step 1: Load your main and new datasets ---
@st.cache_data # Cache the data so it doesn't reload on every interaction
def load_data():
    df = pd.read_csv("augmented_outpatient_data.csv", parse_dates=["Visit_Date"])
    outbreaks_df = pd.read_csv("Outbreak_events.csv", parse_dates=["EVENT_DATE"])
    return df, outbreaks_df

df, outbreaks_df = load_data()

# --- Step 2: Create the daily aggregated DataFrame ---
daily_df = df.groupby("Visit_Date").agg(
    Outpatient_Count=("Patient_ID", "count")
    # You can add other aggregations here if needed
).reset_index()

# --- Step 3: Feature Engineering for Outbreaks ---
OUTBREAK_DURATION_DAYS = 21 
daily_df['is_outbreak_active'] = 0
daily_df['outbreak_severity'] = 'None'
severity_mapping = {'None': 0, 'Moderate': 1, 'Severe': 2}

for _, outbreak in outbreaks_df.iterrows():
    start_date = outbreak['EVENT_DATE']
    end_date = start_date + pd.Timedelta(days=OUTBREAK_DURATION_DAYS)
    severity = outbreak['SEVERITY_LEVEL']
    
    active_period_mask = (daily_df['Visit_Date'] >= start_date) & (daily_df['Visit_Date'] <= end_date)
    
    daily_df.loc[active_period_mask, 'is_outbreak_active'] = 1
    daily_df.loc[active_period_mask, 'outbreak_severity'] = severity

daily_df['outbreak_severity_encoded'] = daily_df['outbreak_severity'].map(severity_mapping)

# --- Step 4: Display the results in the Streamlit App ---

st.header("📈 Daily Patient Count with Outbreak Periods")
st.write("This chart shows the daily number of outpatients. The red line indicates the severity of any active outbreak on that day (0=None, 1=Moderate, 2=Severe).")

# Prepare data for charting
chart_data = daily_df.set_index('Visit_Date')
st.line_chart(chart_data[['Outpatient_Count', 'outbreak_severity_encoded']])


st.header("📋 Data View: Days with Active Outbreaks")
st.write("Below is the filtered data showing only the days that fall within an active outbreak period.")

# Show the dataframe of active outbreak days
active_days_df = daily_df[daily_df['is_outbreak_active'] == 1]
st.dataframe(active_days_df)


st.header("📄 Raw Outbreak Events Data")
st.dataframe(outbreaks_df)
import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Hospital Resource Analyzer")

st.title("🏥 Hospital Resource Utilization Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    # Note the date format here is DD-MMM-YY, so we specify it
    df = pd.read_csv("Hospital_resources.csv", parse_dates=["RESOURCE_DATE"], date_format="%d-%b-%y")
    return df

resources_df = load_data()

# --- Data Transformation ---
# Calculate Occupancy Rate
resources_df["OCCUPANCY_RATE"] = (resources_df["TOTAL_OCCUPIED"] / resources_df["TOTAL_AVAILABLE"]) * 100

# --- Sidebar Filters ---
st.sidebar.header("Filters")
selected_resource = st.sidebar.selectbox(
    "Select Resource Type",
    resources_df["RESOURCE_TYPE"].unique()
)

# Filter data based on selection
filtered_df = resources_df[resources_df["RESOURCE_TYPE"] == selected_resource]

# --- Main Page Display ---

# 1. Key Performance Indicators (KPIs)
st.header(f"KPIs for: {selected_resource.replace('_', ' ').title()}")
avg_occupancy = filtered_df["OCCUPANCY_RATE"].mean()
peak_occupancy_day = filtered_df.loc[filtered_df["OCCUPANCY_RATE"].idxmax()]

col1, col2 = st.columns(2)
col1.metric("Average Occupancy", f"{avg_occupancy:.1f}%")
col2.metric(
    f"Peak Occupancy Date",
    f"{peak_occupancy_day['RESOURCE_DATE'].strftime('%d-%b-%Y')}",
    f"{peak_occupancy_day['OCCUPANCY_RATE']:.1f}%"
)


# 2. Occupancy Rate Over Time (Interactive Chart)
st.header("Occupancy Rate Over Time")

fig = px.line(
    filtered_df,
    x="RESOURCE_DATE",
    y="OCCUPANCY_RATE",
    title=f"Daily Occupancy Rate for {selected_resource.replace('_', ' ').title()}",
    markers=True,
    labels={"RESOURCE_DATE": "Date", "OCCUPANCY_RATE": "Occupancy Rate (%)"}
)
# Add a threshold line for high occupancy
fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="90% Threshold")
st.plotly_chart(fig, use_container_width=True)


# 3. Raw Data View
with st.expander("View Raw Data"):
    st.dataframe(filtered_df)
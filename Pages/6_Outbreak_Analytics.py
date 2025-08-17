# In pages/5_Outbreak_Analytics.py

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Outbreak Analytics", layout="wide")
st.title("☣️ Outbreak Analytics Map")
st.write("This page shows the locations and severity of historical outbreaks.")

try:
    # We need two files: one with outbreak events, and one with lat/lon for each location
    outbreaks_df = pd.read_csv('Outbreak_events.csv')
    locations_df = pd.read_csv('locations.csv')

    # Merge the two dataframes to get coordinates for each outbreak
    # FIX: The key in outbreaks_df is 'DISTRICT' and in locations_df it is 'Location'
    map_df = pd.merge(outbreaks_df, locations_df, left_on='DISTRICT', right_on='Location')

    st.subheader("Map of Outbreak Events")
    # Add a size column for better visualization on the map
    severity_size_mapping = {'Moderate': 15, 'Severe': 30}
    map_df['size'] = map_df['SEVERITY_LEVEL'].map(severity_size_mapping).fillna(10) # .fillna handles any other values

    st.map(map_df, latitude='lat', longitude='lon', size='size', color='#ff0000') # Display map with red dots

    st.subheader("Outbreak Data with Coordinates")
    st.dataframe(map_df[['EVENT_DATE', 'DISTRICT', 'SEVERITY_LEVEL', 'lat', 'lon']])

except FileNotFoundError:
    st.error("Error: 'Outbreak_events.csv' or 'locations.csv' not found.")
except Exception as e:
    st.error(f"An error occurred: {e}. Please ensure your files are correct.")
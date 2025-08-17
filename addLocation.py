import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
import random
from tqdm import tqdm # A library to show a progress bar!

# --- SETUP ---
# To use tqdm, you might need to install it first. In your active terminal, run:
# pip install tqdm

# Step 1: Load original data
df = pd.read_csv("patient.csv")

# Step 2: Replace with outpatient-relevant conditions
outpatient_conditions = [
    "Common Cold", "Diabetes Checkup", "Hypertension", "Flu",
    "Skin Rash", "Migraine", "Joint Pain", "Allergy", "Throat Infection", "Asthma"
]
df["Condition"] = np.random.choice(outpatient_conditions, size=len(df))

# Step 3: Assign random visit dates in June 2025
def random_june_date():
    day = random.randint(1, 30)
    return datetime(2025, 6, day).strftime("%Y-%m-%d")
df["Visit_Date"] = [random_june_date() for _ in range(len(df))]

# Step 4: Assign random locations around Chennai
chennai_locations = [
    "Chennai", "Tambaram", "Anna Nagar", "Velachery", "T Nagar",
    "Kodambakkam", "Ambattur", "Pallavaram", "Porur", "Thiruvanmiyur"
]
df["Location"] = np.random.choice(chennai_locations, size=len(df))

# --- OPTIMIZED WEATHER FETCHING ---
print("Fetching weather data efficiently...")
location_coords = {
    "Chennai": (13.0827, 80.2707), "Tambaram": (12.9246, 80.1277),
    "Anna Nagar": (13.0878, 80.2206), "Velachery": (12.9789, 80.2217),
    "T Nagar": (13.0424, 80.2338), "Kodambakkam": (13.0484, 80.2306),
    "Ambattur": (13.1143, 80.1615), "Pallavaram": (12.9683, 80.1496),
    "Porur": (13.0322, 80.1588), "Thiruvanmiyur": (12.9866, 80.2596),
}
weather_cache = {} # To store results and avoid re-fetching for the same location/date

# Group by location to make fewer API calls
unique_locations = df['Location'].unique()

# Using tqdm to show a progress bar
for location in tqdm(unique_locations, desc="Fetching weather for locations"):
    lat, lon = location_coords.get(location)
    
    # We only need weather data for June 2025
    start_date = "2025-05-25" # Start a bit earlier to cover all 7-day lookbacks
    end_date = "2025-06-30"
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max&timezone=Asia%2FKolkata"
    
    try:
        response = requests.get(url)
        data = response.json()
        daily_data = data.get("daily", {})
        weather_df = pd.DataFrame({'time': pd.to_datetime(daily_data.get('time')), 'temp': daily_data.get('temperature_2m_max')})
        weather_cache[location] = weather_df.set_index('time')
    except Exception as e:
        print(f"⚠️ Weather fetch failed for {location}: {e}")

# --- MAP RESULTS BACK TO PATIENTS (This part is very fast) ---
print("Mapping weather data to patients...")
for i in range(1, 8):
    df[f"Temp_Day_-{i}"] = None

for idx, row in df.iterrows():
    visit_date = pd.to_datetime(row['Visit_Date'])
    location_weather = weather_cache.get(row['Location'])
    if location_weather is not None:
        for i in range(1, 8):
            lookup_date = visit_date - timedelta(days=i)
            if lookup_date in location_weather.index:
                df.at[idx, f"Temp_Day_-{i}"] = location_weather.loc[lookup_date, 'temp']


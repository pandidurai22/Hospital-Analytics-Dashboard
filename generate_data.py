import pandas as pd
import random
from datetime import date, timedelta

# --- CONFIGURATION ---
NUM_ROWS_TO_GENERATE = 1500
OUTPUT_FILENAME = 'generated_large_outpatient_data.csv'

# --- DATA SOURCES ---
PATIENT_IDS = [f'P{i:04d}' for i in range(1, 101)]
LOCATIONS = ['General Hospital', 'Downtown Clinic', 'Suburban Medical Center', 'Eastside Specialty Care', 'Westend Urgent Care']
OUTCOMES = ['Recovered', 'Stable', 'Needs Follow-up']
READMISSIONS = ['Yes', 'No', 'No', 'No', 'Yes']

# --- DATE RANGE ---
START_DATE = date(2024, 1, 1)
END_DATE = date(2026, 12, 31)
total_days_in_range = (END_DATE - START_DATE).days

all_visits_data = []
print(f"Generating {NUM_ROWS_TO_GENERATE} rows of data...")

# --- DATA GENERATION LOOP ---
for i in range(NUM_ROWS_TO_GENERATE):
    visit_date = START_DATE + timedelta(days=random.randint(0, total_days_in_range))
    
    # --- THIS IS THE FIX: The 'Satisfaction' key is now correctly included ---
    new_row = {
        'Patient_ID': random.choice(PATIENT_IDS),
        'Visit_Date': visit_date,
        'Satisfaction': random.randint(1, 5),  # This column is now guaranteed to exist
        'Outcome': random.choice(OUTCOMES),
        'Readmission': random.choice(READMISSIONS),
        'Location': random.choice(LOCATIONS)
    }
    all_visits_data.append(new_row)

# --- CREATE AND SAVE THE DATAFRAME ---
large_df = pd.DataFrame(all_visits_data)
large_df.to_csv(OUTPUT_FILENAME, index=False, date_format='%Y-%m-%d')

print(f"\n--- SUCCESS! ---")
print(f"New, correct dataset '{OUTPUT_FILENAME}' has been created with a 'Satisfaction' column.")
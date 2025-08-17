import pandas as pd
import numpy as np

print("Starting process to create a perfectly matching 'augmented_patients.csv'...")

try:
    # Step 1: Load the main visit data to see which patient IDs are actually used
    visits_df = pd.read_csv('generated_large_outpatient_data.csv')
    print(f"Loaded 'generated_large_outpatient_data.csv' with {len(visits_df)} rows.")
except FileNotFoundError:
    print("ERROR: 'generated_large_outpatient_data.csv' not found. Please run 'generate_data.py' first.")
    exit()

# Step 2: Clean the patient IDs from the visits file and find all unique IDs
visits_df.columns = [col.strip().lower() for col in visits_df.columns]

# Clean the patient_id column and convert to integer
visits_df['patient_id_int'] = visits_df['patient_id'].str.replace('p', '', case=False).astype(int)
unique_patient_ids = visits_df['patient_id_int'].unique()
print(f"Found {len(unique_patient_ids)} unique patient IDs in the visits data.")


# Step 3: Create a new DataFrame for these unique patients
patient_data = []

# Define possible values for demographics
genders = ['Male', 'Female']
conditions = ['Diabetes', 'Cancer', 'Heart Disease', 'Hypertension', 'Arthritis', 'Stroke']
locations = ['T Nagar', 'Pallavaram', 'Porur', 'Velachery', 'Anna Nagar', 'Kodambakkam']

for patient_id in unique_patient_ids:
    patient_data.append({
        'patient_id': patient_id,
        'age': np.random.randint(18, 85),
        'gender': np.random.choice(genders),
        'location': np.random.choice(locations),
        'condition': np.random.choice(conditions),
        'cost': np.random.randint(1000, 20000)
    })

augmented_patients_df = pd.DataFrame(patient_data)
print(f"Created new patient data with {len(augmented_patients_df)} rows.")

# Step 4: Save the new, corrected augmented_patients.csv file
output_filename = 'augmented_patients.csv'
augmented_patients_df.to_csv(output_filename, index=False)

print("\n--- SUCCESS! ---")
print(f"A new, perfectly matching '{output_filename}' has been created.")
print("Please clear the cache in your Streamlit app by pressing 'C' and then rerun the Visit Analytics page.")
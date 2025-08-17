import pandas as pd
import numpy as np

print("Starting process to add physician data to visits...")

# --- PHYSICIAN DATA ---
# Let's create a list of fictional physician names and their specialties
physicians_data = {
    'physician_id': [101, 102, 103, 104, 105, 106, 107, 108],
    'physician_name': ['Dr. Arul', 'Dr. Priya', 'Dr. Suresh', 'Dr. Meena', 'Dr. Anand', 'Dr. Kavitha', 'Dr. Rajesh', 'Dr. Divya'],
    'specialty': ['Cardiology', 'Oncology', 'Cardiology', 'General Medicine', 'Oncology', 'Orthopedics', 'General Medicine', 'Orthopedics']
}
physicians_df = pd.DataFrame(physicians_data)

# Save the physicians list to its own CSV
physicians_df.to_csv('physicians.csv', index=False)
print("SUCCESS: 'physicians.csv' created successfully.")


# --- ASSIGN PHYSICIANS TO VISITS ---
try:
    visits_df = pd.read_csv('generated_large_outpatient_data.csv')
    print(f"Loaded 'generated_large_outpatient_data.csv' with {len(visits_df)} rows.")
except FileNotFoundError:
    print("ERROR: 'generated_large_outpatient_data.csv' not found. Please run 'generate_data.py' first.")
    exit()

# Randomly assign one of our physician IDs to each visit
visits_df['physician_id'] = np.random.choice(physicians_df['physician_id'], size=len(visits_df))

# Save this new, richer version of the visits data
output_filename = 'visits_with_physicians.csv'
visits_df.to_csv(output_filename, index=False)

print(f"SUCCESS: A new file '{output_filename}' has been created with a 'physician_id' for each visit.")
print("\nNext step: Run the 'data_to_db.py' script to get this new data into your database.")
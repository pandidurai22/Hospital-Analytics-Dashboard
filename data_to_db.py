import pandas as pd
from sqlalchemy import create_engine

# --- CONFIGURATION ---
DB_PASSWORD = "password" # Replace with your password
DB_NAME = "hospital_db"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    engine = create_engine(DATABASE_URL)
    print("Successfully connected to the PostgreSQL database.")
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit()

# --- THE MAIN FUNCTION (NOW WITH PROPER CLEANING) ---
def load_csv_to_db(filepath, table_name, db_engine, id_column=None):
    try:
        print(f"\nProcessing '{filepath}'...")
        df = pd.read_csv(filepath)
        df.columns = [col.strip().lower() for col in df.columns]
        
        # --- THIS IS THE CRUCIAL FIX ---
        # If an id_column is specified, clean it to be a pure integer.
        if id_column and id_column in df.columns:
            print(f"Cleaning the '{id_column}' column...")
            # Convert to string, remove the letter 'p' (case-insensitive), and convert to integer
            df[id_column] = df[id_column].astype(str).str.replace('p', '', case=False).astype(int)
            print(f"'{id_column}' column has been successfully converted to integers.")

        print(f"Loading {len(df)} rows into the '{table_name}' table...")
        df.to_sql(table_name, db_engine, if_exists='replace', index=False)
        print(f"--- SUCCESS: Data from '{filepath}' has been loaded into the '{table_name}' table. ---")
    
    except FileNotFoundError:
        print(f"ERROR: File not found at '{filepath}'. Skipping.")
    except Exception as e:
        print(f"An error occurred while loading '{filepath}': {e}")

# --- RUN THE CORRECTED MIGRATION ---
# We now specify the ID columns that need to be cleaned.
load_csv_to_db("augmented_patients.csv", "patients", engine, id_column='patient_id')
load_csv_to_db("Hospital_resources.csv", "resources", engine)
load_csv_to_db("physicians.csv", "physicians", engine, id_column='physician_id')
# The visits_with_physicians file has TWO id columns to clean! 
# Our function handles one, so we'll do the other manually here first.
try:
    visits_df = pd.read_csv("visits_with_physicians.csv")
    visits_df.columns = [col.strip().lower() for col in visits_df.columns]
    # Manually clean the physician_id before passing to the function
    if 'physician_id' in visits_df.columns:
        visits_df['physician_id'] = visits_df['physician_id'].astype(str).astype(int)
    # Save it to a temporary file to load
    visits_df.to_csv("temp_visits.csv", index=False)
    load_csv_to_db("temp_visits.csv", "visits", engine, id_column='patient_id')
except Exception as e:
    print(f"An error occurred while processing the visits file: {e}")
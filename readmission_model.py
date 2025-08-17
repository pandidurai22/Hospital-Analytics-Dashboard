import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("--- Starting Patient Readmission Prediction Model Training ---")

# --- 1. Load Datasets ---
print("Step 1: Loading patient visit and demographics data...")
visits_df = pd.read_csv("augmented_outpatient_data.csv")
demographics_df = pd.read_csv("demographics.csv")

# --- 2. Clean and Merge Data ---
print("Step 2: Cleaning and merging datasets...")
# Standardize the key column name in both dataframes before merging
demographics_df.rename(columns={'PATIENT_ID': 'ID_String'}, inplace=True)
demographics_df['Patient_ID'] = demographics_df['ID_String'].str.replace('P', '').astype(int)

# Now, the 'Patient_ID' column exists in both and is numeric, ensuring a correct merge.
merged_df = pd.merge(visits_df, demographics_df, on='Patient_ID', how='inner')

# --- 3. Feature Engineering ---
print("Step 3: Engineering features for the model...")
# A. Target Variable: Convert 'Readmission' from 'Yes'/'No' to 1/0
merged_df['Readmitted'] = merged_df['Readmission'].apply(lambda x: 1 if x == 'Yes' else 0)

# B. Comorbidities: Count the number of co-existing conditions for each patient
merged_df['COMORBIDITIES'] = merged_df['COMORBIDITIES'].fillna('')
merged_df['num_comorbidities'] = merged_df['COMORBIDITIES'].apply(lambda x: len(x.split(',')) if x else 0)

# C. Handle Missing BMI: Fill missing BMI values with the median of the column (Best Practice)
median_bmi = merged_df['BMI'].median()
merged_df.loc[:, 'BMI'] = merged_df['BMI'].fillna(median_bmi)

# --- 4. Select Features and Target ---
print("Step 4: Selecting final features and target variable...")
# Define which columns are categorical and which are numeric
categorical_features = ['Gender', 'Condition', 'Procedure', 'Location', 'SOCIOECONOMIC_STATUS', 'SMOKING_STATUS']
numeric_features = ['Age', 'Cost', 'Length_of_Stay', 'BMI', 'num_comorbidities']
target = 'Readmitted'

features = categorical_features + numeric_features
X = merged_df[features]
y = merged_df[target]

# --- 5. Create a Preprocessing Pipeline ---
# This is the best practice for handling mixed data types.
# It scales numeric features and one-hot encodes categorical features.
print("Step 5: Building a preprocessing pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 6. Train/Test Split ---
print("Step 6: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 7. Define and Train the RandomForest Model ---
# We create a full pipeline that first preprocesses the data, then trains the model.
print("Step 7: Training the RandomForestClassifier model...")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

model_pipeline.fit(X_train, y_train)

# --- 8. Evaluate the Model ---
print("\n--- MODEL PERFORMANCE ---")
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("-------------------------\n")

# --- 9. Analyze Feature Importance ---
# We need to get the feature names from the one-hot encoder to label the importances correctly.
print("Step 9: Analyzing feature importances...")
ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(ohe_feature_names)
importances = model_pipeline.named_steps['classifier'].feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features for Predicting Readmission:")
print(importance_df.head(10))

# --- 10. Save the Trained Model ---
# The entire pipeline (preprocessor + model) is saved as a single file.
model_filename = 'readmission_model.joblib'
joblib.dump(model_pipeline, model_filename)
print(f"\nStep 10: Model successfully trained and saved as '{model_filename}'")
print("--- Script Finished ---")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

print("--- Starting Patient Readmission Prediction Project ---")

# --- Step 1: Load and Clean the Dataset ---
try:
    df = pd.read_csv('augmented_patients.csv')
    print("Successfully loaded 'augmented_patients.csv'.")
except FileNotFoundError:
    print("Error: 'augmented_patients.csv' not found. Please make sure it's in the directory.")
    exit()

# Clean up column names by removing leading/trailing spaces
df.columns = df.columns.str.strip()
print("Successfully cleaned column names.\n")


# --- Step 2: Define Features (X) and Target (y) ---

# All the columns we will use to make the prediction
features = [
    'Age',  # <-- THIS IS THE FINAL FIX (Capital 'A')
    'Gender',
    'Condition',
    'Procedure',
    'Cost',
    'Length_of_Stay',
    'Outcome',
    'Satisfaction',
    'Location'
]

# The column we want to predict
target = 'Readmission'

X = df[features]
y = df[target]

print(f"Features for prediction: {features}")
print(f"Target to predict: '{target}'\n")


# --- Step 3: Prepare the Data Processing Pipeline ---

categorical_features = [
    'Gender',
    'Condition',
    'Procedure',
    'Outcome',
    'Satisfaction',
    'Location'
]
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)


# --- Step 4: Define and Train the Classification Model ---

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Training the Classification Model ---")
model.fit(X_train, y_train)
print("Model training complete.\n")


# --- Step 5: Evaluate the Model's Performance ---

print("--- Evaluating Model Performance ---")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2%}")
print("This is the percentage of readmissions the model predicted correctly.\n")

print("Classification Report:")
print(classification_report(y_test, predictions))


# --- Step 6: Save Your Trained Model ---

model_filename = 'readmission_predictor_model.joblib'
joblib.dump(model, model_filename)

print("--- Model Saved ---")
print(f"Model successfully trained and saved as '{model_filename}'")
print("This model is ready to be used in a new dashboard!")
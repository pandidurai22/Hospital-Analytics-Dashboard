import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from prophet import Prophet
import joblib
from datetime import datetime, timedelta

# Load and prepare data
df = pd.read_csv("augmented_outpatient_data.csv")
df['Visit_Date'] = pd.to_datetime(df['Visit_Date'])

# 1. PATIENT-LEVEL READMISSION PREDICTION
# Clean data for ML model
ml_df = df.drop(columns=['Patient_ID', 'Outcome', 'Satisfaction'])
ml_df['Days_Until_Readmission'] = (pd.to_datetime(df['Readmission_Date']) - df['Visit_Date']).dt.days
ml_df['Readmission_Risk'] = np.where(ml_df['Days_Until_Readmission'] <= 30, 1, 0)

# Feature engineering
ml_df['Visit_DayOfWeek'] = ml_df['Visit_Date'].dt.dayofweek
ml_df['Visit_Month'] = ml_df['Visit_Date'].dt.month
temp_cols = [f"Temp_Day_{i}" for i in range(-1, -8, -1)]
ml_df['Temp_Avg'] = ml_df[temp_cols].mean(axis=1)

# Train/test split with time series validation
X = ml_df.drop(columns=['Readmission_Risk', 'Visit_Date', 'Readmission_Date'] + temp_cols)
y = ml_df['Readmission_Risk']

# Preprocessing pipeline
numeric_features = ['Age', 'Cost', 'Length_of_Stay', 'Temp_Avg', 'Visit_DayOfWeek', 'Visit_Month']
categorical_features = ['Gender', 'Condition', 'Procedure', 'Location']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Train model
readmission_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])
readmission_model.fit(X, y)

# 2. DAILY VOLUME FORECASTING
# Prepare time series data
volume_df = df.groupby('Visit_Date').size().reset_index(name='Patient_Count')
volume_df = volume_df.rename(columns={'Visit_Date': 'ds', 'Patient_Count': 'y'})

# Train Prophet model
volume_model = Prophet(seasonality_mode='multiplicative')
volume_model.fit(volume_df)

# 3. PREDICTION FUNCTION
def predict_healthcare_demand(date_input):
    """Predicts daily volume and patient-level risks for a given date"""
    try:
        target_date = pd.to_datetime(date_input)
        
        # Forecast volume
        future = volume_model.make_future_dataframe(periods=1, include_history=False)
        forecast = volume_model.predict(future)
        predicted_volume = int(round(forecast['yhat'].iloc[0]))
        
        # Generate synthetic patient data for prediction (in real use, this would be real data)
        synthetic_patients = generate_synthetic_patients(target_date, predicted_volume)
        
        # Predict readmission risks
        synthetic_patients['Readmission_Risk'] = readmission_model.predict_proba(
            synthetic_patients[X.columns])[:, 1]
        
        # Generate insights
        high_risk = synthetic_patients[synthetic_patients['Readmission_Risk'] > 0.7]
        risk_by_procedure = synthetic_patients.groupby('Procedure')['Readmission_Risk'].mean()
        
        # Visualizations
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        risk_by_procedure.sort_values().plot(kind='barh')
        plt.title('Average Readmission Risk by Procedure')
        
        plt.subplot(1, 2, 2)
        plt.scatter(synthetic_patients['Age'], synthetic_patients['Readmission_Risk'], 
                   c=synthetic_patients['Length_of_Stay'], cmap='viridis')
        plt.colorbar(label='Length of Stay (days)')
        plt.xlabel('Age')
        plt.ylabel('Readmission Risk')
        plt.title('Patient Risk Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'predicted_volume': predicted_volume,
            'high_risk_patients': len(high_risk),
            'top_risk_procedure': risk_by_procedure.idxmax(),
            'sample_patient': synthetic_patients.iloc[0][['Age', 'Gender', 'Condition', 'Readmission_Risk']].to_dict()
        }
    
    except Exception as e:
        return f"Prediction error: {str(e)}"

def generate_synthetic_patients(date, n):
    """Helper function to create sample patient data"""
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 90, n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Condition': np.random.choice(['Diabetes', 'Hypertension', 'CHF', 'COPD'], n),
        'Procedure': np.random.choice(['Checkup', 'Surgery', 'Therapy', 'Scan'], n),
        'Location': np.random.choice(['Main', 'North', 'South'], n),
        'Cost': np.random.uniform(100, 5000, n),
        'Length_of_Stay': np.random.randint(1, 14, n),
        'Temp_Avg': np.random.uniform(36, 39, n),
        'Visit_DayOfWeek': [date.weekday()] * n,
        'Visit_Month': [date.month] * n
    }
    return pd.DataFrame(data)

# Example usage
prediction = predict_healthcare_demand('2023-06-15')
print("\n=== Healthcare Demand Prediction ===")
for k, v in prediction.items():
    print(f"{k.replace('_', ' ').title()}: {v}")
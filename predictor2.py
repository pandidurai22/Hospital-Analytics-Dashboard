import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv("augmented_outpatient_data.csv")  # Replace with your actual file name if different

# Target column to predict
target_col = 'Outcome'

# Drop irrelevant or identifier columns (e.g., Patient_ID, Visit_Date)
df = df.drop(columns=['Patient_ID', 'Visit_Date'])

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical columns
categorical_cols = X.select_dtypes(include='object').columns
X = pd.get_dummies(X, columns=categorical_cols)

# Encode the target if it's categorical
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

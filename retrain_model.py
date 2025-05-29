import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv('C:\\Users\\kesha\\Documents\\fraud-detection-system\\data\\creditcard.csv')

# Use only Time and Amount
X = df[['Time', 'Amount']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train IsolationForest
model = IsolationForest(contamination=0.001, random_state=42)
model.fit(X_scaled)

# Overwrite model and scaler files
joblib.dump(model, 'C:\\Users\\kesha\\Documents\\fraud-detection-system\\backend\\model\\fraud_detection_model.pkl')
joblib.dump(scaler, 'C:\\Users\\kesha\\Documents\\fraud-detection-system\\backend\\model\\scaler.pkl')

print("Model and scaler retrained and saved.")
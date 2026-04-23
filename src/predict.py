import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/model.pkl")

# Load dataset (for testing)
data = pd.read_csv("data/creditcard.csv")

# Preprocessing (same as training)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])
data['Time'] = scaler.fit_transform(data[['Time']])

# Take a sample transaction
sample = data.drop('Class', axis=1).iloc[0:1]

# Predict
prediction = model.predict(sample)

# Output result
if prediction[0] == 1:
    print("🚨 Fraud Transaction Detected!")
else:
    print("✅ Normal Transaction")
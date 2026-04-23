import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load("models/model.pkl")

# Load dataset
data = pd.read_csv("data/creditcard.csv")

# Preprocessing
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])
data['Time'] = scaler.fit_transform(data[['Time']])

# UI
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 AI Fraud Detection System")
st.markdown("### 🚀 Test Real Transactions with ML Model")

# Buttons
col1, col2 = st.columns(2)

with col1:
    test_normal = st.button("Test Normal Transaction")

with col2:
    test_fraud = st.button("Test Fraud Transaction")

# Function to predict
def run_prediction(sample):
    X = sample.drop('Class', axis=1)
    y = sample['Class'].values[0]

    prediction = model.predict(X)
    probability = model.predict_proba(X)[0][1]  # fraud probability

    return y, prediction[0], probability

# -------------------------
# NORMAL TRANSACTION
# -------------------------
if test_normal:
    sample = data[data['Class'] == 0].sample(1)

    y, pred, prob = run_prediction(sample)

    st.subheader("🔍 Normal Transaction Test")

    st.write("Actual:", "Normal")
    st.write("Predicted:", "Fraud" if pred == 1 else "Normal")

    st.progress(float(prob))

    st.write(f"Fraud Probability: {prob*100:.2f}%")

    st.success("✅ Normal Transaction Detected")

# -------------------------
# FRAUD TRANSACTION
# -------------------------
if test_fraud:
    sample = data[data['Class'] == 1].sample(1)

    y, pred, prob = run_prediction(sample)

    st.subheader("🚨 Fraud Transaction Test")

    st.write("Actual:", "Fraud")
    st.write("Predicted:", "Fraud" if pred == 1 else "Normal")

    st.progress(float(prob))

    st.write(f"Fraud Probability: {prob*100:.2f}%")

    st.error("🚨 Fraud Detected!")

# -------------------------
# GRAPH SECTION
# -------------------------
st.markdown("---")
st.subheader("📊 Dataset Overview")

fraud_count = data['Class'].value_counts()

fig, ax = plt.subplots()
ax.bar(["Normal", "Fraud"], fraud_count.values)

st.pyplot(fig)
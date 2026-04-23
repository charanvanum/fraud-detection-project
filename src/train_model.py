import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load data
data = pd.read_csv("data/creditcard.csv")

# -------------------------------
# Preprocessing (same as before)
# -------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])
data['Time'] = scaler.fit_transform(data[['Time']])

# -------------------------------
# Handle imbalance (undersampling)
# -------------------------------
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

normal_sample = normal.sample(n=len(fraud), random_state=42)
new_data = pd.concat([fraud, normal_sample])
new_data = new_data.sample(frac=1, random_state=42)

# -------------------------------
# Split data
# -------------------------------
X = new_data.drop('Class', axis=1)
y = new_data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Save model
# -------------------------------
joblib.dump(model, "models/model.pkl")
print("\nModel saved successfully!")
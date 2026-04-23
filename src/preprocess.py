import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data/creditcard.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nClass Distribution:")
print(data['Class'].value_counts())

# -------------------------------
# Step 1: Check missing values
# -------------------------------
print("\nMissing Values:")
print(data.isnull().sum())

# -------------------------------
# Step 2: Feature Scaling
# -------------------------------
scaler = StandardScaler()

data['Amount'] = scaler.fit_transform(data[['Amount']])
data['Time'] = scaler.fit_transform(data[['Time']])

print("\nAfter Scaling:")
print(data.head())

# -------------------------------
# Step 3: Handle Imbalanced Data
# -------------------------------

# Separate fraud and normal
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

print("\nBefore Sampling:")
print(data['Class'].value_counts())

# Undersample normal data
normal_sample = normal.sample(n=len(fraud), random_state=42)

# Combine
new_data = pd.concat([fraud, normal_sample])

# Shuffle dataset
new_data = new_data.sample(frac=1, random_state=42)

print("\nAfter Sampling:")
print(new_data['Class'].value_counts())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load dataset
data = pd.read_csv("german_credit_data.csv")

# Display first rows
print("\nFIRST 5 ROWS:\n")
print(data.head())

# Dataset info
print("\nDATASET INFO:\n")
print(data.info())

# Missing values
print("\nMISSING VALUES:\n")
print(data.isnull().sum())

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Convert categorical columns into numerical
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Features and target
X = data.drop("Risk", axis=1)
y = data["Risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:\n")
print(accuracy)

# Classification report
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nCONFUSION MATRIX:\n")
print(confusion_matrix(y_test, y_pred))

# Histogram plots
data.hist(figsize=(12,10))
plt.show()

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

# Save model
joblib.dump(model, "credit_scoring_model.pkl")

print("\nModel saved successfully!")
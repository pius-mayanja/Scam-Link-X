import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load datasets
legitimate_urls = pd.read_csv('data/legitimate-urls.csv')
phishing_urls = pd.read_csv('data/phishing-urls.csv')

# Assign labels
legitimate_urls["label"] = 0  # Legitimate sites
phishing_urls["label"] = 1  # Phishing sites

# Concatenate both datasets
#data = pd.concat([legitimate_urls, phishing_urls], ignore_index=True)
data = pd.read_csv('data/urls.csv')

# Apply label encoding to categorical columns
label_encoders = {}
for column in data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))  # Convert to numbers
    label_encoders[column] = le  # Save encoders for future use if needed

# Extract features and target
X = data.drop(columns=["status"])  # Features
y = data["status"]  # Target

# Ensure there are no missing values
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
feature_importances = model.feature_importances_
features = X.columns

# Create DataFrame for visualization
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.gca().invert_yaxis()  # Highest importance at top
plt.xlabel("Importance Score")
plt.title("Phishing URL Detection - Feature Importance")
plt.show()

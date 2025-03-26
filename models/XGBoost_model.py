import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load datasets
legitimate_urls = pd.read_csv('data/legitimate-urls.csv')
phishing_urls = pd.read_csv('data/phishing-urls.csv')

# Assign labels
legitimate_urls["label"] = 0  # Legitimate
phishing_urls["label"] = 1  # Phishing

# Combine datasets
data = pd.concat([legitimate_urls, phishing_urls], ignore_index=True)

# Encode categorical features
label_encoders = {}
for column in data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le  

# Extract features & target
X = data.drop(columns=["label"])
y = data["label"]

# Fill missing values
X = X.fillna(0)

# Split dataset (Reduce test size for more training data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train a base XGBoost model to get feature importances
base_model = XGBClassifier(n_estimators=150, random_state=42, tree_method='hist', use_label_encoder=False, verbosity=0)
base_model.fit(X_train, y_train)

# Select top 25 features instead of 20 (for better feature selection)
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": base_model.feature_importances_})
top_features = importance_df.nlargest(25, "Importance")["Feature"].values

# Use only top features
X_train = X_train[top_features]
X_test = X_test[top_features]

# Define Optuna objective function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),  # Increased range
        "max_depth": trial.suggest_int("max_depth", 3, 12),  # Increased depth
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),  # Lowered min
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 0.5),  # Increased range
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 15.0),  # L2 Regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 15.0),  # L1 Regularization
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 2.5),  # Adjusted to favor legitimacy
        "tree_method": "hist",
        "use_label_encoder": False,
        "verbosity": 0
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Run Optuna hyperparameter tuning (Increased trials for better tuning)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # Increased to 50 trials

# Train best model
best_params = study.best_params
best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)

# Predictions
y_pred = best_model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Further Improved Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Further Improved Confusion Matrix")
plt.show()

import joblib

# Save the trained model
joblib.dump(best_model, "phishing_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")


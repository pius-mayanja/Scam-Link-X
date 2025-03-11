import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, classification_report
import optuna
import joblib
from preprocess import load_data, preprocess_data


data = load_data()
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handle Class Imbalance
smt = SMOTETomek(random_state=42)
X_train, y_train = smt.fit_resample(X_train, y_train)

# Hyperparameter Optimization using Optuna
def objective(trial):
    model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0)
    )
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params

# Train Models
models = {
    'XGBoost': XGBClassifier(**best_params),
    'RandomForest': RandomForestClassifier(n_estimators=200),
    'CatBoost': CatBoostClassifier(iterations=200, silent=True),
    'LightGBM': LGBMClassifier(n_estimators=200)
}

best_model, best_score = None, 0
for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_proba)
    print(f"{name} AUC: {score:.4f}")
    if score > best_score:
        best_model, best_score = model, score

# Optimize Classification Threshold
precision, recall, thresholds = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
optimal_threshold = thresholds[np.argmax(precision * recall)]

# Final Predictions
y_pred = (best_model.predict_proba(X_test)[:, 1] > optimal_threshold).astype(int)

# Final Evaluation
print("\nBest Model:", type(best_model).__name__)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC Score:", best_score)
print(classification_report(y_test, y_pred))


joblib.dump(best_model, "scam_link_detector.pkl")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# Load Data
def load_data():
    legitimate = pd.read_csv('data/legitimate-urls.csv')
    phishing = pd.read_csv('data/phishing-urls.csv')
    legitimate['label'] = 0
    phishing['label'] = 1
    return pd.concat([legitimate, phishing], ignore_index=True)

# Feature Engineering
def add_features(df):
    df['num_digits'] = df['Domain'].str.count(r'\d')
    df['special_char_count'] = df['Domain'].apply(lambda x: sum(1 for c in x if c in "-_?="))
    df['num_subdomains'] = df['Domain'].str.count(r'\.')
    df['digit_to_letter_ratio'] = df['num_digits'] / df['Domain'].apply(len)
    df.drop(columns=['Domain', 'Protocol', 'Path'], inplace=True)
    return df

# Preprocessing
def preprocess_data(df):
    df.dropna(inplace=True)
    df = add_features(df)
    X, y = df.drop(columns=['label']), df['label']
    scaler = StandardScaler()
    X[X.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(X.select_dtypes(include=['int64', 'float64']))
    
    joblib.dump(scaler, 'scaler.joblib')
    return X, y

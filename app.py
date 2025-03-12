from fastapi import FastAPI
from pydantic import BaseModel
from urllib.parse import urlparse
import pandas as pd
from src.model import load_model_artifacts, train_and_save_model


app = FastAPI()

# Load model artifacts
model, scaler, optimal_threshold, feature_columns = load_model_artifacts()

class URLRequest(BaseModel):
    url: str

def parse_url(url: str):
    parsed = urlparse(url)
    return {
        'Protocol': parsed.scheme,
        'Domain': parsed.netloc,
        'Path': parsed.path
    }

def extract_features(parsed_url):
    df = pd.DataFrame([parsed_url])
    df['num_digits'] = df['Domain'].str.count(r'\d')
    df['special_char_count'] = df['Domain'].apply(lambda x: sum(1 for c in x if c in "-_?="))
    df['num_subdomains'] = df['Domain'].str.count(r'\.')
    df['digit_to_letter_ratio'] = df['num_digits'] / (df['Domain'].str.len() + 1e-6)
    df.drop(['Domain', 'Protocol', 'Path'], axis=1, inplace=True)
    return df

@app.post('/predict')
async def predict(request: URLRequest):
    try:
        parsed = parse_url(request.url)
        features = extract_features(parsed)
        features = features.reindex(columns=feature_columns, fill_value=0)
        scaled = scaler.transform(features)
        proba = model.predict_proba(scaled)[0][1]
        prediction_int = 1 if proba > optimal_threshold else 0
        prediction_str = "Scam" if prediction_int == 1 else "Safe"
        return {"prediction": prediction_str, "probability": float(proba)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
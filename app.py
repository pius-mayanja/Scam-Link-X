# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from preprocess import add_features

app = FastAPI()

# Define Pydantic model for incoming data
class URLData(BaseModel):
    Domain: str

# Load the trained model
def load_trained_model():
    with open('scam_link_detector.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Preprocess input data
def preprocess_input(data: URLData):
    input_df = pd.DataFrame([data.dict()])
    input_df = add_features(input_df)
    scaler = StandardScaler()
    input_df[input_df.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(input_df.select_dtypes(include=['int64', 'float64']))
    return input_df

# Define prediction endpoint
@app.post("/predict/")
async def predict(data: URLData):
    model = load_trained_model()
    
    # Preprocess input data
    input_df = preprocess_input(data)
    
    # Predict using the trained model
    prediction = model.predict(input_df)
    
    return {"prediction": int(prediction[0])}

# Run the FastAPI app (use `uvicorn` to run the API in the command line)
# uvicorn main:app --reload

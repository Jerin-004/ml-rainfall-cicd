from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json

class PredictRequest(BaseModel):
    features: list

app = FastAPI()

# Load trained model
model = joblib.load("artifacts/model.joblib")

# Load feature names from metadata.json
with open("artifacts/metadata.json") as f:
    meta = json.load(f)
feature_names = meta["features"]

@app.get("/")
def home():
    return {"message": "Rainfall Prediction API is running ✅"}

@app.post("/predict")
def predict(request: PredictRequest):
    # Convert input to DataFrame with correct column names
    df = pd.DataFrame([request.features], columns=feature_names)
    pred = model.predict(df)[0]
    return {"rainfall_prediction": "yes" if pred == 1 else "no"}

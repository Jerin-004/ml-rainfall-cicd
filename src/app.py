from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Pydantic request model
class PredictRequest(BaseModel):
    features: list

app = FastAPI()

# Load the trained model
model = joblib.load("artifacts/model.joblib")

@app.get("/")
def home():
    return {"message": "Rainfall Prediction API is running ✅"}

@app.post("/predict")
def predict(request: PredictRequest):
    arr = np.array(request.features).reshape(1, -1)
    pred = model.predict(arr)[0]
    return {"rainfall_prediction": "yes" if pred == 1 else "no"}

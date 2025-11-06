from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model at server start
model = joblib.load("artifacts/model.joblib")

@app.get("/")
def home():
    return {"message": "Rainfall Prediction Model is Live!"}

@app.post("/predict")
def predict(features: list):
    # features must be in the same order as training columns
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)[0]
    return {"rainfall_prediction": "yes" if pred == 1 else "no"}

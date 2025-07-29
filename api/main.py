from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and feature list
model = joblib.load("model.joblib")
features = joblib.load("features.joblib")

app = FastAPI(title="Churn Prediction API")

# Define request body format using Pydantic
class ChurnRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: int  # encoded as integer (e.g., 0 = Month-to-month)

@app.post("/predict")
def predict_churn(data: ChurnRequest):
    try:
        # Convert request data to input format
        input_data = np.array([[
            data.tenure,
            data.MonthlyCharges,
            data.TotalCharges,
            data.Contract
        ]])

        # Ensure input matches model features
        if len(input_data[0]) != len(features):
            raise ValueError("Incorrect number of features")

        # Make prediction and return both label and probability
        prediction = model.predict(input_data)[0]
        probability = float(model.predict_proba(input_data)[0][1])

        return {
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

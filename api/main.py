
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and features
model = joblib.load("model.joblib")
features = joblib.load("features.joblib")

# Define Pydantic model to match all features used in training

class ChurnRequest(BaseModel):



app = FastAPI(title="Churn Prediction API")

@app.post("/predict")
def predict_churn(data: ChurnRequest):
    try:
        input_array = np.array([[getattr(data, f) for f in features]])
        if input_array.shape[1] != len(features):
            raise ValueError("Incorrect number of features")
        prediction = model.predict(input_array)[0]
        probability = float(model.predict_proba(input_array)[0][1])
        return {
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

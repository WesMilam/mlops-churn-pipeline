from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import joblib
import numpy as np
import csv
from datetime import datetime
import os

# Load model and features
model = joblib.load("model.joblib")
features = joblib.load("features.joblib")

# Enums for categorical fields
class Contract(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"

class InternetService(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"

class PaymentMethod(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"

class Gender(str, Enum):
    male = "Male"
    female = "Female"

# Input schema
class ChurnRequest(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: Contract
    InternetService: InternetService
    OnlineSecurity: float
    OnlineBackup: float
    DeviceProtection: float
    TechSupport: float
    StreamingTV: float
    StreamingMovies: float
    PaperlessBilling: float
    PaymentMethod: PaymentMethod
    SeniorCitizen: float
    Partner: float
    Dependents: float
    PhoneService: float
    MultipleLines: float
    gender: Gender

# Mappings
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}
gender_map = {"Female": 0, "Male": 1}

# App
app = FastAPI(title="Churn Prediction API with Logging")

LOG_FILE = "inference_log.csv"

def log_to_csv(input_dict, prediction, probability):
    header = list(input_dict.keys()) + ["churn_prediction", "churn_probability", "timestamp_utc"]
    row = list(input_dict.values()) + [prediction, round(probability, 4), datetime.utcnow().isoformat()]
    write_header = not os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

@app.post("/predict")
def predict_churn(data: ChurnRequest):
    try:
        input_data = [
            data.tenure,
            data.MonthlyCharges,
            data.TotalCharges,
            contract_map[data.Contract.value],
            internet_map[data.InternetService.value],
            data.OnlineSecurity,
            data.OnlineBackup,
            data.DeviceProtection,
            data.TechSupport,
            data.StreamingTV,
            data.StreamingMovies,
            data.PaperlessBilling,
            payment_map[data.PaymentMethod.value],
            data.SeniorCitizen,
            data.Partner,
            data.Dependents,
            data.PhoneService,
            data.MultipleLines,
            gender_map[data.gender.value]
        ]

        input_array = np.array([input_data])
        prediction = model.predict(input_array)[0]
        probability = float(model.predict_proba(input_array)[0][1])

        input_dict = data.dict()
        log_to_csv(input_dict, int(prediction), probability)

        return {
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import joblib
import numpy as np

# Load model and features
model = joblib.load("model.joblib")
features = joblib.load("features.joblib")

# Define Enums for categorical fields
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

# Define input model
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

# Category mappings
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}
gender_map = {"Female": 0, "Male": 1}

# FastAPI app
app = FastAPI(title="Churn Prediction API (Human-Friendly)")

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

        return {
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

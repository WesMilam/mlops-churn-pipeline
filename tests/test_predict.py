from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

valid_payload = {
    "tenure": 24,
    "MonthlyCharges": 75.25,
    "TotalCharges": 1805,
    "Contract": "One year",
    "InternetService": "Fiber optic",
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 1,
    "PaperlessBilling": 1,
    "PaymentMethod": "Credit card (automatic)",
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "PhoneService": 1,
    "MultipleLines": 0,
    "gender": "Male"
}

def test_predict_valid_input():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data

def test_predict_missing_field():
    invalid_payload = valid_payload.copy()
    invalid_payload.pop("gender")
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_predict_invalid_type():
    invalid_payload = valid_payload.copy()
    invalid_payload["tenure"] = "twenty-four"  # Should be float
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_predict_extra_field():
    extended_payload = valid_payload.copy()
    extended_payload["random_extra_field"] = "not_allowed"
    response = client.post("/predict", json=extended_payload)
    assert response.status_code == 200  # FastAPI should ignore extra fields unless forbidden
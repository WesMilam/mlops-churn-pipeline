import streamlit as st
import requests

st.title("📊 Churn Prediction App")
st.subheader("Predict customer churn based on contract and service behavior")

API_URL = "https://mlops-churn-pipeline-production.up.railway.app/predict"

with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges", value=70.0)
    TotalCharges = st.number_input("Total Charges", value=1400.0)
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.checkbox("Online Security")
    OnlineBackup = st.checkbox("Online Backup")
    DeviceProtection = st.checkbox("Device Protection")
    TechSupport = st.checkbox("Tech Support")
    StreamingTV = st.checkbox("Streaming TV")
    StreamingMovies = st.checkbox("Streaming Movies")
    PaperlessBilling = st.checkbox("Paperless Billing")
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    SeniorCitizen = st.checkbox("Senior Citizen")
    Partner = st.checkbox("Partner")
    Dependents = st.checkbox("Dependents")
    PhoneService = st.checkbox("Phone Service")
    MultipleLines = st.checkbox("Multiple Lines")

    submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "gender": gender,
            "tenure": tenure,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
            "Contract": Contract,
            "InternetService": InternetService,
            "OnlineSecurity": float(OnlineSecurity),
            "OnlineBackup": float(OnlineBackup),
            "DeviceProtection": float(DeviceProtection),
            "TechSupport": float(TechSupport),
            "StreamingTV": float(StreamingTV),
            "StreamingMovies": float(StreamingMovies),
            "PaperlessBilling": float(PaperlessBilling),
            "PaymentMethod": PaymentMethod,
            "SeniorCitizen": float(SeniorCitizen),
            "Partner": float(Partner),
            "Dependents": float(Dependents),
            "PhoneService": float(PhoneService),
            "MultipleLines": float(MultipleLines)
        }

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()

            st.success(f"Prediction: {'Will Churn' if result['churn_prediction'] else 'Will Not Churn'}")
            st.metric(label="Churn Probability", value=f"{result['churn_probability']:.2%}")

        except Exception as e:
            st.error(f"API error: {e}")
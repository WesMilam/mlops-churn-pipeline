# 🧠 MLOps Churn Prediction Pipeline

This project demonstrates a full end-to-end MLOps pipeline for predicting customer churn using modern tools like FastAPI, MLflow, Docker, GitHub Actions, and Streamlit.

---

## 🚀 Features

- ✅ Model training with Scikit-learn and experiment tracking via MLflow
- ✅ RESTful API served with FastAPI (Dockerized)
- ✅ Live deployment via Railway with `/predict` and `/docs` endpoints
- ✅ Logging of every prediction request (CSV audit trail)
- ✅ CI/CD pipeline with linting and tests via GitHub Actions
- ✅ Streamlit frontend for human-friendly interaction

---

## 📦 Tech Stack

- Python 3.10
- Scikit-learn, Pandas, NumPy
- FastAPI, Uvicorn
- MLflow, Joblib
- Docker
- GitHub Actions
- Streamlit
- Railway (Cloud hosting)

---

## 🧪 Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/mlops-churn-pipeline.git
cd mlops-churn-pipeline
```

### 2. Create and Activate a Virtual Environment

```bash
conda create -n mlops-churn python=3.10
conda activate mlops-churn
pip install -r requirements.txt
```

### 3. Start the FastAPI Server

```bash
uvicorn api.main:app --reload
```

Then open your browser to:
```
http://localhost:8000/docs
```

---

## 📊 Run the Streamlit Frontend

```bash
streamlit run app.py
```

Then visit:
```
http://localhost:8501
```

---

## 🌐 Live Demo (Deployed)

- 🚀 **API**: [https://mlops-churn-pipeline-production.up.railway.app/docs](https://mlops-churn-pipeline-production.up.railway.app/docs)
- 💻 **Frontend**: Run `streamlit run app.py` locally (or deploy to Streamlit Cloud)

---

## ✅ CI/CD Pipeline

Every push to `main` triggers GitHub Actions:

- Linting with flake8
- Tests for API health and `/predict` route
- Auto-deployment to Railway

Badge (add this after first public push):

```
![CI](https://github.com/YOUR_USERNAME/mlops-churn-pipeline/actions/workflows/deploy.yml/badge.svg)
```

---

## 📁 Project Structure

```
├── api/
│   ├── main.py
│   └── __init__.py
├── tests/
│   ├── test_api.py
│   └── test_predict.py
├── model.joblib
├── features.joblib
├── app.py
├── Dockerfile
├── requirements.txt
├── .github/workflows/deploy.yml
└── README.md
```

---

## ✨ Credits

Built by @DrWesMilam as part of a professional MLOps portfolio.
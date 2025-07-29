import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import os

def load_data(filepath=None):
    if filepath and os.path.exists(filepath):
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
    else:
        print("No data file found. Generating synthetic dataset.")
        np.random.seed(42)
        size = 1000
        df = pd.DataFrame({
            'tenure': np.random.randint(1, 72, size),
            'MonthlyCharges': np.random.uniform(20.0, 120.0, size),
            'TotalCharges': np.random.uniform(20.0, 9000.0, size),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size),
            'Churn': np.random.choice(['Yes', 'No'], size)
        })
    return df

def preprocess(df):
    df = df.copy()

    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Convert TotalCharges to numeric (real Telco data issue)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col == 'Churn':
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop rows with missing values
    df = df.dropna()

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    return clf, acc, f1

def main(filepath=None):
    mlflow.set_experiment("Churn-Prediction")
    with mlflow.start_run():
        df = load_data(filepath)
        X_train, X_test, y_train, y_test = preprocess(df)
        model, acc, f1 = train_and_evaluate(X_train, X_test, y_train, y_test)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        print(f"Model Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    main(filepath)

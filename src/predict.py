import joblib
import numpy as np

MODEL_PATH = "C:\\Users\\sajan\\Documents\\GitHub\\End-to-End-Customer-Churn-Prediction-System\\models\\churn_model.pkl"
SCALER_PATH = "C:\\Users\\sajan\\Documents\\GitHub\\End-to-End-Customer-Churn-Prediction-System\\models\\scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_churn(features: list) -> float:
    """
    features: list of numerical feature values in correct order
    returns: churn probability
    """
    features_scaled = scaler.transform([features])
    probability = model.predict_proba(features_scaled)[0][1]
    return probability

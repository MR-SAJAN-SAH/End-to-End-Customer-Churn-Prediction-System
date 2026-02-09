import os
import json
import streamlit as st
import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "models", "feature_names.json")


#Load Artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

#Streamlit UI

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction System")
st.write(
    "Predicts the probability of a customer leaving a service "
    "based on account and usage information."
)

st.markdown("---")
st.subheader("Enter Customer Details")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges (€)", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges (€)", 0.0, 10000.0, 2000.0)

contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

#Encoding Logic

def encode_inputs():
    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract_One year": int(contract_type == "One year"),
        "Contract_Two year": int(contract_type == "Two year"),
        "InternetService_Fiber optic": int(internet_service == "Fiber optic"),
        "InternetService_No": int(internet_service == "No"),
        "PaymentMethod_Credit card (automatic)": int(payment_method == "Credit card (automatic)"),
        "PaymentMethod_Electronic check": int(payment_method == "Electronic check"),
        "PaymentMethod_Mailed check": int(payment_method == "Mailed check"),
    }

    input_df = pd.DataFrame([data])

    for col in FEATURE_NAMES:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[FEATURE_NAMES]


if st.button("Predict Churn"):
    input_df = encode_inputs()
    scaled_input = scaler.transform(input_df)

    churn_prob = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if churn_prob > 0.5:
        st.error(f"High Churn Risk: {churn_prob:.2%}")
    else:
        st.success(f"Low Churn Risk: {churn_prob:.2%}")

    st.caption("Prediction generated using a trained machine learning model.")

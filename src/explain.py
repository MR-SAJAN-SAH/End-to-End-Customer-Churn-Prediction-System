import shap
import joblib
import pandas as pd

pipeline = joblib.load("models/churn_pipeline.pkl")
model = pipeline.named_steps["model"]

explainer = shap.Explainer(model)
print("SHAP explainer ready")

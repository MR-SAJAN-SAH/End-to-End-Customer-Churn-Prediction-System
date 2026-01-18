import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)

from preprocessing import preprocess_pipeline


# =========================
# Path Configuration
# =========================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "C:\\Users\\sajan\\Documents\\GitHub\\End-to-End-Customer-Churn-Prediction-System\\models\\churn_model.pkl"
DATA_PATH = "C:\\Users\\sajan\\Documents\\GitHub\\End-to-End-Customer-Churn-Prediction-System\\data\\raw\\telco_customer_churn_dataset.csv"



def evaluate_model():
    print("\n🔍 Loading model and data...\n")

    model = joblib.load(MODEL_PATH)

    X_train, X_test, y_train, y_test, _ = preprocess_pipeline(DATA_PATH)

    # =========================
    # Predictions
    # =========================
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # =========================
    # Classification Metrics
    # =========================
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc:.4f}\n")

    # =========================
    # Confusion Matrix
    # =========================
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Customer Churn Prediction")
    plt.show()

    # =========================
    # Model Explainability
    # =========================
    explain_model(model, X_train)


def explain_model(model, X_train):
    """
    Explain Logistic Regression model using feature coefficients
    """
    if model.__class__.__name__ != "LogisticRegression":
        print("Explainability currently supported only for Logistic Regression.")
        return

    print("Model Explainability (Top Features)\n")

    feature_names = model.feature_names_in_
    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "AbsCoefficient": np.abs(coefficients)
    }).sort_values(by="AbsCoefficient", ascending=False)

    print(importance_df.head(10).to_string(index=False))

    # Plot top features
    plt.figure(figsize=(8, 5))
    plt.barh(
        importance_df["Feature"].head(10),
        importance_df["Coefficient"].head(10)
    )
    plt.gca().invert_yaxis()
    plt.title("Top 10 Features Influencing Churn")
    plt.xlabel("Coefficient Value")
    plt.show()


if __name__ == "__main__":
    evaluate_model()


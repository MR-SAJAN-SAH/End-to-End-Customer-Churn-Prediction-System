import os
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from preprocessing import preprocess_pipeline


# =========================
# Path Configuration
# =========================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "telco_customer_churn_dataset.csv"
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.json")


def train_models():
    print("\n Starting model training...\n")

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load and preprocess data (with feature names)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(
        DATA_PATH,
        return_feature_names=True
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    }

    best_model = None
    best_score = 0.0

    # =========================
    # Model Training & Selection
    # =========================
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, probs)

        print(f"{name} ROC-AUC: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model

    # =========================
    # Save Artifacts
    # =========================
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f)

    print("\nTraining completed successfully")
    print(f"Best ROC-AUC: {best_score:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Feature names saved to: {FEATURES_PATH}")


if __name__ == "__main__":
    train_models()

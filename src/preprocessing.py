import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop ID column
    df.drop(columns=["customerID"], inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop missing values
    df.dropna(inplace=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded


def split_and_scale(df, return_feature_names=False):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if return_feature_names:
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def preprocess_pipeline(path, return_feature_names=False):
    df = load_data(path)
    df = clean_data(df)
    df = encode_features(df)

    return split_and_scale(df, return_feature_names)


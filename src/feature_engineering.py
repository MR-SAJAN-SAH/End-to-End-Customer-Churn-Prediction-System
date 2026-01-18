import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Example engineered feature
    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df["ChargesPerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    return df

"""
Feature Engineering for Telco Churn Prediction.

Adds domain-driven derived features that capture business logic
not directly expressed in the raw columns.
"""

import pandas as pd


SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CONTRACT_RISK_MAP = {
    "Two year": 0,
    "One year": 1,
    "Month-to-month": 2,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the dataframe (in-place copy).

    New columns:
    - charges_per_tenure_month : MonthlyCharges / (tenure + 1)
        Captures cost burden relative to how long the customer has been around.
    - service_count : number of active add-on services (0–8)
        Customers with more services tend to be more engaged.
    - is_high_value : 1 if MonthlyCharges > 70th percentile
        Flag for customers generating above-average revenue.
    - is_long_tenure : 1 if tenure > 24 months
        Long-tenured customers are significantly less likely to churn.
    - contract_risk : ordinal encoding (Two year=0, One year=1, M2M=2)
        Numeric proxy for contract commitment level.

    Args:
        df: Raw or partially processed DataFrame.

    Returns:
        New DataFrame with additional feature columns.
    """
    df = df.copy()

    # Charges-to-tenure ratio
    df["charges_per_tenure_month"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Count active services (Yes = active; other values = inactive)
    df["service_count"] = sum(
        (df[col] == "Yes").astype(int) for col in SERVICE_COLUMNS if col in df.columns
    )

    # High-value customer flag
    threshold = df["MonthlyCharges"].quantile(0.70)
    df["is_high_value"] = (df["MonthlyCharges"] > threshold).astype(int)

    # Long-tenure flag
    df["is_long_tenure"] = (df["tenure"] > 24).astype(int)

    # Contract risk (ordinal)
    df["contract_risk"] = df["Contract"].map(CONTRACT_RISK_MAP).fillna(2).astype(int)

    return df


def get_engineered_feature_names() -> list[str]:
    """Return list of feature names added by engineer_features()."""
    return [
        "charges_per_tenure_month",
        "service_count",
        "is_high_value",
        "is_long_tenure",
        "contract_risk",
    ]

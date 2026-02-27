"""
Data Preprocessing Pipeline.

Provides a DataPreprocessor class that wraps a sklearn ColumnTransformer
to handle numeric scaling and categorical encoding in a single, reproducible
pipeline that can be fit on train data and applied to test data.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Features used for modeling (excludes customerID and target)
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Engineered feature names added by features.py
ENGINEERED_NUMERIC = [
    "charges_per_tenure_month",
    "service_count",
    "is_high_value",
    "is_long_tenure",
    "contract_risk",
]

TARGET = "Churn"


class DataPreprocessor:
    """
    Sklearn-compatible preprocessor for the Telco Churn dataset.

    Applies:
    - Median imputation + standard scaling to numeric columns
    - Mode imputation + one-hot encoding to categorical columns

    Usage:
        preprocessor = DataPreprocessor()
        X_train = preprocessor.fit_transform(df_train)
        X_test  = preprocessor.transform(df_test)
    """

    def __init__(self, include_engineered: bool = True):
        self.include_engineered = include_engineered
        self._transformer: ColumnTransformer | None = None
        self._feature_names: list[str] = []

    def _build_transformer(self, numeric_cols: list[str]) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in self._input_columns]
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, cat_cols),
            ],
            remainder="drop",
        )

    def _get_numeric_cols(self, df: pd.DataFrame) -> list[str]:
        base = [c for c in NUMERIC_FEATURES if c in df.columns]
        if self.include_engineered:
            base += [c for c in ENGINEERED_NUMERIC if c in df.columns]
        return base

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit the preprocessor on df and return the transformed array."""
        self._input_columns = df.columns.tolist()
        numeric_cols = self._get_numeric_cols(df)
        self._transformer = self._build_transformer(numeric_cols)
        X = self._transformer.fit_transform(df)
        self._feature_names = self._build_feature_names(numeric_cols)
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform df using the already-fitted preprocessor."""
        if self._transformer is None:
            raise RuntimeError("Call fit_transform() before transform().")
        return self._transformer.transform(df)

    def get_feature_names(self) -> list[str]:
        """Return the list of output feature names after encoding."""
        return self._feature_names

    def _build_feature_names(self, numeric_cols: list[str]) -> list[str]:
        names = list(numeric_cols)
        cat_encoder = self._transformer.named_transformers_["cat"]["encoder"]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in self._input_columns]
        for col, cats in zip(cat_cols, cat_encoder.categories_):
            names += [f"{col}_{cat}" for cat in cats]
        return names


def prepare_target(df: pd.DataFrame) -> pd.Series:
    """Convert Churn column to binary (1=Yes, 0=No)."""
    return (df[TARGET] == "Yes").astype(int)

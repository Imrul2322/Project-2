"""Unit tests for src/features.py."""

import numpy as np
import pandas as pd
import pytest

from src.data_generation import generate_telco_data
from src.features import engineer_features, get_engineered_feature_names

NEW_FEATURES = [
    "charges_per_tenure_month",
    "service_count",
    "is_high_value",
    "is_long_tenure",
    "contract_risk",
]


@pytest.fixture(scope="module")
def df_raw():
    return generate_telco_data(n_samples=300, random_state=42)


@pytest.fixture(scope="module")
def df_eng(df_raw):
    return engineer_features(df_raw)


def test_new_columns_added(df_eng):
    """All engineered feature columns must be present."""
    for col in NEW_FEATURES:
        assert col in df_eng.columns, f"Missing column: {col}"


def test_original_columns_preserved(df_raw, df_eng):
    """Original columns should still be present after engineering."""
    for col in df_raw.columns:
        assert col in df_eng.columns


def test_no_new_missing_values(df_raw, df_eng):
    """Feature engineering should not introduce NaN values."""
    original_nulls = df_raw.isnull().sum().sum()
    new_nulls = df_eng[NEW_FEATURES].isnull().sum().sum()
    assert new_nulls == 0, f"Feature engineering introduced {new_nulls} nulls"


def test_charges_per_tenure_positive(df_eng):
    """charges_per_tenure_month should always be positive."""
    assert (df_eng["charges_per_tenure_month"] > 0).all()


def test_service_count_range(df_eng):
    """service_count should be in [0, 8]."""
    assert df_eng["service_count"].min() >= 0
    assert df_eng["service_count"].max() <= 8


def test_is_high_value_binary(df_eng):
    """is_high_value should only contain 0 or 1."""
    assert set(df_eng["is_high_value"].unique()).issubset({0, 1})


def test_is_long_tenure_binary(df_eng):
    """is_long_tenure should only contain 0 or 1."""
    assert set(df_eng["is_long_tenure"].unique()).issubset({0, 1})


def test_long_tenure_threshold(df_eng):
    """Customers with tenure > 24 must have is_long_tenure == 1."""
    long = df_eng[df_eng["tenure"] > 24]["is_long_tenure"]
    assert (long == 1).all(), "Some long-tenure customers are not flagged"

    short = df_eng[df_eng["tenure"] <= 24]["is_long_tenure"]
    assert (short == 0).all(), "Some short-tenure customers are incorrectly flagged"


def test_contract_risk_values(df_eng):
    """contract_risk must be in {0, 1, 2}."""
    assert set(df_eng["contract_risk"].unique()).issubset({0, 1, 2})


def test_contract_risk_ordering(df_eng):
    """Two-year=0 < One-year=1 < Month-to-month=2."""
    two_yr = df_eng[df_eng["Contract"] == "Two year"]["contract_risk"].unique()
    one_yr = df_eng[df_eng["Contract"] == "One year"]["contract_risk"].unique()
    m2m = df_eng[df_eng["Contract"] == "Month-to-month"]["contract_risk"].unique()
    assert two_yr[0] == 0
    assert one_yr[0] == 1
    assert m2m[0] == 2


def test_get_engineered_feature_names():
    """Should return the list of 5 engineered feature names."""
    names = get_engineered_feature_names()
    assert names == NEW_FEATURES


def test_input_not_mutated(df_raw):
    """engineer_features should not modify the input DataFrame."""
    original_cols = df_raw.columns.tolist()
    engineer_features(df_raw)  # should return a copy
    assert df_raw.columns.tolist() == original_cols

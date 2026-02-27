"""Unit tests for src/data_generation.py."""

import numpy as np
import pandas as pd
import pytest

from src.data_generation import generate_telco_data

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


@pytest.fixture(scope="module")
def sample_df():
    return generate_telco_data(n_samples=500, random_state=0)


def test_shape(sample_df):
    """Dataset should have the requested number of rows and 21 columns."""
    assert sample_df.shape == (500, 21)


def test_columns(sample_df):
    """All expected columns must be present."""
    assert list(sample_df.columns) == EXPECTED_COLUMNS


def test_no_missing_values(sample_df):
    """Generated data should contain no nulls."""
    assert sample_df.isnull().sum().sum() == 0


def test_churn_rate_realistic(sample_df):
    """Churn rate should be in the 20–45% range for any seed."""
    rate = (sample_df["Churn"] == "Yes").mean()
    assert 0.20 <= rate <= 0.45, f"Churn rate {rate:.2%} is outside expected range"


def test_churn_binary(sample_df):
    """Churn column should only contain 'Yes' and 'No'."""
    assert set(sample_df["Churn"].unique()) == {"Yes", "No"}


def test_tenure_range(sample_df):
    """Tenure should be between 1 and 72 months."""
    assert sample_df["tenure"].min() >= 1
    assert sample_df["tenure"].max() <= 72


def test_monthly_charges_range(sample_df):
    """Monthly charges should be non-negative and capped at 120."""
    assert sample_df["MonthlyCharges"].min() >= 0
    assert sample_df["MonthlyCharges"].max() <= 120


def test_total_charges_positive(sample_df):
    """Total charges should be non-negative."""
    assert (sample_df["TotalCharges"] >= 0).all()


def test_contract_values(sample_df):
    """Contract column must contain exactly 3 valid categories."""
    expected = {"Month-to-month", "One year", "Two year"}
    assert set(sample_df["Contract"].unique()) == expected


def test_reproducibility():
    """Same seed should produce identical DataFrames."""
    df1 = generate_telco_data(n_samples=100, random_state=99)
    df2 = generate_telco_data(n_samples=100, random_state=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ():
    """Different seeds should produce different data."""
    df1 = generate_telco_data(n_samples=100, random_state=1)
    df2 = generate_telco_data(n_samples=100, random_state=2)
    assert not df1["Churn"].equals(df2["Churn"])


def test_contract_churn_correlation(sample_df):
    """Month-to-month churn rate should exceed two-year churn rate."""
    m2m_rate = (
        sample_df[sample_df["Contract"] == "Month-to-month"]["Churn"] == "Yes"
    ).mean()
    two_yr_rate = (
        sample_df[sample_df["Contract"] == "Two year"]["Churn"] == "Yes"
    ).mean()
    assert m2m_rate > two_yr_rate, (
        f"M2M churn {m2m_rate:.2%} should exceed 2yr churn {two_yr_rate:.2%}"
    )

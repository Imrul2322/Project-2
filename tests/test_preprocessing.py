"""Unit tests for src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data_generation import generate_telco_data
from src.features import engineer_features
from src.preprocessing import DataPreprocessor, prepare_target


@pytest.fixture(scope="module")
def splits():
    df = generate_telco_data(n_samples=400, random_state=0)
    df = engineer_features(df)
    y = prepare_target(df)
    feature_df = df.drop(columns=["customerID", "Churn"])
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.25, stratify=y, random_state=0
    )
    return X_train, X_test, y_train, y_test


def test_fit_transform_returns_array(splits):
    """fit_transform should return a numpy array."""
    X_train, _, _, _ = splits
    prep = DataPreprocessor()
    X = prep.fit_transform(X_train)
    assert isinstance(X, np.ndarray)


def test_output_shape_no_nan(splits):
    """Transformed output must have no NaN values."""
    X_train, X_test, _, _ = splits
    prep = DataPreprocessor()
    X_tr = prep.fit_transform(X_train)
    X_te = prep.transform(X_test)
    assert not np.isnan(X_tr).any(), "Train set has NaN values"
    assert not np.isnan(X_te).any(), "Test set has NaN values"


def test_train_test_same_n_features(splits):
    """Train and test must have the same number of features."""
    X_train, X_test, _, _ = splits
    prep = DataPreprocessor()
    X_tr = prep.fit_transform(X_train)
    X_te = prep.transform(X_test)
    assert X_tr.shape[1] == X_te.shape[1]


def test_feature_names_length(splits):
    """Feature names list length must match the matrix width."""
    X_train, _, _, _ = splits
    prep = DataPreprocessor()
    X = prep.fit_transform(X_train)
    names = prep.get_feature_names()
    assert len(names) == X.shape[1]


def test_transform_before_fit_raises():
    """transform() before fit_transform() must raise RuntimeError."""
    prep = DataPreprocessor()
    df = generate_telco_data(n_samples=50, random_state=0)
    with pytest.raises(RuntimeError):
        prep.transform(df)


def test_prepare_target_binary():
    """prepare_target should produce a binary Series."""
    df = generate_telco_data(n_samples=200, random_state=0)
    y = prepare_target(df)
    assert set(y.unique()).issubset({0, 1})


def test_prepare_target_matches_churn(splits):
    """prepare_target: Churn==Yes  <->  y==1."""
    df = generate_telco_data(n_samples=200, random_state=0)
    y = prepare_target(df)
    assert (y[df["Churn"] == "Yes"] == 1).all()
    assert (y[df["Churn"] == "No"] == 0).all()

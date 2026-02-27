"""Unit tests for src/models.py."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.data_generation import generate_telco_data
from src.features import engineer_features
from src.models import MODELS, ModelTrainer
from src.preprocessing import DataPreprocessor, prepare_target


@pytest.fixture(scope="module")
def trained_trainer():
    """Train a ModelTrainer on a small dataset once for all tests in this module."""
    df = generate_telco_data(n_samples=600, random_state=42)
    df = engineer_features(df)
    y = prepare_target(df)
    feature_df = df.drop(columns=["customerID", "Churn"])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.25, stratify=y, random_state=42
    )
    prep = DataPreprocessor()
    X_train = prep.fit_transform(X_train_raw)
    X_test = prep.transform(X_test_raw)

    trainer = ModelTrainer(cv_folds=3)
    trainer.train_all(X_train, y_train)
    return trainer, X_test, y_test


def test_models_dict_not_empty():
    """MODELS dict must contain at least 4 entries."""
    assert len(MODELS) >= 4


def test_all_models_fitted(trained_trainer):
    """All models should be present in fitted_models_ after training."""
    trainer, _, _ = trained_trainer
    for name in MODELS:
        assert name in trainer.fitted_models_


def test_best_model_name_set(trained_trainer):
    """best_model_name_ must be one of the known model keys."""
    trainer, _, _ = trained_trainer
    assert trainer.best_model_name_ in MODELS


def test_evaluate_keys(trained_trainer):
    """evaluate() must return all expected metric keys."""
    trainer, X_test, y_test = trained_trainer
    metrics = trainer.evaluate(trainer.best_model_, X_test, y_test)
    for key in ["roc_auc", "f1", "precision", "recall", "confusion_matrix", "y_pred", "y_prob"]:
        assert key in metrics, f"Missing key: {key}"


def test_roc_auc_above_random(trained_trainer):
    """ROC-AUC on test set should exceed 0.6 (better than random)."""
    trainer, X_test, y_test = trained_trainer
    metrics = trainer.evaluate(trainer.best_model_, X_test, y_test)
    assert metrics["roc_auc"] > 0.6, f"AUC {metrics['roc_auc']:.3f} is too low"


def test_evaluate_all_returns_dataframe(trained_trainer):
    """evaluate_all() should return a DataFrame with a Model column."""
    trainer, X_test, y_test = trained_trainer
    df = trainer.evaluate_all(X_test, y_test)
    assert "Model" in df.columns
    assert len(df) == len(MODELS)


def test_business_value_net_savings():
    """Net savings should equal gross savings minus intervention cost."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    bv = ModelTrainer.compute_business_value(y_true, y_pred, cost_churn=500, cost_intervention=50)
    expected_net = bv["gross_savings"] - bv["intervention_cost"]
    assert bv["net_savings"] == pytest.approx(expected_net)


def test_business_value_perfect_model():
    """A perfect model should have zero false positives and zero false negatives."""
    y = np.array([1, 0, 1, 0, 1])
    bv = ModelTrainer.compute_business_value(y, y)
    assert bv["false_positives"] == 0
    assert bv["false_negatives"] == 0
    assert bv["net_savings"] > 0

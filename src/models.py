"""
Model Training and Evaluation for Telco Churn Prediction.

Trains four classifiers (Logistic Regression, Random Forest,
Gradient Boosting, XGBoost), compares them via cross-validation,
and exposes evaluation utilities including a business-value calculator.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ── Model Definitions ─────────────────────────────────────────────────────────

MODELS: dict[str, Any] = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    ),
}


# ── ModelTrainer ──────────────────────────────────────────────────────────────

class ModelTrainer:
    """Train, compare, and persist churn prediction models."""

    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
        self.cv_results_: dict[str, dict] = {}
        self.fitted_models_: dict[str, Any] = {}
        self.best_model_name_: str = ""
        self.best_model_: Any = None

    def train_all(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> pd.DataFrame:
        """
        Train all models with stratified k-fold CV and return a comparison table.

        Args:
            X_train: Preprocessed feature matrix.
            y_train: Binary target vector.

        Returns:
            DataFrame with CV ROC-AUC mean and std for each model.
        """
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        records = []

        for name, model in MODELS.items():
            scores = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
            )
            self.cv_results_[name] = {"mean": scores.mean(), "std": scores.std()}
            # Fit on full training set
            model.fit(X_train, y_train)
            self.fitted_models_[name] = model
            records.append(
                {
                    "Model": name,
                    "CV ROC-AUC (mean)": round(scores.mean(), 4),
                    "CV ROC-AUC (std)": round(scores.std(), 4),
                }
            )
            print(f"  {name:25s}  AUC={scores.mean():.4f} ± {scores.std():.4f}")

        results_df = pd.DataFrame(records).sort_values("CV ROC-AUC (mean)", ascending=False)

        # Identify best model
        self.best_model_name_ = results_df.iloc[0]["Model"]
        self.best_model_ = self.fitted_models_[self.best_model_name_]
        print(f"\nBest model: {self.best_model_name_}")

        return results_df

    def evaluate(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5
    ) -> dict:
        """
        Evaluate a fitted model on the test set.

        Returns:
            Dict with roc_auc, f1, precision, recall, confusion_matrix,
            y_pred, y_prob.
        """
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        return {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    def evaluate_all(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        """Evaluate all fitted models and return a summary DataFrame."""
        records = []
        for name, model in self.fitted_models_.items():
            metrics = self.evaluate(model, X_test, y_test)
            records.append(
                {
                    "Model": name,
                    "ROC-AUC": round(metrics["roc_auc"], 4),
                    "F1": round(metrics["f1"], 4),
                    "Precision": round(metrics["precision"], 4),
                    "Recall": round(metrics["recall"], 4),
                }
            )
        return pd.DataFrame(records).sort_values("ROC-AUC", ascending=False)

    @staticmethod
    def compute_business_value(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cost_churn: float = 500.0,
        cost_intervention: float = 50.0,
    ) -> dict:
        """
        Compute the net business value of applying the model.

        Assumptions:
        - Each correctly identified churner who is retained saves `cost_churn` ($500).
        - Each customer contacted for retention costs `cost_intervention` ($50).
        - False positives incur intervention cost but no recovery benefit.
        - False negatives incur full churn cost.

        Args:
            y_true: Ground-truth binary labels.
            y_pred: Model predictions.
            cost_churn: Revenue/cost lost per unretained churner.
            cost_intervention: Cost of a retention outreach.

        Returns:
            Dict with true positives, false positives, false negatives,
            gross_savings, intervention_cost, net_savings.
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        gross_savings = tp * cost_churn
        intervention_cost = (tp + fp) * cost_intervention
        net_savings = gross_savings - intervention_cost
        baseline_cost = (tp + fn) * cost_churn  # cost if no model

        return {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "gross_savings": gross_savings,
            "intervention_cost": intervention_cost,
            "net_savings": net_savings,
            "baseline_cost": baseline_cost,
            "roi_pct": round(net_savings / intervention_cost * 100, 1) if intervention_cost else 0,
        }

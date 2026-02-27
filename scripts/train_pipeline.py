"""
CLI script: Full end-to-end training pipeline.

Loads data → engineers features → preprocesses → trains all models →
prints evaluation table → saves best model to models/best_model.joblib.

Usage:
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --data data/raw/telco_churn.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import generate_telco_data
from src.features import engineer_features
from src.models import ModelTrainer
from src.preprocessing import DataPreprocessor, prepare_target

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "telco_churn.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.joblib"


def main():
    parser = argparse.ArgumentParser(description="Train churn prediction models.")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to CSV dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    data_file = Path(args.data)
    if data_file.exists():
        print(f"Loading data from {data_file} ...")
        df = pd.read_csv(data_file)
    else:
        print("Dataset not found — generating synthetic data...")
        df = generate_telco_data()

    print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Churn rate: {(df['Churn']=='Yes').mean():.1%}\n")

    # ── Feature Engineering ────────────────────────────────────────────────────
    print("Engineering features...")
    df = engineer_features(df)

    # ── Prepare target & split ────────────────────────────────────────────────
    y = prepare_target(df)
    feature_df = df.drop(columns=["customerID", "Churn"], errors="ignore")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        feature_df, y, test_size=args.test_size, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train_raw):,}  |  Test: {len(X_test_raw):,}\n")

    # ── Preprocessing ──────────────────────────────────────────────────────────
    print("Preprocessing...")
    preprocessor = DataPreprocessor(include_engineered=True)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names()
    print(f"Feature matrix: {X_train.shape[1]} features\n")

    # ── Training ───────────────────────────────────────────────────────────────
    print("Training models (5-fold CV):")
    trainer = ModelTrainer(cv_folds=5)
    cv_results = trainer.train_all(X_train, y_train)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    print("\n── Test Set Results ──────────────────────────────────────")
    test_results = trainer.evaluate_all(X_test, y_test)
    print(test_results.to_string(index=False))

    # ── Business Value ─────────────────────────────────────────────────────────
    best_metrics = trainer.evaluate(trainer.best_model_, X_test, y_test)
    bv = ModelTrainer.compute_business_value(y_test, best_metrics["y_pred"])
    print(f"\n── Business Value ({trainer.best_model_name_}) ──────────────────")
    print(f"  Correctly identified churners : {bv['true_positives']:,}")
    print(f"  Customers contacted           : {bv['true_positives'] + bv['false_positives']:,}")
    print(f"  Gross savings                 : ${bv['gross_savings']:,.0f}")
    print(f"  Intervention cost             : ${bv['intervention_cost']:,.0f}")
    print(f"  Net savings                   : ${bv['net_savings']:,.0f}")
    print(f"  ROI                           : {bv['roi_pct']}%")

    # ── Save artefacts ─────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artefact = {
        "model": trainer.best_model_,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "model_name": trainer.best_model_name_,
        "test_auc": best_metrics["roc_auc"],
    }
    joblib.dump(artefact, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()

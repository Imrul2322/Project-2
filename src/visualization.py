"""
Reusable Visualization Functions for Telco Churn Analysis.

All functions return matplotlib Figure objects and optionally save
to the reports/figures/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
PALETTE = {"No": "#2ecc71", "Yes": "#e74c3c"}
STYLE = "seaborn-v0_8-whitegrid"


def _save(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight", dpi=150)


# ── EDA Plots ─────────────────────────────────────────────────────────────────

def plot_churn_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Pie + bar chart of overall churn distribution."""
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        counts = df["Churn"].value_counts()

        axes[0].pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=[PALETTE["No"], PALETTE["Yes"]],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        axes[0].set_title("Churn Distribution", fontsize=14, fontweight="bold")

        sns.countplot(x="Churn", data=df, palette=PALETTE, ax=axes[1], order=["No", "Yes"])
        axes[1].set_title("Customer Count by Churn", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("")
        for p in axes[1].patches:
            axes[1].annotate(
                f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11,
            )

        fig.suptitle("Target Variable: Customer Churn", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
    if save:
        _save(fig, "01_churn_distribution.png")
    return fig


def plot_churn_rate_by_feature(
    df: pd.DataFrame, feature: str, save: bool = True
) -> plt.Figure:
    """Grouped bar chart showing churn rate per category of a feature."""
    with plt.style.context(STYLE):
        churn_rate = (
            df.groupby(feature)["Churn"]
            .apply(lambda x: (x == "Yes").mean())
            .reset_index()
            .rename(columns={"Churn": "Churn Rate"})
            .sort_values("Churn Rate", ascending=False)
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            churn_rate[feature],
            churn_rate["Churn Rate"],
            color=sns.color_palette("RdYlGn_r", len(churn_rate)),
            edgecolor="white",
        )
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(f"Churn Rate by {feature}", fontsize=14, fontweight="bold")
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel("Churn Rate", fontsize=12)
        ax.tick_params(axis="x", rotation=25)

        for bar, rate in zip(bars, churn_rate["Churn Rate"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{rate:.1%}",
                ha="center", va="bottom", fontsize=10,
            )

        plt.tight_layout()
    if save:
        _save(fig, f"churn_rate_{feature.lower()}.png")
    return fig


def plot_tenure_vs_churn(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """KDE + histogram of tenure split by churn status."""
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for label, color in PALETTE.items():
            subset = df[df["Churn"] == label]["tenure"]
            axes[0].hist(subset, bins=30, alpha=0.6, label=f"Churn={label}", color=color, edgecolor="white")
        axes[0].set_title("Tenure Distribution by Churn", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Tenure (months)")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        for label, color in PALETTE.items():
            subset = df[df["Churn"] == label]["MonthlyCharges"]
            axes[1].hist(subset, bins=30, alpha=0.6, label=f"Churn={label}", color=color, edgecolor="white")
        axes[1].set_title("Monthly Charges Distribution by Churn", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Monthly Charges ($)")
        axes[1].set_ylabel("Count")
        axes[1].legend()

        plt.tight_layout()
    if save:
        _save(fig, "02_tenure_charges_by_churn.png")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Heatmap of correlations among numeric features."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[numeric_cols].corr()

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
    if save:
        _save(fig, "03_correlation_heatmap.png")
    return fig


# ── Model Evaluation Plots ────────────────────────────────────────────────────

def plot_roc_curves(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Overlay ROC curves for multiple fitted models."""
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")

        for (name, model), color in zip(models.items(), colors):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_curve.__module__ and __import__(
                "sklearn.metrics", fromlist=["roc_auc_score"]
            ).roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.3f})")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
    if save:
        _save(fig, "04_roc_curves.png")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "Best Model",
    save: bool = True,
) -> plt.Figure:
    """Annotated confusion matrix heatmap."""
    labels = ["No Churn", "Churn"]
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            linewidths=1,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
    if save:
        _save(fig, "05_confusion_matrix.png")
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    model_name: str = "Model",
    top_n: int = 15,
    save: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances."""
    idx = np.argsort(importances)[-top_n:]
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = sns.color_palette("viridis", top_n)
        ax.barh(
            [feature_names[i] for i in idx],
            importances[idx],
            color=colors,
            edgecolor="white",
        )
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=12)
        plt.tight_layout()
    if save:
        _save(fig, "06_feature_importance.png")
    return fig


def plot_model_comparison(results_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Bar chart comparing ROC-AUC scores across models."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("RdYlGn", len(results_df))[::-1]
        bars = ax.bar(
            results_df["Model"],
            results_df["ROC-AUC"],
            color=colors,
            edgecolor="white",
            width=0.5,
        )
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("ROC-AUC Score", fontsize=12)
        ax.set_title("Model Comparison — ROC-AUC on Test Set", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, results_df["ROC-AUC"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        plt.tight_layout()
    if save:
        _save(fig, "07_model_comparison.png")
    return fig

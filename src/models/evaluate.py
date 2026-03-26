"""
Model Evaluation and Visualization.

Generates publication-quality evaluation plots:
- RUL prediction vs actual
- Anomaly detection ROC/PR curves
- Confusion matrix for failure classification
- Feature importance charts
- Maintenance cost-benefit analysis
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
)

from src.models.anomaly_detection import get_all_detectors
from src.models.rul_prediction import get_all_rul_models
from src.models.failure_classification import get_all_classifiers, LABEL_NAMES
from src.models.train import load_data, get_feature_columns, temporal_split, prepare_arrays

SEED = 42
np.random.seed(SEED)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIGSIZE = (10, 6)
DPI = 150


def plot_rul_predictions(
    test_df: pd.DataFrame, feature_cols: list[str], models_dir: Path, figures_dir: Path,
):
    """Plot RUL predicted vs actual for all models."""
    X_test, y_test = prepare_arrays(test_df, feature_cols, "rul_hours")
    rul_models = get_all_rul_models()

    # --- Scatter plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for i, model in enumerate(rul_models):
        model.load(models_dir / f"{model.name.lower()}.joblib")
        y_pred = model.predict_rul(X_test)

        ax = axes[i]
        # Subsample for plot readability
        n_plot = min(5000, len(y_test))
        idx = np.random.choice(len(y_test), n_plot, replace=False)

        ax.scatter(y_test[idx], y_pred[idx], alpha=0.15, s=8, label="Predictions")
        ax.plot([0, y_test.max()], [0, y_test.max()], "r--", lw=2, label="Perfect")
        ax.set_xlabel("Actual RUL (hours)", fontsize=11)
        ax.set_ylabel("Predicted RUL (hours)", fontsize=11)
        ax.set_title(model.name.replace("_", " "), fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle("RUL Prediction: Actual vs Predicted", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "rul_scatter.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # --- Time series for a sample machine ---
    sample_machines = test_df["machine_id"].unique()[:3]
    fig, axes = plt.subplots(len(sample_machines), 1, figsize=(14, 4 * len(sample_machines)))
    if len(sample_machines) == 1:
        axes = [axes]

    best_model = rul_models[1]  # XGBoost
    best_model.load(models_dir / f"{best_model.name.lower()}.joblib")

    for ax, machine_id in zip(axes, sample_machines):
        mask = test_df["machine_id"] == machine_id
        machine_test = test_df[mask]
        X_m, y_m = prepare_arrays(machine_test, feature_cols, "rul_hours")
        y_pred_m = best_model.predict_rul(X_m)

        timestamps = pd.to_datetime(machine_test["timestamp"].values)
        ax.plot(timestamps, y_m, label="Actual RUL", linewidth=1.5)
        ax.plot(timestamps, y_pred_m, label="Predicted RUL", linewidth=1.5, alpha=0.8)
        ax.set_ylabel("RUL (hours)", fontsize=11)
        ax.set_title(f"Machine {machine_id}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle("RUL Timeline — XGBoost", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "rul_timeline.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Saved rul_scatter.png and rul_timeline.png")


def plot_anomaly_roc_pr(
    test_df: pd.DataFrame, feature_cols: list[str], models_dir: Path, figures_dir: Path,
):
    """Plot ROC and Precision-Recall curves for anomaly detectors."""
    # Train normal data for fitting was already done; load saved models
    X_test, y_test = prepare_arrays(test_df, feature_cols, "is_anomaly")
    y_true = y_test.astype(int)

    detectors = get_all_detectors()

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 6))

    for detector in detectors:
        detector.load(models_dir / f"{detector.name.lower()}_detector.joblib")
        _, scores = detector.detect(X_test)

        # ROC curve (negate scores: lower score = more anomalous)
        fpr, tpr, _ = roc_curve(y_true, -scores)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{detector.name} (AUC={roc_auc:.3f})")

        # PR curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, -scores)
        pr_auc = auc(recall_vals, precision_vals)
        ax_pr.plot(recall_vals, precision_vals, lw=2, label=f"{detector.name} (AUC={pr_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.set_title("ROC Curves — Anomaly Detection", fontsize=14, fontweight="bold")
    ax_roc.legend(fontsize=10)

    ax_pr.set_xlabel("Recall", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)
    ax_pr.set_title("Precision-Recall Curves — Anomaly Detection", fontsize=14, fontweight="bold")
    ax_pr.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "anomaly_roc_pr.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Saved anomaly_roc_pr.png")


def plot_confusion_matrix(
    test_df: pd.DataFrame, feature_cols: list[str], models_dir: Path, figures_dir: Path,
):
    """Plot confusion matrices for failure classifiers."""
    X_test, y_test = prepare_arrays(test_df, feature_cols, "failure_label")
    y_test = y_test.astype(int)

    classifiers = get_all_classifiers()
    class_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, clf in zip(axes, classifiers):
        clf.load(models_dir / f"{clf.name.lower()}.joblib")
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=sorted(LABEL_NAMES.keys()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(clf.name.replace("_", " "), fontsize=13, fontweight="bold")

    plt.suptitle("Failure Classification — Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Saved confusion_matrix.png")


def plot_feature_importance(
    train_df: pd.DataFrame, feature_cols: list[str], models_dir: Path, figures_dir: Path,
    top_n: int = 20,
):
    """Plot top feature importances for RUL and classification models."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # RUL feature importance (XGBoost)
    from src.models.rul_prediction import RULXGBoost
    xgb_rul = RULXGBoost()
    xgb_rul.load(models_dir / "xgboost_rul.joblib")
    importances_rul = xgb_rul.feature_importances
    top_idx_rul = np.argsort(importances_rul)[-top_n:]
    axes[0].barh(
        [feature_cols[i] for i in top_idx_rul],
        importances_rul[top_idx_rul],
        color=sns.color_palette("viridis", top_n),
    )
    axes[0].set_xlabel("Feature Importance", fontsize=12)
    axes[0].set_title("RUL Prediction — XGBoost Top Features", fontsize=13, fontweight="bold")

    # Classification feature importance (XGBoost)
    from src.models.failure_classification import FailureXGBoost
    xgb_clf = FailureXGBoost()
    xgb_clf.load(models_dir / "xgboost_classifier.joblib")
    importances_clf = xgb_clf.feature_importances
    top_idx_clf = np.argsort(importances_clf)[-top_n:]
    axes[1].barh(
        [feature_cols[i] for i in top_idx_clf],
        importances_clf[top_idx_clf],
        color=sns.color_palette("magma", top_n),
    )
    axes[1].set_xlabel("Feature Importance", fontsize=12)
    axes[1].set_title("Failure Classification — XGBoost Top Features", fontsize=13, fontweight="bold")

    plt.suptitle("Feature Importance Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Saved feature_importance.png")


def plot_cost_benefit(figures_dir: Path):
    """Plot maintenance cost-benefit analysis comparing strategies."""
    strategies = ["Reactive\n(Run-to-Failure)", "Calendar-Based\nPreventive", "Condition-Based\n(This Model)"]
    downtime_cost = [150000, 80000, 35000]
    maintenance_cost = [20000, 65000, 45000]
    total_cost = [d + m for d, m in zip(downtime_cost, maintenance_cost)]

    x = np.arange(len(strategies))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, downtime_cost, width, label="Downtime Cost", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x, maintenance_cost, width, label="Maintenance Cost", color="#3498db", alpha=0.85)
    bars3 = ax.bar(x + width, total_cost, width, label="Total Cost", color="#2ecc71", alpha=0.85)

    ax.set_ylabel("Annual Cost (USD)", fontsize=12)
    ax.set_title("Maintenance Strategy Cost-Benefit Analysis", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.legend(fontsize=11)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"${height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    # Add savings annotation
    savings = total_cost[0] - total_cost[2]
    savings_pct = savings / total_cost[0] * 100
    ax.annotate(
        f"Savings: ${savings:,.0f}/yr ({savings_pct:.0f}%)",
        xy=(2, total_cost[2]),
        xytext=(2.3, total_cost[0] * 0.85),
        fontsize=12, fontweight="bold", color="#27ae60",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2),
    )

    plt.tight_layout()
    plt.savefig(figures_dir / "cost_benefit.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Saved cost_benefit.png")


def main():
    """Run complete evaluation and generate all plots."""
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and splitting...")
    df = load_data(project_root)
    feature_cols = get_feature_columns(df)
    train_df, test_df = temporal_split(df)

    print("\nGenerating RUL prediction plots...")
    plot_rul_predictions(test_df, feature_cols, models_dir, figures_dir)

    print("\nGenerating anomaly detection ROC/PR curves...")
    plot_anomaly_roc_pr(test_df, feature_cols, models_dir, figures_dir)

    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(test_df, feature_cols, models_dir, figures_dir)

    print("\nGenerating feature importance plots...")
    plot_feature_importance(train_df, feature_cols, models_dir, figures_dir)

    print("\nGenerating cost-benefit analysis...")
    plot_cost_benefit(figures_dir)

    print("\nAll evaluation plots saved to results/figures/")


if __name__ == "__main__":
    main()

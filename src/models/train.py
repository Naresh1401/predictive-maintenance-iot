"""
Training Pipeline for Predictive Maintenance Models.

Orchestrates training of:
- Anomaly detection models
- RUL prediction models
- Failure classification models

Uses temporal train/test split to respect time-series ordering.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, roc_auc_score,
)

from src.models.anomaly_detection import get_all_detectors
from src.models.rul_prediction import get_all_rul_models
from src.models.failure_classification import get_all_classifiers, LABEL_NAMES

SEED = 42
np.random.seed(SEED)

TEMPORAL_SPLIT_RATIO = 0.8  # First 80% of time for training


def load_data(project_root: Path) -> pd.DataFrame:
    """Load processed feature data."""
    processed_dir = project_root / "data" / "processed"
    df = pd.read_csv(processed_dir / "features.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature column names (exclude targets and metadata)."""
    exclude = {"timestamp", "machine_id", "rul_hours", "failure_label", "is_anomaly"}
    return [c for c in df.columns if c not in exclude]


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally — first 80% of timestamps for training."""
    timestamps = sorted(df["timestamp"].unique())
    split_idx = int(len(timestamps) * TEMPORAL_SPLIT_RATIO)
    split_time = timestamps[split_idx]

    train = df[df["timestamp"] < split_time].copy()
    test = df[df["timestamp"] >= split_time].copy()
    return train, test


def prepare_arrays(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare feature matrix and target array, handling NaN/inf."""
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    # Replace inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def train_anomaly_detectors(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str],
    models_dir: Path,
) -> dict:
    """Train anomaly detection models and return metrics."""
    # Train on normal data only
    train_normal = train_df[train_df["is_anomaly"] == 0]
    X_train, _ = prepare_arrays(train_normal, feature_cols, "is_anomaly")
    X_test, y_test = prepare_arrays(test_df, feature_cols, "is_anomaly")

    # For evaluation, convert labels: is_anomaly=1 -> anomaly
    y_true = y_test.astype(int)

    results = {}
    detectors = get_all_detectors()

    for detector in detectors:
        print(f"  Training {detector.name}...")
        detector.fit(X_train)
        labels, scores = detector.detect(X_test)

        # Convert detector labels: -1=anomaly->1, 1=normal->0
        y_pred = (labels == -1).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # AUC-ROC (use inverted scores — more negative = more anomalous)
        try:
            auc = roc_auc_score(y_true, -scores)
        except ValueError:
            auc = 0.0

        results[detector.name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc, 4),
        }
        print(f"    Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} AUC={auc:.4f}")

        detector.save(models_dir / f"{detector.name.lower()}_detector.joblib")

    return results


def train_rul_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str],
    models_dir: Path,
) -> dict:
    """Train RUL prediction models and return metrics."""
    X_train, y_train = prepare_arrays(train_df, feature_cols, "rul_hours")
    X_test, y_test = prepare_arrays(test_df, feature_cols, "rul_hours")

    results = {}
    rul_models = get_all_rul_models()

    for model in rul_models:
        print(f"  Training {model.name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict_rul(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[model.name] = {
            "mae_hours": round(mae, 2),
            "rmse_hours": round(rmse, 2),
            "r2_score": round(r2, 4),
        }
        print(f"    MAE={mae:.2f}h RMSE={rmse:.2f}h R²={r2:.4f}")

        model.save(models_dir / f"{model.name.lower()}.joblib")

    return results


def train_failure_classifiers(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str],
    models_dir: Path,
) -> dict:
    """Train failure classification models and return metrics."""
    X_train, y_train = prepare_arrays(train_df, feature_cols, "failure_label")
    X_test, y_test = prepare_arrays(test_df, feature_cols, "failure_label")
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    results = {}
    classifiers = get_all_classifiers()

    for clf in classifiers:
        print(f"  Training {clf.name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Per-class F1
        class_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
        per_class_f1 = f1_score(y_test, y_pred, labels=class_labels, average=None, zero_division=0)
        per_class = {
            LABEL_NAMES.get(int(lbl), str(lbl)): round(float(f), 4)
            for lbl, f in zip(class_labels, per_class_f1)
        }

        results[clf.name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": per_class,
        }
        print(f"    Accuracy={acc:.4f} Macro-F1={macro_f1:.4f}")

        clf.save(models_dir / f"{clf.name.lower()}.joblib")

    return results


def main():
    """Run complete training pipeline."""
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = project_root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed data...")
    df = load_data(project_root)
    feature_cols = get_feature_columns(df)
    print(f"  {len(df):,} rows, {len(feature_cols)} features")

    print("\nPerforming temporal train/test split...")
    train_df, test_df = temporal_split(df)
    print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    all_metrics = {}

    print("\n=== Anomaly Detection ===")
    all_metrics["anomaly_detection"] = train_anomaly_detectors(
        train_df, test_df, feature_cols, models_dir
    )

    print("\n=== RUL Prediction ===")
    all_metrics["rul_prediction"] = train_rul_models(
        train_df, test_df, feature_cols, models_dir
    )

    print("\n=== Failure Classification ===")
    all_metrics["failure_classification"] = train_failure_classifiers(
        train_df, test_df, feature_cols, models_dir
    )

    # Save all metrics
    with open(metrics_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {metrics_dir / 'all_metrics.json'}")

    # Save feature columns for API
    pd.Series(feature_cols).to_csv(models_dir / "feature_columns.csv", index=False, header=False)

    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()

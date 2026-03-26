"""
Failure Type Classification Models.

Multi-class classification: normal vs bearing vs motor vs overheating.
Handles class imbalance via class weighting.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from pathlib import Path


LABEL_NAMES = {0: "normal", 1: "bearing", 2: "motor", 3: "overheating"}


class FailureRandomForest:
    """Failure classification using Random Forest with class weights."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 15, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.name = "RandomForest_Classifier"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FailureRandomForest":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    @property
    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: Path):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: Path) -> "FailureRandomForest":
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self


class FailureXGBoost:
    """Failure classification using XGBoost with class weights."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.name = "XGBoost_Classifier"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FailureXGBoost":
        X_scaled = self.scaler.fit_transform(X)

        # Compute sample weights for class imbalance
        classes = np.unique(y)
        class_weights = compute_class_weight("balanced", classes=classes, y=y)
        weight_map = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_map[label] for label in y])

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=len(classes),
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric="mlogloss",
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    @property
    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: Path):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: Path) -> "FailureXGBoost":
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self


def get_all_classifiers() -> list:
    """Return instances of all failure classification models."""
    return [
        FailureRandomForest(),
        FailureXGBoost(),
    ]

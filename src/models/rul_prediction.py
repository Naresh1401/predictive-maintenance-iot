"""
Remaining Useful Life (RUL) Prediction Models.

Implements regression models to predict hours until equipment failure.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


class RULRandomForest:
    """RUL prediction using Random Forest Regressor."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 15, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.name = "RandomForest_RUL"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RULRandomForest":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_rul(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0, None)

    @property
    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: Path):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: Path) -> "RULRandomForest":
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self


class RULXGBoost:
    """RUL prediction using XGBoost Regressor."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self.name = "XGBoost_RUL"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RULXGBoost":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_rul(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0, None)

    @property
    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: Path):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: Path) -> "RULXGBoost":
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self


class RULLightGBM:
    """RUL prediction using LightGBM Regressor."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 10,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.scaler = StandardScaler()
        self.name = "LightGBM_RUL"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RULLightGBM":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_rul(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0, None)

    @property
    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: Path):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: Path) -> "RULLightGBM":
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self


class RULGradientBoosting:
    """RUL prediction using Gradient Boosting Regressor."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.name = "GradientBoosting_RUL"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RULGradientBoosting":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_rul(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0, None)

    @property
    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: Path):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: Path) -> "RULGradientBoosting":
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self


def get_all_rul_models() -> list:
    """Return instances of all RUL prediction models."""
    return [
        RULRandomForest(),
        RULXGBoost(),
        RULLightGBM(),
        RULGradientBoosting(),
    ]

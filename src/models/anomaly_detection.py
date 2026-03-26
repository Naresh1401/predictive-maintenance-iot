"""
Anomaly Detection Models for Industrial Equipment.

Implements multiple anomaly detection approaches:
- Isolation Forest
- One-Class SVM
- Autoencoder-based (MLP reconstruction)
- Statistical Process Control (Hotelling T² and Q-statistic)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from pathlib import Path


class IsolationForestDetector:
    """Anomaly detection using Isolation Forest."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.name = "IsolationForest"

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def detect(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (labels, scores). Labels: 1=normal, -1=anomaly. Scores: lower=more anomalous."""
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        return labels, scores

    def save(self, path: Path):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: Path) -> "IsolationForestDetector":
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self


class OneClassSVMDetector:
    """Anomaly detection using One-Class SVM."""

    def __init__(self, kernel: str = "rbf", nu: float = 0.05, gamma: str = "scale"):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.scaler = StandardScaler()
        self.name = "OneClassSVM"

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        X_scaled = self.scaler.fit_transform(X)
        # Subsample for SVM scalability
        if len(X_scaled) > 10000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_scaled), 10000, replace=False)
            X_scaled = X_scaled[idx]
        self.model.fit(X_scaled)
        return self

    def detect(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        return labels, scores

    def save(self, path: Path):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: Path) -> "OneClassSVMDetector":
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self


class AutoencoderDetector:
    """Anomaly detection using MLP-based autoencoder (reconstruction error)."""

    def __init__(
        self,
        hidden_layers: tuple = (64, 32, 16, 32, 64),
        max_iter: int = 200,
        threshold_percentile: float = 95.0,
        random_state: int = 42,
    ):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate="adaptive",
        )
        self.scaler = StandardScaler()
        self.threshold = None
        self.threshold_percentile = threshold_percentile
        self.name = "Autoencoder"

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        """Train autoencoder on normal data. The model learns to reconstruct inputs."""
        X_scaled = self.scaler.fit_transform(X)
        # Train: input = output (autoencoder learns identity for normal data)
        self.model.fit(X_scaled, X_scaled)
        # Set threshold from training reconstruction error
        recon = self.model.predict(X_scaled)
        errors = np.mean((X_scaled - recon) ** 2, axis=1)
        self.threshold = np.percentile(errors, self.threshold_percentile)
        return self

    def detect(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (labels, scores). Labels: 1=normal, -1=anomaly. Scores: reconstruction error."""
        X_scaled = self.scaler.transform(X)
        recon = self.model.predict(X_scaled)
        errors = np.mean((X_scaled - recon) ** 2, axis=1)
        labels = np.where(errors > self.threshold, -1, 1)
        # Invert scores so lower = more anomalous (consistent with IF)
        scores = -errors
        return labels, scores

    def save(self, path: Path):
        joblib.dump({
            "model": self.model, "scaler": self.scaler, "threshold": self.threshold
        }, path)

    def load(self, path: Path) -> "AutoencoderDetector":
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.threshold = data["threshold"]
        return self


class SPCDetector:
    """Statistical Process Control using Hotelling T² and Q-statistic."""

    def __init__(self, n_components: float = 0.95, alpha: float = 0.01):
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.alpha = alpha
        self.t2_threshold = None
        self.q_threshold = None
        self.name = "HotellingT2"

    def fit(self, X: np.ndarray) -> "SPCDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)

        # Compute T² and Q statistics on training data
        t2_scores, q_scores = self._compute_statistics(X_scaled)

        # Set thresholds at (1 - alpha) percentile
        percentile = (1 - self.alpha) * 100
        self.t2_threshold = np.percentile(t2_scores, percentile)
        self.q_threshold = np.percentile(q_scores, percentile)
        return self

    def _compute_statistics(self, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute Hotelling T² and Q (SPE) statistics."""
        scores = self.pca.transform(X_scaled)
        eigenvalues = self.pca.explained_variance_

        # Hotelling T²: sum of (score_i / eigenvalue_i)²
        t2 = np.sum((scores ** 2) / eigenvalues, axis=1)

        # Q-statistic (Squared Prediction Error)
        reconstructed = self.pca.inverse_transform(scores)
        residuals = X_scaled - reconstructed
        q = np.sum(residuals ** 2, axis=1)

        return t2, q

    def detect(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (labels, scores). Combined T² and Q for anomaly detection."""
        X_scaled = self.scaler.transform(X)
        t2_scores, q_scores = self._compute_statistics(X_scaled)

        # Anomaly if either T² or Q exceeds threshold
        t2_anomaly = t2_scores > self.t2_threshold
        q_anomaly = q_scores > self.q_threshold
        is_anomaly = t2_anomaly | q_anomaly

        labels = np.where(is_anomaly, -1, 1)
        # Combined score (normalized, lower means more anomalous)
        combined = -(t2_scores / self.t2_threshold + q_scores / self.q_threshold)
        return labels, combined

    def save(self, path: Path):
        joblib.dump({
            "pca": self.pca, "scaler": self.scaler,
            "t2_threshold": self.t2_threshold, "q_threshold": self.q_threshold,
        }, path)

    def load(self, path: Path) -> "SPCDetector":
        data = joblib.load(path)
        self.pca = data["pca"]
        self.scaler = data["scaler"]
        self.t2_threshold = data["t2_threshold"]
        self.q_threshold = data["q_threshold"]
        return self


def get_all_detectors() -> list:
    """Return instances of all anomaly detection models."""
    return [
        IsolationForestDetector(),
        OneClassSVMDetector(),
        AutoencoderDetector(),
        SPCDetector(),
    ]

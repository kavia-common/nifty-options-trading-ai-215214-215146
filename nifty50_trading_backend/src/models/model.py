from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np


@dataclass
class LinearModelState:
    weights: List[float]
    bias: float
    mu: Dict[str, float]
    sigma: Dict[str, float]
    feature_order: List[str]


class LinearRegressionModel:
    """A very small CPU-friendly linear regression with z-score normalization and ridge regularization."""

    def __init__(self, l2_reg: float = 1e-6):
        self.l2_reg = l2_reg
        self.state: LinearModelState | None = None

    # PUBLIC_INTERFACE
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Fit the linear model using the normal equation with L2 regularization.

        Returns minimal training metrics (MSE).
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D matrix")
        if y.ndim != 1:
            y = y.reshape(-1)

        # z-score normalization
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0  # avoid divide by zero
        Xn = (X - mu) / sigma

        # closed-form solution: (X^T X + λI) w = X^T y
        xtx = Xn.T @ Xn
        regI = self.l2_reg * np.eye(xtx.shape[0])
        w = np.linalg.pinv(xtx + regI) @ (Xn.T @ y)

        # bias as mean residual
        y_pred = Xn @ w
        b = float(np.mean(y - y_pred))

        self.state = LinearModelState(
            weights=w.tolist(),
            bias=b,
            mu={name: float(m) for name, m in zip(feature_names, mu)},
            sigma={name: float(s) for name, s in zip(feature_names, sigma)},
            feature_order=list(feature_names),
        )

        mse = float(np.mean((y - (y_pred + b)) ** 2))
        return {"mse": mse}

    # PUBLIC_INTERFACE
    def predict(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Predict using the stored state; requires matching feature order."""
        if self.state is None:
            raise RuntimeError("Model is not trained/loaded")

        # Reorder/align columns as per training
        order = self.state.feature_order
        indices = [feature_names.index(name) for name in order]
        Xa = X[:, indices]

        mu = np.array([self.state.mu[name] for name in order])
        sigma = np.array([self.state.sigma[name] for name in order])
        sigma[sigma == 0] = 1.0

        Xn = (Xa - mu) / sigma
        w = np.array(self.state.weights)
        preds = Xn @ w + self.state.bias
        return preds

    # PUBLIC_INTERFACE
    def save(self, path: str) -> str:
        """Persist the model state to JSON on disk."""
        if self.state is None:
            raise RuntimeError("No state to save")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self.state), f)
        return path

    # PUBLIC_INTERFACE
    def load(self, path: str) -> "LinearRegressionModel":
        """Load model state from JSON on disk."""
        with open(path, "r") as f:
            data = json.load(f)
        state = LinearModelState(**data)
        self.state = state
        return self

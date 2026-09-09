"""
Week 7: Regression — Linear Regression (Closed-Form Normal Equation & Gradient Descent)

Implements:
1. Analytical Closed-Form Normal Equation: theta = (X^T * X)^(-1) * X^T * y
2. Batch Gradient Descent with cost history convergence:
   J(theta) = (1 / 2m) * sum((h_theta(x) - y)^2)
3. Metrics: MSE, RMSE, MAE, R^2 Score
"""

import numpy as np
from typing import Tuple, List, Optional


class LinearRegressionScratch:
    """Linear regression model implemented with dual solvers: 'normal' and 'gd'."""

    def __init__(self, solver: str = "normal", lr: float = 0.01, max_iter: int = 1000, tol: float = 1e-6):
        self.solver = solver
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.theta: Optional[np.ndarray] = None  # Includes intercept [theta_0, theta_1, ...]
        self.cost_history: List[float] = []

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        return np.column_stack([np.ones((m, 1)), X])

    def compute_cost(self, X_b: np.ndarray, y: np.ndarray) -> float:
        m = len(y)
        predictions = X_b @ self.theta
        errors = predictions - y
        return float((1.0 / (2.0 * m)) * np.sum(errors ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionScratch':
        X_b = self._add_intercept(X)
        m, n = X_b.shape

        if self.solver == "normal":
            # Normal Equation: theta = (X_b^T * X_b)^(-1) * X_b^T * y
            # Using pinv to guarantee numerical stability even if rank deficient
            self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            self.cost_history = [self.compute_cost(X_b, y)]

        elif self.solver == "gd":
            # Initialize weights to zeros
            self.theta = np.zeros(n)
            prev_cost = float('inf')

            for _ in range(self.max_iter):
                predictions = X_b @ self.theta
                errors = predictions - y
                gradient = (1.0 / m) * (X_b.T @ errors)
                self.theta -= self.lr * gradient

                cost = self.compute_cost(X_b, y)
                self.cost_history.append(cost)

                if abs(prev_cost - cost) < self.tol:
                    break
                prev_cost = cost
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Model has not been fitted.")
        X_b = self._add_intercept(X)
        return X_b @ self.theta


# =====================================================================
# Regression Metrics from Scratch
# =====================================================================
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - (ss_res / ss_tot))


if __name__ == "__main__":
    np.random.seed(42)
    # Synthetic 1D regression: y = 3.5 * x + 7.2 + noise
    X_sample = 2.0 * np.random.rand(100, 1)
    y_sample = 7.2 + 3.5 * X_sample.squeeze() + np.random.randn(100) * 0.4

    print("=== Linear Regression Demonstration ===")
    # 1. Closed Form
    model_normal = LinearRegressionScratch(solver="normal").fit(X_sample, y_sample)
    preds_normal = model_normal.predict(X_sample)
    print(f"Normal Equation Intercept: {model_normal.theta[0]:.4f} (True: 7.2)")
    print(f"Normal Equation Slope:     {model_normal.theta[1]:.4f} (True: 3.5)")
    print(f"R^2 Score:                 {r2_score_scratch(y_sample, preds_normal):.4f}")

    # 2. Gradient Descent
    model_gd = LinearRegressionScratch(solver="gd", lr=0.1, max_iter=2000).fit(X_sample, y_sample)
    print(f"GD Intercept:              {model_gd.theta[0]:.4f}")
    print(f"GD Slope:                  {model_gd.theta[1]:.4f}")

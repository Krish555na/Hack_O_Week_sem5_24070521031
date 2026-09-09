"""
Week 7: Regression — Ridge (L2) & Lasso (L1) Regularization from Scratch

Implements:
1. Ridge Regression (Tikhonov / L2 Regularization):
   Closed-form: theta = (X^T * X + alpha * I')^(-1) * X^T * y
   (Where I' does not penalize intercept theta_0)
2. Lasso Regression (L1 Regularization):
   Solved via Coordinate Descent with Soft-Thresholding:
   S(rho, lambda) = sign(rho) * max(0, |rho| - lambda)
   Demonstrates automatic feature selection / coefficient sparsity!
"""

import numpy as np
from typing import Optional


class RidgeRegressionScratch:
    """Ridge regression with L2 weight shrinkage."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.theta: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegressionScratch':
        m = X.shape[0]
        X_b = np.column_stack([np.ones((m, 1)), X])
        n = X_b.shape[1]

        # Regularization matrix: do NOT regularize intercept (index 0)
        I_reg = np.eye(n)
        I_reg[0, 0] = 0.0

        # Normal equation with L2 penalty
        self.theta = np.linalg.pinv(X_b.T @ X_b + self.alpha * I_reg) @ X_b.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        X_b = np.column_stack([np.ones((m, 1)), X])
        return X_b @ self.theta


class LassoRegressionScratch:
    """Lasso regression with L1 penalty solved via Coordinate Descent."""

    def __init__(self, alpha: float = 0.1, max_iter: int = 1000, tol: float = 1e-5):
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.tol = tol
        self.theta: Optional[np.ndarray] = None

    def _soft_threshold(self, rho: float, lam: float) -> float:
        """Soft-thresholding operator S(rho, lam)."""
        if rho < -lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegressionScratch':
        m, p = X.shape
        # Intercept is handled by centering or explicit bias column
        X_b = np.column_stack([np.ones((m, 1)), X])
        n = X_b.shape[1]

        self.theta = np.zeros(n)

        # Precompute column norms squared: z_j = sum(x_{ij}^2)
        z = np.sum(X_b ** 2, axis=0)

        for _ in range(self.max_iter):
            max_change = 0.0

            for j in range(n):
                old_theta_j = self.theta[j]

                # Compute partial residual: y - sum_{k != j} (X_{ik} * theta_k)
                y_pred = X_b @ self.theta
                residual = y - y_pred + self.theta[j] * X_b[:, j]

                # rho_j = X_j^T * residual
                rho_j = float(np.dot(X_b[:, j], residual))

                if j == 0:
                    # Do not regularize intercept
                    self.theta[j] = rho_j / z[j]
                else:
                    # Apply L1 soft-thresholding
                    lam = self.alpha * m
                    self.theta[j] = self._soft_threshold(rho_j, lam) / (z[j] + 1e-12)

                change = abs(self.theta[j] - old_theta_j)
                if change > max_change:
                    max_change = change

            if max_change < self.tol:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        X_b = np.column_stack([np.ones((m, 1)), X])
        return X_b @ self.theta


if __name__ == "__main__":
    np.random.seed(42)
    # Generate dataset with 10 features, but ONLY 3 are truly informative!
    n_samples = 100
    n_features = 10
    X_raw = np.random.randn(n_samples, n_features)

    # True weights: [5.0, -3.0, 2.0, 0, 0, 0, 0, 0, 0, 0]
    true_weights = np.array([5.0, -3.0, 2.0] + [0.0] * 7)
    y_raw = 4.5 + X_raw @ true_weights + np.random.randn(n_samples) * 0.5

    print("=== Regularization: Ridge vs Lasso ===")
    ridge = RidgeRegressionScratch(alpha=10.0).fit(X_raw, y_raw)
    lasso = LassoRegressionScratch(alpha=0.15, max_iter=2000).fit(X_raw, y_raw)

    print(f"True non-zero features: [w1, w2, w3] = {true_weights[:3]}")
    print("\nRidge Coefficients (theta_1 to theta_10):")
    print(np.round(ridge.theta[1:], 4))

    print("\nLasso Coefficients (theta_1 to theta_10):")
    print(np.round(lasso.theta[1:], 4))
    print(f"Number of exact zero features in Lasso: {np.sum(lasso.theta[1:] == 0.0)} of {n_features}")

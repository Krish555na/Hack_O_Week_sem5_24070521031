"""
Week 7: Regression — Polynomial Feature Mapping & Bias-Variance Tradeoff

Demonstrates:
1. Polynomial Feature Expansion from scratch without external dependencies
2. Modeling non-linear functions (e.g., sine waves, curves) via linear models
3. Overfitting vs. Underfitting diagnostics across increasing polynomial degrees
"""

import numpy as np
from typing import Tuple, Dict, Any
from linear_regression import LinearRegressionScratch, root_mean_squared_error


def generate_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Expands 1D feature array X into polynomial power columns [x, x^2, ..., x^degree].
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    features = [X]
    for d in range(2, degree + 1):
        features.append(X ** d)

    return np.hstack(features)


def analyze_degree_tradeoff(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    degrees: Tuple[int, ...] = (1, 2, 3, 5, 10)
) -> Dict[int, Dict[str, float]]:
    """Evaluates train and validation RMSE for each degree to diagnose bias and variance."""
    results = {}

    for d in degrees:
        X_tr_poly = generate_polynomial_features(X_train, degree=d)
        X_v_poly = generate_polynomial_features(X_val, degree=d)

        # Standardize features for numerical stability when degrees are high
        mean = np.mean(X_tr_poly, axis=0)
        std = np.std(X_tr_poly, axis=0)
        std[std == 0] = 1.0

        X_tr_norm = (X_tr_poly - mean) / std
        X_v_norm = (X_v_poly - mean) / std

        model = LinearRegressionScratch(solver="normal").fit(X_tr_norm, y_train)

        train_rmse = root_mean_squared_error(y_train, model.predict(X_tr_norm))
        val_rmse = root_mean_squared_error(y_val, model.predict(X_v_norm))

        results[d] = {
            "train_rmse": round(train_rmse, 4),
            "val_rmse": round(val_rmse, 4)
        }

    return results


if __name__ == "__main__":
    np.random.seed(42)
    # True function: y = sin(x) + noise
    x = np.linspace(-3, 3, 60)
    y = np.sin(x) + np.random.normal(0, 0.2, size=len(x))

    # Split 70/30 train/val
    perm = np.random.permutation(len(x))
    train_idx, val_idx = perm[:42], perm[42:]

    X_train, y_train = x[train_idx], y[train_idx]
    X_val, y_val = x[val_idx], y[val_idx]

    print("=== Polynomial Regression & Degree Analysis ===")
    analysis = analyze_degree_tradeoff(X_train, y_train, X_val, y_val)
    for deg, metrics in analysis.items():
        print(f"Degree {deg:2d} -> Train RMSE: {metrics['train_rmse']:.4f} | Val RMSE: {metrics['val_rmse']:.4f}")

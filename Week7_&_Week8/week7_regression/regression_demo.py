"""
Week 7: Regression Demo & Benchmark

Trains and compares:
- Ordinary Least Squares (Normal Equation)
- Ridge Regression (L2)
- Lasso Regression (L1)
Plots comparative regularization paths to 'output_plots/'.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegressionScratch, root_mean_squared_error, r2_score_scratch
from regularization_ridge_lasso import RidgeRegressionScratch, LassoRegressionScratch


def run_housing_price_demo(output_dir: str = "output_plots"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Generate Synthetic Real Estate Dataset (200 houses)
    # Features: [Area_sqft, Bedrooms, Age_years, Distance_to_transit_km, Uncorrelated_Noise_Feature]
    n_samples = 200
    area = np.random.uniform(500, 3500, n_samples)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples)
    age = np.random.uniform(0, 50, n_samples)
    transit = np.random.uniform(0.2, 15.0, n_samples)
    noise_feat = np.random.normal(0, 1.0, n_samples)

    X = np.column_stack([area, bedrooms, age, transit, noise_feat])
    # True relationship
    y = (
        150_000
        + 180.0 * area
        + 25_000.0 * bedrooms
        - 1_200.0 * age
        - 4_500.0 * transit
        + np.random.normal(0, 15_000, n_samples)
    )

    # Train / Test split (80/20)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Standardization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    # Fit models
    ols = LinearRegressionScratch(solver="normal").fit(X_train_scaled, y_train)
    ridge = RidgeRegressionScratch(alpha=20.0).fit(X_train_scaled, y_train)
    lasso = LassoRegressionScratch(alpha=500.0, max_iter=2000).fit(X_train_scaled, y_train)

    # Metrics
    models = [("OLS", ols), ("Ridge (L2)", ridge), ("Lasso (L1)", lasso)]
    metrics = {}

    for name, m in models:
        preds = m.predict(X_test_scaled)
        metrics[name] = {
            "RMSE": root_mean_squared_error(y_test, preds),
            "R2": r2_score_scratch(y_test, preds)
        }

    # Plot Actual vs Predicted & Regularization Paths
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Subplot 1: Predictions
    for name, m in models:
        preds = m.predict(X_test_scaled)
        axes[0].scatter(y_test / 1e3, preds / 1e3, alpha=0.6, label=f"{name} (R²={metrics[name]['R2']:.3f})")

    min_val = min(y_test) / 1e3
    max_val = max(y_test) / 1e3
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Fit")
    axes[0].set_title("Housing Price: Actual vs. Predicted ($k)", fontweight='bold')
    axes[0].set_xlabel("Actual Price ($k)")
    axes[0].set_ylabel("Predicted Price ($k)")
    axes[0].legend()

    # Subplot 2: Regularization Path across Alphas for Ridge
    alphas = np.logspace(-1, 3, 30)
    coefs = []
    for a in alphas:
        r = RidgeRegressionScratch(alpha=a).fit(X_train_scaled, y_train)
        coefs.append(r.theta[1:])
    coefs = np.array(coefs)

    feature_names = ["Area", "Bedrooms", "Age", "Transit Dist", "Noise Feat"]
    for i in range(5):
        axes[1].plot(alphas, coefs[:, i], label=feature_names[i], linewidth=1.8)
    axes[1].set_xscale("log")
    axes[1].set_title("Ridge Shrinkage Path across Regularization Penalty (α)", fontweight='bold')
    axes[1].set_xlabel("Alpha (Regularization Strength)")
    axes[1].set_ylabel("Coefficients")
    axes[1].legend()

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "07_regression_benchmark.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return metrics, chart_path


if __name__ == "__main__":
    m, p = run_housing_price_demo()
    print("=== Housing Price Regression Benchmark ===")
    for k, v in m.items():
        print(f"{k:12s} -> RMSE: ${v['RMSE']:,.2f} | R²: {v['R2']:.4f}")
    print(f"Saved plot: {p}")

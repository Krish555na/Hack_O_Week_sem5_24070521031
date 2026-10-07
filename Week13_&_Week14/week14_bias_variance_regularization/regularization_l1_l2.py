"""
Week 14 - Regularization Deep Dive: L1 (Lasso), L2 (Ridge) & ElasticNet from Scratch
===================================================================================
Topics Covered:
  1. Why Regularization works: constraining hypothesis space & shrinking weights
  2. L2 Regularization (Ridge / Tikhonov):
     - Analytical solution: w* = (X^T X + lambda * I)^(-1) X^T y
     - Weight shrinkage geometry (circular L2 ball)
  3. L1 Regularization (Lasso):
     - Subgradient & Soft-Thresholding operator: S_lambda(z) = sign(z) * max(|z| - lambda, 0)
     - Coordinate Descent optimization from scratch
     - Why L1 induces SPARSITY (diamond geometry corners on axes)
  4. ElasticNet: combining L1 and L2 for correlated feature stability
  5. Regularization path comparison & visualization dashboard

Dependencies: numpy, matplotlib (zero external ML dependencies)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# =====================================================================
# 1. RIDGE (L2) REGRESSION FROM SCRATCH
# =====================================================================

class RidgeRegressionScratch:
    """
    L2 Regularized Linear Regression:
      Objective: ||y - Xw||^2 + lambda * ||w||_2^2
      Closed-form solution: w = (X^T X + lambda * I)^(-1) X^T y
    """
    def __init__(self, l2_lambda=1.0):
        self.l2_lambda = l2_lambda
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        # Center data so intercept is not regularized
        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y)
        X_centered = X - self.x_mean_
        y_centered = y - self.y_mean_

        # Ridge closed form with identity matrix
        reg_matrix = self.l2_lambda * np.eye(n_features)
        self.coef_ = np.linalg.solve(X_centered.T @ X_centered + reg_matrix, X_centered.T @ y_centered)
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


# =====================================================================
# 2. LASSO (L1) REGRESSION VIA COORDINATE DESCENT & SOFT THRESHOLDING
# =====================================================================

class LassoRegressionScratch:
    """
    L1 Regularized Linear Regression using Coordinate Descent:
      Objective: 0.5 * ||y - Xw||^2 + lambda * ||w||_1
      Soft-Thresholding Operator: S_lambda(z) = sign(z) * max(|z| - lambda, 0)
    """
    def __init__(self, l1_lambda=0.1, max_iter=1000, tol=1e-5):
        self.l1_lambda = l1_lambda
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def soft_threshold(z, gamma):
        """Proximal soft-thresholding operator S_gamma(z)."""
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        else:
            return 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        # Center data
        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y)
        X_c = X - self.x_mean_
        y_c = y - self.y_mean_

        # Initialize weights
        w = np.zeros(n_features)
        col_sq_norms = np.sum(X_c ** 2, axis=0)

        for iteration in range(self.max_iter):
            w_old = np.copy(w)
            for j in range(n_features):
                if col_sq_norms[j] < 1e-12:
                    continue
                # Partial residual excluding feature j
                r = y_c - (X_c @ w - X_c[:, j] * w[j])
                rho_j = X_c[:, j] @ r
                # Update w_j via soft thresholding
                w[j] = self.soft_threshold(rho_j, self.l1_lambda) / col_sq_norms[j]

            # Convergence check
            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.coef_ = w
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


# =====================================================================
# 3. ELASTIC NET (L1 + L2) REGRESSION SCRATCH
# =====================================================================

class ElasticNetScratch:
    """
    Elastic Net combining L1 and L2 penalties:
      Objective: 0.5 * ||y - Xw||^2 + lambda * [ l1_ratio * ||w||_1 + 0.5 * (1 - l1_ratio) * ||w||_2^2 ]
    """
    def __init__(self, alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y)
        X_c = X - self.x_mean_
        y_c = y - self.y_mean_

        w = np.zeros(n_features)
        col_sq_norms = np.sum(X_c ** 2, axis=0)

        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1.0 - self.l1_ratio)

        for iteration in range(self.max_iter):
            w_old = np.copy(w)
            for j in range(n_features):
                if col_sq_norms[j] < 1e-12:
                    continue
                r = y_c - (X_c @ w - X_c[:, j] * w[j])
                rho_j = X_c[:, j] @ r
                denom = col_sq_norms[j] + l2_penalty
                w[j] = LassoRegressionScratch.soft_threshold(rho_j, l1_penalty) / denom

            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.coef_ = w
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


# =====================================================================
# 4. BENCHMARK & VISUALIZATION DASHBOARD
# =====================================================================

def run_regularization_demo():
    print("=" * 70)
    print("WEEK 14: REGULARIZATION DEEP DIVE - L1, L2 & ELASTICNET")
    print("=" * 70)

    # 1. Dataset with Sparse True Signal + Collinear Irrelevant Features
    # 8 features total: only first 3 have true signal, last 5 are noise/correlated
    np.random.seed(42)
    n_samples, n_features = 100, 8
    X = np.random.randn(n_samples, n_features)
    # Add multicollinearity: feature 4 is correlated with feature 1
    X[:, 3] = X[:, 0] * 0.9 + np.random.normal(0, 0.1, n_samples)
    true_beta = np.array([3.0, -2.5, 1.8, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ true_beta + np.random.normal(0, 0.5, n_samples)

    # Models
    ridge = RidgeRegressionScratch(l2_lambda=10.0).fit(X, y)
    lasso = LassoRegressionScratch(l1_lambda=15.0).fit(X, y)
    enet = ElasticNetScratch(alpha=12.0, l1_ratio=0.6).fit(X, y)

    print("\n[Feature Recovery & Sparsity Inspection]")
    print(f"{'Feature':<10} | {'True Beta':<10} | {'Ridge (L2)':<12} | {'Lasso (L1)':<12} | {'ElasticNet':<12}")
    print("-" * 65)
    for i in range(n_features):
        print(f"X{i+1:<9} | {true_beta[i]:<10.2f} | {ridge.coef_[i]:<12.3f} | {lasso.coef_[i]:<12.3f} | {enet.coef_[i]:<12.3f}")

    n_zeros_ridge = np.sum(np.abs(ridge.coef_) < 1e-4)
    n_zeros_lasso = np.sum(np.abs(lasso.coef_) < 1e-4)
    print(f"\nExact Zeros: Ridge={n_zeros_ridge}/{n_features} | Lasso={n_zeros_lasso}/{n_features} (Automatic Feature Selection)")

    # 2. Regularization Paths across Penalty Strengths
    lambdas = np.logspace(-1, 2.5, 30)
    ridge_paths = []
    lasso_paths = []

    for lam in lambdas:
        r = RidgeRegressionScratch(l2_lambda=lam).fit(X, y)
        l = LassoRegressionScratch(l1_lambda=lam).fit(X, y)
        ridge_paths.append(r.coef_)
        lasso_paths.append(l.coef_)

    ridge_paths = np.array(ridge_paths)
    lasso_paths = np.array(lasso_paths)

    # 3. Generate Visualization Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel A: Ridge Path (Smooth Asymptotic Shrinkage)
    for feat in range(n_features):
        label = f"X{feat+1}" if feat < 3 else None
        axes[0, 0].plot(lambdas, ridge_paths[:, feat], linewidth=1.8, label=label)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel(r"L2 Penalty $\lambda$ (log scale)", color="gray")
    axes[0, 0].set_ylabel("Weight Values", color="gray")
    axes[0, 0].set_title("Ridge Path (L2): Continuous Shrinkage Towards 0", color="white", fontsize=11, fontweight="bold")
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8)

    # Panel B: Lasso Path (Sparsity / Exact Zero Knockouts)
    for feat in range(n_features):
        label = f"X{feat+1}" if feat < 3 else None
        axes[0, 1].plot(lambdas, lasso_paths[:, feat], linewidth=1.8, label=label)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel(r"L1 Penalty $\lambda$ (log scale)", color="gray")
    axes[0, 1].set_ylabel("Weight Values", color="gray")
    axes[0, 1].set_title("Lasso Path (L1): Weights Truncate Exactly to Zero", color="white", fontsize=11, fontweight="bold")
    if axes[0, 1].get_legend_handles_labels()[0]:
        axes[0, 1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8)

    # Panel C: Geometric Intuition (L1 Diamond vs L2 Circle Contour)
    w1 = np.linspace(-2.2, 2.2, 100)
    w2 = np.linspace(-2.2, 2.2, 100)
    W1, W2 = np.meshgrid(w1, w2)
    # Elliptical loss contours centered at (1.5, 1.2)
    Loss = 2.0 * (W1 - 1.5)**2 + 0.8 * (W2 - 1.2)**2 - 0.5 * (W1 - 1.5)*(W2 - 1.2)

    axes[1, 0].contour(W1, W2, Loss, levels=8, colors="#8b949e", alpha=0.5)
    # L1 Diamond (|w1| + |w2| <= 1.0)
    axes[1, 0].plot([1.0, 0.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, -1.0, 0.0], color="#f85149", linewidth=2.5, label="L1 Diamond Constraint")
    # L2 Circle (w1^2 + w2^2 <= 1.0)
    theta = np.linspace(0, 2*np.pi, 100)
    axes[1, 0].plot(np.cos(theta), np.sin(theta), color="#58a6ff", linewidth=2.5, label="L2 Circular Constraint")
    # Mark sharp corner on axis
    axes[1, 0].scatter([1.0], [0.0], color="#f85149", s=80, zorder=5, label="L1 Corner Hit (w2=0, Sparse)")
    axes[1, 0].axhline(0, color="#30363d", linestyle=":")
    axes[1, 0].axvline(0, color="#30363d", linestyle=":")
    axes[1, 0].set_xlabel("Weight w1", color="gray")
    axes[1, 0].set_ylabel("Weight w2", color="gray")
    axes[1, 0].set_title("Geometric Reason: Why L1 Yields Sparsity", color="white", fontsize=11, fontweight="bold")
    axes[1, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8)

    # Panel D: Soft-Thresholding Operator Visualization
    z = np.linspace(-3, 3, 200)
    gamma = 1.0
    st_vals = [LassoRegressionScratch.soft_threshold(val, gamma) for val in z]
    axes[1, 1].plot(z, z, color="#8b949e", linestyle="--", label="Identity (Unregularized)")
    axes[1, 1].plot(z, st_vals, color="#3fb950", linewidth=2.4, label=f"Soft-Thresholding ($\\gamma={gamma}$)")
    axes[1, 1].axhline(0, color="#30363d", linestyle=":")
    axes[1, 1].axvline(0, color="#30363d", linestyle=":")
    axes[1, 1].fill_between([-gamma, gamma], -3, 3, color="#f85149", alpha=0.15, label="Dead Zone (Zero Output)")
    axes[1, 1].set_xlabel("Raw Correlation z", color="gray")
    axes[1, 1].set_ylabel("Thresholded Weight", color="gray")
    axes[1, 1].set_title("L1 Soft-Thresholding Function", color="white", fontsize=11, fontweight="bold")
    axes[1, 1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8.5)

    plt.tight_layout()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(curr_dir, "regularization_l1_l2_dashboard.png")
    plt.savefig(out_path, dpi=140, facecolor="#0d1117")
    plt.close()
    print(f"\n[SAVED] Dashboard written to {out_path}")


if __name__ == "__main__":
    run_regularization_demo()

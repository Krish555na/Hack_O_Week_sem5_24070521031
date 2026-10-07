"""
Week 14 - Bias-Variance Decomposition & Trade-Off Analysis from Scratch
=======================================================================
Topics Covered:
  1. The Bias-Variance Dilemma:
     - Underfitting (High Bias, Low Variance)
     - Overfitting (Low Bias, High Variance)
     - Sweet Spot / Optimal Generalization
  2. Mathematical Derivation & Proof:
     E[(y - f_hat(x))^2] = Bias(f_hat(x))^2 + Var(f_hat(x)) + sigma^2 (Noise)
  3. Monte Carlo Empirical Decomposition:
     - Resampling M distinct training sets from ground-truth data generator
     - Fitting polynomials of increasing degrees (1 to 10)
     - Directly measuring Bias^2, Variance, and Irreducible Noise
  4. Visualization Dashboard

Dependencies: numpy, matplotlib (zero external ML dependencies)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# =====================================================================
# 1. GROUND TRUTH DATA GENERATION & POLYNOMIAL MODEL
# =====================================================================

def true_function(x):
    """Underlying physical signal: non-linear harmonic function."""
    return np.cos(1.5 * np.pi * x)


def generate_dataset(n_samples=30, noise_std=0.25):
    """Draws a random noisy dataset from the ground truth."""
    x = np.random.uniform(0, 1, n_samples)
    noise = np.random.normal(0, noise_std, n_samples)
    y = true_function(x) + noise
    return x, y


class PolynomialRegressionScratch:
    """Least-squares polynomial regression using Normal Equation."""
    def __init__(self, degree=1):
        self.degree = degree
        self.weights = None

    def _vandermonde(self, x):
        return np.vander(x, N=self.degree + 1, increasing=True)

    def fit(self, x, y):
        X = self._vandermonde(x)
        # Normal equation with slight Tikhonov damping for numerical stability
        reg = 1e-6 * np.eye(X.shape[1])
        self.weights = np.linalg.solve(X.T @ X + reg, X.T @ y)
        return self

    def predict(self, x):
        X = self._vandermonde(x)
        return X @ self.weights


# =====================================================================
# 2. MONTE CARLO BIAS-VARIANCE DECOMPOSITION SIMULATOR
# =====================================================================

def perform_bias_variance_decomposition(
    n_datasets=100,
    n_train_samples=25,
    noise_std=0.25,
    degrees=range(1, 11)
):
    """
    Empirically decomposes total test error across degrees into:
      Total MSE = Bias^2 + Variance + Irreducible Noise
    """
    # Fixed test grid for evaluation
    x_test = np.linspace(0, 1, 100)
    y_true = true_function(x_test)
    irreducible_noise = noise_std ** 2

    bias_squared_list = []
    variance_list = []
    total_mse_list = []

    for deg in degrees:
        # Collect predictions across multiple independently sampled training sets
        predictions = np.zeros((n_datasets, len(x_test)))

        for m in range(n_datasets):
            x_train, y_train = generate_dataset(n_samples=n_train_samples, noise_std=noise_std)
            model = PolynomialRegressionScratch(degree=deg).fit(x_train, y_train)
            predictions[m, :] = model.predict(x_test)

        # E[f_hat(x)] across all dataset realisations
        expected_pred = np.mean(predictions, axis=0)

        # Bias^2 = (E[f_hat(x)] - f(x))^2
        bias_sq = np.mean((expected_pred - y_true) ** 2)

        # Variance = E[(f_hat(x) - E[f_hat(x)])^2]
        variance = np.mean(np.var(predictions, axis=0))

        # Expected Total MSE = E[(y - f_hat(x))^2]
        total_mse = bias_sq + variance + irreducible_noise

        bias_squared_list.append(bias_sq)
        variance_list.append(variance)
        total_mse_list.append(total_mse)

    return list(degrees), bias_squared_list, variance_list, irreducible_noise, total_mse_list


# =====================================================================
# 3. BENCHMARK & VISUALIZATION DASHBOARD
# =====================================================================

def run_bias_variance_demo():
    print("=" * 70)
    print("WEEK 14: BIAS-VARIANCE DECOMPOSITION & GENERALIZATION DYNAMICS")
    print("=" * 70)

    degrees, bias_sq, variance, noise, total_mse = perform_bias_variance_decomposition()

    print("\n[Empirical Decomposition Table]")
    print(f"{'Degree':<8} | {'Bias^2':<10} | {'Variance':<10} | {'Noise':<8} | {'Total MSE':<10} | {'Regime'}")
    print("-" * 65)
    for d, b, v, tot in zip(degrees, bias_sq, variance, total_mse):
        if d <= 2:
            regime = "High Bias (Underfitting)"
        elif d == 3 or d == 4:
            regime = "Optimal Trade-Off"
        else:
            regime = "High Variance (Overfitting)"
        print(f"{d:<8} | {b:<10.4f} | {v:<10.4f} | {noise:<8.4f} | {tot:<10.4f} | {regime}")

    # Generate Visualization Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel A: The Bias-Variance Tradeoff Curves
    axes[0, 0].plot(degrees, bias_sq, marker="o", color="#f85149", linewidth=2.2, label=r"$\mathrm{Bias}^2$ (Underfitting)")
    axes[0, 0].plot(degrees, variance, marker="s", color="#58a6ff", linewidth=2.2, label="Variance (Overfitting)")
    axes[0, 0].axhline(noise, color="#8b949e", linestyle="--", linewidth=1.5, label=r"Irreducible Noise $\sigma^2$")
    axes[0, 0].plot(degrees, total_mse, marker="^", color="#3fb950", linewidth=2.6, label="Total Expected Test MSE")
    opt_deg = degrees[np.argmin(total_mse)]
    axes[0, 0].axvline(opt_deg, color="#d29922", linestyle=":", label=f"Optimal Complexity (d={opt_deg})")
    axes[0, 0].set_xlabel("Model Complexity (Polynomial Degree)", color="gray")
    axes[0, 0].set_ylabel("Expected Error / Loss", color="gray")
    axes[0, 0].set_title("The Fundamental Bias-Variance Tradeoff", color="white", fontsize=11, fontweight="bold")
    axes[0, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8.5)

    # Panel B: Underfitting Realisation (Degree 1)
    x_grid = np.linspace(0, 1, 150)
    y_true_grid = true_function(x_grid)
    for _ in range(12):
        x_samp, y_samp = generate_dataset(25, 0.25)
        m_under = PolynomialRegressionScratch(degree=1).fit(x_samp, y_samp)
        axes[0, 1].plot(x_grid, m_under.predict(x_grid), color="#f85149", alpha=0.35, linewidth=1.2)
    axes[0, 1].plot(x_grid, y_true_grid, color="#3fb950", linewidth=2.5, linestyle="--", label="True Function")
    axes[0, 1].set_title("Degree 1: High Bias (Models tightly grouped, all wrong)", color="#f85149", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylim(-1.6, 1.6)
    axes[0, 1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    # Panel C: Overfitting Realisation (Degree 9)
    for _ in range(12):
        x_samp, y_samp = generate_dataset(25, 0.25)
        m_over = PolynomialRegressionScratch(degree=9).fit(x_samp, y_samp)
        axes[1, 0].plot(x_grid, m_over.predict(x_grid), color="#58a6ff", alpha=0.35, linewidth=1.2)
    axes[1, 0].plot(x_grid, y_true_grid, color="#3fb950", linewidth=2.5, linestyle="--", label="True Function")
    axes[1, 0].set_title("Degree 9: High Variance (Fits noise, huge fluctuations)", color="#58a6ff", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylim(-1.8, 1.8)
    axes[1, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    # Panel D: Optimal Realisation (Degree 4)
    for _ in range(12):
        x_samp, y_samp = generate_dataset(25, 0.25)
        m_opt = PolynomialRegressionScratch(degree=4).fit(x_samp, y_samp)
        axes[1, 1].plot(x_grid, m_opt.predict(x_grid), color="#3fb950", alpha=0.35, linewidth=1.2)
    axes[1, 1].plot(x_grid, y_true_grid, color="white", linewidth=2.5, linestyle="--", label="True Function")
    axes[1, 1].set_title(f"Degree {opt_deg}: Balanced Sweet Spot (Accurate & Stable)", color="#3fb950", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylim(-1.6, 1.6)
    axes[1, 1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    plt.tight_layout()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(curr_dir, "bias_variance_tradeoff_dashboard.png")
    plt.savefig(out_path, dpi=140, facecolor="#0d1117")
    plt.close()
    print(f"\n[SAVED] Dashboard written to {out_path}")


if __name__ == "__main__":
    run_bias_variance_demo()

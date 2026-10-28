"""
Week 16 - Loss Functions & Mathematical Derivatives from Scratch
================================================================
Topics Covered:
  1. Mean Squared Error (MSE / L2 Loss):
     - Regression standard
     - Gradient: dL/dy_hat = 1/m * (y_hat - y)
  2. Binary Cross-Entropy (BCE / Logistic Loss):
     - Binary classification standard
     - Numerically stabilized with epsilon clipping
     - Gradient w.r.t logits (combined with Sigmoid): dL/dz = 1/m * (y_hat - y)
  3. Categorical Cross-Entropy (CCE / Multi-Class Log Loss):
     - One-hot encoded multi-class standard
     - Gradient w.r.t logits (combined with Softmax): dL/dz = 1/m * (Y_hat - Y)
  4. Convexity, Hessian & Loss Landscape comparisons

Dependencies: numpy, matplotlib (zero external ML dependencies)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# =====================================================================
# 1. LOSS FUNCTIONS ENGINE (WITH ANALYTICAL DERIVATIVES)
# =====================================================================

class LossFunctions:
    """Standard Machine Learning Loss functions and gradients."""

    EPSILON = 1e-12

    # --- Mean Squared Error (MSE) ---
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error: 0.5/m * sum((y_pred - y_true)^2)"""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        m = len(y_true)
        return float(np.sum((y_pred - y_true) ** 2) / (2.0 * m))

    @staticmethod
    def mse_derivative(y_true, y_pred):
        """dL / dy_pred = 1/m * (y_pred - y_true)"""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        m = len(y_true)
        return (y_pred - y_true) / m

    # --- Binary Cross-Entropy (BCE) ---
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """Binary Cross-Entropy with clipping for numerical stability."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), LossFunctions.EPSILON, 1.0 - LossFunctions.EPSILON)
        m = len(y_true)
        loss = - (1.0 / m) * np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        """dL / dy_pred = 1/m * ( (y_pred - y_true) / (y_pred * (1 - y_pred)) )"""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), LossFunctions.EPSILON, 1.0 - LossFunctions.EPSILON)
        m = len(y_true)
        return (1.0 / m) * ((y_pred - y_true) / (y_pred * (1.0 - y_pred)))

    @staticmethod
    def bce_sigmoid_logit_derivative(y_true, y_pred):
        """dL / dz (Combined Sigmoid + BCE gradient: clean A - Y without saturation division)."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        m = len(y_true)
        return (y_pred - y_true) / m

    # --- Categorical Cross-Entropy (CCE) ---
    @staticmethod
    def categorical_cross_entropy(Y_true, Y_pred):
        """Multi-class cross-entropy for one-hot encoded targets."""
        Y_true = np.asarray(Y_true, dtype=float)
        Y_pred = np.clip(np.asarray(Y_pred, dtype=float), LossFunctions.EPSILON, 1.0 - LossFunctions.EPSILON)
        m = len(Y_true)
        loss = - (1.0 / m) * np.sum(Y_true * np.log(Y_pred))
        return float(loss)

    @staticmethod
    def cce_softmax_logit_derivative(Y_true, Y_pred):
        """dL / dz (Combined Softmax + CCE gradient: clean Y_pred - Y_true)."""
        Y_true = np.asarray(Y_true, dtype=float)
        Y_pred = np.asarray(Y_pred, dtype=float)
        m = len(Y_true)
        return (Y_pred - Y_true) / m


# =====================================================================
# 2. BENCHMARK & DEMONSTRATION RUNNER
# =====================================================================

def run_loss_demo():
    print("=" * 70)
    print("WEEK 16: LOSS FUNCTIONS & MATHEMATICAL GRADIENTS")
    print("=" * 70)

    # 1. Comparative Loss Values for Binary Prediction
    p_grid = np.linspace(0.01, 0.99, 100)
    target_1 = np.ones_like(p_grid)
    
    mse_vals = [LossFunctions.mse(np.array([1.0]), np.array([p])) for p in p_grid]
    bce_vals = [LossFunctions.binary_cross_entropy(np.array([1.0]), np.array([p])) for p in p_grid]

    print(f"\n[1] Penalty for Confident Wrong Prediction (y_true=1.0, y_pred=0.01):")
    print(f"    MSE Penalty: {LossFunctions.mse([1.0], [0.01]):.4f} (Bounded quad penalty)")
    print(f"    BCE Penalty: {LossFunctions.binary_cross_entropy([1.0], [0.01]):.4f} (Logarithmic steep penalty -> forces rapid learning!)")

    # 2. Multi-Class CCE Verification
    Y_true_onehot = np.array([[1, 0, 0], [0, 1, 0]])
    Y_pred_good = np.array([[0.8, 0.1, 0.1], [0.1, 0.85, 0.05]])
    Y_pred_poor = np.array([[0.2, 0.5, 0.3], [0.6, 0.2, 0.2]])

    loss_good = LossFunctions.categorical_cross_entropy(Y_true_onehot, Y_pred_good)
    loss_poor = LossFunctions.categorical_cross_entropy(Y_true_onehot, Y_pred_poor)
    print(f"\n[2] Categorical Cross-Entropy Multi-Class Benchmark:")
    print(f"    Well-calibrated predictions CCE: {loss_good:.4f}")
    print(f"    Poorly-calibrated predictions CCE: {loss_poor:.4f}")

    # 3. Generate Visualization Dashboard
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel A: MSE vs BCE Loss curves when y_true = 1
    axes[0].plot(p_grid, mse_vals, color="#58a6ff", linewidth=2.4, label="MSE Loss: $0.5 (y - \hat{y})^2$")
    axes[0].plot(p_grid, bce_vals, color="#f85149", linewidth=2.4, label="BCE Loss: $-\log(\hat{y})$")
    axes[0].set_xlabel("Predicted Probability r$\hat{y}$ (when True Class $y=1$)", color="gray")
    axes[0].set_ylabel("Loss Magnitude", color="gray")
    axes[0].set_title("Why Classification Uses BCE Over MSE", color="white", fontsize=11, fontweight="bold")
    axes[0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)
    axes[0].grid(True, color="#21262d", linestyle="--")

    # Panel B: Gradient Comparison (Backprop Driving Force)
    bce_grads = [LossFunctions.bce_sigmoid_logit_derivative([1.0], [p])[0] for p in p_grid]
    mse_grads = [LossFunctions.mse_derivative([1.0], [p])[0] for p in p_grid]
    axes[1].plot(p_grid, bce_grads, color="#f85149", linewidth=2.2, label=r"BCE + Sigmoid Logit Gradient ($\hat{y} - y$)")
    axes[1].plot(p_grid, mse_grads, color="#58a6ff", linewidth=2.2, label=r"MSE Gradient ($\hat{y} - y$)")
    axes[1].set_xlabel("Predicted Probability r$\hat{y}$ (when $y=1$)", color="gray")
    axes[1].set_ylabel("Gradient Magnitude", color="gray")
    axes[1].set_title("Gradient Signal Strength vs Predicted Probability", color="white", fontsize=11, fontweight="bold")
    axes[1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)
    axes[1].grid(True, color="#21262d", linestyle="--")

    plt.tight_layout()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_plot = os.path.join(curr_dir, "loss_functions_dashboard.png")
    plt.savefig(out_plot, dpi=140, facecolor="#0d1117")
    plt.close()
    print(f"\n[SAVED] Dashboard written to {out_plot}")


if __name__ == "__main__":
    run_loss_demo()

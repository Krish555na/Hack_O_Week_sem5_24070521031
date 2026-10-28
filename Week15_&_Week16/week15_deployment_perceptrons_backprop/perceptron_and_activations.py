"""
Week 15 - The Perceptron Algorithm, XOR Limitation & Activation Functions
========================================================================
Topics Covered:
  1. Frank Rosenblatt's Perceptron (1958) from scratch:
     - Mathematical formulation: y = step(w^T x + b)
     - Learning rule: w <- w + lr * (y - y_hat) * x
     - Perceptron Convergence Theorem
  2. The XOR Dilemma (Minsky & Papert, 1969):
     - Analytical proof of linear inseparability
     - Why single-layer perceptrons triggered the first "AI Winter"
  3. Modern Activation Functions & Exact Analytical Derivatives:
     - Sigmoid: f(z) = 1 / (1 + e^-z), f'(z) = f(z) * (1 - f(z))
     - Tanh: f(z) = tanh(z), f'(z) = 1 - f(z)^2
     - ReLU: f(z) = max(0, z), f'(z) = 1 (z > 0) else 0
     - LeakyReLU: f(z) = max(alpha * z, z), f'(z) = 1 (z > 0) else alpha
     - Softmax: multi-class probability distribution
  4. Vanishing Gradient Analysis (Sigmoid max grad = 0.25 vs ReLU = 1.0)

Dependencies: numpy, matplotlib (zero external ML dependencies)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# =====================================================================
# 1. ROSENBLATT PERCEPTRON (1958)
# =====================================================================

class RosenblattPerceptron:
    """
    Classical Single-Layer Perceptron:
      Decision function: y_hat = 1 if (w^T x + b >= 0) else 0
      Update rule: w <- w + lr * (y - y_hat) * x
                   b <- b + lr * (y - y_hat)
    """
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = 0.0
        self.errors_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_history = []

        for epoch in range(self.max_epochs):
            errors = 0
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_hat = 1 if linear_output >= 0.0 else 0
                error = y[i] - y_hat
                if error != 0:
                    self.weights += self.lr * error * X[i]
                    self.bias += self.lr * error
                    errors += 1
            self.errors_history.append(errors)
            if errors == 0:
                # Early stop on perfect linear separation
                break

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        linear_output = X @ self.weights + self.bias
        return (linear_output >= 0.0).astype(int)


# =====================================================================
# 2. ACTIVATION FUNCTIONS & DERIVATIVES ENGINE
# =====================================================================

class Activations:
    """Mathematical activations and their exact derivatives."""

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = Activations.sigmoid(z)
        return s * (1.0 - s)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        t = np.tanh(z)
        return 1.0 - t ** 2

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0.0).astype(float)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0.0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        return np.where(z > 0.0, 1.0, alpha)

    @staticmethod
    def softmax(z):
        # Numerically stable softmax with max subtraction
        shift_z = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(shift_z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


# =====================================================================
# 3. BENCHMARK & DEMONSTRATION RUNNER
# =====================================================================

def run_perceptron_activations_demo():
    print("=" * 70)
    print("WEEK 15: PERCEPTRONS, XOR PARADOX & ACTIVATION FUNCTIONS")
    print("=" * 70)

    # 1. Perceptron on Boolean Logic Gates
    X_gates = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    y_or  = np.array([0, 1, 1, 1])
    y_xor = np.array([0, 1, 1, 0])

    p_and = RosenblattPerceptron(learning_rate=0.1, max_epochs=30).fit(X_gates, y_and)
    p_or  = RosenblattPerceptron(learning_rate=0.1, max_epochs=30).fit(X_gates, y_or)
    p_xor = RosenblattPerceptron(learning_rate=0.1, max_epochs=50).fit(X_gates, y_xor)

    acc_and = np.mean(p_and.predict(X_gates) == y_and)
    acc_or  = np.mean(p_or.predict(X_gates) == y_or)
    acc_xor = np.mean(p_xor.predict(X_gates) == y_xor)

    print(f"\n[1] Perceptron Gate Solving Capability:")
    print(f"    AND Gate Accuracy: {acc_and*100:.1f}% (Converged in {len(p_and.errors_history)} epochs)")
    print(f"    OR  Gate Accuracy: {acc_or*100:.1f}% (Converged in {len(p_or.errors_history)} epochs)")
    print(f"    XOR Gate Accuracy: {acc_xor*100:.1f}% (FAILS! Linear boundary cannot separate XOR)")

    # 2. Activation Function Derivative Comparison (Vanishing Gradient Proof)
    z_test = np.linspace(-6, 6, 300)
    sig_grad = Activations.sigmoid_derivative(z_test)
    tanh_grad = Activations.tanh_derivative(z_test)
    relu_grad = Activations.relu_derivative(z_test)

    print(f"\n[2] Peak Gradient Magnitudes (Backprop Signal Retention):")
    print(f"    Sigmoid Peak Derivative:   {np.max(sig_grad):.4f} (Causes vanishing gradient when stacked!)")
    print(f"    Tanh    Peak Derivative:   {np.max(tanh_grad):.4f} (Zero-centered, but saturates at tails)")
    print(f"    ReLU    Active Derivative: {np.max(relu_grad):.4f} (Perfect 1.0 gradient flow for positive activations)")

    # 3. Generate Visualization Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel A: Activation Curves
    axes[0, 0].plot(z_test, Activations.sigmoid(z_test), color="#58a6ff", linewidth=2.2, label="Sigmoid $\sigma(z)$")
    axes[0, 0].plot(z_test, Activations.tanh(z_test), color="#3fb950", linewidth=2.2, label="Tanh $\\tanh(z)$")
    axes[0, 0].plot(z_test, Activations.relu(z_test), color="#f0883e", linewidth=2.2, label="ReLU $\max(0, z)$")
    axes[0, 0].plot(z_test, Activations.leaky_relu(z_test, 0.1), color="#a371f7", linestyle="--", linewidth=1.8, label="Leaky ReLU ($\\alpha=0.1$)")
    axes[0, 0].set_ylim(-1.5, 4.0)
    axes[0, 0].axhline(0, color="#30363d", linestyle=":")
    axes[0, 0].axvline(0, color="#30363d", linestyle=":")
    axes[0, 0].set_title("Activation Functions $f(z)$", color="white", fontsize=11, fontweight="bold")
    axes[0, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8.5)

    # Panel B: Activation Derivatives (Vanishing Gradient Inspection)
    axes[0, 1].plot(z_test, sig_grad, color="#58a6ff", linewidth=2.2, label="Sigmoid' (Peak = 0.25)")
    axes[0, 1].plot(z_test, tanh_grad, color="#3fb950", linewidth=2.2, label="Tanh' (Peak = 1.0)")
    axes[0, 1].plot(z_test, relu_grad, color="#f0883e", linewidth=2.2, label="ReLU' (Step: 0 or 1)")
    axes[0, 1].set_ylim(-0.1, 1.2)
    axes[0, 1].axhline(0, color="#30363d", linestyle=":")
    axes[0, 1].axvline(0, color="#30363d", linestyle=":")
    axes[0, 1].set_title("Derivatives $f'(z)$ (Gradient Flow)", color="white", fontsize=11, fontweight="bold")
    axes[0, 1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8.5)

    # Panel C: Perceptron Decision Boundary on OR Gate (Solvable)
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z_or = p_or.predict(grid).reshape(xx.shape)
    axes[1, 0].contourf(xx, yy, Z_or, alpha=0.3, cmap="coolwarm")
    axes[1, 0].scatter(X_gates[:, 0], X_gates[:, 1], c=y_or, cmap="coolwarm", s=100, edgecolor="white", linewidth=1.5)
    axes[1, 0].set_title("OR Gate: Linearly Separable (Solved 100%)", color="#3fb950", fontsize=11, fontweight="bold")

    # Panel D: Perceptron Decision Boundary on XOR Gate (Unsolvable)
    Z_xor = p_xor.predict(grid).reshape(xx.shape)
    axes[1, 1].contourf(xx, yy, Z_xor, alpha=0.3, cmap="coolwarm")
    axes[1, 1].scatter(X_gates[:, 0], X_gates[:, 1], c=y_xor, cmap="coolwarm", s=100, edgecolor="white", linewidth=1.5)
    axes[1, 1].set_title("XOR Gate: Linearly Inseparable (Requires MLP Hidden Layer)", color="#f85149", fontsize=11, fontweight="bold")

    plt.tight_layout()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_plot = os.path.join(curr_dir, "perceptron_and_activations_dashboard.png")
    plt.savefig(out_plot, dpi=140, facecolor="#0d1117")
    plt.close()
    print(f"\n[SAVED] Dashboard written to {out_plot}")


if __name__ == "__main__":
    run_perceptron_activations_demo()

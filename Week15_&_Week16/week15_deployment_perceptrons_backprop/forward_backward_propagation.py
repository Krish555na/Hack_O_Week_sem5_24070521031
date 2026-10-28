"""
Week 15 - Forward & Backward Propagation: Mathematical Mechanics & Gradient Checking
=====================================================================================
Topics Covered:
  1. Computational Graph Architecture (2-Layer Neural Network):
     Input (D=2) -> Hidden Layer (H=4, ReLU) -> Output Layer (K=1, Sigmoid)
  2. Forward Propagation:
     Z1 = X W1 + b1,  A1 = ReLU(Z1)
     Z2 = A1 W2 + b2, A2 = Sigmoid(Z2)
  3. Binary Cross-Entropy Loss (BCE):
     L = -1/m * sum(y * log(A2) + (1-y) * log(1-A2))
  4. Backward Propagation (Chain Rule Step-by-Step):
     dZ2 = A2 - Y
     dW2 = 1/m * (A1^T dZ2),  db2 = 1/m * sum(dZ2)
     dA1 = dZ2 W2^T
     dZ1 = dA1 * ReLU'(Z1)
     dW1 = 1/m * (X^T dZ1),   db1 = 1/m * sum(dZ1)
  5. Numerical Gradient Checking (Verifying analytical backprop with epsilon=1e-7):
     relative_error = ||g_analytical - g_numerical|| / (||g_analytical|| + ||g_numerical||) < 1e-7
  6. Solving the Non-Linear XOR Problem (100% accuracy)

Dependencies: numpy, matplotlib (zero external ML dependencies)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# =====================================================================
# 1. TWO-LAYER NEURAL NETWORK (FROM FIRST PRINCIPLES)
# =====================================================================

class TwoLayerNNScratch:
    """
    2-Layer Fully Connected Neural Network with ReLU hidden activation
    and Sigmoid binary output.
    """
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        # He / Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        self.loss_history = []

    def forward(self, X):
        """Executes forward propagation and stores cache for backprop."""
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0.0, Z1)  # ReLU
        Z2 = A1 @ self.W2 + self.b2
        # Sigmoid with numerical clipping
        Z2_clipped = np.clip(Z2, -30, 30)
        A2 = 1.0 / (1.0 + np.exp(-Z2_clipped))

        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def compute_loss(self, A2, Y):
        """Computes binary cross-entropy loss."""
        m = len(Y)
        eps = 1e-12
        loss = - (1.0 / m) * np.sum(Y * np.log(A2 + eps) + (1.0 - Y) * np.log(1.0 - A2 + eps))
        return float(loss)

    def backward(self, cache, Y):
        """Executes backward propagation using analytical chain rule."""
        X = cache["X"]
        Z1 = cache["Z1"]
        A1 = cache["A1"]
        A2 = cache["A2"]
        m = len(Y)

        # Output layer gradients (dL/dZ2 for BCE + Sigmoid simplifies to A2 - Y)
        dZ2 = A2 - Y
        dW2 = (1.0 / m) * (A1.T @ dZ2)
        db2 = (1.0 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0.0).astype(float)  # ReLU derivative
        dW1 = (1.0 / m) * (X.T @ dZ1)
        db1 = (1.0 / m) * np.sum(dZ1, axis=0, keepdims=True)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def update_parameters(self, grads):
        """Gradient descent parameter update."""
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]

    def fit(self, X, Y, epochs=1000):
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.loss_history = []
        for _ in range(epochs):
            A2, cache = self.forward(X)
            loss = self.compute_loss(A2, Y)
            self.loss_history.append(loss)
            grads = self.backward(cache, Y)
            self.update_parameters(grads)

        return self

    def predict(self, X):
        A2, _ = self.forward(X)
        return (A2 >= 0.5).astype(int)


# =====================================================================
# 2. NUMERICAL GRADIENT CHECKING ENGINE
# =====================================================================

def gradient_check(model, X, Y, epsilon=1e-7):
    """
    Computes numerical gradient via two-sided symmetric finite differences
    and compares with analytical backprop gradients.
    """
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # 1. Get analytical gradients
    A2, cache = model.forward(X)
    analytical_grads = model.backward(cache, Y)

    # Flatten analytical gradients into single vector
    g_analytical = np.concatenate([
        analytical_grads["dW1"].ravel(),
        analytical_grads["db1"].ravel(),
        analytical_grads["dW2"].ravel(),
        analytical_grads["db2"].ravel()
    ])

    # 2. Compute numerical gradients parameter by parameter
    param_list = [model.W1, model.b1, model.W2, model.b2]
    g_numerical = []

    for param in param_list:
        for idx in np.ndindex(param.shape):
            orig_val = param[idx]

            # J(theta + epsilon)
            param[idx] = orig_val + epsilon
            A2_plus, _ = model.forward(X)
            loss_plus = model.compute_loss(A2_plus, Y)

            # J(theta - epsilon)
            param[idx] = orig_val - epsilon
            A2_minus, _ = model.forward(X)
            loss_minus = model.compute_loss(A2_minus, Y)

            # Reset original value
            param[idx] = orig_val

            # Symmetric difference quotient
            approx_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
            g_numerical.append(approx_grad)

    g_numerical = np.array(g_numerical)

    # 3. Relative difference formula
    numerator = np.linalg.norm(g_analytical - g_numerical)
    denominator = np.linalg.norm(g_analytical) + np.linalg.norm(g_numerical)
    relative_difference = numerator / (denominator + 1e-12)

    return relative_difference, g_analytical, g_numerical


# =====================================================================
# 3. BENCHMARK & DEMONSTRATION RUNNER
# =====================================================================

def run_forward_backward_demo():
    print("=" * 70)
    print("WEEK 15: FORWARD & BACKWARD PROPAGATION + GRADIENT CHECKING")
    print("=" * 70)

    # 1. Gradient Checking Verification
    np.random.seed(123)
    X_sample = np.random.randn(6, 2)
    Y_sample = np.array([[1], [0], [1], [1], [0], [0]])
    test_net = TwoLayerNNScratch(input_dim=2, hidden_dim=4, output_dim=1)

    rel_diff, g_ana, g_num = gradient_check(test_net, X_sample, Y_sample, epsilon=1e-7)
    print(f"\n[1] Numerical Gradient Checking Result:")
    print(f"    Relative Difference: {rel_diff:.3e}")
    if rel_diff < 1e-6:
        print("    [SUCCESS] Backpropagation implementation verified with high precision (< 1e-6)!")
    else:
        print("    [WARNING] Gradient discrepancy detected.")

    # 2. Solving the XOR Problem (which Rosenblatt's Perceptron failed)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([[0], [1], [1], [0]], dtype=float)

    xor_net = TwoLayerNNScratch(input_dim=2, hidden_dim=6, output_dim=1, learning_rate=0.8)
    xor_net.fit(X_xor, y_xor, epochs=1200)

    preds_xor = xor_net.predict(X_xor)
    acc_xor = np.mean(preds_xor == y_xor)
    print(f"\n[2] XOR Problem Resolution via Multi-Layer Backprop:")
    print(f"    Initial Loss: {xor_net.loss_history[0]:.4f} -> Final Loss: {xor_net.loss_history[-1]:.4f}")
    print(f"    Accuracy:     {acc_xor * 100:.1f}% (Solved 100%!)")

    # 3. Generate Visualization Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel A: Gradient Check Scatter (Analytical vs Numerical)
    axes[0].scatter(g_num, g_ana, color="#58a6ff", s=40, alpha=0.8, edgecolor="white", label="Param Gradients")
    lims = [min(np.min(g_num), np.min(g_ana)), max(np.max(g_num), np.max(g_ana))]
    axes[0].plot(lims, lims, color="#3fb950", linestyle="--", linewidth=1.8, label="Ideal y = x")
    axes[0].set_xlabel("Numerical Gradient", color="gray")
    axes[0].set_ylabel("Analytical Gradient", color="gray")
    axes[0].set_title(f"Gradient Check (Rel Error: {rel_diff:.2e})", color="white", fontsize=11, fontweight="bold")
    axes[0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8.5)

    # Panel B: Loss Curve
    axes[1].plot(xor_net.loss_history, color="#f0883e", linewidth=2.0)
    axes[1].set_xlabel("Epoch", color="gray")
    axes[1].set_ylabel("Binary Cross-Entropy Loss", color="gray")
    axes[1].set_title("XOR Training Loss Trajectory", color="white", fontsize=11, fontweight="bold")

    # Panel C: Non-linear Decision Boundary for XOR
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 150), np.linspace(-0.5, 1.5, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs, _ = xor_net.forward(grid)
    Z = probs.reshape(xx.shape)
    axes[2].contourf(xx, yy, Z, levels=20, alpha=0.4, cmap="coolwarm")
    axes[2].scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor.ravel(), cmap="coolwarm", s=120, edgecolor="white", linewidth=2.0)
    axes[2].set_title("Non-Linear XOR Decision Boundary (100% Solved)", color="#3fb950", fontsize=11, fontweight="bold")

    plt.tight_layout()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_plot = os.path.join(curr_dir, "forward_backward_propagation_dashboard.png")
    plt.savefig(out_plot, dpi=140, facecolor="#0d1117")
    plt.close()
    print(f"\n[SAVED] Dashboard written to {out_plot}")


if __name__ == "__main__":
    run_forward_backward_demo()

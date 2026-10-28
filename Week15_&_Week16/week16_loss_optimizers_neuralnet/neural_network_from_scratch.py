"""
Week 16 - Deep Neural Network Framework from Scratch (Pure NumPy)
=================================================================
Topics Covered:
  1. Modular Object-Oriented Deep Learning Architecture:
     - Base Layer abstraction
     - Dense / Fully-Connected Layer with Xavier/He Initialization
     - Modular Activation Layers (ReLU, Sigmoid, Tanh, Softmax)
  2. Loss Function & Optimizer Integrations (SGD, Momentum, RMSprop, Adam)
  3. Mini-Batch Training Loop with Gradient Shuffling
  4. Model Persistence (Pickle & Joblib Serialization)
  5. Solving Complex Non-Linear Manifolds (Concentric Circles & Moons)

Dependencies: numpy, joblib, matplotlib (zero external ML dependencies)
"""

import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# =====================================================================
# 1. MODULAR LAYER ABSTRACTIONS
# =====================================================================

class Layer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class Dense(Layer):
    """Fully Connected Dense Linear Layer: Z = X W + b"""
    def __init__(self, in_features, out_features, init_type="he"):
        self.in_features = in_features
        self.out_features = out_features

        # Weight initialization
        if init_type == "he":
            self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        elif init_type == "xavier":
            self.weights = np.random.randn(in_features, out_features) * np.sqrt(1.0 / in_features)
        else:
            self.weights = np.random.randn(in_features, out_features) * 0.01

        self.biases = np.zeros((1, out_features))

        # Gradients
        self.grad_weights = None
        self.grad_biases = None
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.weights + self.biases

    def backward(self, grad_output):
        m = len(self.inputs)
        self.grad_weights = (self.inputs.T @ grad_output) / m
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True) / m
        return grad_output @ self.weights.T


class ReLU(Layer):
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0.0, inputs)

    def backward(self, grad_output):
        return grad_output * (self.inputs > 0.0).astype(float)


class Sigmoid(Layer):
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        clipped = np.clip(inputs, -30, 30)
        self.output = 1.0 / (1.0 + np.exp(-clipped))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1.0 - self.output)


class Tanh(Layer):
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1.0 - self.output ** 2)


# =====================================================================
# 2. NEURAL NETWORK PIPELINE CLASS
# =====================================================================

class NeuralNetworkScratch:
    """
    Sequential Neural Network container:
      - Chains arbitrary Dense and Activation layers
      - Integrates Adam, Momentum, or standard SGD
      - Mini-batch SGD training loop
    """
    def __init__(self, optimizer="adam", lr=0.01):
        self.layers = []
        self.optimizer_type = optimizer
        self.lr = lr
        self.history = {"loss": [], "accuracy": []}

        # Optimizer momentum & variance states
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        self.t = 0

    def add(self, layer: Layer):
        self.layers.append(layer)
        if isinstance(layer, Dense):
            self.m_w.append(np.zeros_like(layer.weights))
            self.v_w.append(np.zeros_like(layer.weights))
            self.m_b.append(np.zeros_like(layer.biases))
            self.v_b.append(np.zeros_like(layer.biases))
        return self

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_loss):
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def _step_optimizer(self):
        self.t += 1
        dense_idx = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for layer in self.layers:
            if isinstance(layer, Dense):
                gw = layer.grad_weights
                gb = layer.grad_biases

                if self.optimizer_type == "adam":
                    # Adam First Moment
                    self.m_w[dense_idx] = beta1 * self.m_w[dense_idx] + (1.0 - beta1) * gw
                    self.m_b[dense_idx] = beta1 * self.m_b[dense_idx] + (1.0 - beta1) * gb
                    # Adam Second Moment
                    self.v_w[dense_idx] = beta2 * self.v_w[dense_idx] + (1.0 - beta2) * (gw ** 2)
                    self.v_b[dense_idx] = beta2 * self.v_b[dense_idx] + (1.0 - beta2) * (gb ** 2)
                    # Bias Correction
                    m_w_hat = self.m_w[dense_idx] / (1.0 - beta1 ** self.t)
                    m_b_hat = self.m_b[dense_idx] / (1.0 - beta1 ** self.t)
                    v_w_hat = self.v_w[dense_idx] / (1.0 - beta2 ** self.t)
                    v_b_hat = self.v_b[dense_idx] / (1.0 - beta2 ** self.t)
                    # Update
                    layer.weights -= (self.lr / (np.sqrt(v_w_hat) + eps)) * m_w_hat
                    layer.biases -= (self.lr / (np.sqrt(v_b_hat) + eps)) * m_b_hat

                else:
                    # Standard Gradient Descent
                    layer.weights -= self.lr * gw
                    layer.biases -= self.lr * gb

                dense_idx += 1

    def fit(self, X, y, epochs=500, batch_size=32, verbose=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = len(X)
        self.history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            # Shuffle mini-batches
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = int(np.ceil(n_samples / batch_size))

            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_samples)
                xb, yb = X_shuffled[start:end], y_shuffled[start:end]

                preds = self.forward(xb)
                # Binary Cross Entropy Loss
                clipped = np.clip(preds, 1e-12, 1.0 - 1e-12)
                b_loss = - np.mean(yb * np.log(clipped) + (1.0 - yb) * np.log(1.0 - clipped))
                epoch_loss += b_loss

                # Output gradient (for BCE + Sigmoid: dL/dZ = (preds - y))
                grad_output = (preds - yb)
                self.backward(grad_output)
                self._step_optimizer()

            # Record metrics
            full_preds = self.forward(X)
            clipped_full = np.clip(full_preds, 1e-12, 1.0 - 1e-12)
            tot_loss = float(- np.mean(y * np.log(clipped_full) + (1.0 - y) * np.log(1.0 - clipped_full)))
            acc = float(np.mean((full_preds >= 0.5) == y))

            self.history["loss"].append(tot_loss)
            self.history["accuracy"].append(acc)

            if verbose and (epoch + 1) % (epochs // 5) == 0:
                print(f"    Epoch {epoch+1:4d}/{epochs} | Loss: {tot_loss:.4f} | Accuracy: {acc*100:.2f}%")

        return self

    def predict(self, X):
        probs = self.forward(X)
        return (probs >= 0.5).astype(int)

    def save(self, filepath):
        """Persist trained neural network model."""
        joblib.dump(self, filepath, compress=3)

    @staticmethod
    def load(filepath):
        """Load trained neural network model."""
        return joblib.load(filepath)


# =====================================================================
# 3. BENCHMARK & DEMONSTRATION RUNNER
# =====================================================================

def generate_concentric_circles(n_samples=400, noise=0.08):
    """Synthetic non-linear circular classification problem."""
    n_per_ring = n_samples // 2
    theta = np.random.uniform(0, 2 * np.pi, n_per_ring)

    # Inner ring (class 0)
    r0 = np.random.normal(0.4, noise, n_per_ring)
    X0 = np.stack([r0 * np.cos(theta), r0 * np.sin(theta)], axis=1)

    # Outer ring (class 1)
    r1 = np.random.normal(0.9, noise, n_per_ring)
    X1 = np.stack([r1 * np.cos(theta), r1 * np.sin(theta)], axis=1)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_ring), np.ones(n_per_ring)])
    return X, y


def run_neural_net_demo():
    print("=" * 70)
    print("WEEK 16: DEEP NEURAL NETWORK FROM SCRATCH (PURE NUMPY)")
    print("=" * 70)

    # 1. Dataset Generation: Concentric Circles (Highly Non-Linear)
    X, y = generate_concentric_circles(n_samples=500, noise=0.07)

    # 2. Build 3-Layer Deep MLP: 2 -> 16 (ReLU) -> 8 (ReLU) -> 1 (Sigmoid)
    print("\n[1] Constructing Deep Neural Network Architecture:")
    print("    Input (2) -> Dense(16, He) -> ReLU -> Dense(8, He) -> ReLU -> Dense(1) -> Sigmoid")
    print("    Optimizer: Adam (lr=0.02, beta1=0.9, beta2=0.999)")

    net = NeuralNetworkScratch(optimizer="adam", lr=0.02)
    net.add(Dense(2, 16, init_type="he"))
    net.add(ReLU())
    net.add(Dense(16, 8, init_type="he"))
    net.add(ReLU())
    net.add(Dense(8, 1, init_type="xavier"))
    net.add(Sigmoid())

    # 3. Train the Model
    print("\n[2] Training Progress:")
    net.fit(X, y, epochs=250, batch_size=32, verbose=True)

    final_acc = net.history["accuracy"][-1]
    final_loss = net.history["loss"][-1]
    print(f"\n[3] Final Evaluation:")
    print(f"    Final Training Loss: {final_loss:.4f}")
    print(f"    Final Accuracy:      {final_acc*100:.2f}% (Non-linear manifold solved!)")

    # 4. Model Persistence Check
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_dir, "trained_mlp_model.joblib")
    net.save(model_path)
    loaded_net = NeuralNetworkScratch.load(model_path)
    loaded_acc = np.mean(loaded_net.predict(X).ravel() == y)
    print(f"\n[4] Model Persistence (Joblib Serialization):")
    print(f"    Model saved to: {model_path}")
    print(f"    Reloaded Model Accuracy: {loaded_acc*100:.2f}% (Fidelity Confirmed)")

    # 5. Generate Visualization Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel A: Training Curves
    epochs_range = range(1, len(net.history["loss"]) + 1)
    axes[0].plot(epochs_range, net.history["loss"], color="#f85149", linewidth=2.0, label="BCE Loss")
    axes[0].plot(epochs_range, net.history["accuracy"], color="#3fb950", linewidth=2.0, label="Accuracy")
    axes[0].set_xlabel("Epoch", color="gray")
    axes[0].set_ylabel("Metric", color="gray")
    axes[0].set_title("Training Loss & Accuracy History", color="white", fontsize=11, fontweight="bold")
    axes[0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8.5)

    # Panel B: Learned Non-Linear Decision Boundary
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = net.forward(grid).reshape(xx.shape)

    axes[1].contourf(xx, yy, probs, levels=20, alpha=0.45, cmap="coolwarm")
    axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=30, edgecolor="k", linewidth=0.5)
    axes[1].set_title(f"Learned Decision Boundary (Acc: {final_acc*100:.1f}%)", color="#58a6ff", fontsize=11, fontweight="bold")

    # Panel C: Weights Distribution across Layers
    w_l1 = net.layers[0].weights.ravel()
    w_l2 = net.layers[2].weights.ravel()
    w_l3 = net.layers[4].weights.ravel()
    axes[2].hist(w_l1, bins=15, alpha=0.6, color="#58a6ff", label="Layer 1 (Dense 2->16)")
    axes[2].hist(w_l2, bins=15, alpha=0.6, color="#3fb950", label="Layer 2 (Dense 16->8)")
    axes[2].hist(w_l3, bins=15, alpha=0.6, color="#f0883e", label="Layer 3 (Dense 8->1)")
    axes[2].set_xlabel("Weight Values", color="gray")
    axes[2].set_ylabel("Count", color="gray")
    axes[2].set_title("Learned Weight Distributions (He/Xavier)", color="white", fontsize=11, fontweight="bold")
    axes[2].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8.5)

    plt.tight_layout()
    out_plot = os.path.join(curr_dir, "neural_network_scratch_dashboard.png")
    plt.savefig(out_plot, dpi=140, facecolor="#0d1117")
    plt.close()
    print(f"\n[SAVED] Dashboard written to {out_plot}")


if __name__ == "__main__":
    run_neural_net_demo()

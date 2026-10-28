"""
Week 16 - Deep Learning Optimizers from Scratch: SGD, Momentum, RMSprop & Adam
==============================================================================
Topics Covered:
  1. Standard Stochastic Gradient Descent (SGD):
     w <- w - lr * grad
  2. SGD with Classical Momentum:
     v <- beta * v + (1 - beta) * grad
     w <- w - lr * v
     (Damps oscillations, speeds through ravines)
  3. RMSprop (Adaptive Learning Rate):
     s <- beta * s + (1 - beta) * grad^2
     w <- w - (lr / (sqrt(s) + eps)) * grad
  4. Adam (Adaptive Moment Estimation):
     m <- beta1 * m + (1 - beta1) * grad
     v <- beta2 * v + (1 - beta2) * grad^2
     m_hat <- m / (1 - beta1^t)  (bias correction)
     v_hat <- v / (1 - beta2^t)  (bias correction)
     w <- w - (lr / (sqrt(v_hat) + eps)) * m_hat
  5. Optimization Trajectory Benchmark on an Elliptical Ravine Surface

Dependencies: numpy, matplotlib (zero external ML dependencies)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# =====================================================================
# 1. OPTIMIZER CLASSES (MODULAR STATEFUL ARCHITECTURE)
# =====================================================================

class SGDOptimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, param, grad, key=None):
        return param - self.lr * grad


class MomentumOptimizer:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = {}

    def update(self, param, grad, key="w"):
        if key not in self.velocities:
            self.velocities[key] = np.zeros_like(param)
        self.velocities[key] = self.beta * self.velocities[key] + (1.0 - self.beta) * grad
        return param - self.lr * self.velocities[key]


class RMSpropOptimizer:
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.squares = {}

    def update(self, param, grad, key="w"):
        if key not in self.squares:
            self.squares[key] = np.zeros_like(param)
        self.squares[key] = self.beta * self.squares[key] + (1.0 - self.beta) * (grad ** 2)
        step = (self.lr / (np.sqrt(self.squares[key]) + self.eps)) * grad
        return param - step


class AdamOptimizer:
    """
    Adam (Kingma & Ba, 2014) with first/second moment tracking
    and bias correction.
    """
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, param, grad, key="w"):
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)
            self.t[key] = 0

        self.t[key] += 1
        t = self.t[key]

        # 1st moment (momentum)
        self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
        # 2nd moment (uncentered variance)
        self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad ** 2)

        # Bias correction
        m_hat = self.m[key] / (1.0 - self.beta1 ** t)
        v_hat = self.v[key] / (1.0 - self.beta2 ** t)

        step = (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
        return param - step


# =====================================================================
# 2. TEST BENCH: NON-ISOTROPIC RAVINE LOSS FUNCTION
# =====================================================================

def ravine_loss(w):
    """
    Steep anisotropic valley:
    L(w1, w2) = 0.5 * (10.0 * w1^2 + 0.5 * w2^2)
    Gradient has 20x higher curvature in w1 than w2.
    """
    return 0.5 * (10.0 * w[0] ** 2 + 0.5 * w[1] ** 2)


def ravine_gradient(w):
    return np.array([10.0 * w[0], 0.5 * w[1]])


def simulate_optimizer_trajectory(optimizer_cls, opt_kwargs, start_point=np.array([4.0, 7.0]), steps=80):
    """Tracks parameter trajectory through the ravine surface."""
    opt = optimizer_cls(**opt_kwargs)
    point = np.copy(start_point).astype(float)
    trajectory = [np.copy(point)]
    losses = [ravine_loss(point)]

    for _ in range(steps):
        grad = ravine_gradient(point)
        point = opt.update(point, grad, key="coord")
        trajectory.append(np.copy(point))
        losses.append(ravine_loss(point))

    return np.array(trajectory), np.array(losses)


# =====================================================================
# 3. BENCHMARK & DEMONSTRATION RUNNER
# =====================================================================

def run_optimizers_demo():
    print("=" * 70)
    print("WEEK 16: OPTIMIZERS BENCHMARK (SGD, MOMENTUM, RMSPROP, ADAM)")
    print("=" * 70)

    start = np.array([3.5, 6.0])
    steps = 90

    traj_sgd, loss_sgd = simulate_optimizer_trajectory(SGDOptimizer, {"lr": 0.08}, start, steps)
    traj_mom, loss_mom = simulate_optimizer_trajectory(MomentumOptimizer, {"lr": 0.08, "beta": 0.85}, start, steps)
    traj_rms, loss_rms = simulate_optimizer_trajectory(RMSpropOptimizer, {"lr": 0.35, "beta": 0.9}, start, steps)
    traj_adam, loss_adam = simulate_optimizer_trajectory(AdamOptimizer, {"lr": 0.45, "beta1": 0.9, "beta2": 0.999}, start, steps)

    print(f"\n[1] Final Ravine Loss after {steps} steps:")
    print(f"    Standard SGD:   {loss_sgd[-1]:.5f} (Oscillates across steep walls)")
    print(f"    SGD + Momentum: {loss_mom[-1]:.5f} (Dampens oscillations, accelerates along valley)")
    print(f"    RMSprop:        {loss_rms[-1]:.5f} (Scales step sizes inversely with curvature)")
    print(f"    Adam:           {loss_adam[-1]:.5f} (Combines momentum + adaptive learning rates)")

    # Generate Visualization Dashboard
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel A: 2D Contour Trajectory Plot
    w1_vals = np.linspace(-4.5, 4.5, 120)
    w2_vals = np.linspace(-1.0, 7.5, 120)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = 0.5 * (10.0 * W1 ** 2 + 0.5 * W2 ** 2)

    axes[0].contour(W1, W2, Z, levels=np.logspace(-1, 2.5, 20), cmap="viridis", alpha=0.45)
    axes[0].plot(traj_sgd[:, 0], traj_sgd[:, 1], color="#8b949e", marker=".", markersize=4, linewidth=1.5, label="Standard SGD")
    axes[0].plot(traj_mom[:, 0], traj_mom[:, 1], color="#f0883e", marker=".", markersize=4, linewidth=1.8, label="Momentum")
    axes[0].plot(traj_rms[:, 0], traj_rms[:, 1], color="#a371f7", marker=".", markersize=4, linewidth=1.8, label="RMSprop")
    axes[0].plot(traj_adam[:, 0], traj_adam[:, 1], color="#58a6ff", marker="o", markersize=4, linewidth=2.4, label="Adam (Fastest)")
    axes[0].scatter([0], [0], color="#3fb950", s=100, marker="*", zorder=5, label="Global Minimum (0, 0)")
    axes[0].set_xlabel("Parameter $w_1$ (Steep Curvature)", color="gray")
    axes[0].set_ylabel("Parameter $w_2$ (Flat Ravine)", color="gray")
    axes[0].set_title("Optimization Paths in a Steep Ravine Surface", color="white", fontsize=11, fontweight="bold")
    axes[0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8.5)

    # Panel B: Loss vs Step Curves
    axes[1].plot(loss_sgd, color="#8b949e", linewidth=1.6, label="SGD")
    axes[1].plot(loss_mom, color="#f0883e", linewidth=1.8, label="Momentum")
    axes[1].plot(loss_rms, color="#a371f7", linewidth=1.8, label="RMSprop")
    axes[1].plot(loss_adam, color="#58a6ff", linewidth=2.4, label="Adam")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Optimization Step", color="gray")
    axes[1].set_ylabel("Loss (log scale)", color="gray")
    axes[1].set_title("Convergence Speed (Loss vs Steps)", color="white", fontsize=11, fontweight="bold")
    axes[1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8.5)
    axes[1].grid(True, color="#21262d", linestyle="--")

    plt.tight_layout()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_plot = os.path.join(curr_dir, "optimizers_benchmark_dashboard.png")
    plt.savefig(out_plot, dpi=140, facecolor="#0d1117")
    plt.close()
    print(f"\n[SAVED] Dashboard written to {out_plot}")


if __name__ == "__main__":
    run_optimizers_demo()

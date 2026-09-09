"""
Week 6: Calculus — Gradient Descent & Advanced Optimizers (Momentum, Adam)

Compares 3 optimization algorithms minimizing a 2D loss surface:
1. Standard / Vanilla Gradient Descent: theta = theta - lr * grad
2. Momentum: v = beta * v + lr * grad; theta = theta - v
3. Adam (Adaptive Moment Estimation): m = beta1*m + (1-beta1)*g; v = beta2*v + (1-beta2)*g^2
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def loss_function(w: np.ndarray) -> float:
    """Anisotropic quadratic valley: L(w0, w1) = 0.1 * w0^2 + 2.0 * w1^2"""
    return float(0.1 * w[0]**2 + 2.0 * w[1]**2)


def loss_gradient(w: np.ndarray) -> np.ndarray:
    """Analytical gradient: [0.2*w0, 4.0*w1]"""
    return np.array([0.2 * w[0], 4.0 * w[1]])


def optimize_vanilla_gd(w_init: np.ndarray, lr: float = 0.4, steps: int = 50) -> List[np.ndarray]:
    w = w_init.copy()
    trajectory = [w.copy()]
    for _ in range(steps):
        g = loss_gradient(w)
        w -= lr * g
        trajectory.append(w.copy())
    return trajectory


def optimize_momentum(w_init: np.ndarray, lr: float = 0.15, beta: float = 0.85, steps: int = 50) -> List[np.ndarray]:
    w = w_init.copy()
    v = np.zeros_like(w)
    trajectory = [w.copy()]
    for _ in range(steps):
        g = loss_gradient(w)
        v = beta * v + lr * g
        w -= v
        trajectory.append(w.copy())
    return trajectory


def optimize_adam(
    w_init: np.ndarray,
    lr: float = 0.3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    steps: int = 50
) -> List[np.ndarray]:
    w = w_init.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    trajectory = [w.copy()]

    for t in range(1, steps + 1):
        g = loss_gradient(w)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g ** 2)

        # Bias correction
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        trajectory.append(w.copy())

    return trajectory


def plot_optimizer_trajectories(output_dir: str = "output_plots") -> str:
    """Plots and saves contour map comparing optimizer convergence paths."""
    os.makedirs(output_dir, exist_ok=True)
    w_start = np.array([4.0, 3.0])
    steps = 40

    traj_gd = np.array(optimize_vanilla_gd(w_start, lr=0.3, steps=steps))
    traj_mom = np.array(optimize_momentum(w_start, lr=0.15, beta=0.85, steps=steps))
    traj_adam = np.array(optimize_adam(w_start, lr=0.4, steps=steps))

    # Grid for contour
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)
    Z = 0.1 * X**2 + 2.0 * Y**2

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.contour(X, Y, Z, levels=np.logspace(-1, 2, 25), cmap="viridis", alpha=0.6)

    ax.plot(traj_gd[:, 0], traj_gd[:, 1], 'ro-', label="Vanilla GD (oscillating)", markersize=4, linewidth=1.5)
    ax.plot(traj_mom[:, 0], traj_mom[:, 1], 'bs-', label="Momentum (accelerated)", markersize=4, linewidth=1.5)
    ax.plot(traj_adam[:, 0], traj_adam[:, 1], 'g^-', label="Adam (adaptive)", markersize=4, linewidth=1.5)
    ax.plot([0], [0], 'k*', markersize=14, label="Global Optimum (0,0)")

    ax.set_title("Optimization Dynamics: Vanilla GD vs. Momentum vs. Adam", fontweight='bold')
    ax.set_xlabel("w0 (Gentle slope)")
    ax.set_ylabel("w1 (Steep ravines)")
    ax.legend()
    plt.tight_layout()

    out_file = os.path.join(output_dir, "06_optimizers_trajectory.png")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


if __name__ == "__main__":
    w0 = np.array([4.0, 3.0])
    print(f"Starting Loss: {loss_function(w0):.4f}")
    gd = optimize_vanilla_gd(w0)[-1]
    mom = optimize_momentum(w0)[-1]
    adm = optimize_adam(w0)[-1]

    print(f"GD final:   {gd.round(4)}, Loss: {loss_function(gd):.6f}")
    print(f"Mom final:  {mom.round(4)}, Loss: {loss_function(mom):.6f}")
    print(f"Adam final: {adm.round(4)}, Loss: {loss_function(adm):.6f}")

    saved_plot = plot_optimizer_trajectories()
    print(f"Saved trajectory plot: {saved_plot}")

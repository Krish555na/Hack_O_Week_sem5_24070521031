"""
Week 6: Calculus — Derivatives, Gradients & Gradient Checking

Implements:
1. Numerical Differentiation (Forward, Backward, and Central Difference formulas)
2. Analytical Derivatives for standard ML Activations & Losses:
   - Sigmoid: d/dx sigma(x) = sigma(x) * (1 - sigma(x))
   - ReLU: d/dx relu(x) = 1 if x > 0 else 0
   - Tanh: d/dx tanh(x) = 1 - tanh(x)^2
   - MSE Loss: grad_w (1/N) * ||Xw - y||^2 = (2/N) * X^T (Xw - y)
3. Gradient Checking: Verifying analytical gradients against numerical approximations
"""

import math
import numpy as np
from typing import Callable, Tuple, Dict, Any


# =====================================================================
# 1. Numerical Differentiation
# =====================================================================
def numerical_derivative_1d(f: Callable[[float], float], x: float, h: float = 1e-5, method: str = "central") -> float:
    """
    Computes numerical derivative of scalar function f(x):
    - Forward:  (f(x + h) - f(x)) / h            [Error O(h)]
    - Backward: (f(x) - f(x - h)) / h            [Error O(h)]
    - Central:  (f(x + h) - f(x - h)) / (2 * h)  [Error O(h^2) - Higher accuracy]
    """
    if method == "forward":
        return (f(x + h) - f(x)) / h
    elif method == "backward":
        return (f(x) - f(x - h)) / h
    elif method == "central":
        return (f(x + h) - f(x - h)) / (2.0 * h)
    else:
        raise ValueError(f"Unknown method: {method}")


def numerical_gradient_nd(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Computes multidimensional gradient vector via central difference along each coordinate."""
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]

        # f(x + h)
        x[idx] = orig_val + h
        fx_plus = f(x)

        # f(x - h)
        x[idx] = orig_val - h
        fx_minus = f(x)

        # restore
        x[idx] = orig_val

        grad[idx] = (fx_plus - fx_minus) / (2.0 * h)
        it.iternext()

    return grad


# =====================================================================
# 2. Analytical Activations & Derivatives
# =====================================================================
def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def sigmoid_derivative(x: float) -> float:
    """Analytical derivative: sigma'(x) = sigma(x) * (1 - sigma(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x: float) -> float:
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    return 1.0 if x > 0 else 0.0


def tanh(x: float) -> float:
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    t = math.tanh(x)
    return 1.0 - t * t


# =====================================================================
# 3. Gradient Checking Routine
# =====================================================================
def check_gradient(
    loss_fn: Callable[[np.ndarray], float],
    analytical_grad_fn: Callable[[np.ndarray], np.ndarray],
    weights: np.ndarray,
    epsilon: float = 1e-5
) -> Tuple[bool, float]:
    """
    Performs rigorous gradient checking:
    relative_difference = ||grad_approx - grad_analytical|| / (||grad_approx|| + ||grad_analytical||)
    """
    grad_approx = numerical_gradient_nd(loss_fn, weights, h=epsilon)
    grad_analytical = analytical_grad_fn(weights)

    diff = np.linalg.norm(grad_approx - grad_analytical)
    norm_sum = np.linalg.norm(grad_approx) + np.linalg.norm(grad_analytical)

    rel_diff = diff / max(norm_sum, 1e-12)
    is_valid = bool(rel_diff < 1e-4)

    return is_valid, float(rel_diff)


if __name__ == "__main__":
    print("=== Numerical vs Analytical Derivatives ===")
    test_points = [-2.0, 0.0, 1.5, 3.0]

    print("\n--- Sigmoid Activation ---")
    for pt in test_points:
        num_d = numerical_derivative_1d(sigmoid, pt, method="central")
        ana_d = sigmoid_derivative(pt)
        print(f"x = {pt:4.1f} | Numerical: {num_d:.7f} | Analytical: {ana_d:.7f} | Diff: {abs(num_d - ana_d):.2e}")

    print("\n--- Gradient Checking on Quadratic Loss Surface ---")
    # Loss: L(w) = 0.5 * (w0^2 + 3*w1^2 + 2*w0*w1)
    def test_loss(w):
        return 0.5 * (w[0]**2 + 3.0 * w[1]**2 + 2.0 * w[0] * w[1])

    def test_grad(w):
        return np.array([w[0] + w[1], 3.0 * w[1] + w[0]])

    w0 = np.array([1.5, -2.0])
    passed, rel_diff = check_gradient(test_loss, test_grad, w0)
    print(f"Gradient Check at {w0}: Passed={passed} (Relative Diff: {rel_diff:.2e})")

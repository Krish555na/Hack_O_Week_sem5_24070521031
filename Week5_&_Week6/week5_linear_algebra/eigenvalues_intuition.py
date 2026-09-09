"""
Week 5: Linear Algebra — Eigenvalues, Eigenvectors & PCA Intuition

Builds intuitive and computational understanding of:
1. The Fundamental Characteristic Equation: A * v = lambda * v
2. Power Iteration Algorithm from scratch to find dominant eigenvalue & eigenvector
3. Eigendecomposition of symmetric matrices (Covariance matrices)
4. PCA Intuition: Eigenvectors as orthogonal axes of maximum variance
"""

import numpy as np
from typing import Tuple, Dict, Any


def power_iteration(A: np.ndarray, num_simulations: int = 100, tol: float = 1e-7) -> Tuple[float, np.ndarray]:
    """
    Computes dominant eigenvalue and corresponding eigenvector via Power Iteration:
    b_{k+1} = (A * b_k) / ||A * b_k||
    Rayleigh quotient: lambda = (b^T * A * b) / (b^T * b)
    """
    n = A.shape[0]
    # Initialize non-zero random vector
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)

    last_eigenvalue = 0.0

    for _ in range(num_simulations):
        # Multiply by matrix A
        b_k1 = A @ b_k

        # Re-normalize
        norm = np.linalg.norm(b_k1)
        if norm == 0:
            break
        b_k = b_k1 / norm

        # Rayleigh quotient
        eigenvalue = float((b_k.T @ A @ b_k) / (b_k.T @ b_k))
        if abs(eigenvalue - last_eigenvalue) < tol:
            break
        last_eigenvalue = eigenvalue

    return float(last_eigenvalue), b_k


def pca_intuition_demo(n_samples: int = 200) -> Dict[str, Any]:
    """
    Demonstrates how eigendecomposition connects to PCA:
    1. Generates 2D correlated Gaussian data
    2. Centers data (zero mean)
    3. Computes sample covariance matrix C = (X^T * X) / (N - 1)
    4. Eigendecomposition reveals principal axes of variance!
    """
    np.random.seed(42)

    # Correlated data: x2 = 1.8 * x1 + noise
    x1 = np.random.normal(0, 2.0, n_samples)
    x2 = 1.8 * x1 + np.random.normal(0, 1.0, n_samples)
    X = np.column_stack([x1, x2])

    # Center data
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix (2x2)
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Explained variance ratios
    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_var

    return {
        "covariance_matrix": cov_matrix,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance_ratio": explained_variance_ratio,
        "first_principal_component": eigenvectors[:, 0]
    }


if __name__ == "__main__":
    print("=== Eigenvalues & Eigenvectors Intuition ===")
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0]
    ])

    dom_val, dom_vec = power_iteration(A)
    print(f"Matrix A:\n{A}")
    print(f"Power Iteration Dominant Eigenvalue: {dom_val:.5f}")
    print(f"Dominant Eigenvector: {np.round(dom_vec, 5)}")

    # Verify A * v = lambda * v
    Av = A @ dom_vec
    lv = dom_val * dom_vec
    print(f"A @ v:      {np.round(Av, 5)}")
    print(f"lambda * v: {np.round(lv, 5)}")
    print(f"A*v equals lambda*v within tolerance? {np.allclose(Av, lv, atol=1e-4)}")

    print("\n=== PCA Intuition via Covariance Eigendecomposition ===")
    pca_res = pca_intuition_demo()
    print("Covariance Matrix:\n", pca_res["covariance_matrix"])
    print("Eigenvalues (Variance along axes):", np.round(pca_res["eigenvalues"], 4))
    print("Explained Variance Ratio:", np.round(pca_res["explained_variance_ratio"] * 100, 2), "%")
    print("PC1 Direction (Unit Vector):", np.round(pca_res["first_principal_component"], 4))

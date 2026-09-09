"""
Week 5: Linear Algebra — Geometric Transformations, Vector Projections & Gram-Schmidt

Demonstrates:
1. 2D Linear Transformations (Rotation, Scaling, Shearing, Reflection)
2. Vector Projection: proj_u(v) = ((v · u) / (u · u)) * u
3. Gram-Schmidt Orthogonalization: Converting arbitrary linearly independent vectors into an orthonormal basis
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any


def get_rotation_matrix(degrees: float) -> np.ndarray:
    """Returns 2x2 rotation matrix for angle theta in degrees."""
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, -s],
        [s,  c]
    ])


def get_shear_matrix(k_x: float = 0.0, k_y: float = 0.0) -> np.ndarray:
    """Returns 2x2 shearing matrix along X and/or Y axes."""
    return np.array([
        [1.0, k_x],
        [k_y, 1.0]
    ])


def project_vector_onto_vector(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Computes orthogonal projection of vector v onto subspace spanned by u:
    proj_u(v) = (v · u / ||u||^2) * u
    """
    u_norm_sq = np.dot(u, u)
    if u_norm_sq == 0:
        raise ValueError("Cannot project onto a zero vector.")
    scalar_proj = np.dot(v, u) / u_norm_sq
    return scalar_proj * u


def gram_schmidt_orthogonalization(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """
    Executes the Gram-Schmidt process to produce an orthonormal basis from linearly independent vectors:
    u_1 = v_1
    e_1 = u_1 / ||u_1||
    u_k = v_k - sum_{j=1}^{k-1} proj_{u_j}(v_k)
    e_k = u_k / ||u_k||
    """
    orthonormal_basis: List[np.ndarray] = []

    for v in vectors:
        u_k = v.astype(float).copy()
        for e_j in orthonormal_basis:
            # subtract projection onto previous orthonormal vectors
            u_k -= np.dot(v, e_j) * e_j

        norm = np.linalg.norm(u_k)
        if norm < 1e-10:
            raise ValueError("Input vectors are linearly dependent; cannot form basis.")

        e_k = u_k / norm
        orthonormal_basis.append(e_k)

    return orthonormal_basis


if __name__ == "__main__":
    print("=== Geometric Transformations & Projections ===")
    v = np.array([3.0, 4.0])
    u = np.array([1.0, 0.0])  # X-axis

    proj = project_vector_onto_vector(v, u)
    print(f"Vector v: {v}")
    print(f"Projection of v onto x-axis u: {proj}")

    R45 = get_rotation_matrix(45)
    rotated_v = R45 @ v
    print(f"v rotated by 45 degrees: {np.round(rotated_v, 3)}")

    # Gram-Schmidt on 3D vectors
    v1 = np.array([1.0, 1.0, 0.0])
    v2 = np.array([1.0, 0.0, 1.0])
    v3 = np.array([0.0, 1.0, 1.0])
    basis = gram_schmidt_orthogonalization([v1, v2, v3])

    print("\nOrthonormal Basis via Gram-Schmidt:")
    for i, e in enumerate(basis):
        print(f"e_{i+1}: {np.round(e, 4)}, norm: {np.linalg.norm(e):.4f}")

    print("Dot product e_1 · e_2 (should be 0):", np.round(np.dot(basis[0], basis[1]), 6))

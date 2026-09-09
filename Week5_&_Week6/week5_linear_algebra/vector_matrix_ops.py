"""
Week 5: Linear Algebra — Vectors, Matrices, Dot Products & Systems of Equations

Provides both from-scratch mathematical implementations and vectorized NumPy counterparts
to build core intuition for Machine Learning:
- Vector norms (L1, L2, Linf) and unit normalization
- Dot product, cosine similarity, angle computation & orthogonality
- Matrix-Vector and Matrix-Matrix multiplication
- Transposition, Trace, Determinant, and Matrix Inversion
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any


# =====================================================================
# 1. Vector Operations
# =====================================================================
def vector_norm(v: List[float], order: int = 2) -> float:
    """Computes vector Lp norm: ||v||_p = (sum(|v_i|^p))^(1/p)."""
    if order == 1:
        return sum(abs(x) for x in v)
    elif order == 2:
        return math.sqrt(sum(x * x for x in v))
    elif order == float('inf'):
        return max(abs(x) for x in v)
    return sum(abs(x) ** order for x in v) ** (1.0 / order)


def unit_vector(v: List[float]) -> List[float]:
    """Normalizes vector to unit length ||u|| = 1."""
    norm = vector_norm(v, order=2)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return [x / norm for x in v]


def dot_product_scratch(u: List[float], v: List[float]) -> float:
    """Computes dot product u · v = sum(u_i * v_i)."""
    if len(u) != len(v):
        raise ValueError(f"Vector dimensions must match: {len(u)} vs {len(v)}")
    return sum(a * b for a, b in zip(u, v))


def cosine_similarity(u: List[float], v: List[float]) -> float:
    """Computes cos(theta) = (u · v) / (||u|| * ||v||)."""
    norm_u = vector_norm(u, 2)
    norm_v = vector_norm(v, 2)
    if norm_u == 0 or norm_v == 0:
        raise ValueError("Vectors must be non-zero to compute cosine similarity.")
    cos_theta = dot_product_scratch(u, v) / (norm_u * norm_v)
    # Clamp for floating point edge cases [-1.0, 1.0]
    return max(-1.0, min(1.0, cos_theta))


def angle_between_vectors(u: List[float], v: List[float], degrees: bool = True) -> float:
    """Computes angle theta between two vectors in radians or degrees."""
    cos_sim = cosine_similarity(u, v)
    rad = math.acos(cos_sim)
    return math.degrees(rad) if degrees else rad


# =====================================================================
# 2. Matrix Operations
# =====================================================================
def matrix_mult_scratch(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiplies matrix A (m x k) with matrix B (k x n) yielding (m x n)."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError(f"Cannot multiply ({rows_A}x{cols_A}) by ({rows_B}x{cols_B})")

    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return C


def matrix_transpose(A: List[List[float]]) -> List[List[float]]:
    """Transposes matrix: A_T[j][i] = A[i][j]."""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def determinant_2x2(A: List[List[float]]) -> float:
    """Determinant of 2x2 matrix: ad - bc."""
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def inverse_2x2(A: List[List[float]]) -> List[List[float]]:
    """Analytical inverse of a 2x2 matrix."""
    det = determinant_2x2(A)
    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular (non-invertible).")
    inv_det = 1.0 / det
    return [
        [A[1][1] * inv_det, -A[0][1] * inv_det],
        [-A[1][0] * inv_det, A[0][0] * inv_det]
    ]


# =====================================================================
# 3. Vectorized NumPy Comparison & Verification
# =====================================================================
def verify_with_numpy(u: List[float], v: List[float], A: List[List[float]], B: List[List[float]]) -> Dict[str, Any]:
    """Cross-verifies scratch implementations against highly optimized NumPy LAPACK."""
    u_np, v_np = np.array(u), np.array(v)
    A_np, B_np = np.array(A), np.array(B)

    # Dot product
    dot_scratch = dot_product_scratch(u, v)
    dot_np = float(np.dot(u_np, v_np))

    # Matrix Mult
    C_scratch = matrix_mult_scratch(A, B)
    C_np = (A_np @ B_np).tolist()

    # Invert 2x2
    inv_scratch = inverse_2x2(A)
    inv_np = np.linalg.inv(A_np).tolist()

    return {
        "dot_match": math.isclose(dot_scratch, dot_np, rel_tol=1e-7),
        "matmul_match": np.allclose(C_scratch, C_np),
        "inv_match": np.allclose(inv_scratch, inv_np),
        "scratch_matmul": C_scratch,
        "numpy_matmul": C_np
    }


if __name__ == "__main__":
    print("=== Linear Algebra: Vectors & Matrices ===")
    u = [1.0, 2.0, 3.0]
    v = [4.0, 5.0, 6.0]
    print(f"Vector u: {u}, Norm: {vector_norm(u):.4f}")
    print(f"Dot product u · v: {dot_product_scratch(u, v)}")
    print(f"Angle between u & v: {angle_between_vectors(u, v):.2f}°")

    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    print("\nMatrix A:\n", A)
    print("Matrix B:\n", B)
    print("Matrix Multiplication A @ B:\n", matrix_mult_scratch(A, B))
    print("Inverse A^-1:\n", inverse_2x2(A))

    res = verify_with_numpy(u, v, A, B)
    print(f"\nNumPy verification check: All operations match? {res['matmul_match'] and res['inv_match']}")

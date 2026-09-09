"""
Week 4: NumPy Mastery — Arrays, Broadcasting & Vectorized Operations

This module provides clear explanations, implementations, and micro-benchmarks for:
1. Multidimensional array manipulation, slicing, and strided views
2. Broadcasting mechanics across dissimilar dimensions
3. Vectorized mathematical computations vs pure Python loop benchmarks
4. Boolean masking, conditional filtering, and aggregation
"""

import time
import numpy as np
from typing import Dict, Tuple, Any


def demonstrate_array_creation_and_slicing() -> Dict[str, Any]:
    """Demonstrates multidimensional array creation, reshaping, and fancy slicing."""
    # 1. 2D grid creation
    grid_2d = np.arange(1, 26).reshape(5, 5)

    # 2. Sub-matrix slicing (rows 1-3, cols 2-4)
    sub_matrix = grid_2d[1:4, 2:5]

    # 3. Strided step slicing
    strided_view = grid_2d[::2, ::2]

    # 4. Fancy integer indexing (extract specific coordinates)
    row_indices = np.array([0, 2, 4])
    col_indices = np.array([1, 3, 0])
    selected_elements = grid_2d[row_indices, col_indices]

    # 5. Boolean masking
    mask = (grid_2d % 2 == 0) & (grid_2d > 10)
    even_gt_10 = grid_2d[mask]

    return {
        "grid_2d": grid_2d,
        "sub_matrix": sub_matrix,
        "strided_view": strided_view,
        "selected_elements": selected_elements,
        "even_gt_10": even_gt_10
    }


def demonstrate_broadcasting() -> Dict[str, Any]:
    """
    Demonstrates NumPy broadcasting rules.
    Rule 1: If arrays have different ndim, prepend 1s to the smaller shape.
    Rule 2: Arrays are compatible along a dimension if sizes match or one size is 1.
    """
    # (3, 1) column vector
    col_vector = np.array([[10], [20], [30]])  # shape (3, 1)

    # (1, 4) row vector
    row_vector = np.array([[1, 2, 3, 4]])      # shape (1, 4)

    # Broadcasting yields (3, 4) outer addition grid
    outer_sum = col_vector + row_vector        # shape (3, 4)

    # Feature Normalization via broadcasting (Z-Score Standardization)
    data_matrix = np.array([
        [150.0, 50.0, 22.0],
        [175.0, 70.0, 35.0],
        [160.0, 62.0, 28.0],
        [190.0, 95.0, 45.0],
        [168.0, 68.0, 31.0]
    ])
    mean = np.mean(data_matrix, axis=0)          # shape (3,) -> broadcasted to (1, 3)
    std = np.std(data_matrix, axis=0)            # shape (3,) -> broadcasted to (1, 3)
    standardized = (data_matrix - mean) / std    # broadcasted element-wise

    return {
        "col_vector_shape": col_vector.shape,
        "row_vector_shape": row_vector.shape,
        "outer_sum": outer_sum,
        "outer_sum_shape": outer_sum.shape,
        "standardized_mean": np.round(np.mean(standardized, axis=0), 4),
        "standardized_std": np.round(np.std(standardized, axis=0), 4)
    }


def benchmark_vectorization(n_elements: int = 1_000_000) -> Dict[str, float]:
    """
    Benchmarks pure Python loops against NumPy vectorized SIMD operations.
    Computes Euclidean distance between two N-dimensional vectors.
    """
    # Generate random arrays
    arr_a = np.random.uniform(0.0, 10.0, size=n_elements)
    arr_b = np.random.uniform(0.0, 10.0, size=n_elements)

    list_a = arr_a.tolist()
    list_b = arr_b.tolist()

    # 1. Pure Python Loop
    start_py = time.perf_counter()
    diff_squared_sum = 0.0
    for i in range(n_elements):
        diff = list_a[i] - list_b[i]
        diff_squared_sum += diff * diff
    dist_py = diff_squared_sum ** 0.5
    py_time = time.perf_counter() - start_py

    # 2. NumPy Vectorized
    start_np = time.perf_counter()
    dist_np = np.sqrt(np.sum((arr_a - arr_b) ** 2))
    np_time = time.perf_counter() - start_np

    speedup = py_time / np_time if np_time > 0 else float('inf')

    return {
        "n_elements": n_elements,
        "python_loop_time_sec": py_time,
        "numpy_vectorized_time_sec": np_time,
        "speedup_factor": speedup,
        "results_match": bool(np.isclose(dist_py, dist_np))
    }


if __name__ == "__main__":
    print("=== 1. NumPy Array Creation & Slicing ===")
    slicing_res = demonstrate_array_creation_and_slicing()
    print("Full 5x5 Grid:\n", slicing_res["grid_2d"])
    print("Sub-matrix (rows 1:4, cols 2:5):\n", slicing_res["sub_matrix"])
    print("Masked (even and > 10):", slicing_res["even_gt_10"])

    print("\n=== 2. Broadcasting Mechanics ===")
    bc_res = demonstrate_broadcasting()
    print("Col (3,1) + Row (1,4) Outer Grid:\n", bc_res["outer_sum"])
    print("Standardized column means (approx 0):", bc_res["standardized_mean"])
    print("Standardized column stds (approx 1):", bc_res["standardized_std"])

    print("\n=== 3. Vectorization Benchmark (1,000,000 elements) ===")
    bench = benchmark_vectorization(1_000_000)
    print(f"Pure Python Loop:      {bench['python_loop_time_sec']:.4f} seconds")
    print(f"NumPy Vectorized:      {bench['numpy_vectorized_time_sec']:.4f} seconds")
    print(f"🚀 Speedup Factor:     {bench['speedup_factor']:.1f}x faster!")
    print(f"Results match within precision: {bench['results_match']}")

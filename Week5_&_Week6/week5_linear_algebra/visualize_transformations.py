"""
Week 5: Linear Algebra Visualizer — Transformations & Eigenvector Preservation

Plots 2D linear transformations and illustrates the core geometric intuition:
- Grid shearing / stretching
- Eigenvectors remain on their own span (only scaled by lambda)!
Saves figures to 'output_plots/'.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def visualize_eigen_preservation(output_dir: str = "output_plots"):
    """Visualizes how a matrix stretches space and preserves eigenvector lines."""
    os.makedirs(output_dir, exist_ok=True)

    # 2x2 Transformation Matrix
    A = np.array([
        [3.0, 1.0],
        [1.0, 2.0]
    ])

    # Compute eigenvalues & eigenvectors
    vals, vecs = np.linalg.eig(A)

    # Test arbitrary vectors
    test_vectors = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        vecs[:, 0],  # Eigenvector 1
        vecs[:, 1]   # Eigenvector 2
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax in axes:
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)

    axes[0].set_title("Original Space (v)", fontweight='bold')
    axes[1].set_title("Transformed Space (A · v)", fontweight='bold')

    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    labels = ['Unit X', 'Unit Y', 'Arbitrary [1,1]', f'Eigenvector 1 (λ={vals[0]:.2f})', f'Eigenvector 2 (λ={vals[1]:.2f})']

    for v, c, l in zip(test_vectors, colors, labels):
        # Original
        axes[0].quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=c, label=l, width=0.015)
        # Transformed
        Av = A @ v
        axes[1].quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1, color=c, label=l, width=0.015)

    axes[0].legend(loc='upper left', fontsize=8)
    axes[1].legend(loc='upper left', fontsize=8)
    plt.tight_layout()

    out_file = os.path.join(output_dir, "05_eigen_transformation_visualizer.png")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


if __name__ == "__main__":
    p = visualize_eigen_preservation()
    print(f"Generated linear algebra visualizer: {p}")

"""
Week 10: Clustering Benchmark — K-Means vs. Agglomerative vs. DBSCAN

Compares the 3 foundational clustering algorithms across 3 challenging data geometries:
1. Spherical Blobs: Standard isotropic clusters
2. Interlocking Moons: Non-linear continuous manifolds
3. Concentric Rings: Nested non-convex shapes
Generates side-by-side comparison figure saved to 'output_plots/'.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from kmeans_clustering import KMeansScratch
from hierarchical_clustering import AgglomerativeClusteringScratch
from dbscan_clustering import DBSCANScratch


def generate_synthetic_geometries(n_samples: int = 150):
    """Generates 3 synthetic clustering datasets."""
    np.random.seed(42)

    # 1. Blobs
    b1 = np.random.normal(loc=[-2, -2], scale=0.45, size=(n_samples // 2, 2))
    b2 = np.random.normal(loc=[2, 2], scale=0.45, size=(n_samples // 2, 2))
    blobs = np.vstack([b1, b2])

    # 2. Moons
    n_half = n_samples // 2
    theta = np.linspace(0, np.pi, n_half)
    m1 = np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.normal(0, 0.08, (n_half, 2))
    m2 = np.column_stack([1 - np.cos(theta), 1 - np.sin(theta) - 0.5]) + np.random.normal(0, 0.08, (n_half, 2))
    moons = np.vstack([m1, m2])

    # 3. Concentric Rings
    r_inner = 0.8 + np.random.normal(0, 0.08, n_half)
    t_inner = np.random.uniform(0, 2 * np.pi, n_half)
    ring_inner = np.column_stack([r_inner * np.cos(t_inner), r_inner * np.sin(t_inner)])

    r_outer = 2.2 + np.random.normal(0, 0.08, n_half)
    t_outer = np.random.uniform(0, 2 * np.pi, n_half)
    ring_outer = np.column_stack([r_outer * np.cos(t_outer), r_outer * np.sin(t_outer)])
    rings = np.vstack([ring_inner, ring_outer])

    return [("Blobs", blobs), ("Moons", moons), ("Concentric Rings", rings)]


def run_clustering_benchmark(output_dir: str = "output_plots") -> str:
    os.makedirs(output_dir, exist_ok=True)
    datasets = generate_synthetic_geometries(160)

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    algo_names = ["K-Means (k=2)", "Agglomerative (k=2)", "DBSCAN (density)"]

    for row_idx, (dataset_name, X) in enumerate(datasets):
        # 1. K-Means
        km = KMeansScratch(n_clusters=2).fit(X)
        axes[row_idx, 0].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="viridis", s=30, edgecolors="k", linewidth=0.5)
        axes[row_idx, 0].scatter(km.centroids[:, 0], km.centroids[:, 1], c="red", marker="X", s=90, label="Centroids")

        # 2. Agglomerative (Single linkage handles non-convex manifolds well)
        linkage_type = "single" if dataset_name in ("Moons", "Concentric Rings") else "average"
        agg = AgglomerativeClusteringScratch(n_clusters=2, linkage=linkage_type).fit(X)
        axes[row_idx, 1].scatter(X[:, 0], X[:, 1], c=agg.labels_, cmap="viridis", s=30, edgecolors="k", linewidth=0.5)

        # 3. DBSCAN
        eps_val = 0.35 if dataset_name == "Concentric Rings" else 0.4
        db = DBSCANScratch(eps=eps_val, min_samples=4).fit(X)
        # Handle noise points (-1) colored with distinct alpha
        axes[row_idx, 2].scatter(X[:, 0], X[:, 1], c=db.labels_, cmap="tab10", s=30, edgecolors="k", linewidth=0.5)

        for col_idx in range(3):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(algo_names[col_idx], fontweight="bold", fontsize=12)

        axes[row_idx, 0].set_ylabel(dataset_name, fontweight="bold", fontsize=11)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "10_clustering_benchmark.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


if __name__ == "__main__":
    p = run_clustering_benchmark()
    print(f"Generated clustering benchmark grid: {p}")

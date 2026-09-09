"""
Week 10: Clustering — Agglomerative Hierarchical Clustering from Scratch & Dendrograms

Implements:
1. Bottom-up Agglomerative Clustering:
   - Starts with each point as its own singleton cluster
   - Iteratively merges the two closest clusters according to linkage criteria
2. Linkage Criteria:
   - Single Linkage: min_{x in A, y in B} dist(x, y)
   - Complete Linkage: max_{x in A, y in B} dist(x, y)
   - Average Linkage: mean_{x in A, y in B} dist(x, y)
3. SciPy linkage matrix and dendrogram plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import List, Set, Optional, Dict, Tuple, Any


class AgglomerativeClusteringScratch:
    """Agglomerative Hierarchical Clustering with flexible linkage rules."""

    def __init__(self, n_clusters: int = 2, linkage: str = "average"):
        self.n_clusters = int(n_clusters)
        self.linkage = linkage.lower()
        self.labels_: Optional[np.ndarray] = None
        self.history_: List[Dict[str, Any]] = []

    def _cluster_distance(self, c1: List[int], c2: List[int], dist_matrix: np.ndarray) -> float:
        """Computes distance between two clusters based on linkage method."""
        sub_dists = dist_matrix[np.ix_(c1, c2)]

        if self.linkage == "single":
            return float(np.min(sub_dists))
        elif self.linkage == "complete":
            return float(np.max(sub_dists))
        elif self.linkage == "average":
            return float(np.mean(sub_dists))
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")

    def fit(self, X: np.ndarray) -> 'AgglomerativeClusteringScratch':
        X = np.array(X, dtype=float)
        n_samples = len(X)

        # Precompute initial pairwise distance matrix
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)

        # Track clusters: list of member indices
        clusters: List[List[int]] = [[i] for i in range(n_samples)]
        current_dists = dist_matrix.copy()

        while len(clusters) > self.n_clusters:
            # Find closest pair (i, j) with min distance
            min_idx = np.argmin(current_dists)
            i, j = np.unravel_index(min_idx, current_dists.shape)
            if i > j:
                i, j = j, i

            min_dist = float(current_dists[i, j])
            self.history_.append({
                "merged_size": len(clusters[i]) + len(clusters[j]),
                "distance": min_dist
            })

            # Update distance row i using Lance-Williams formulas
            size_i, size_j = len(clusters[i]), len(clusters[j])
            for k in range(len(clusters)):
                if k == i or k == j:
                    continue
                d_ik = current_dists[i, k]
                d_jk = current_dists[j, k]

                if self.linkage == "single":
                    new_d = min(d_ik, d_jk)
                elif self.linkage == "complete":
                    new_d = max(d_ik, d_jk)
                elif self.linkage == "average":
                    new_d = (size_i * d_ik + size_j * d_jk) / (size_i + size_j)
                else:
                    new_d = min(d_ik, d_jk)

                current_dists[i, k] = new_d
                current_dists[k, i] = new_d

            current_dists[i, i] = np.inf

            # Merge cluster j into cluster i
            clusters[i].extend(clusters[j])
            clusters.pop(j)

            # Remove row and col j from distance matrix
            current_dists = np.delete(current_dists, j, axis=0)
            current_dists = np.delete(current_dists, j, axis=1)

        # Assign final cluster labels
        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, member_indices in enumerate(clusters):
            labels[member_indices] = cluster_id

        self.labels_ = labels
        return self


def plot_dendrogram_scipy(X: np.ndarray, output_path: str = "dendrogram.png"):
    """Generates a hierarchical tree dendrogram using SciPy."""
    Z = linkage(X, method="average")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=12, leaf_rotation=45, show_contracted=True)
    ax.set_title("Hierarchical Clustering Dendrogram (Average Linkage)", fontweight="bold")
    ax.set_xlabel("Cluster Sample Index / Count")
    ax.set_ylabel("Distance (Dissimilarity)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    np.random.seed(42)
    pts = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
        [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
    ])

    agg = AgglomerativeClusteringScratch(n_clusters=2, linkage="average").fit(pts)
    print("=== Agglomerative Hierarchical Clustering Scratch ===")
    print("Points:\n", pts)
    print("Assigned Clusters:", agg.labels_)
    print("Merge Steps History:", len(agg.history_))

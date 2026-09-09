"""
Week 10: Clustering — K-Means Clustering from Scratch (K-Means++, Lloyd's Algorithm & Silhouette)

Implements:
1. K-Means++ smart centroid initialization (probabilistic sampling ~ D(x)^2)
2. Lloyd's iterative expectation-maximization algorithm:
   - Assignment step: assign each point to closest centroid
   - Update step: recompute centroid as mean of cluster members
3. Inertia / WCSS (Within-Cluster Sum of Squares) & Elbow method helper
4. Silhouette Coefficient calculator from scratch
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class KMeansScratch:
    """K-Means clustering with K-Means++ seeding and convergence tolerance."""

    def __init__(self, n_clusters: int = 3, init: str = "k-means++", max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        self.k = int(n_clusters)
        self.init = init.lower()
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        n_samples, _ = X.shape

        if self.init == "random":
            indices = rng.choice(n_samples, self.k, replace=False)
            return X[indices].copy()

        elif self.init == "k-means++":
            # 1. Choose first center uniformly at random
            centroids = [X[rng.choice(n_samples)]]

            # 2. Choose remaining k - 1 centers
            for _ in range(1, self.k):
                # Compute distance of each point to nearest chosen center
                dists_sq = np.array([
                    min(np.sum((x - c) ** 2) for c in centroids)
                    for x in X
                ])
                # Probabilities proportional to D(x)^2
                probs = dists_sq / np.sum(dists_sq)
                next_idx = rng.choice(n_samples, p=probs)
                centroids.append(X[next_idx])

            return np.array(centroids)
        else:
            raise ValueError(f"Unknown init scheme: {self.init}")

    def fit(self, X: np.ndarray) -> 'KMeansScratch':
        X = np.array(X, dtype=float)
        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            # Assignment step
            # dists shape: (n_samples, k)
            dists = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            # Update step
            new_centroids = np.zeros_like(self.centroids)
            for j in range(self.k):
                cluster_members = X[new_labels == j]
                if len(cluster_members) > 0:
                    new_centroids[j] = np.mean(cluster_members, axis=0)
                else:
                    # Re-seed empty cluster to an arbitrary data point
                    new_centroids[j] = X[np.random.choice(len(X))]

            # Check convergence
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.labels_ = new_labels

            if shift < self.tol:
                break

        # Calculate final inertia
        final_dists = np.linalg.norm(X - self.centroids[self.labels_], axis=1)
        self.inertia_ = float(np.sum(final_dists ** 2))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        return np.argmin(dists, axis=1)


def silhouette_score_scratch(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes mean Silhouette Coefficient over all samples:
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where a(i) = mean intra-cluster distance, b(i) = mean nearest-cluster distance.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    n_samples = len(X)
    s_scores = []

    for i in range(n_samples):
        curr_label = labels[i]
        curr_pt = X[i]

        # a(i): Mean distance to other points in the same cluster
        same_cluster_pts = X[labels == curr_label]
        if len(same_cluster_pts) > 1:
            a_i = np.mean([np.linalg.norm(curr_pt - p) for p in same_cluster_pts if not np.array_equal(curr_pt, p)])
        else:
            a_i = 0.0

        # b(i): Min mean distance to points in any other cluster
        other_clusters_means = []
        for other_label in unique_labels:
            if other_label == curr_label:
                continue
            other_pts = X[labels == other_label]
            if len(other_pts) > 0:
                mean_dist = np.mean(np.linalg.norm(other_pts - curr_pt, axis=1))
                other_clusters_means.append(mean_dist)

        b_i = min(other_clusters_means) if other_clusters_means else 0.0

        denom = max(a_i, b_i)
        s_i = (b_i - a_i) / denom if denom > 0 else 0.0
        s_scores.append(s_i)

    return float(np.mean(s_scores))


if __name__ == "__main__":
    np.random.seed(42)
    # Generate 3 distinct blobs
    c1 = np.random.normal(loc=[0, 0], scale=0.6, size=(40, 2))
    c2 = np.random.normal(loc=[4, 4], scale=0.6, size=(40, 2))
    c3 = np.random.normal(loc=[0, 5], scale=0.6, size=(40, 2))
    X_toy = np.vstack([c1, c2, c3])

    kmeans = KMeansScratch(n_clusters=3).fit(X_toy)
    sil = silhouette_score_scratch(X_toy, kmeans.labels_)
    print("=== K-Means Clustering Scratch ===")
    print(f"Final Inertia (WCSS):    {kmeans.inertia_:.2f}")
    print(f"Silhouette Score:        {sil:.4f}")
    print("Cluster Centroids:\n", kmeans.centroids.round(2))

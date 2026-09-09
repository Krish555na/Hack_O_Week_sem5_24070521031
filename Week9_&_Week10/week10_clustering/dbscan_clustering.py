"""
Week 10: Clustering — DBSCAN (Density-Based Spatial Clustering) from Scratch

Implements:
1. Epsilon-neighborhood identification: N_eps(p) = {q in D | dist(p, q) <= eps}
2. Point classification:
   - Core points: |N_eps(p)| >= min_samples
   - Border points: reachable from core point but |N_eps| < min_samples
   - Noise points: not reachable from any core point (labeled -1)
3. Cluster expansion via density-connectivity
"""

import numpy as np
from collections import deque
from typing import Optional, List, Set


class DBSCANScratch:
    """Density-Based Spatial Clustering of Applications with Noise."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: List[int] = []

    def _region_query(self, X: np.ndarray, point_idx: int) -> List[int]:
        """Returns indices of all points within eps distance of point_idx."""
        diff = X - X[point_idx]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        return list(np.where(dists <= self.eps)[0])

    def fit(self, X: np.ndarray) -> 'DBSCANScratch':
        X = np.array(X, dtype=float)
        n_samples = len(X)

        # -1 represents noise, None/unvisited represented as -2
        labels = np.full(n_samples, -2, dtype=int)
        self.core_sample_indices_ = []

        current_cluster_id = 0

        for p_idx in range(n_samples):
            # Skip if already visited
            if labels[p_idx] != -2:
                continue

            neighbors = self._region_query(X, p_idx)

            if len(neighbors) < self.min_samples:
                # Tentatively mark as noise
                labels[p_idx] = -1
            else:
                # Core point discovered: expand cluster
                self.core_sample_indices_.append(p_idx)
                labels[p_idx] = current_cluster_id

                # BFS queue for expanding cluster
                queue = deque([idx for idx in neighbors if idx != p_idx])

                while queue:
                    neighbor_idx = queue.popleft()

                    if labels[neighbor_idx] == -1:
                        # Border point previously classified as noise -> change to current cluster
                        labels[neighbor_idx] = current_cluster_id

                    if labels[neighbor_idx] != -2:
                        continue

                    labels[neighbor_idx] = current_cluster_id
                    secondary_neighbors = self._region_query(X, neighbor_idx)

                    if len(secondary_neighbors) >= self.min_samples:
                        self.core_sample_indices_.append(neighbor_idx)
                        for sec_idx in secondary_neighbors:
                            if labels[sec_idx] in (-2, -1):
                                queue.append(sec_idx)

                current_cluster_id += 1

        self.labels_ = labels
        return self


if __name__ == "__main__":
    np.random.seed(42)
    # Generate dense circle + scattered noise
    theta = np.linspace(0, 2 * np.pi, 80)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    noise = np.random.uniform(-2, 2, size=(15, 2))
    X_toy = np.vstack([circle, noise])

    db = DBSCANScratch(eps=0.3, min_samples=3).fit(X_toy)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = int(np.sum(db.labels_ == -1))

    print("=== DBSCAN Clustering Scratch ===")
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points detected: {n_noise} of {len(X_toy)}")
    print(f"Core points count: {len(db.core_sample_indices_)}")

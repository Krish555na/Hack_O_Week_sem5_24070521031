"""
Week 8: Classification — K-Nearest Neighbors (KNN) from Scratch

Implements:
1. Distance metrics: Euclidean (L2), Manhattan (L1), and arbitrary Minkowski (Lp)
2. Uniform Majority Voting vs. Distance-Weighted Voting (1 / (d + eps))
3. Probability estimates based on neighborhood frequency
"""

import numpy as np
from collections import Counter
from typing import Optional, List, Dict, Any


class KNNClassifierScratch:
    """K-Nearest Neighbors non-parametric classification model."""

    def __init__(self, n_neighbors: int = 5, distance_metric: str = "euclidean", p: float = 2.0, weights: str = "uniform"):
        self.k = int(n_neighbors)
        self.metric = distance_metric.lower()
        self.p = float(p)
        self.weights = weights.lower()
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifierScratch':
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        self.classes = np.unique(self.y_train)
        return self

    def _compute_distances(self, x_query: np.ndarray) -> np.ndarray:
        """Computes distances from query point to all training points."""
        diff = self.X_train - x_query

        if self.metric == "euclidean":
            return np.sqrt(np.sum(diff ** 2, axis=1))
        elif self.metric == "manhattan":
            return np.sum(np.abs(diff), axis=1)
        elif self.metric == "minkowski":
            return np.sum(np.abs(diff) ** self.p, axis=1) ** (1.0 / self.p)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _predict_single(self, x_query: np.ndarray) -> Any:
        distances = self._compute_distances(x_query)
        # Find indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        k_distances = distances[k_indices]

        if self.weights == "uniform":
            # Majority voting
            counts = Counter(k_nearest_labels)
            return counts.most_common(1)[0][0]

        elif self.weights == "distance":
            # Weighted by inverse distance
            weights = 1.0 / (k_distances + 1e-8)
            label_scores: Dict[Any, float] = {}
            for label, w in zip(k_nearest_labels, weights):
                label_scores[label] = label_scores.get(label, 0.0) + w
            return max(label_scores.items(), key=lambda item: item[1])[0]
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weights}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("KNN model is not fitted.")
        X = np.array(X, dtype=float)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probability estimates based on k nearest neighbors."""
        X = np.array(X, dtype=float)
        probas = []

        for x in X:
            distances = self._compute_distances(x)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]

            class_probs = []
            for cls in self.classes:
                prob = np.mean(k_nearest_labels == cls)
                class_probs.append(prob)
            probas.append(class_probs)

        return np.array(probas)


if __name__ == "__main__":
    X_sample = np.array([
        [1.0, 1.0], [1.2, 0.9], [0.8, 1.1],  # Class 0
        [5.0, 5.0], [5.2, 4.8], [4.8, 5.1]   # Class 1
    ])
    y_sample = np.array([0, 0, 0, 1, 1, 1])

    knn = KNNClassifierScratch(n_neighbors=3, distance_metric="euclidean").fit(X_sample, y_sample)
    test_pts = np.array([[1.1, 1.0], [4.9, 5.0], [3.0, 3.0]])
    print("=== KNN Classifier Scratch ===")
    print("Test Points:\n", test_pts)
    print("Predicted Classes:", knn.predict(test_pts))
    print("Class Probabilities:\n", knn.predict_proba(test_pts).round(2))

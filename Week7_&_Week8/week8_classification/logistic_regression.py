"""
Week 8: Classification — Logistic Regression & One-vs-Rest Multiclass from Scratch

Implements:
1. Binary Logistic Regression:
   - Sigmoid hypothesis: P(y=1|x) = 1 / (1 + exp(-X * theta))
   - Log Loss / Binary Cross-Entropy (BCE)
   - Gradient Descent optimization
2. Multi-class Extension via One-vs-Rest (OvR)
"""

import numpy as np
from typing import List, Optional, Tuple, Any


class LogisticRegressionScratch:
    """Binary Logistic Regression with vectorized Gradient Descent."""

    def __init__(self, lr: float = 0.1, max_iter: int = 1000, tol: float = 1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.theta: Optional[np.ndarray] = None
        self.cost_history: List[float] = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Numerically stable clipping to prevent overflow in exp
        z = np.clip(z, -250.0, 250.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones((X.shape[0], 1)), X])

    def compute_cost(self, X_b: np.ndarray, y: np.ndarray) -> float:
        m = len(y)
        h = self._sigmoid(X_b @ self.theta)
        eps = 1e-15
        h = np.clip(h, eps, 1.0 - eps)
        # Log loss
        loss = -(1.0 / m) * np.sum(y * np.log(h) + (1.0 - y) * np.log(1.0 - h))
        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionScratch':
        X_b = self._add_bias(X)
        m, n = X_b.shape
        self.theta = np.zeros(n)
        prev_cost = float('inf')

        for _ in range(self.max_iter):
            h = self._sigmoid(X_b @ self.theta)
            gradient = (1.0 / m) * (X_b.T @ (h - y))
            self.theta -= self.lr * gradient

            cost = self.compute_cost(X_b, y)
            self.cost_history.append(cost)

            if abs(prev_cost - cost) < self.tol:
                break
            prev_cost = cost

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Model not fitted yet.")
        X_b = self._add_bias(X)
        return self._sigmoid(X_b @ self.theta)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class MulticlassLogisticRegressionOvR:
    """Multi-class classifier wrapping One-vs-Rest binary models."""

    def __init__(self, lr: float = 0.1, max_iter: int = 1000):
        self.lr = lr
        self.max_iter = max_iter
        self.models: List[Tuple[Any, LogisticRegressionScratch]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MulticlassLogisticRegressionOvR':
        self.classes = np.unique(y)
        self.models = []

        for cls in self.classes:
            # Binary target: 1 for current class, 0 for all others
            y_binary = (y == cls).astype(int)
            clf = LogisticRegressionScratch(lr=self.lr, max_iter=self.max_iter).fit(X, y_binary)
            self.models.append((cls, clf))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = np.column_stack([model.predict_proba(X) for _, model in self.models])
        # Normalize across classes
        row_sums = np.sum(probas, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probas / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = np.column_stack([model.predict_proba(X) for _, model in self.models])
        best_indices = np.argmax(probas, axis=1)
        return self.classes[best_indices]


if __name__ == "__main__":
    np.random.seed(42)
    # Generate 2 separable Gaussian clusters
    c0 = np.random.normal(loc=[-1.5, -1.5], scale=0.8, size=(50, 2))
    c1 = np.random.normal(loc=[1.5, 1.5], scale=0.8, size=(50, 2))
    X_bin = np.vstack([c0, c1])
    y_bin = np.array([0] * 50 + [1] * 50)

    clf = LogisticRegressionScratch(lr=0.5, max_iter=1000).fit(X_bin, y_bin)
    preds = clf.predict(X_bin)
    acc = np.mean(preds == y_bin)
    print("=== Binary Logistic Regression ===")
    print(f"Learned Weights [bias, w1, w2]: {clf.theta.round(4)}")
    print(f"Training Accuracy: {acc * 100:.1f}%")

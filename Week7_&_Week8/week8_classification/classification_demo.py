"""
Week 8: Classification Demo & Decision Boundary Visualization

Trains and compares:
1. Logistic Regression (Linear Decision Boundary)
2. KNN with k=1 (High variance, complex Voronoi-like boundary)
3. KNN with k=15 (Smooth non-linear boundary)
Generates decision boundary contours saved to 'output_plots/'.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegressionScratch
from knn_classifier import KNNClassifierScratch
from metrics import compute_classification_metrics, roc_curve_and_auc_scratch


def generate_synthetic_dataset(n_samples: int = 250):
    """Generates two interlocking half-moon non-linear clusters."""
    np.random.seed(42)
    n_per_cluster = n_samples // 2

    # Cluster 0: Upper half circle
    theta0 = np.linspace(0, np.pi, n_per_cluster)
    r0 = 1.0 + np.random.normal(0, 0.12, n_per_cluster)
    x0 = r0 * np.cos(theta0)
    y0 = r0 * np.sin(theta0)

    # Cluster 1: Lower shifted half circle
    theta1 = np.linspace(0, np.pi, n_per_cluster)
    r1 = 1.0 + np.random.normal(0, 0.12, n_per_cluster)
    x1 = 1.0 - r1 * np.cos(theta1)
    y1 = 0.5 - r1 * np.sin(theta1)

    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    y = np.array([0] * n_per_cluster + [1] * n_per_cluster)

    # Shuffle
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


def run_classification_benchmark(output_dir: str = "output_plots"):
    os.makedirs(output_dir, exist_ok=True)
    X, y = generate_synthetic_dataset(260)

    # Train / Test split (75/25)
    split = int(0.75 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Models
    models = {
        "Logistic Regression": LogisticRegressionScratch(lr=0.8, max_iter=1500).fit(X_train, y_train),
        "KNN (k=1)": KNNClassifierScratch(n_neighbors=1).fit(X_train, y_train),
        "KNN (k=15)": KNNClassifierScratch(n_neighbors=15).fit(X_train, y_train)
    }

    # Evaluate
    reports = {}
    for name, clf in models.items():
        preds = clf.predict(X_test)
        metrics = compute_classification_metrics(y_test, preds)
        reports[name] = metrics

    # Plot Decision Boundaries
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    for idx, (name, clf) in enumerate(models.items()):
        ax = axes[idx]
        Z = clf.predict(grid_points).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k", s=35)

        acc = reports[name]["Accuracy"]
        f1 = reports[name]["F1_Score"]
        ax.set_title(f"{name}\nAcc: {acc*100:.1f}% | F1: {f1:.3f}", fontweight="bold")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "08_classification_boundaries.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return reports, chart_path


if __name__ == "__main__":
    reps, path = run_classification_benchmark()
    print("=== Classification Model Comparison ===")
    for name, m in reps.items():
        print(f"{name:22s} -> Acc: {m['Accuracy']*100:.2f}% | F1: {m['F1_Score']:.4f} | Prec: {m['Precision']:.4f} | Rec: {m['Recall']:.4f}")
    print(f"Saved decision boundaries plot: {path}")

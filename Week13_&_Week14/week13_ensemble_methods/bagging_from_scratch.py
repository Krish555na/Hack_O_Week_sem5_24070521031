"""
Week 13 - Bagging (Bootstrap Aggregating) & Random Forest from Scratch
=====================================================================
Topics Covered:
  1. Why single Decision Trees have HIGH VARIANCE
  2. Bootstrap Sampling with replacement (~63.2% unique samples)
  3. Out-Of-Bag (OOB) error estimation - free validation without a holdout set
  4. Bagging Classifier & Regressor implementation
  5. Random Forest feature subsampling (decorrelating ensemble trees)
  6. Variance reduction proof & empirical verification

Dependencies: numpy, matplotlib (zero external ML dependencies)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# =====================================================================
# 1. CORE DECISION TREE (Scratch Implementation for Ensemble Base)
# =====================================================================

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeScratch:
    """
    Lightweight Decision Tree capable of Classification (Gini) and Regression (MSE).
    Supports max_features for Random Forest feature subspacing.
    """
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, task="classification"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # None (all), 'sqrt', or integer
        self.task = task
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]
        if self.max_features is None:
            self.n_sub_features = n_features
        elif self.max_features == "sqrt":
            self.n_sub_features = max(1, int(np.sqrt(n_features)))
        else:
            self.n_sub_features = min(n_features, int(self.max_features))

        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) <= 1):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Feature subsampling (Random Forest decorrelation mechanism)
        feat_indices = np.random.choice(n_features, self.n_sub_features, replace=False)

        best_feat, best_thresh = self._best_split(X, y, feat_indices)
        if best_feat is None:
            return Node(value=self._calculate_leaf_value(y))

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _calculate_leaf_value(self, y):
        if len(y) == 0:
            return 0.0
        if self.task == "classification":
            # Majority vote
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]
        else:
            # Mean for regression
            return np.mean(y)

    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        if self.task == "classification":
            # Gini Impurity = 1 - sum(p_i^2)
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1.0 - np.sum(probs ** 2)
        else:
            # Variance (MSE)
            return np.var(y)

    def _best_split(self, X, y, feat_indices):
        best_gain = -1.0
        split_feat, split_thresh = None, None
        parent_impurity = self._impurity(y)

        for feat in feat_indices:
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                n = len(y)
                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                imp_l = self._impurity(y[left_mask])
                imp_r = self._impurity(y[right_mask])
                child_impurity = (n_l / n) * imp_l + (n_r / n) * imp_r
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    split_feat = feat
                    split_thresh = thresh

        return split_feat, split_thresh

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root) for row in X])

    def _predict_row(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)


# =====================================================================
# 2. BAGGING ENSEMBLE WITH OUT-OF-BAG (OOB) EVALUATION
# =====================================================================

class BaggingEnsembleScratch:
    """
    Bootstrap Aggregating (Bagging) Ensemble:
      - Fits B base estimators on bootstrap samples (sampling with replacement)
      - Computes Out-Of-Bag (OOB) score automatically
      - Supports Random Forest mode via max_features='sqrt'
    """
    def __init__(self, n_estimators=25, max_depth=6, max_features=None, task="classification"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.task = task
        self.estimators_ = []
        self.oob_indices_ = []
        self.oob_score_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = len(X)

        self.estimators_ = []
        self.oob_indices_ = []

        # Track out-of-bag predictions: for each sample, store list of predictions
        oob_preds = [[] for _ in range(n_samples)]

        for _ in range(self.n_estimators):
            # Bootstrap sample (sample with replacement, size N)
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False
            oob_idx = np.where(oob_mask)[0]

            self.oob_indices_.append(oob_idx)

            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                max_features=self.max_features,
                task=self.task
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.estimators_.append(tree)

            # Record predictions on out-of-bag samples
            if len(oob_idx) > 0:
                preds_oob = tree.predict(X[oob_idx])
                for idx, pred in zip(oob_idx, preds_oob):
                    oob_preds[idx].append(pred)

        # Compute aggregate OOB score
        valid_oob_samples = [i for i in range(n_samples) if len(oob_preds[i]) > 0]
        if len(valid_oob_samples) > 0:
            final_oob_preds = []
            for i in valid_oob_samples:
                if self.task == "classification":
                    vals, counts = np.unique(oob_preds[i], return_counts=True)
                    final_oob_preds.append(vals[np.argmax(counts)])
                else:
                    final_oob_preds.append(np.mean(oob_preds[i]))

            y_valid = y[valid_oob_samples]
            final_oob_preds = np.array(final_oob_preds)
            if self.task == "classification":
                self.oob_score_ = np.mean(final_oob_preds == y_valid)
            else:
                self.oob_score_ = -np.mean((final_oob_preds - y_valid) ** 2)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])  # (n_trees, n_samples)

        if self.task == "classification":
            final_preds = []
            for col in all_preds.T:
                vals, counts = np.unique(col, return_counts=True)
                final_preds.append(vals[np.argmax(counts)])
            return np.array(final_preds)
        else:
            return np.mean(all_preds, axis=0)


# =====================================================================
# 3. DEMONSTRATION & BENCHMARK SUITE
# =====================================================================

def generate_noisy_circles(n_samples=250, noise=0.15):
    """Synthetic non-linear classification dataset."""
    n_per_ring = n_samples // 2
    theta1 = np.random.uniform(0, 2 * np.pi, n_per_ring)
    r1 = np.random.normal(0.4, noise, n_per_ring)
    X1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)

    theta2 = np.random.uniform(0, 2 * np.pi, n_per_ring)
    r2 = np.random.normal(0.9, noise, n_per_ring)
    X2 = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis=1)

    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_per_ring), np.ones(n_per_ring)])
    return X, y


def generate_sine_regression(n_samples=150, noise=0.25):
    """Synthetic 1D sine regression with high noise."""
    X = np.sort(np.random.uniform(-3, 3, (n_samples, 1)), axis=0)
    y = np.sin(X[:, 0]) + np.random.normal(0, noise, n_samples)
    return X, y


def run_bagging_demo():
    print("=" * 70)
    print("WEEK 13: BAGGING & RANDOM FOREST (SCRATCH ENGINE)")
    print("=" * 70)

    # 1. Theoretical Bootstrap OOB Coverage
    n = 1000
    boot_counts = np.array([len(np.unique(np.random.choice(n, size=n, replace=True))) for _ in range(100)])
    empirical_fraction = np.mean(boot_counts / n)
    theory_fraction = 1.0 - 1.0 / np.e
    print(f"\n[1] Bootstrap Mathematical Property:")
    print(f"    Theoretical unique samples: 1 - 1/e = {theory_fraction*100:.2f}%")
    print(f"    Empirical unique samples:             {empirical_fraction*100:.2f}%")
    print(f"    Out-Of-Bag (OOB) samples:   ~{ (1 - empirical_fraction)*100:.2f}% per tree")

    # 2. Classification comparison: Single Tree vs. Bagging vs. Random Forest
    X_clf, y_clf = generate_noisy_circles(n_samples=300, noise=0.12)

    # Train / Test Split
    train_mask = np.random.rand(len(X_clf)) < 0.75
    X_train, y_train = X_clf[train_mask], y_clf[train_mask]
    X_test, y_test = X_clf[~train_mask], y_clf[~train_mask]

    single_tree = DecisionTreeScratch(max_depth=10, task="classification").fit(X_train, y_train)
    bagging_ens = BaggingEnsembleScratch(n_estimators=30, max_depth=10, task="classification").fit(X_train, y_train)
    rf_ens = BaggingEnsembleScratch(n_estimators=30, max_depth=10, max_features="sqrt", task="classification").fit(X_train, y_train)

    acc_tree = np.mean(single_tree.predict(X_test) == y_test)
    acc_bag = np.mean(bagging_ens.predict(X_test) == y_test)
    acc_rf = np.mean(rf_ens.predict(X_test) == y_test)

    print(f"\n[2] Non-Linear Classification Accuracy:")
    print(f"    Single Decision Tree (Depth 10): {acc_tree*100:.2f}% (High variance)")
    print(f"    Bagging Ensemble (30 Trees):      {acc_bag*100:.2f}% (OOB: {bagging_ens.oob_score_*100:.2f}%)")
    print(f"    Random Forest (30 Subspaced):     {acc_rf*100:.2f}% (OOB: {rf_ens.oob_score_*100:.2f}%)")

    # 3. Variance Reduction on Regression
    X_reg, y_reg = generate_sine_regression(n_samples=120, noise=0.3)
    reg_tree = DecisionTreeScratch(max_depth=6, task="regression").fit(X_reg, y_reg)
    reg_bag = BaggingEnsembleScratch(n_estimators=35, max_depth=6, task="regression").fit(X_reg, y_reg)

    mse_tree = np.mean((reg_tree.predict(X_reg) - y_reg) ** 2)
    mse_bag = np.mean((reg_bag.predict(X_reg) - y_reg) ** 2)
    print(f"\n[3] Sine Regression Smoothing & MSE:")
    print(f"    Single Tree MSE:  {mse_tree:.4f} (Step-like jerky fit)")
    print(f"    Bagging (35) MSE: {mse_bag:.4f} (Smooth ensemble curve)")

    # 4. Generate Visualization Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel A: Single Tree Decision Boundary
    x_min, x_max = X_clf[:, 0].min() - 0.3, X_clf[:, 0].max() + 0.3
    y_min, y_max = X_clf[:, 1].min() - 0.3, X_clf[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z_tree = single_tree.predict(grid).reshape(xx.shape)
    axes[0, 0].contourf(xx, yy, Z_tree, alpha=0.35, cmap="coolwarm")
    axes[0, 0].scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap="coolwarm", edgecolor="k", s=30)
    axes[0, 0].set_title(f"Single Tree (Overfitting, Acc: {acc_tree*100:.1f}%)", color="white", fontsize=11, fontweight="bold")

    # Panel B: Bagging Ensemble Decision Boundary
    Z_bag = bagging_ens.predict(grid).reshape(xx.shape)
    axes[0, 1].contourf(xx, yy, Z_bag, alpha=0.35, cmap="coolwarm")
    axes[0, 1].scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap="coolwarm", edgecolor="k", s=30)
    axes[0, 1].set_title(f"Bagging (30 Trees, Acc: {acc_bag*100:.1f}%)", color="#58a6ff", fontsize=11, fontweight="bold")

    # Panel C: Regression Smoothing
    X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    axes[1, 0].scatter(X_reg, y_reg, color="#8b949e", alpha=0.6, s=25, label="Noisy Points")
    axes[1, 0].plot(X_plot, np.sin(X_plot), color="#3fb950", linewidth=2.0, linestyle="--", label="True Sine")
    axes[1, 0].plot(X_plot, reg_tree.predict(X_plot), color="#f85149", linewidth=1.5, label="Single Tree")
    axes[1, 0].plot(X_plot, reg_bag.predict(X_plot), color="#58a6ff", linewidth=2.2, label="Bagging (35 Trees)")
    axes[1, 0].set_title("Regression Variance Reduction", color="white", fontsize=11, fontweight="bold")
    axes[1, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    # Panel D: OOB Score vs. Ensemble Size
    tree_counts = [1, 3, 5, 10, 15, 25, 40]
    oob_scores = []
    test_accs = []
    for tc in tree_counts:
        b = BaggingEnsembleScratch(n_estimators=tc, max_depth=8).fit(X_train, y_train)
        oob_scores.append(b.oob_score_ if b.oob_score_ is not None else 0.5)
        test_accs.append(np.mean(b.predict(X_test) == y_test))

    axes[1, 1].plot(tree_counts, oob_scores, marker="o", color="#d29922", label="OOB Accuracy")
    axes[1, 1].plot(tree_counts, test_accs, marker="s", color="#58a6ff", label="Test Accuracy")
    axes[1, 1].set_xlabel("Number of Estimators (Trees)", color="gray")
    axes[1, 1].set_ylabel("Accuracy", color="gray")
    axes[1, 1].set_title("OOB Score as Reliable Generalization Metric", color="white", fontsize=11, fontweight="bold")
    axes[1, 1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    plt.tight_layout()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(curr_dir, "bagging_random_forest_dashboard.png")
    plt.savefig(out_path, dpi=140, facecolor="#0d1117")
    plt.close()
    print(f"\n[SAVED] Dashboard written to {out_path}")


if __name__ == "__main__":
    run_bagging_demo()

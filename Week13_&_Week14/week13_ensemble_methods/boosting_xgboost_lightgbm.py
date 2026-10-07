"""
Week 13 - Boosting Algorithms: Gradient Boosting, XGBoost & LightGBM Mechanics
=============================================================================
Topics Covered:
  1. The Boosting philosophy: converting weak learners into strong learners sequentially
  2. Gradient Boosting (GBM): Pseudo-residuals & learning rate shrinkage
  3. XGBoost core math:
     - 2nd-order Taylor expansion (gradients g_i and hessians h_i)
     - Optimal leaf weight: w* = -G / (H + lambda)
     - Exact split gain formula with gamma penalty:
       Gain = 0.5 * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma
  4. LightGBM core innovations:
     - Histogram-based binning (discretization of features)
     - GOSS (Gradient-based One-Side Sampling with weight amplification)
     - Leaf-wise (best-first) tree growth vs. Level-wise (depth-first) growth
  5. Empirical comparison & convergence tracking

Dependencies: numpy, matplotlib (zero external ML dependencies)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# =====================================================================
# 1. GRADIENT BOOSTING REGRESSOR (MSE Loss Scratch Engine)
# =====================================================================

class DecisionStump:
    """Fast 1-level decision tree (stump) for boosting weak learners."""
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_mse = float("inf")

        for feat in range(n_features):
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            # Sample thresholds for speed if too many
            if len(thresholds) > 20:
                thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 20))

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                l_val = np.mean(y[left_mask])
                r_val = np.mean(y[right_mask])
                mse = np.sum((y[left_mask] - l_val)**2) + np.sum((y[right_mask] - r_val)**2)

                if mse < best_mse:
                    best_mse = mse
                    self.feature = feat
                    self.threshold = thresh
                    self.left_value = l_val
                    self.right_value = r_val

        if self.feature is None:
            self.left_value = np.mean(y)
            self.right_value = np.mean(y)
            self.threshold = 0.0
            self.feature = 0

        return self

    def predict(self, X):
        preds = np.zeros(len(X))
        left_mask = X[:, self.feature] <= self.threshold
        preds[left_mask] = self.left_value
        preds[~left_mask] = self.right_value
        return preds


class GradientBoostingScratch:
    """
    Standard Gradient Boosting for Regression:
      - Initial prediction F0 = mean(y)
      - At step m: residual = y - F_{m-1}(X)
      - Fit weak learner to residuals
      - Update F_m = F_{m-1} + lr * h_m(X)
    """
    def __init__(self, n_estimators=40, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.initial_pred = None
        self.train_loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.initial_pred = np.mean(y)
        f_hat = np.full_like(y, self.initial_pred)
        self.trees = []
        self.train_loss_history = []

        for _ in range(self.n_estimators):
            residuals = y - f_hat
            stump = DecisionStump()
            stump.fit(X, residuals)
            step_pred = stump.predict(X)
            f_hat += self.learning_rate * step_pred

            self.trees.append(stump)
            self.train_loss_history.append(np.mean((y - f_hat)**2))

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        f_hat = np.full(len(X), self.initial_pred)
        for stump in self.trees:
            f_hat += self.learning_rate * stump.predict(X)
        return f_hat


# =====================================================================
# 2. XGBOOST ARCHITECTURE: 2ND-ORDER TAYLOR, GAIN & REGULARIZATION
# =====================================================================

class XGBoostNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, weight=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.weight = weight  # Optimal leaf weight w* = -G / (H + lambda)

    @property
    def is_leaf(self):
        return self.weight is not None


class XGBoostTreeScratch:
    """
    XGBoost Tree implementing exact 2nd-order Taylor optimization:
      - g_i = gradient of loss w.r.t prediction
      - h_i = hessian of loss w.r.t prediction
      - lambda = L2 leaf regularization
      - gamma = split complexity penalty (pruning threshold)
    """
    def __init__(self, max_depth=3, l2_reg=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.l2_reg = l2_reg
        self.gamma = gamma
        self.root = None

    def fit(self, X, g, h):
        self.root = self._build_node(X, g, h, depth=0)
        return self

    def _calc_leaf_weight(self, G, H):
        """w* = -G / (H + lambda)"""
        return -G / (H + self.l2_reg)

    def _calc_split_gain(self, G_L, H_L, G_R, H_R):
        """
        XGBoost Split Gain formula:
        Gain = 0.5 * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma
        """
        denom_l = H_L + self.l2_reg
        denom_r = H_R + self.l2_reg
        denom_tot = H_L + H_R + self.l2_reg

        gain = 0.5 * (
            (G_L ** 2) / denom_l +
            (G_R ** 2) / denom_r -
            ((G_L + G_R) ** 2) / denom_tot
        ) - self.gamma
        return gain

    def _build_node(self, X, g, h, depth):
        G, H = np.sum(g), np.sum(h)

        if depth >= self.max_depth or len(g) <= 2:
            return XGBoostNode(weight=self._calc_leaf_weight(G, H))

        n_samples, n_features = X.shape
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        for feat in range(n_features):
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            if len(thresholds) > 15:
                thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 15))

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                G_L, H_L = np.sum(g[left_mask]), np.sum(h[left_mask])
                G_R, H_R = np.sum(g[right_mask]), np.sum(h[right_mask])

                gain = self._calc_split_gain(G_L, H_L, G_R, H_R)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None or best_gain <= 0.0:
            # Prune split if gain does not exceed gamma
            return XGBoostNode(weight=self._calc_leaf_weight(G, H))

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left = self._build_node(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right = self._build_node(X[right_mask], g[right_mask], h[right_mask], depth + 1)
        return XGBoostNode(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def predict(self, X):
        return np.array([self._predict_row(row, self.root) for row in X])

    def _predict_row(self, x, node):
        if node.is_leaf:
            return node.weight
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)


class XGBoostScratchRegressor:
    """
    XGBoost Regressor using Squared Loss:
      L(y, y_hat) = 0.5 * (y - y_hat)^2
      g_i = y_hat - y
      h_i = 1.0 (constant curvature for squared loss)
    """
    def __init__(self, n_estimators=30, learning_rate=0.15, max_depth=3, l2_reg=1.0, gamma=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_reg = l2_reg
        self.gamma = gamma
        self.trees = []
        self.base_pred = 0.0
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.base_pred = np.mean(y)
        y_hat = np.full_like(y, self.base_pred)

        self.trees = []
        self.loss_history = []

        for _ in range(self.n_estimators):
            # 1st and 2nd derivatives for squared error
            g = y_hat - y
            h = np.ones_like(y)

            tree = XGBoostTreeScratch(max_depth=self.max_depth, l2_reg=self.l2_reg, gamma=self.gamma)
            tree.fit(X, g, h)

            update = tree.predict(X)
            y_hat += self.learning_rate * update
            self.trees.append(tree)
            self.loss_history.append(np.mean((y - y_hat)**2))

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        y_hat = np.full(len(X), self.base_pred)
        for tree in self.trees:
            y_hat += self.learning_rate * tree.predict(X)
        return y_hat


# =====================================================================
# 3. LIGHTGBM INNOVATIONS: GOSS & HISTOGRAM BINNING
# =====================================================================

class LightGBMSimulator:
    """
    Demonstrates the two core engineering breakthroughs in LightGBM:
      1. Gradient-based One-Side Sampling (GOSS):
         - Keep top 'a' fraction of instances with high gradients
         - Randomly sample 'b' fraction from remaining small gradients
         - Amplify small gradient weights by (1 - a) / b to preserve distribution
      2. Histogram Binning:
         - Converts continuous float features into K discrete integer bins
    """
    @staticmethod
    def goss_sampling(g, a=0.2, b=0.2):
        """
        Gradient-based One-Side Sampling (GOSS).
        Returns sampled indices and weights.
        """
        n = len(g)
        abs_g = np.abs(g)
        sorted_indices = np.argsort(abs_g)[::-1]

        top_k = int(a * n)
        large_gradient_indices = sorted_indices[:top_k]
        rest_indices = sorted_indices[top_k:]

        sub_k = int(b * len(rest_indices))
        small_gradient_indices = np.random.choice(rest_indices, size=sub_k, replace=False)

        selected_indices = np.concatenate([large_gradient_indices, small_gradient_indices])
        weights = np.ones(len(selected_indices))
        # Amplify small gradient weights
        weights[top_k:] = (1.0 - a) / b

        return selected_indices, weights

    @staticmethod
    def histogram_binning(X, max_bins=32):
        """Discretizes continuous columns into max_bins integers."""
        X_binned = np.zeros_like(X, dtype=int)
        bin_edges = []
        for feat in range(X.shape[1]):
            col = X[:, feat]
            quantiles = np.linspace(0, 1, max_bins + 1)
            edges = np.unique(np.quantile(col, quantiles))
            X_binned[:, feat] = np.digitize(col, edges[1:-1])
            bin_edges.append(edges)
        return X_binned, bin_edges


# =====================================================================
# 4. BENCHMARK & COMPARISON DEMONSTRATION
# =====================================================================

def run_boosting_demo():
    print("=" * 70)
    print("WEEK 13: BOOSTING ALGORITHMS - GBM, XGBOOST & LIGHTGBM")
    print("=" * 70)

    # 1. Dataset: High Non-linearity with Step and Sine combination
    np.random.seed(10)
    X = np.sort(np.random.uniform(-4, 4, (180, 1)), axis=0)
    y = np.sin(1.5 * X[:, 0]) + 0.5 * (X[:, 0] > 0) + np.random.normal(0, 0.2, 180)

    # Models
    gbm = GradientBoostingScratch(n_estimators=45, learning_rate=0.1).fit(X, y)
    xgb_reg = XGBoostScratchRegressor(n_estimators=45, learning_rate=0.1, max_depth=3, l2_reg=1.5, gamma=0.05).fit(X, y)

    mse_gbm = np.mean((gbm.predict(X) - y)**2)
    mse_xgb = np.mean((xgb_reg.predict(X) - y)**2)

    print(f"\n[1] Regression Benchmark:")
    print(f"    Gradient Boosting (GBM Stumps) MSE: {mse_gbm:.4f}")
    print(f"    XGBoost (2nd Order + L2 Reg)   MSE: {mse_xgb:.4f}")

    # 2. GOSS Sampling Verification
    residuals = y - gbm.predict(X)
    g = residuals  # loss derivative
    idx_goss, w_goss = LightGBMSimulator.goss_sampling(g, a=0.25, b=0.25)
    print(f"\n[2] LightGBM GOSS Feature Mechanics:")
    print(f"    Total Instances: {len(X)}")
    print(f"    Retained via GOSS: {len(idx_goss)} (Speedup: {len(X)/len(idx_goss):.1f}x)")
    print(f"    Amplification weight on small gradients: {w_goss[-1]:.2f}")

    # 3. Histogram Discretization
    X_binned, edges = LightGBMSimulator.histogram_binning(X, max_bins=16)
    print(f"\n[3] LightGBM Histogram Binning:")
    print(f"    Continuous feature mapped into {len(edges[0]) - 1} bins.")
    print(f"    Enables cache-friendly integer operations and O(#bins) split search.")

    # 4. Generate Visualization Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel A: Predictions vs Ground Truth
    X_plot = np.linspace(-4, 4, 300).reshape(-1, 1)
    y_true = np.sin(1.5 * X_plot[:, 0]) + 0.5 * (X_plot[:, 0] > 0)
    axes[0, 0].scatter(X, y, color="#8b949e", alpha=0.5, s=25, label="Samples with noise")
    axes[0, 0].plot(X_plot, y_true, color="#3fb950", linewidth=2.0, linestyle="--", label="Ground Truth")
    axes[0, 0].plot(X_plot, gbm.predict(X_plot), color="#f0883e", linewidth=1.8, label="Standard GBM (Stumps)")
    axes[0, 0].plot(X_plot, xgb_reg.predict(X_plot), color="#58a6ff", linewidth=2.2, label="XGBoost (2nd Order + Reg)")
    axes[0, 0].set_title("Regression Fit: GBM vs. XGBoost", color="white", fontsize=11, fontweight="bold")
    axes[0, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    # Panel B: Training Loss Convergence
    axes[0, 1].plot(gbm.train_loss_history, color="#f0883e", linewidth=2, label="GBM MSE Loss")
    axes[0, 1].plot(xgb_reg.loss_history, color="#58a6ff", linewidth=2, label="XGBoost MSE Loss")
    axes[0, 1].set_xlabel("Boosting Iteration (Trees Added)", color="gray")
    axes[0, 1].set_ylabel("Mean Squared Error (MSE)", color="gray")
    axes[0, 1].set_title("Loss Reduction per Boosting Step", color="white", fontsize=11, fontweight="bold")
    axes[0, 1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    # Panel C: GOSS Sampling Distribution
    axes[1, 0].hist(np.abs(g), bins=20, color="#8b949e", alpha=0.4, label="All Sample Gradients |g|")
    axes[1, 0].hist(np.abs(g[idx_goss]), bins=20, color="#a371f7", alpha=0.7, label="GOSS Retained Samples")
    axes[1, 0].set_title("LightGBM GOSS: Retain Top Gradients + Sample Tail", color="white", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("|Gradient Magnitude|", color="gray")
    axes[1, 0].set_ylabel("Count", color="gray")
    axes[1, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    # Panel D: Architectural Comparison
    axes[1, 1].axis("off")
    table_data = [
        ["Mechanism", "Gradient Boosting (GBM)", "XGBoost", "LightGBM"],
        ["Taylor Expansion", "1st Order (Gradients)", "2nd Order (Grad + Hessian)", "2nd Order (Grad + Hessian)"],
        ["Split Finding", "Exact Greedy O(N x D)", "Quantile / Exact", "Histogram-based O(K x D)"],
        ["Tree Growth", "Level-wise (Balanced)", "Level-wise (Pruned by gamma)", "Leaf-wise (Best-first)"],
        ["Regularization", "Shrinkage (lr)", "L1 (alpha) + L2 (lambda)", "L1 + L2 + Min-data-in-leaf"],
        ["Data Subsampling", "Uniform Bagging", "Uniform Subsample", "GOSS (Gradient-based)"],
        ["Categorical Handling", "One-Hot / Integer", "One-Hot / Experimental", "Native optimal split"]
    ]
    table = axes[1, 1].table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#58a6ff", weight="bold")
        else:
            cell.set_facecolor("#161b22")
            cell.set_text_props(color="white")
        cell.set_edgecolor("#30363d")

    plt.tight_layout()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(curr_dir, "boosting_xgboost_lightgbm_dashboard.png")
    plt.savefig(out_path, dpi=140, facecolor="#0d1117")
    plt.close()
    print(f"\n[SAVED] Dashboard written to {out_path}")


if __name__ == "__main__":
    run_boosting_demo()

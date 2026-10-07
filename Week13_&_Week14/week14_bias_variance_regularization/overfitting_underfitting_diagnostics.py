"""
Week 14 - Overfitting vs. Underfitting: Diagnostics, Learning Curves & Early Stopping
=====================================================================================
Topics Covered:
  1. Diagnostic principles:
     - High Bias / Underfitting: Train loss high, Val loss high, small gap
     - High Variance / Overfitting: Train loss low, Val loss high, large gap
  2. Learning Curves from scratch (Error vs Training Set Size N)
  3. Validation Curves from scratch (Error vs Model Capacity)
  4. Early Stopping Engine (Patience, checkpointing, and generalization safeguard)
  5. K-Fold Cross-Validation from scratch

Dependencies: numpy, matplotlib (zero external ML dependencies)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# =====================================================================
# 1. ITERATIVE GRADIENT-BASED MODEL WITH EARLY STOPPING
# =====================================================================

class PolynomialRegressorGD:
    """Polynomial Regression trained via Gradient Descent with Early Stopping."""
    def __init__(self, degree=5, learning_rate=0.01, max_epochs=2000, patience=20):
        self.degree = degree
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.train_history = []
        self.val_history = []
        self.best_epoch = None
        self.best_weights = None

    def _transform(self, x):
        X = np.vander(x, N=self.degree + 1, increasing=True)
        # Normalize polynomial features (excluding bias column) for stable GD
        return X

    def fit(self, x_train, y_train, x_val, y_val):
        X_tr = self._transform(x_train)
        X_va = self._transform(x_val)

        # Feature normalization
        self.means = np.mean(X_tr[:, 1:], axis=0)
        self.stds = np.std(X_tr[:, 1:], axis=0) + 1e-8

        X_tr_norm = np.copy(X_tr)
        X_tr_norm[:, 1:] = (X_tr[:, 1:] - self.means) / self.stds

        X_va_norm = np.copy(X_va)
        X_va_norm[:, 1:] = (X_va[:, 1:] - self.means) / self.stds

        n_samples, n_features = X_tr_norm.shape
        self.weights = np.zeros(n_features)
        best_val_loss = float("inf")
        patience_counter = 0

        self.train_history = []
        self.val_history = []

        for epoch in range(self.max_epochs):
            # Compute predictions
            y_pred_tr = X_tr_norm @ self.weights
            y_pred_va = X_va_norm @ self.weights

            # Compute losses
            train_loss = np.mean((y_train - y_pred_tr) ** 2)
            val_loss = np.mean((y_val - y_pred_va) ** 2)

            self.train_history.append(train_loss)
            self.val_history.append(val_loss)

            # Gradient step
            grad = (-2.0 / n_samples) * (X_tr_norm.T @ (y_train - y_pred_tr))
            self.weights -= self.lr * grad

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_weights = np.copy(self.weights)
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    # Early stop triggered
                    self.weights = np.copy(self.best_weights)
                    break

        return self

    def predict(self, x):
        X = self._transform(x)
        X_norm = np.copy(X)
        X_norm[:, 1:] = (X[:, 1:] - self.means) / self.stds
        return X_norm @ self.weights


# =====================================================================
# 2. DIAGNOSTIC TOOLS: LEARNING CURVES & K-FOLD CROSS-VALIDATION
# =====================================================================

def compute_learning_curves(model_cls, x, y, train_fractions=np.linspace(0.15, 0.9, 12)):
    """Computes train and test error as training dataset size grows."""
    n = len(x)
    indices = np.random.permutation(n)
    split = int(0.75 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    x_train_full, y_train_full = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    train_sizes = []
    train_errors = []
    val_errors = []

    for frac in train_fractions:
        sub_size = max(4, int(frac * len(x_train_full)))
        sub_idx = np.random.choice(len(x_train_full), sub_size, replace=False)
        x_sub, y_sub = x_train_full[sub_idx], y_train_full[sub_idx]

        model = model_cls().fit(x_sub, y_sub)
        pred_tr = model.predict(x_sub)
        pred_va = model.predict(x_val)

        train_sizes.append(sub_size)
        train_errors.append(np.mean((pred_tr - y_sub)**2))
        val_errors.append(np.mean((pred_va - y_val)**2))

    return np.array(train_sizes), np.array(train_errors), np.array(val_errors)


def k_fold_cv(x, y, degree=3, k=5):
    """K-Fold Cross-Validation from scratch."""
    n = len(x)
    indices = np.random.permutation(n)
    folds = np.array_split(indices, k)
    scores = []

    for fold_i in range(k):
        val_idx = folds[fold_i]
        train_idx = np.setdiff1d(indices, val_idx)

        X_tr = np.vander(x[train_idx], N=degree + 1, increasing=True)
        X_va = np.vander(x[val_idx], N=degree + 1, increasing=True)

        w = np.linalg.lstsq(X_tr, y[train_idx], rcond=None)[0]
        preds = X_va @ w
        mse = np.mean((preds - y[val_idx]) ** 2)
        scores.append(mse)

    return np.mean(scores), np.std(scores)


# =====================================================================
# 3. BENCHMARK & VISUALIZATION DASHBOARD
# =====================================================================

def run_diagnostics_demo():
    print("=" * 70)
    print("WEEK 14: OVERFITTING/UNDERFITTING DIAGNOSTICS & EARLY STOPPING")
    print("=" * 70)

    # 1. Dataset Generation
    x = np.sort(np.random.uniform(-1, 1, 120))
    y = np.cos(np.pi * x) + np.random.normal(0, 0.25, len(x))

    # Split for Early Stopping
    val_mask = np.random.rand(len(x)) < 0.25
    x_tr, y_tr = x[~val_mask], y[~val_mask]
    x_va, y_va = x[val_mask], y[val_mask]

    # Fit GD with early stopping on high degree model (degree 8)
    model_es = PolynomialRegressorGD(degree=8, learning_rate=0.03, max_epochs=2000, patience=30)
    model_es.fit(x_tr, y_tr, x_va, y_va)

    print(f"\n[1] Early Stopping Triggered:")
    print(f"    Total Epochs Run:   {len(model_es.train_history)}")
    print(f"    Best Validation At: Epoch {model_es.best_epoch}")
    print(f"    Best Val Loss:      {model_es.val_history[model_es.best_epoch]:.4f}")
    print(f"    Final Train Loss:   {model_es.train_history[-1]:.4f}")

    # 2. 5-Fold Cross Validation across degrees
    print(f"\n[2] K-Fold Cross Validation Across Capacities:")
    cv_means = []
    cv_stds = []
    degrees = [1, 2, 3, 4, 6, 8]
    for d in degrees:
        mean_s, std_s = k_fold_cv(x, y, degree=d, k=5)
        cv_means.append(mean_s)
        cv_stds.append(std_s)
        print(f"    Degree {d}: 5-Fold CV MSE = {mean_s:.4f} (+/- {std_s:.4f})")

    # 3. Generate Diagnostics Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel A: Early Stopping Training Trajectory
    axes[0, 0].plot(model_es.train_history, color="#58a6ff", linewidth=2.0, label="Training Loss")
    axes[0, 0].plot(model_es.val_history, color="#f85149", linewidth=2.0, label="Validation Loss")
    axes[0, 0].axvline(model_es.best_epoch, color="#3fb950", linestyle="--", linewidth=1.8, label=f"Best Model (Epoch {model_es.best_epoch})")
    axes[0, 0].set_xlabel("Epoch", color="gray")
    axes[0, 0].set_ylabel("Mean Squared Error", color="gray")
    axes[0, 0].set_title("Early Stopping: Preventing Overfitting During Iteration", color="white", fontsize=11, fontweight="bold")
    axes[0, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    # Panel B: Fitted Model at Early Stop Checkpoint
    x_plot = np.linspace(-1, 1, 200)
    axes[0, 1].scatter(x_tr, y_tr, color="#58a6ff", alpha=0.5, s=25, label="Train Samples")
    axes[0, 1].scatter(x_va, y_va, color="#f85149", alpha=0.8, s=35, marker="^", label="Validation Samples")
    axes[0, 1].plot(x_plot, np.cos(np.pi * x_plot), color="#3fb950", linestyle="--", linewidth=2, label="True Signal")
    axes[0, 1].plot(x_plot, model_es.predict(x_plot), color="#d29922", linewidth=2.2, label="Early-Stopped Fit")
    axes[0, 1].set_title("Restored Model at Lowest Validation Error", color="white", fontsize=11, fontweight="bold")
    axes[0, 1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    # Panel C: K-Fold Cross Validation Curve
    axes[1, 0].errorbar(degrees, cv_means, yerr=cv_stds, fmt="-o", color="#a371f7", ecolor="#8b949e", capsize=4, linewidth=2, label="CV MSE (+/- 1 std)")
    best_d = degrees[np.argmin(cv_means)]
    axes[1, 0].axvline(best_d, color="#3fb950", linestyle=":", label=f"Optimal Degree ({best_d})")
    axes[1, 0].set_xlabel("Polynomial Degree (Complexity)", color="gray")
    axes[1, 0].set_ylabel("5-Fold Cross Validation MSE", color="gray")
    axes[1, 0].set_title("Validation Curve via 5-Fold Cross Validation", color="white", fontsize=11, fontweight="bold")
    axes[1, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    # Panel D: Diagnostic Decision Matrix
    axes[1, 1].axis("off")
    diag_data = [
        ["Diagnostic Pattern", "Root Cause", "Action to Take"],
        ["High Train Loss, High Val Loss", "Underfitting (High Bias)", "Increase capacity, add features, reduce reg"],
        ["Low Train Loss, High Val Loss", "Overfitting (High Variance)", "Add L1/L2 reg, early stopping, more data, dropout"],
        ["Loss diverges to NaN / Inf", "Exploding Gradient / LR high", "Lower learning rate, clip gradients, normalize X"],
        ["Train error keeps dropping, Val stable", "Model saturating capacity", "Monitor generalization gap, early stop"]
    ]
    diag_table = axes[1, 1].table(cellText=diag_data, loc="center", cellLoc="center")
    diag_table.auto_set_font_size(False)
    diag_table.set_fontsize(8.5)
    diag_table.scale(1.0, 1.7)
    for (row, col), cell in diag_table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#58a6ff", weight="bold")
        else:
            cell.set_facecolor("#161b22")
            cell.set_text_props(color="white")
        cell.set_edgecolor("#30363d")

    plt.tight_layout()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(curr_dir, "overfitting_underfitting_diagnostics_dashboard.png")
    plt.savefig(out_path, dpi=140, facecolor="#0d1117")
    plt.close()
    print(f"\n[SAVED] Dashboard written to {out_path}")


if __name__ == "__main__":
    run_diagnostics_demo()

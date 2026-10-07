"""
Week 11 - Principal Component Analysis (PCA) from Scratch
==========================================================
Topics Covered:
  1. The curse of dimensionality - why we reduce dimensions
  2. Variance, Covariance Matrix and what they tell us
  3. Eigenvectors & Eigenvalues - the core math of PCA
  4. PCA algorithm step-by-step from scratch (NumPy only)
  5. Explained Variance Ratio & Scree Plot
  6. Reconstruction & Information Loss
  7. Verification via SVD (no sklearn needed)
  8. Real-world application on Iris dataset (bundled)

Dependencies: numpy, matplotlib (stdlib only, no sklearn/scipy)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

print("=" * 65)
print("  WEEK 11 - PCA (Principal Component Analysis) from Scratch")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# IRIS DATASET  (hardcoded - no sklearn dependency)
# 150 samples x 4 features: sepal_len, sepal_wid, petal_len, petal_wid
# ─────────────────────────────────────────────────────────────
IRIS_DATA = np.array([
    [5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],
    [5.0,3.6,1.4,0.2],[5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5.0,3.4,1.5,0.2],
    [4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],[5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],
    [4.8,3.0,1.4,0.1],[4.3,3.0,1.1,0.1],[5.8,4.0,1.2,0.2],[5.7,4.4,1.5,0.4],
    [5.4,3.9,1.3,0.4],[5.1,3.5,1.4,0.3],[5.7,3.8,1.7,0.3],[5.1,3.8,1.5,0.3],
    [5.4,3.4,1.7,0.2],[5.1,3.7,1.5,0.4],[4.6,3.6,1.0,0.2],[5.1,3.3,1.7,0.5],
    [4.8,3.4,1.9,0.2],[5.0,3.0,1.6,0.2],[5.0,3.4,1.6,0.4],[5.2,3.5,1.5,0.2],
    [5.2,3.4,1.4,0.2],[4.7,3.2,1.6,0.2],[4.8,3.1,1.6,0.2],[5.4,3.4,1.5,0.4],
    [5.2,4.1,1.5,0.1],[5.5,4.2,1.4,0.2],[4.9,3.1,1.5,0.2],[5.0,3.2,1.2,0.2],
    [5.5,3.5,1.3,0.2],[4.9,3.6,1.4,0.1],[4.4,3.0,1.3,0.2],[5.1,3.4,1.5,0.2],
    [5.0,3.5,1.3,0.3],[4.5,2.3,1.3,0.3],[4.4,3.2,1.3,0.2],[5.0,3.5,1.6,0.6],
    [5.1,3.8,1.9,0.4],[4.8,3.0,1.4,0.3],[5.1,3.8,1.6,0.2],[4.6,3.2,1.4,0.2],
    [5.3,3.7,1.5,0.2],[5.0,3.3,1.4,0.2],
    [7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],[5.5,2.3,4.0,1.3],
    [6.5,2.8,4.6,1.5],[5.7,2.8,4.5,1.3],[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1.0],
    [6.6,2.9,4.6,1.3],[5.2,2.7,3.9,1.4],[5.0,2.0,3.5,1.0],[5.9,3.0,4.2,1.5],
    [6.0,2.2,4.0,1.0],[6.1,2.9,4.7,1.4],[5.6,2.9,3.6,1.3],[6.7,3.1,4.4,1.4],
    [5.6,3.0,4.5,1.5],[5.8,2.7,4.1,1.0],[6.2,2.2,4.5,1.5],[5.6,2.5,3.9,1.1],
    [5.9,3.2,4.8,1.8],[6.1,2.8,4.0,1.3],[6.3,2.5,4.9,1.5],[6.1,2.8,4.7,1.2],
    [6.4,2.9,4.3,1.3],[6.6,3.0,4.4,1.4],[6.8,2.8,4.8,1.4],[6.7,3.0,5.0,1.7],
    [6.0,2.9,4.5,1.5],[5.7,2.6,3.5,1.0],[5.5,2.4,3.8,1.1],[5.5,2.4,3.7,1.0],
    [5.8,2.7,3.9,1.2],[6.0,2.7,5.1,1.6],[5.4,3.0,4.5,1.5],[6.0,3.4,4.5,1.6],
    [6.7,3.1,4.7,1.5],[6.3,2.3,4.4,1.3],[5.6,3.0,4.1,1.3],[5.5,2.5,4.0,1.3],
    [5.5,2.6,4.4,1.2],[6.1,3.0,4.6,1.4],[5.8,2.6,4.0,1.2],[5.0,2.3,3.3,1.0],
    [5.6,2.7,4.2,1.3],[5.7,3.0,4.2,1.2],[5.7,2.9,4.2,1.3],[6.2,2.9,4.3,1.3],
    [5.1,2.5,3.0,1.1],[5.7,2.8,4.1,1.3],
    [6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],[6.3,2.9,5.6,1.8],
    [6.5,3.0,5.8,2.2],[7.6,3.0,6.6,2.1],[4.9,2.5,4.5,1.7],[7.3,2.9,6.3,1.8],
    [6.7,2.5,5.8,1.8],[7.2,3.6,6.1,2.5],[6.5,3.2,5.1,2.0],[6.4,2.7,5.3,1.9],
    [6.8,3.0,5.5,2.1],[5.7,2.5,5.0,2.0],[5.8,2.8,5.1,2.4],[6.4,3.2,5.3,2.3],
    [6.5,3.0,5.5,1.8],[7.7,3.8,6.7,2.2],[7.7,2.6,6.9,2.3],[6.0,2.2,5.0,1.5],
    [6.9,3.2,5.7,2.3],[5.6,2.8,4.9,2.0],[7.7,2.8,6.7,2.0],[6.3,2.7,4.9,1.8],
    [6.7,3.3,5.7,2.1],[7.2,3.2,6.0,1.8],[6.2,2.8,4.8,1.8],[6.1,3.0,4.9,1.8],
    [6.4,2.8,5.6,2.1],[7.2,3.0,5.8,1.6],[7.4,2.8,6.1,1.9],[7.9,3.8,6.4,2.0],
    [6.4,2.8,5.6,2.2],[6.3,2.8,5.1,1.5],[6.1,2.6,5.6,1.4],[7.7,3.0,6.1,2.3],
    [6.3,3.4,5.6,2.4],[6.4,3.1,5.5,1.8],[6.0,3.0,4.8,1.8],[6.9,3.1,5.4,2.1],
    [6.7,3.1,5.6,2.4],[6.9,3.1,5.1,2.3],[5.8,2.7,5.1,1.9],[6.8,3.2,5.9,2.3],
    [6.7,3.3,5.7,2.5],[6.7,3.0,5.2,2.3],[6.3,2.5,5.0,1.9],[6.5,3.0,5.2,2.0],
    [6.2,3.4,5.4,2.3],[5.9,3.0,5.1,1.8],
], dtype=float)
IRIS_LABELS = np.array([0]*50 + [1]*50 + [2]*50)
IRIS_TARGET_NAMES = ["setosa", "versicolor", "virginica"]
IRIS_FEATURE_NAMES = ["S-Len", "S-Wid", "P-Len", "P-Wid"]

# ─────────────────────────────────────────────────────────────
# 1.  CURSE OF DIMENSIONALITY
# ─────────────────────────────────────────────────────────────
print("\n-- 1. Curse of Dimensionality --")
print("""
Imagine data points in a unit hypercube:
  * 1-D  -> avg distance between random points ~= 0.33
  * 2-D  -> ~= 0.52
  * 10-D -> ~= 1.81   (points become 'equally far from everything')
  * 100-D-> ~= 7.07

As dimensions grow:
  [!] Distance metrics lose meaning
  [!] Models need exponentially more data
  [!] Visualization becomes impossible

Solution -> PCA finds DIRECTIONS of maximum variance and
            projects data into a lower-dimensional subspace.
""")

dims = [1, 2, 5, 10, 20, 50, 100]
avg_distances = []
for d in dims:
    pts = np.random.rand(300, d)
    diffs = pts[None, :, :] - pts[:, None, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    avg_distances.append(dists[np.triu_indices(300, k=1)].mean())

print("Avg pairwise distance vs dimensions:")
for d, dist in zip(dims, avg_distances):
    bar = "#" * int(dist * 10)
    print(f"  {d:>4}-D  {dist:.3f}  {bar}")

# ─────────────────────────────────────────────────────────────
# 2.  PCA CLASS  (pure NumPy)
# ─────────────────────────────────────────────────────────────
class PCAFromScratch:
    """
    PCA via eigendecomposition of the covariance matrix.
    Steps:
      1. Centre data
      2. Compute covariance matrix
      3. Eigendecompose
      4. Sort eigenvectors descending by eigenvalue
      5. Project data
    """
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray) -> "PCAFromScratch":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = np.cov(Xc, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / eigenvalues.sum()
        self._all_eigenvalues = eigenvalues
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Xt: np.ndarray) -> np.ndarray:
        return Xt @ self.components_ + self.mean_


# ─────────────────────────────────────────────────────────────
# 3.  TOY 2-D EXAMPLE
# ─────────────────────────────────────────────────────────────
print("\n-- 2. Toy Example: 2-D data with PCA directions --")
X_toy = np.random.multivariate_normal([3, 2], [[1.5, 1.2], [1.2, 1.2]], 200)
pca_toy = PCAFromScratch(n_components=2)
pca_toy.fit(X_toy)
pc1, pc2 = pca_toy.components_[0], pca_toy.components_[1]
print(f"  PC1 direction : [{pc1[0]:.4f}, {pc1[1]:.4f}]")
print(f"  PC2 direction : [{pc2[0]:.4f}, {pc2[1]:.4f}]")
print(f"  PC1 dot PC2   : {np.dot(pc1, pc2):.2e}  (orthogonal [OK])")
print(f"  Explained var : PC1={pca_toy.explained_variance_ratio_[0]*100:.1f}%  "
      f"PC2={pca_toy.explained_variance_ratio_[1]*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# 4.  IRIS DATASET  4-D -> 2-D
# ─────────────────────────────────────────────────────────────
print("\n-- 3. Iris Dataset: 4-D -> 2-D --")
X_iris = IRIS_DATA.copy()
y_iris = IRIS_LABELS
# Standardize
X_std = (X_iris - X_iris.mean(0)) / X_iris.std(0)

pca_full = PCAFromScratch(n_components=4)
pca_full.fit(X_std)

print("\nExplained Variance per Component:")
cumulative = 0.0
for i, (ev, evr) in enumerate(zip(pca_full.explained_variance_,
                                   pca_full.explained_variance_ratio_)):
    cumulative += evr
    bar = "#" * int(evr * 50)
    print(f"  PC{i+1}  eigenvalue={ev:.4f}  {evr*100:5.1f}%  "
          f"cumulative={cumulative*100:.1f}%  {bar}")

pca_2d = PCAFromScratch(n_components=2)
X_proj = pca_2d.fit_transform(X_std)
print(f"\nOriginal : {X_std.shape}  ->  Projected : {X_proj.shape}")
print(f"2 PCs capture {pca_2d.explained_variance_ratio_.sum()*100:.1f}% of variance")

# ─────────────────────────────────────────────────────────────
# 5.  RECONSTRUCTION ERROR vs k
# ─────────────────────────────────────────────────────────────
print("\n-- 4. Reconstruction Error vs. k --")
errors = []
for k in range(1, 5):
    pk = PCAFromScratch(n_components=k)
    Xr = pk.inverse_transform(pk.fit_transform(X_std))
    mse = np.mean((X_std - Xr) ** 2)
    errors.append(mse)
    print(f"  k={k}  MSE={mse:.6f}")

# ─────────────────────────────────────────────────────────────
# 6.  VERIFY via SVD  (alternative derivation, no sklearn)
# ─────────────────────────────────────────────────────────────
print("\n-- 5. Verification via SVD --")
Xc = X_std - X_std.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
evr_svd = (S**2) / (S**2).sum()

evr_our = np.sort(pca_2d.explained_variance_ratio_)[::-1]
evr_svd_top2 = np.sort(evr_svd[:2])[::-1]

match = np.allclose(evr_our, evr_svd_top2, atol=1e-6)
print(f"  Our EVR (top 2) : {evr_our}")
print(f"  SVD EVR (top 2) : {evr_svd_top2}")
print(f"  Match           : {'[OK]' if match else '[FAIL]'}")

# ─────────────────────────────────────────────────────────────
# 7.  VISUALIZATION
# ─────────────────────────────────────────────────────────────
colors = ["#6366f1", "#06b6d4", "#f59e0b"]
fig = plt.figure(figsize=(18, 12), facecolor="#0a0d14")
fig.suptitle("Week 11 - PCA Analysis Dashboard", fontsize=18,
             color="white", fontweight="bold", y=0.97)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Panel A: Toy 2-D data + PC arrows
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor("#0f1320")
ax_a.scatter(X_toy[:, 0], X_toy[:, 1], alpha=0.5, s=20, color="#6366f1")
cx, cy = pca_toy.mean_
scale = 1.5
for vec, col, label in zip(pca_toy.components_, ["#f59e0b", "#06b6d4"], ["PC1", "PC2"]):
    ev = pca_toy.explained_variance_[0 if label == "PC1" else 1]
    ax_a.annotate("", xy=(cx + vec[0]*scale*ev, cy + vec[1]*scale*ev),
                  xytext=(cx, cy),
                  arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
    ax_a.text(cx + vec[0]*scale*ev*1.12, cy + vec[1]*scale*ev*1.12,
              label, color=col, fontsize=11, fontweight="bold")
ax_a.set_title("Toy 2-D Data + Principal Components", color="white", fontsize=10)
ax_a.tick_params(colors="gray"); ax_a.spines[:].set_edgecolor("#2a2a3e")

# Panel B: Scree Plot
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor("#0f1320")
ks = np.arange(1, 5)
ax_b.bar(ks, pca_full.explained_variance_ratio_ * 100,
         color=["#6366f1", "#8b5cf6", "#06b6d4", "#f59e0b"], width=0.5)
cumline = np.cumsum(pca_full.explained_variance_ratio_) * 100
ax_b.plot(ks, cumline, "o--", color="#f59e0b", linewidth=2, label="Cumulative %")
ax_b.axhline(95, linestyle=":", color="#ef4444", alpha=0.7, label="95% threshold")
ax_b.set_xlabel("Principal Component", color="gray")
ax_b.set_ylabel("Explained Variance (%)", color="gray")
ax_b.set_title("Scree Plot - Iris (4 features)", color="white", fontsize=10)
ax_b.legend(facecolor="#1a1f2e", edgecolor="#2a2a3e", labelcolor="white", fontsize=8)
ax_b.tick_params(colors="gray"); ax_b.spines[:].set_edgecolor("#2a2a3e")

# Panel C: 2-D Projection
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_facecolor("#0f1320")
for cls, col, name in zip([0, 1, 2], colors, IRIS_TARGET_NAMES):
    mask = y_iris == cls
    ax_c.scatter(X_proj[mask, 0], X_proj[mask, 1],
                 s=40, alpha=0.85, color=col, label=name)
ax_c.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", color="gray")
ax_c.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", color="gray")
ax_c.set_title("Iris: 4-D -> 2-D PCA Projection", color="white", fontsize=10)
ax_c.legend(facecolor="#1a1f2e", edgecolor="#2a2a3e", labelcolor="white", fontsize=8)
ax_c.tick_params(colors="gray"); ax_c.spines[:].set_edgecolor("#2a2a3e")

# Panel D: Reconstruction Error
ax_d = fig.add_subplot(gs[1, 0])
ax_d.set_facecolor("#0f1320")
ax_d.plot(range(1, 5), errors, "o-", color="#6366f1", linewidth=2, markersize=8)
ax_d.fill_between(range(1, 5), errors, alpha=0.15, color="#6366f1")
ax_d.set_xlabel("Number of Components (k)", color="gray")
ax_d.set_ylabel("Reconstruction MSE", color="gray")
ax_d.set_title("Reconstruction Error vs. k", color="white", fontsize=10)
ax_d.tick_params(colors="gray"); ax_d.spines[:].set_edgecolor("#2a2a3e")

# Panel E: Covariance Heatmap
ax_e = fig.add_subplot(gs[1, 1])
ax_e.set_facecolor("#0f1320")
cov_iris = np.cov(X_std, rowvar=False)
im = ax_e.imshow(cov_iris, cmap="RdBu_r", vmin=-1, vmax=1)
ax_e.set_xticks(range(4)); ax_e.set_xticklabels(IRIS_FEATURE_NAMES, color="gray", fontsize=8)
ax_e.set_yticks(range(4)); ax_e.set_yticklabels(IRIS_FEATURE_NAMES, color="gray", fontsize=8)
for i in range(4):
    for j in range(4):
        ax_e.text(j, i, f"{cov_iris[i,j]:.2f}", ha="center", va="center",
                  color="white", fontsize=7)
ax_e.set_title("Covariance Matrix (standardized Iris)", color="white", fontsize=10)
plt.colorbar(im, ax=ax_e, fraction=0.046, pad=0.04)

# Panel F: Loadings
ax_f = fig.add_subplot(gs[1, 2])
ax_f.set_facecolor("#0f1320")
loadings = pca_2d.components_
x_pos = np.arange(4)
width = 0.35
ax_f.bar(x_pos - width/2, loadings[0], width, label="PC1", color="#6366f1", alpha=0.85)
ax_f.bar(x_pos + width/2, loadings[1], width, label="PC2", color="#06b6d4", alpha=0.85)
ax_f.set_xticks(x_pos); ax_f.set_xticklabels(IRIS_FEATURE_NAMES, color="gray", fontsize=9)
ax_f.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax_f.set_ylabel("Loading (contribution)", color="gray")
ax_f.set_title("Feature Loadings on PC1 & PC2", color="white", fontsize=10)
ax_f.legend(facecolor="#1a1f2e", edgecolor="#2a2a3e", labelcolor="white", fontsize=8)
ax_f.tick_params(colors="gray"); ax_f.spines[:].set_edgecolor("#2a2a3e")

outfile = "week11_pca_dashboard.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#0a0d14")
print(f"\n[SAVED] {outfile}")

print("""
+--------------------------------------------------------------+
|              PCA - Key Takeaways                             |
+--------------------------------------------------------------+
|  Algorithm:                                                  |
|    1. Standardize features (zero mean, unit variance)        |
|    2. Compute covariance matrix Sigma                        |
|    3. Eigendecompose: Sigma v = lambda v                     |
|    4. Sort eigenvectors by eigenvalue (descending)           |
|    5. Project: Z = X_centered @ W  (W = top-k eigenvecs)    |
|                                                              |
|  Intuition:                                                  |
|    * Eigenvalues  = variance captured by each component      |
|    * Eigenvectors = directions of maximum variance           |
|    * Orthogonal components = no redundancy between PCs       |
|                                                              |
|  When to use PCA:                                            |
|    [Y] High-dimensional data (tabular, images, gene data)    |
|    [Y] Visualize clusters in 2-D / 3-D                       |
|    [Y] Speed up ML by reducing feature count                 |
|    [N] Don't use when feature interpretability matters       |
+--------------------------------------------------------------+
""")

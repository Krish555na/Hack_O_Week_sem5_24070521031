"""
Week 12 - t-SNE: Intuition, Mechanics & Visualization
=======================================================
Topics Covered:
  1. Why PCA isn't always enough (linear vs non-linear structure)
  2. t-SNE intuition step-by-step
  3. The t-distribution trick - solving the crowding problem
  4. KL Divergence - what t-SNE minimizes
  5. t-SNE from scratch (gradient descent, pure NumPy)
  6. Key hyperparameters: perplexity, learning rate, n_iter
  7. PCA vs t-SNE comparison on synthetic datasets
  8. Best practices and pitfalls

Dependencies: numpy, matplotlib (no sklearn/scipy)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

print("=" * 65)
print("  WEEK 12 - t-SNE: Intuition & Visualization")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# BUNDLED IRIS DATA (same as week11, no sklearn needed)
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
IRIS_NAMES  = ["setosa", "versicolor", "virginica"]

# Standardize
X_iris = IRIS_DATA.copy()
X_iris = (X_iris - X_iris.mean(0)) / X_iris.std(0)
y_iris = IRIS_LABELS

# ─────────────────────────────────────────────────────────────
# 1.  WHY PCA ISN'T ALWAYS ENOUGH
# ─────────────────────────────────────────────────────────────
print("""
-- 1. The Limits of PCA --

PCA is LINEAR - it finds straight-line directions of max variance.
It CANNOT unroll:
  * Swiss roll manifolds
  * Concentric rings
  * Spiral clusters

t-SNE (t-distributed Stochastic Neighbor Embedding) is NON-LINEAR.
It preserves LOCAL neighborhood structure in 2-D / 3-D.

Published: van der Maaten & Hinton (2008) - 100k+ citations.
""")

# ─────────────────────────────────────────────────────────────
# 2.  t-SNE INTUITION
# ─────────────────────────────────────────────────────────────
print("""
-- 2. t-SNE Intuition --

HIGH-DIMENSIONAL SPACE:
  "How likely is it that point j is a NEIGHBOR of point i?"
  -> Use a Gaussian bell curve centred on i
  -> Close  points -> HIGH probability
  -> Far    points -> LOW probability
  -> These form matrix P (n x n)

LOW-DIMENSIONAL SPACE (2-D canvas):
  -> Randomly place n points on the canvas
  -> Compute similar probabilities: matrix Q
  -> Use a t-distribution (fatter tails) - the KEY trick

OPTIMIZATION:
  -> Move 2-D points so Q ~= P
  -> Minimize KL(P || Q) via gradient descent
  -> After ~1000 iterations -> clusters emerge!

THE FAT-TAIL t-DISTRIBUTION TRICK:
  In 2-D there is less 'room' than in 784-D, so moderately-far
  points get squished together ('crowding problem').
  The t-distribution's fat tails push these apart naturally.
""")

# ─────────────────────────────────────────────────────────────
# 3.  MATH CORE
# ─────────────────────────────────────────────────────────────
print("""
-- 3. The Math (high-level) --

HIGH-D (Gaussian):
  p(j|i) = exp(-||xi-xj||^2 / 2*sigma^2) / sum_k exp(-||xi-xk||^2 / 2*sigma^2)
  pij = (p(j|i) + p(i|j)) / 2n

  sigma chosen so perplexity(Pi) = user Perplexity (typically 5-50)

LOW-D (Student-t, df=1, i.e. Cauchy):
  qij = (1 + ||yi-yj||^2)^-1 / sum_{k!=l}(1 + ||yk-yl||^2)^-1

Loss (KL Divergence):
  C = KL(P || Q) = sum_ij pij * log(pij / qij)

Gradient:
  dC/dyi = 4 * sum_j (pij - qij)(yi - yj)(1 + ||yi-yj||^2)^-1
""")

# ─────────────────────────────────────────────────────────────
# 4.  t-SNE FROM SCRATCH
# ─────────────────────────────────────────────────────────────
class TSNEFromScratch:
    """
    Simplified t-SNE (pure NumPy).
    O(n^2) - suitable for n < 300.
    """
    def __init__(self, n_components=2, perplexity=30.0,
                 learning_rate=200.0, n_iter=500, random_state=42):
        self.n_components  = n_components
        self.perplexity    = perplexity
        self.learning_rate = learning_rate
        self.n_iter        = n_iter
        self.random_state  = random_state
        self.kl_history_   = []

    @staticmethod
    def _pairwise_sq_dists(X):
        sum_sq = np.sum(X**2, axis=1)
        D = sum_sq[:, None] + sum_sq[None, :] - 2 * X @ X.T
        return np.clip(D, 0, None)

    def _high_dim_affinities(self, X):
        D = self._pairwise_sq_dists(X)
        sigma_sq = 1.0 / (2.0 * np.log(self.perplexity + 1e-10))
        P = np.exp(-D / (2 * sigma_sq))
        np.fill_diagonal(P, 0)
        P /= P.sum()
        P = (P + P.T) / 2
        return np.maximum(P, 1e-12)

    @staticmethod
    def _low_dim_affinities(Y):
        D = TSNEFromScratch._pairwise_sq_dists(Y)
        Q_num = (1 + D) ** -1
        np.fill_diagonal(Q_num, 0)
        Q = Q_num / Q_num.sum()
        return np.maximum(Q, 1e-12), Q_num

    @staticmethod
    def _gradient(P, Q, Q_num, Y):
        PQ   = P - Q
        grad = np.zeros_like(Y)
        for i in range(Y.shape[0]):
            diff    = Y[i] - Y
            grad[i] = 4 * (PQ[i, :, None] * diff * Q_num[i, :, None]).sum(axis=0)
        return grad

    def fit_transform(self, X):
        rng = np.random.default_rng(self.random_state)
        n   = X.shape[0]
        P   = self._high_dim_affinities(X)
        Y   = rng.normal(0, 1e-4, (n, self.n_components))
        vel = np.zeros_like(Y)
        mom = 0.5

        for it in range(self.n_iter):
            Q, Q_num = self._low_dim_affinities(Y)
            kl       = (P * np.log(P / Q)).sum()
            self.kl_history_.append(kl)
            grad     = self._gradient(P, Q, Q_num, Y)
            vel      = mom * vel - self.learning_rate * grad
            Y        = Y + vel
            if it == 100:
                mom = 0.8
            if (it + 1) % 100 == 0:
                print(f"    iter {it+1:4d}/{self.n_iter}  KL={kl:.4f}")

        self.embedding_ = Y
        return Y


# ─────────────────────────────────────────────────────────────
# 5.  PCA from scratch (reused from week11)
# ─────────────────────────────────────────────────────────────
class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
    def fit_transform(self, X):
        Xc  = X - X.mean(0)
        cov = np.cov(Xc, rowvar=False)
        ev, evec = np.linalg.eigh(cov)
        idx  = np.argsort(ev)[::-1]
        evec = evec[:, idx]
        W    = evec[:, :self.n_components]
        return Xc @ W

# ─────────────────────────────────────────────────────────────
# 6.  RUN on Iris
# ─────────────────────────────────────────────────────────────
print("\n-- 4. Running t-SNE (from scratch) on Iris --")
tsne = TSNEFromScratch(n_components=2, perplexity=30,
                        learning_rate=150, n_iter=500)
Y_tsne = tsne.fit_transform(X_iris)
print(f"  Embedding shape: {Y_tsne.shape}")

# PCA 2D for comparison
Y_pca = PCAFromScratch(2).fit_transform(X_iris)

# ─────────────────────────────────────────────────────────────
# 7.  SYNTHETIC NON-LINEAR DATASETS
# ─────────────────────────────────────────────────────────────
print("\n-- 5. Generating non-linear datasets --")

def make_moons(n=100, noise=0.1):
    n_half = n // 2
    t = np.linspace(0, np.pi, n_half)
    X1 = np.column_stack([np.cos(t), np.sin(t)])
    X2 = np.column_stack([1 - np.cos(t), 1 - np.sin(t) - 0.5])
    X  = np.vstack([X1, X2]) + np.random.normal(0, noise, (n, 2))
    y  = np.array([0]*n_half + [1]*n_half)
    return X, y

def make_circles(n=100, r_inner=0.4, noise=0.05):
    n_half  = n // 2
    theta   = np.random.uniform(0, 2*np.pi, n)
    r       = np.array([r_inner]*n_half + [1.0]*n_half)
    X       = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    X      += np.random.normal(0, noise, X.shape)
    y       = np.array([0]*n_half + [1]*n_half)
    return X, y

def make_blobs(n=150, centers=3, std=0.6):
    rng     = np.random.default_rng(7)
    cntrs   = rng.uniform(-3, 3, (centers, 2))
    per_c   = n // centers
    Xs, ys  = [], []
    for i, c in enumerate(cntrs):
        Xs.append(rng.normal(c, std, (per_c, 2)))
        ys.append([i]*per_c)
    return np.vstack(Xs), np.concatenate(ys)

X_moons,   y_moons   = make_moons(100, noise=0.08)
X_circles, y_circles = make_circles(100)
X_blobs,   y_blobs   = make_blobs(150, centers=3)

print("  Datasets ready: moons (100), circles (100), blobs (150)")

# Run t-SNE on moons (embedded in 5-D noise)
np.random.seed(42)
X_moons_5d = np.hstack([X_moons, np.random.randn(100, 3) * 0.3])
print("\n  Running t-SNE on moons (5-D) ...")
tsne_moons = TSNEFromScratch(n_components=2, perplexity=15,
                              learning_rate=100, n_iter=400)
Y_moons_tsne = tsne_moons.fit_transform(X_moons_5d)
Y_moons_pca  = PCAFromScratch(2).fit_transform(X_moons_5d)

# ─────────────────────────────────────────────────────────────
# 8.  PERPLEXITY EFFECT
# ─────────────────────────────────────────────────────────────
print("""
-- 6. Perplexity Hyperparameter --

  Perplexity ~= effective number of neighbors each point cares about

  Low  (5)  -> very local clusters, may fragment real groups
  Med  (30) -> balanced view (most common default)
  High (50) -> more global structure, clusters may merge if too high

  Rule of thumb: 5 <= perplexity <= 50, and n > 3*perplexity

  [!] t-SNE cluster SIZES and DISTANCES between clusters are
      NOT meaningful. Only topology (shape/neighborhood) matters!
""")

# Run t-SNE at 3 perplexities on Iris
perplexity_results = {}
for perp in [5, 30, 50]:
    print(f"  t-SNE perplexity={perp} on Iris ...")
    t = TSNEFromScratch(n_components=2, perplexity=perp,
                         learning_rate=150, n_iter=400)
    perplexity_results[perp] = t.fit_transform(X_iris)

# ─────────────────────────────────────────────────────────────
# 9.  COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print("""
-- 7. PCA vs t-SNE Comparison --

+----------------------+---------------------+---------------------+
| Property             | PCA                 | t-SNE               |
+----------------------+---------------------+---------------------+
| Type                 | Linear              | Non-linear          |
| Preserves            | Global variance     | Local neighbors     |
| Speed                | O(n*d^2) - fast     | O(n^2) - slow       |
| Deterministic        | Yes                 | No (stochastic)     |
| Has transform()      | Yes                 | No                  |
| Cluster distances    | Meaningful          | NOT meaningful      |
| Handles manifolds    | No                  | Yes                 |
+----------------------+---------------------+---------------------+
""")

# ─────────────────────────────────────────────────────────────
# 10.  VISUALIZATION DASHBOARD
# ─────────────────────────────────────────────────────────────
colors_3 = ["#6366f1", "#06b6d4", "#f59e0b"]
colors_2 = ["#6366f1", "#f59e0b"]

fig = plt.figure(figsize=(22, 14), facecolor="#0a0d14")
fig.suptitle("Week 12 - t-SNE Visualization Dashboard", fontsize=18,
             color="white", fontweight="bold", y=0.97)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.32)

def style_ax(ax, title):
    ax.set_facecolor("#0f1320")
    ax.set_title(title, color="white", fontsize=9)
    ax.tick_params(colors="gray")
    ax.spines[:].set_edgecolor("#2a2a3e")

# Panel A: PCA on Iris
ax_a = fig.add_subplot(gs[0, 0])
style_ax(ax_a, "PCA - Iris (4-D -> 2-D)")
for cls, col, name in zip([0,1,2], colors_3, IRIS_NAMES):
    m = y_iris == cls
    ax_a.scatter(Y_pca[m, 0], Y_pca[m, 1], s=35, alpha=0.85, color=col, label=name)
ax_a.legend(facecolor="#1a1f2e", edgecolor="#2a2a3e", labelcolor="white", fontsize=7)

# Panel B: t-SNE (scratch) on Iris
ax_b = fig.add_subplot(gs[0, 1])
style_ax(ax_b, "t-SNE (scratch) - Iris")
for cls, col, name in zip([0,1,2], colors_3, IRIS_NAMES):
    m = y_iris == cls
    ax_b.scatter(Y_tsne[m, 0], Y_tsne[m, 1], s=35, alpha=0.85, color=col, label=name)
ax_b.legend(facecolor="#1a1f2e", edgecolor="#2a2a3e", labelcolor="white", fontsize=7)

# Panel C: KL convergence
ax_c = fig.add_subplot(gs[0, 2])
style_ax(ax_c, "KL Divergence During Optimization")
ax_c.plot(tsne.kl_history_, color="#6366f1", linewidth=1.5)
ax_c.fill_between(range(len(tsne.kl_history_)), tsne.kl_history_, alpha=0.15, color="#6366f1")
ax_c.set_xlabel("Iteration", color="gray")
ax_c.set_ylabel("KL Divergence (loss)", color="gray")

# Panel D: Pitfalls
ax_d = fig.add_subplot(gs[0, 3])
style_ax(ax_d, "Key Pitfalls")
ax_d.axis("off")
note = ("t-SNE PITFALLS\n\n"
        "[X] Cluster SIZES\n"
        "    != data density\n\n"
        "[X] Inter-cluster DIST\n"
        "    != similarity\n\n"
        "[X] Different runs =\n"
        "    different layouts\n\n"
        "[OK] Use PCA/UMAP\n"
        "     for preprocessing\n\n"
        "[OK] Only trust LOCAL\n"
        "     neighborhood shape")
ax_d.text(0.05, 0.95, note, transform=ax_d.transAxes, fontsize=9,
          color="#fca5a5", va="top", fontfamily="monospace",
          bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a0d0d",
                    edgecolor="#ef4444", alpha=0.85))

# Panel E: PCA on moons
ax_e = fig.add_subplot(gs[1, 0])
style_ax(ax_e, "PCA - Moons (5-D -> 2-D)")
for cls, col in zip([0, 1], colors_2):
    m = y_moons == cls
    ax_e.scatter(Y_moons_pca[m, 0], Y_moons_pca[m, 1], s=30, alpha=0.85, color=col)

# Panel F: t-SNE on moons
ax_f = fig.add_subplot(gs[1, 1])
style_ax(ax_f, "t-SNE - Moons (5-D -> 2-D)")
for cls, col in zip([0, 1], colors_2):
    m = y_moons == cls
    ax_f.scatter(Y_moons_tsne[m, 0], Y_moons_tsne[m, 1], s=30, alpha=0.85, color=col)

# Panels G, H: perplexity comparison
for idx, perp in enumerate([5, 50]):
    ax = fig.add_subplot(gs[1, 2 + idx])
    style_ax(ax, f"t-SNE perplexity={perp} - Iris")
    Y_p = perplexity_results[perp]
    for cls, col, name in zip([0,1,2], colors_3, IRIS_NAMES):
        m = y_iris == cls
        ax.scatter(Y_p[m, 0], Y_p[m, 1], s=30, alpha=0.85, color=col, label=name)
    ax.legend(facecolor="#1a1f2e", edgecolor="#2a2a3e", labelcolor="white", fontsize=6)

outfile = "week12_tsne_dashboard.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#0a0d14")
print(f"\n[SAVED] {outfile}")

print("""
+--------------------------------------------------------------+
|              t-SNE - Key Takeaways                           |
+--------------------------------------------------------------+
|  How it works:                                               |
|    1. Compute pij = Gaussian neighbor probs in high-D        |
|    2. Init random 2-D embedding Y                            |
|    3. Compute qij = Cauchy neighbor probs in low-D           |
|    4. Minimize KL(P || Q) via gradient descent               |
|    5. t-distribution fat tails solve crowding problem        |
|                                                              |
|  Best practices:                                             |
|    * Always pre-scale features (StandardScaler)              |
|    * Pre-reduce with PCA to ~30-50 dims for speed            |
|    * Try multiple perplexity values (5, 30, 50)              |
|    * Run multiple times to check stability                   |
|                                                              |
|  What to TRUST in t-SNE output:                              |
|    [Y] Whether clusters exist                                |
|    [Y] Local point neighborhoods                             |
|    [N] Cluster sizes   [N] Inter-cluster distances           |
+--------------------------------------------------------------+
""")

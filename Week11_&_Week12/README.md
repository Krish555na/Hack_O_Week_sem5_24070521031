# Folder 6: Dimensionality Reduction (Weeks 11 & 12)

## Overview
This folder covers **dimensionality reduction** — the art of representing high-dimensional data in fewer dimensions while preserving the structure that matters.

## Projects

### Week 11 — PCA (Principal Component Analysis)
**File**: `week11_pca/pca_from_scratch.py`

Built entirely from NumPy with step-by-step derivation:

| Step | What Happens |
|------|-------------|
| 1. Centre | Subtract mean so data is zero-centred |
| 2. Covariance | Build Σ — captures feature relationships |
| 3. Eigendecompose | Find variance-maximizing directions |
| 4. Sort & Select | Keep top-k eigenvectors (most variance) |
| 5. Project | Multiply centred data by component matrix |

**Key outputs:**
- Scree plot (explained variance per PC)
- 2-D Iris projection
- Reconstruction error vs k
- Feature loadings bar chart
- Covariance heatmap

### Week 12 — t-SNE (Intuition & Application)
**File**: `week12_tsne/tsne_intuition.py`

| Concept | Explanation |
|---------|------------|
| High-D probabilities | Gaussian: close points → high P |
| Low-D probabilities | Cauchy (t-dist): fat tails fix crowding |
| Optimization | Gradient descent on KL(P ∥ Q) |
| Perplexity | Controls effective neighborhood size (5–50) |

**Key outputs:**
- Side-by-side PCA vs t-SNE on Iris
- KL divergence convergence curve
- Perplexity comparison (5 / 30 / 50) on Digits
- Common pitfalls panel

## Running

```bash
# From the folder root:
cd 06_dimensionality_reduction_week11_12

# Week 11 — PCA
python week11_pca/pca_from_scratch.py

# Week 12 — t-SNE
python week12_tsne/tsne_intuition.py

# All tests
python -m pytest tests/test_week11_12.py -v
```

## PCA vs t-SNE Quick Reference

| | PCA | t-SNE |
|---|---|---|
| **Type** | Linear | Non-linear |
| **Preserves** | Global variance | Local neighborhoods |
| **Speed** | Fast O(nd²) | Slow O(n²) |
| **Deterministic** | ✅ | ❌ |
| **Has `transform()`** | ✅ | ❌ |
| **Cluster distances** | Meaningful | NOT meaningful |
| **Best for** | Preprocessing, EDA | Visualization |

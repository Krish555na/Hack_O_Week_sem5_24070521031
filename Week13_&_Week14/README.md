# Weeks 13 & 14: Ensemble Methods & Regularization Foundations

A rigorous, mathematics-backed, and completely self-contained implementation of advanced ensemble learning techniques and generalization theory from scratch using pure NumPy and Matplotlib.

---

## Folder Architecture

```
Week13_&_Week14/
├── README.md
├── __init__.py
├── week13_ensemble_methods/
│   ├── __init__.py
│   ├── bagging_from_scratch.py              # Bootstrap Aggregating, Decision Trees, OOB Error, Random Forest
│   ├── boosting_xgboost_lightgbm.py         # GBM, XGBoost 2nd-order Taylor expansion, LightGBM GOSS & Histograms
│   └── test_week13.py                       # Automated unit tests for Week 13
└── week14_bias_variance_regularization/
    ├── __init__.py
    ├── bias_variance_decomposition.py       # Monte Carlo Bias^2 + Variance + Noise analytical decomposition
    ├── overfitting_underfitting_diagnostics.py # Learning curves, K-Fold CV, early stopping engine
    ├── regularization_l1_l2.py              # Ridge (L2), Lasso (L1 soft-thresholding), ElasticNet
    └── test_week14.py                       # Automated unit tests for Week 14
```

---

## Week 13: Ensemble Methods (Bagging & Boosting)

### 1. Bagging & Random Forest (`bagging_from_scratch.py`)
- **Bootstrap Sampling**: Sampling $N$ points with replacement; theoretically includes $1 - 1/e \approx 63.2\%$ unique points, leaving $\sim 36.8\%$ as Out-Of-Bag (OOB) samples.
- **OOB Score**: Provides an unbiased estimate of generalization error without requiring a dedicated validation split.
- **Random Forest**: Feature subspacing ($\sqrt{p}$ features considered per split) to de-correlate individual trees and maximize ensemble variance reduction.

### 2. Boosting: Gradient Boosting, XGBoost & LightGBM (`boosting_xgboost_lightgbm.py`)
- **Gradient Boosting Machine (GBM)**: Sequential weak learners fitting negative pseudo-residuals $r_{im} = -\left[\frac{\partial L}{\partial F}\right]$ with learning rate shrinkage $\eta$.
- **XGBoost (Extreme Gradient Boosting)**:
  - 2nd-order Taylor expansion of arbitrary differentiable loss functions using gradients $g_i$ and hessians $h_i$.
  - Optimal leaf weight: $w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$.
  - Split Gain formula with tree complexity penalization:
    $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
- **LightGBM Innovations**:
  - **GOSS (Gradient-based One-Side Sampling)**: Retains top $a$ fraction of large gradients and randomly samples $b$ fraction of small gradients with amplification weight $\frac{1-a}{b}$.
  - **Histogram-based Binning**: Continuous variables mapped into discrete bins ($K \le 256$) for cache efficiency and $O(K \times D)$ split speed.
  - **Leaf-wise (Best-First) Tree Growth**: Expands the leaf offering maximum loss reduction.

---

## Week 14: Generalization Dynamics & Regularization

### 1. Bias-Variance Decomposition (`bias_variance_decomposition.py`)
- **Mathematical Decomposition**:
  $$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2(\hat{f}(x)) + \text{Var}(\hat{f}(x)) + \sigma^2$$
- Monte Carlo simulations over 100 resampled datasets proving how polynomial degrees transition from High Bias (Underfitting) to the U-shaped sweet spot, to High Variance (Overfitting).

### 2. Diagnostics & Early Stopping (`overfitting_underfitting_diagnostics.py`)
- **Learning Curves & Validation Curves**: High bias vs high variance gap detection.
- **Early Stopping**: Epoch-by-epoch validation monitoring with patience counter and best-checkpoint restoration.
- **K-Fold Cross-Validation**: Data-efficient validation from scratch.

### 3. Regularization: L1, L2 & ElasticNet (`regularization_l1_l2.py`)
- **Ridge (L2 / Tikhonov)**: Closed-form solution $(X^T X + \lambda I)^{-1} X^T y$. Continuous asymptotic shrinkage towards zero.
- **Lasso (L1)**: Coordinate Descent via proximal Soft-Thresholding operator $S_\lambda(z) = \text{sign}(z) \max(|z| - \lambda, 0)$. Induces exact zeros (automatic feature selection) due to diamond corner intersections.
- **ElasticNet**: Combines $L_1$ and $L_2$ penalties to handle collinear features cleanly.

---

## Running Scripts & Tests

### Execute Scripts
```powershell
python week13_ensemble_methods/bagging_from_scratch.py
python week13_ensemble_methods/boosting_xgboost_lightgbm.py
python week14_bias_variance_regularization/bias_variance_decomposition.py
python week14_bias_variance_regularization/overfitting_underfitting_diagnostics.py
python week14_bias_variance_regularization/regularization_l1_l2.py
```

### Run Automated Unit Tests
```powershell
python week13_ensemble_methods/test_week13.py
python week14_bias_variance_regularization/test_week14.py
```

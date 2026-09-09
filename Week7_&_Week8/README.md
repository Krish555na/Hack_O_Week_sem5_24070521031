# Folder 4: Regression & Classification Suite (Weeks 7 & 8)

This folder contains mathematical and algorithmic implementations of the primary supervised learning paradigms: Continuous Target Prediction (Regression) and Discrete Class Prediction (Classification).

## Projects Included

### 1. Week 7: Regression Suite
- **Path**: `week7_regression/`
- **Topics**:
  - Linear Regression with dual solvers: Closed-form Normal Equation $(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ and Batch Gradient Descent.
  - Regression loss metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ coefficient of determination.
  - Polynomial feature expansion from scratch without black-box libraries, illustrating underfitting (high bias) vs overfitting (high variance).
  - Regularized linear regression:
    - Ridge (L2 penalty) via analytical closed-form with unregularized bias term.
    - Lasso (L1 penalty) solved via Coordinate Descent with Soft-Thresholding operator to produce sparse zero weights.
  - Housing price prediction benchmark and regularization path visualizations.
- **Run**:
  ```powershell
  python linear_regression.py
  python polynomial_regression.py
  python regularization_ridge_lasso.py
  python regression_demo.py
  python -m unittest test_week7.py
  ```

### 2. Week 8: Classification Suite
- **Path**: `week8_classification/`
- **Topics**:
  - Binary Logistic Regression with Sigmoid hypothesis, Log Loss (Binary Cross-Entropy), and gradient updates.
  - Multiclass Logistic Regression using One-vs-Rest (OvR) architecture.
  - K-Nearest Neighbors (KNN) classifier with Euclidean, Manhattan, and Minkowski distance metrics, uniform voting and distance-weighted voting.
  - Complete classification metrics from scratch: Confusion Matrix, Accuracy, Precision, Recall, Specificity, F1-Score, and ROC Curve with trapezoidal AUC calculation.
  - Non-linear decision boundary comparison visualizer (Logistic Regression vs. KNN k=1 vs. KNN k=15).
- **Run**:
  ```powershell
  python logistic_regression.py
  python knn_classifier.py
  python metrics.py
  python classification_demo.py
  python -m unittest test_week8.py
  ```

# Folder 3: Math Foundations for Machine Learning (Weeks 5 & 6)

This folder contains computational and intuitive implementations of the essential mathematics powering modern Machine Learning and Deep Learning: Linear Algebra and Calculus.

## Projects Included

### 1. Week 5: Linear Algebra
- **Path**: `week5_linear_algebra/`
- **Topics**:
  - Vector algebra ($L_1, L_2, L_\infty$ norms, unit normalization).
  - Dot product, cosine similarity, angle calculations, and orthogonality.
  - Matrix operations (from-scratch matrix multiplication vs. LAPACK NumPy `@`, transpose, determinants, analytical inversion).
  - Geometric transformations (2D rotation, shearing) and orthogonal subspace projections.
  - Gram-Schmidt orthonormalization algorithm.
  - Eigenvalues and Eigenvectors intuition ($Av = \lambda v$), Power Iteration algorithm from scratch, and PCA covariance eigendecomposition.
  - Visualizer showing eigenvector preservation under linear transformation.
- **Run**:
  ```powershell
  python vector_matrix_ops.py
  python transformations_and_projections.py
  python eigenvalues_intuition.py
  python visualize_transformations.py
  python -m unittest test_week5.py
  ```

### 2. Week 6: Calculus & Gradients
- **Path**: `week6_calculus_gradients/`
- **Topics**:
  - Numerical differentiation (forward, backward, central difference) and approximation order.
  - Analytical derivatives for ML activations (Sigmoid, ReLU, Tanh) and loss functions.
  - Gradient checking: Rigorous numerical vs. analytical relative error verification.
  - Automatic differentiation engine from scratch (micrograd-style `Scalar` DAG backpropagation).
  - Chain rule backpropagation through an artificial neuron.
  - Comparative optimizer benchmark: Vanilla Gradient Descent, Momentum, and Adam on non-convex/convex surfaces with trajectory contour visualization.
- **Run**:
  ```powershell
  python derivatives_and_gradients.py
  python autograd_engine.py
  python gradient_descent_optimizers.py
  python -m unittest test_week6.py
  ```

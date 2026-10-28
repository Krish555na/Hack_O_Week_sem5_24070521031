# Weeks 15 & 16: Model Deployment, Perceptrons & Deep Neural Networks from Scratch

A comprehensive, mathematically rigorous implementation of deep learning foundations and production deployment from scratch using pure NumPy, Joblib, and Matplotlib.

---

## Folder Architecture

```
Week15_&_Week16/
├── README.md
├── __init__.py
├── week15_deployment_perceptrons_backprop/
│   ├── __init__.py
│   ├── model_persistence_deployment.py    # Pickle vs Joblib, metadata packaging, production inference service
│   ├── perceptron_and_activations.py       # Rosenblatt Perceptron, XOR paradox, Sigmoid, Tanh, ReLU, LeakyReLU, Softmax
│   ├── forward_backward_propagation.py     # 2-layer backprop from scratch, chain rule, numerical gradient check
│   └── test_week15.py                      # Automated unit tests for Week 15
└── week16_loss_optimizers_neuralnet/
    ├── __init__.py
    ├── loss_functions.py                   # MSE, BCE, and CCE derivations and gradient mechanics
    ├── optimizers_from_scratch.py          # SGD, Momentum, RMSprop, Adam on steep ravine surfaces
    ├── neural_network_from_scratch.py      # Modular Deep Neural Network framework (Dense, Activations, Mini-batch, Save/Load)
    └── test_week16.py                      # Automated unit tests for Week 16
```

---

## Week 15: Deployment Basics, Perceptrons & Backpropagation

### 1. Model Deployment Basics (`model_persistence_deployment.py`)
- **Offline vs Online Inference**: Decoupling compute-heavy gradient descent from sub-millisecond REST serving.
- **Pickle vs Joblib**: Benchmarks serialization speed and compressed file footprints for large NumPy parameter arrays.
- **Production Packaging**: Bundles model weights, scaler means/variances, feature names, version strings, and confidence thresholds.
- **Microservice Simulation**: Validates input JSON payloads and measures microsecond inference latency.

### 2. The Perceptron & Activations (`perceptron_and_activations.py`)
- **Frank Rosenblatt's Perceptron (1958)**: Threshold activation with Hebbian update rule $w \leftarrow w + \eta (y - \hat{y}) x$.
- **The XOR Paradox**: Geometric proof of linear inseparability and why single-layer perceptrons triggered the first AI Winter.
- **Modern Activations & Derivatives**:
  - **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$, peak derivative $= 0.25$ (vanishing gradient danger).
  - **Tanh**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, zero-centered, peak derivative $= 1.0$.
  - **ReLU**: $\max(0, z)$, solves vanishing gradients with unit derivative for positive activations.
  - **LeakyReLU**: $\max(\alpha z, z)$, solves the dying ReLU problem.
  - **Softmax**: Multi-class normalized probability distribution.

### 3. Forward & Backward Propagation (`forward_backward_propagation.py`)
- **Computational Graph (2-Layer)**: Input $\to$ Hidden (ReLU) $\to$ Output (Sigmoid).
- **Exact Analytical Chain Rule**:
  $$\frac{\partial L}{\partial Z^{[2]}} = A^{[2]} - Y, \quad \frac{\partial L}{\partial W^{[2]}} = \frac{1}{m} (A^{[1]})^T dZ^{[2]}$$
  $$\frac{\partial L}{\partial Z^{[1]}} = (dZ^{[2]} (W^{[2]})^T) \odot \text{ReLU}'(Z^{[1]}), \quad \frac{\partial L}{\partial W^{[1]}} = \frac{1}{m} X^T dZ^{[1]}$$
- **Numerical Gradient Checking**: Finite difference symmetric quotient verification with relative error $< 10^{-7}$.
- **XOR Solved**: 100% convergence using multi-layer non-linear representation.

---

## Week 16: Loss Functions, Optimizers & Deep Neural Network

### 1. Loss Functions (`loss_functions.py`)
- **Mean Squared Error (MSE)**: $L = \frac{1}{2m} \sum \|y - \hat{y}\|^2$.
- **Binary Cross-Entropy (BCE)**: $L = -\frac{1}{m} \sum [y \log \hat{y} + (1-y) \log(1-\hat{y})]$. Shows why BCE avoids saturation plateaus when combined with Sigmoid ($dL/dz = \frac{1}{m}(\hat{y} - y)$).
- **Categorical Cross-Entropy (CCE)**: Multi-class log loss combined with Softmax.

### 2. Optimizers from Scratch (`optimizers_from_scratch.py`)
- **SGD**: Standard negative gradient steps.
- **Momentum**: Velocity accumulation $v_t = \beta v_{t-1} + (1-\beta) g_t$ to accelerate through ravines.
- **RMSprop**: Second-moment scaling $s_t = \beta s_{t-1} + (1-\beta) g_t^2$ for adaptive coordinate learning rates.
- **Adam (Adaptive Moment Estimation)**: First and second moment tracking with bias correction $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ and $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$.
- **Ravine Benchmark**: Tracks 2D convergence trajectories on high-anisotropy loss surfaces.

### 3. Deep Neural Network Framework (`neural_network_from_scratch.py`)
- Modular object-oriented architecture: `Dense`, `ReLU`, `Sigmoid`, `Tanh`.
- Xavier & He parameter initialization to maintain unit activation variance across deep layers.
- Mini-batch gradient descent loop with dataset shuffling.
- Model persistence (`save()` / `load()`) with Joblib.
- Solves complex non-linear concentric circle manifolds with $>98\%$ accuracy.

---

## Running the Demos & Unit Tests

### Execute Demonstrations & Dashboards
```powershell
python "week15_deployment_perceptrons_backprop/model_persistence_deployment.py"
python "week15_deployment_perceptrons_backprop/perceptron_and_activations.py"
python "week15_deployment_perceptrons_backprop/forward_backward_propagation.py"
python "week16_loss_optimizers_neuralnet/loss_functions.py"
python "week16_loss_optimizers_neuralnet/optimizers_from_scratch.py"
python "week16_loss_optimizers_neuralnet/neural_network_from_scratch.py"
```

### Run Automated Unit Tests
```powershell
python "week15_deployment_perceptrons_backprop/test_week15.py"
python "week16_loss_optimizers_neuralnet/test_week16.py"
```

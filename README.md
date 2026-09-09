# 10-Week Fullstack AI/ML & Web Development Curriculum

Welcome to the comprehensive, 10-week Fullstack and Machine Learning Engineering curriculum. This repository is structured into **5 core folders**, taking you from interactive frontend applications to mathematical foundations, classical algorithms from first principles, and production ML pipelines.

---

## Curriculum Structure

```
ai_ml_fullstack_curriculum/
│
├── 01_web_development_week1_2/           # [WEEKS 1 & 2] Interactive Web Applications
│   ├── week1_library_management/         # AuraLib: Library Management System (HTML5, CSS3, Vanilla JS)
│   │   ├── index.html
│   │   ├── styles.css
│   │   └── app.js
│   └── week2_whiteboard/                 # AuraBoard: Collaborative Studio Whiteboard (Canvas, Undo/Redo, Vectors)
│       ├── index.html
│       ├── styles.css
│       └── app.js
│
├── 02_python_data_essentials_week3_4/    # [WEEKS 3 & 4] Python Core & Data Wrangling
│   ├── week3_python_essentials/          # OOP Hierarchy, Dunder Methods, Comprehensions, Streaming Generators
│   │   ├── inventory_oop.py
│   │   ├── comprehensions_generators.py
│   │   └── test_week3.py
│   └── week4_numpy_pandas_dataviz/       # NumPy Vectorization, Relational Pandas Pipelines, Seaborn/Matplotlib
│       ├── numpy_operations.py
│       ├── pandas_pipeline.py
│       ├── visualizations.py
│       └── test_week4.py
│
├── 03_math_foundations_ml_week5_6/       # [WEEKS 5 & 6] Mathematics for Machine Learning
│   ├── week5_linear_algebra/             # Vectors, Dot Products, Projections, Gram-Schmidt, Eigenvalues & PCA
│   │   ├── vector_matrix_ops.py
│   │   ├── transformations_and_projections.py
│   │   ├── eigenvalues_intuition.py
│   │   ├── visualize_transformations.py
│   │   └── test_week5.py
│   └── week6_calculus_gradients/         # Numerical/Analytical Derivatives, Autograd Engine (DAG Backpropagation)
│       ├── derivatives_and_gradients.py
│       ├── autograd_engine.py
│       ├── gradient_descent_optimizers.py
│       └── test_week6.py
│
├── 04_regression_classification_week7_8/ # [WEEKS 7 & 8] Supervised Statistical Learning from Scratch
│   ├── week7_regression/                 # Normal Equation, Gradient Descent, Polynomial Expansion, Ridge & Lasso
│   │   ├── linear_regression.py
│   │   ├── polynomial_regression.py
│   │   ├── regularization_ridge_lasso.py
│   │   ├── regression_demo.py
│   │   └── test_week7.py
│   └── week8_classification/             # Binary/Multiclass Logistic Regression, KNN, Confusion Matrix, ROC-AUC
│       ├── logistic_regression.py
│       ├── knn_classifier.py
│       ├── metrics.py
│       ├── classification_demo.py
│       └── test_week8.py
│
├── 05_scikit_learn_clustering_week9_10/  # [WEEKS 9 & 10] Production Scikit-learn & Unsupervised Clustering
│   ├── week9_sklearn_workflow/           # ColumnTransformer Preprocessing, Pipeline, GridSearch, Interpretation
│   │   ├── pipeline_churn_prediction.py
│   │   ├── evaluation_and_interpretation.py
│   │   └── test_week9.py
│   └── week10_clustering/                # K-Means++ from scratch, Agglomerative Hierarchical & DBSCAN
│       ├── kmeans_clustering.py
│       ├── hierarchical_clustering.py
│       ├── dbscan_clustering.py
│       ├── clustering_benchmark.py
│       └── test_week10.py
│
├── requirements.txt                      # Python dependencies
└── run_all_tests.py                      # Master automated verification test runner
```

---

## Quick-Start Guide

### 1. Environment Setup
Install the verified dependencies into your Python environment:
```powershell
python -m pip install -r requirements.txt
```

### 2. Launching the Web Applications (Weeks 1 & 2)
The web projects are completely self-contained and require no build tools:
- **Week 1 (Library Management System)**:
  ```powershell
  python -m http.server 8080 --directory 01_web_development_week1_2/week1_library_management
  ```
  Open `http://localhost:8080` in your browser.
- **Week 2 (Whiteboard Suite)**:
  ```powershell
  python -m http.server 8081 --directory 01_web_development_week1_2/week2_whiteboard
  ```
  Open `http://localhost:8081` in your browser.

### 3. Running All Automated Tests
To run all test suites across all 10 weeks simultaneously:
```powershell
python run_all_tests.py
```

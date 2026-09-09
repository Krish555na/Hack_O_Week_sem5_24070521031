# Folder 5: Scikit-learn Workflow & Clustering Suite (Weeks 9 & 10)

This folder contains modern production Machine Learning pipelines with Scikit-learn, along with unsupervised learning and density clustering algorithms implemented from scratch.

## Projects Included

### 1. Week 9: Scikit-learn Workflow
- **Path**: `week9_sklearn_workflow/`
- **Topics**:
  - Modular preprocessing architecture using `ColumnTransformer`:
    - Numerical: `SimpleImputer(strategy='median')` -> `StandardScaler()`
    - Categorical: `SimpleImputer(strategy='most_frequent')` -> `OneHotEncoder(handle_unknown='ignore')`
  - Composite `Pipeline` chaining preprocessing with a tree ensemble (`GradientBoostingClassifier`).
  - Cross-validated hyperparameter tuning using `GridSearchCV` with `StratifiedKFold`.
  - Model serialization and deserialization using `joblib`.
  - Production prediction on un-preprocessed JSON / dictionary customer records.
  - Comprehensive evaluation and interpretation:
    - ROC Curve and Confusion Matrix
    - Gini tree-based feature importances mapped back to one-hot column names
    - Test-set permutation feature importance.
- **Run**:
  ```powershell
  python pipeline_churn_prediction.py
  python evaluation_and_interpretation.py
  python -m unittest test_week9.py
  ```

### 2. Week 10: Clustering Suite
- **Path**: `week10_clustering/`
- **Topics**:
  - K-Means algorithm from scratch:
    - K-Means++ smart seeding ($D(x)^2$ sampling)
    - Lloyd's expectation-maximization updates
    - Inertia (WCSS) tracking
    - Silhouette Coefficient calculated from pairwise distance definitions.
  - Agglomerative Hierarchical Clustering from scratch:
    - Pairwise distance matrix updates
    - Single, Complete, and Average linkage criteria
    - Dendrogram visual tree generation using SciPy linkage.
  - DBSCAN (Density-Based Spatial Clustering) from scratch:
    - $\epsilon$-neighborhood search
    - Core points, border points, and noise detection (labeled `-1`)
    - Cluster expansion via density connectivity.
  - Comprehensive side-by-side benchmark comparing K-Means, Agglomerative, and DBSCAN on difficult geometries (concentric rings, interlocking moons, spherical blobs).
- **Run**:
  ```powershell
  python kmeans_clustering.py
  python hierarchical_clustering.py
  python dbscan_clustering.py
  python clustering_benchmark.py
  python -m unittest test_week10.py
  ```

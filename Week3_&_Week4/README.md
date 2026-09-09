# Folder 2: Python Data Essentials (Weeks 3 & 4)

This folder contains Python core engineering and high-performance data processing pipelines.

## Projects Included

### 1. Week 3: Python Essentials
- **Path**: `week3_python_essentials/`
- **Topics**:
  - Abstract Base Classes (`abc.ABC`), custom exceptions, encapsulation, and property getters/setters.
  - Python Data Model dunder/magic methods: `__len__`, `__getitem__`, `__contains__`, `__iter__`, `__repr__`, `__str__`.
  - Comprehensive comprehensions (nested lists, dictionary mappings, set deduplication).
  - Memory-efficient streaming generators (`yield`), generator expressions, and batching pipelines.
  - Custom decorators (`@timing_decorator`, `@memoize`).
- **Run**:
  ```powershell
  python inventory_oop.py
  python comprehensions_generators.py
  python -m unittest test_week3.py
  ```

### 2. Week 4: NumPy, Pandas & Data Visualization
- **Path**: `week4_numpy_pandas_dataviz/`
- **Topics**:
  - N-dimensional arrays, slicing, boolean masking, strided views.
  - Broadcasting rules and feature standardization (Z-score).
  - SIMD vectorization micro-benchmarks vs pure Python loops.
  - Relational DataFrame pipelines: null imputation, Left/Inner joins, multi-level hierarchical GroupBy aggregations, pivot tables.
  - High-resolution Seaborn & Matplotlib visualizations (KDE distributions, correlation heatmaps, categorical boxplots, relational scatterplots).
- **Run**:
  ```powershell
  python numpy_operations.py
  python pandas_pipeline.py
  python visualizations.py
  python -m unittest test_week4.py
  ```

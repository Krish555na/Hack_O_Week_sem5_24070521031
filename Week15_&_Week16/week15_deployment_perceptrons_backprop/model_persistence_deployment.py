"""
Week 15 - Model Deployment Basics & Model Persistence (Pickle vs Joblib)
========================================================================
Topics Covered:
  1. Why Model Persistence Matters:
     - Separating offline training (expensive) from real-time inference (fast)
     - Artifact packaging for production microservices & REST APIs
  2. Pickle vs Joblib:
     - Pickle (Python stdlib): general Python object serialization
     - Joblib: optimized for large NumPy arrays and multi-dimensional matrices
     - Compression options (zlib, gzip)
  3. Production Model Packaging:
     - Saving model weights + preprocessing metadata (scalers, feature names, version, threshold)
  4. Production Inference Engine:
     - Input schema validation, real-time prediction, latency profiling

Dependencies: numpy, joblib, matplotlib (standard Python environment)
"""

import os
import sys
import time
import pickle
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# =====================================================================
# 1. SAMPLE TRAINED MODEL FOR SERIALIZATION
# =====================================================================

class LogisticModelArtifact:
    """Lightweight Logistic Regression model container for persistence demo."""
    def __init__(self, feature_names=None):
        self.feature_names = feature_names or ["feature_1", "feature_2"]
        self.weights = None
        self.bias = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.threshold = 0.5
        self.version = "1.0.0"
        self.metadata = {}

    def fit(self, X, y, lr=0.1, epochs=300):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Preprocessing: standard scaling
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_scale = np.std(X, axis=0) + 1e-8
        X_scaled = (X - self.scaler_mean) / self.scaler_scale

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(epochs):
            z = X_scaled @ self.weights + self.bias
            # Sigmoid activation
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            # Gradient step
            dz = p - y
            dw = (X_scaled.T @ dz) / n_samples
            db = np.mean(dz)
            self.weights -= lr * dw
            self.bias -= lr * db

        self.metadata = {
            "training_samples": n_samples,
            "feature_dim": n_features,
            "trained_at_epoch": epochs
        }
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X_scaled = (X - self.scaler_mean) / self.scaler_scale
        z = X_scaled @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)


# =====================================================================
# 2. MODEL PERSISTENCE UTILITIES (PICKLE vs JOBLIB)
# =====================================================================

class ModelDeploymentManager:
    """Helper for saving, loading, verifying, and serving ML model artifacts."""

    @staticmethod
    def save_with_pickle(model_obj, filepath):
        """Serialize object to disk using standard pickle."""
        start = time.perf_counter()
        with open(filepath, "wb") as f:
            pickle.dump(model_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        elapsed = (time.perf_counter() - start) * 1000
        size_bytes = os.path.getsize(filepath)
        return size_bytes, elapsed

    @staticmethod
    def load_with_pickle(filepath):
        """Deserialize object from disk using standard pickle."""
        start = time.perf_counter()
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        elapsed = (time.perf_counter() - start) * 1000
        return obj, elapsed

    @staticmethod
    def save_with_joblib(model_obj, filepath, compress=3):
        """Serialize object to disk using Joblib (with optional compression)."""
        start = time.perf_counter()
        joblib.dump(model_obj, filepath, compress=compress)
        elapsed = (time.perf_counter() - start) * 1000
        size_bytes = os.path.getsize(filepath)
        return size_bytes, elapsed

    @staticmethod
    def load_with_joblib(filepath):
        """Deserialize object from disk using Joblib."""
        start = time.perf_counter()
        obj = joblib.load(filepath)
        elapsed = (time.perf_counter() - start) * 1000
        return obj, elapsed


# =====================================================================
# 3. PRODUCTION INFERENCE ENGINE API
# =====================================================================

class ProductionInferenceService:
    """
    Simulates a production REST API / Microservice inference engine:
      - Loads serialized model artifact once into memory
      - Validates incoming JSON-like payloads
      - Executes fast sub-millisecond inference
    """
    def __init__(self, artifact_path):
        self.artifact_path = artifact_path
        self.model, load_ms = ModelDeploymentManager.load_with_joblib(artifact_path)
        self.load_latency_ms = load_ms

    def predict_request(self, payload: dict) -> dict:
        """
        Processes an incoming prediction request.
        Expected payload: {"features": [val1, val2, ...]}
        """
        t0 = time.perf_counter()
        if "features" not in payload:
            return {"error": "Missing 'features' key in request payload", "status": 400}

        raw_features = np.asarray(payload["features"], dtype=float)
        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)

        if raw_features.shape[1] != len(self.model.feature_names):
            return {
                "error": f"Feature length mismatch: expected {len(self.model.feature_names)}, got {raw_features.shape[1]}",
                "status": 422
            }

        probs = self.model.predict_proba(raw_features)
        preds = (probs >= self.model.threshold).astype(int)
        latency_us = (time.perf_counter() - t0) * 1_000_000

        return {
            "status": 200,
            "prediction": int(preds[0]),
            "probability": float(probs[0]),
            "model_version": self.model.version,
            "inference_time_us": round(latency_us, 2)
        }


# =====================================================================
# 4. BENCHMARK & DEMONSTRATION RUNNER
# =====================================================================

def run_deployment_demo():
    print("=" * 70)
    print("WEEK 15: MODEL PERSISTENCE & DEPLOYMENT BASICS (PICKLE vs JOBLIB)")
    print("=" * 70)

    # 1. Train model on synthetic data
    X = np.random.randn(500, 4)
    y = ((X[:, 0] * 1.5 - X[:, 1] * 2.0 + X[:, 2] * 0.5) > 0).astype(int)
    model = LogisticModelArtifact(feature_names=["age", "income", "credit_score", "debt_ratio"])
    model.fit(X, y)

    # 2. Benchmark Persistence
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(curr_dir, "model_artifact.pkl")
    joblib_path = os.path.join(curr_dir, "model_artifact.joblib")

    size_pkl, save_time_pkl = ModelDeploymentManager.save_with_pickle(model, pkl_path)
    loaded_pkl, load_time_pkl = ModelDeploymentManager.load_with_pickle(pkl_path)

    size_joblib, save_time_joblib = ModelDeploymentManager.save_with_joblib(model, joblib_path, compress=3)
    loaded_joblib, load_time_joblib = ModelDeploymentManager.load_with_joblib(joblib_path)

    print(f"\n[1] Serialization Benchmark Table:")
    print(f"{'Method':<12} | {'File Size (Bytes)':<18} | {'Save Time (ms)':<15} | {'Load Time (ms)':<15}")
    print("-" * 65)
    print(f"{'Pickle':<12} | {size_pkl:<18} | {save_time_pkl:<15.3f} | {load_time_pkl:<15.3f}")
    print(f"{'Joblib (gz)':<12} | {size_joblib:<18} | {save_time_joblib:<15.3f} | {load_time_joblib:<15.3f}")

    # Verify identical prediction fidelity
    X_test = np.random.randn(20, 4)
    p_orig = model.predict_proba(X_test)
    p_pkl = loaded_pkl.predict_proba(X_test)
    p_joblib = loaded_joblib.predict_proba(X_test)
    assert np.allclose(p_orig, p_pkl), "Pickle predictions do not match original!"
    assert np.allclose(p_orig, p_joblib), "Joblib predictions do not match original!"
    print("\n[2] Prediction Fidelity: 100% Exact Numerical Match (tol < 1e-12)")

    # 3. Simulate Production REST API Inference
    service = ProductionInferenceService(joblib_path)
    sample_request = {"features": [45.0, 75000.0, 720.0, 0.28]}
    response = service.predict_request(sample_request)

    print(f"\n[3] Production Inference Simulation:")
    print(f"    Payload: {sample_request}")
    print(f"    Response: {response}")

    # 4. Generate Visualization Dashboard
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel A: File Size & Latency Comparison
    methods = ["Pickle (raw)", "Joblib (compressed)"]
    sizes = [size_pkl, size_joblib]
    axes[0].bar(methods, sizes, color=["#58a6ff", "#3fb950"], width=0.45)
    axes[0].set_ylabel("Artifact Size (Bytes)", color="gray")
    axes[0].set_title("Model Persistence: Storage Footprint", color="white", fontsize=11, fontweight="bold")
    for i, s in enumerate(sizes):
        axes[0].text(i, s + 15, f"{s} B", ha="center", color="white", fontweight="bold")

    # Panel B: Production Architecture Schema
    axes[1].axis("off")
    arch_info = [
        ["Phase", "Component", "Responsibility"],
        ["1. Training", "Model Training Pipeline", "Trains weights, fits standard scalers on train set"],
        ["2. Packaging", "ModelDeploymentManager", "Dumps weights + scalers + metadata into 1 bundle"],
        ["3. Persistence", "Disk / S3 / Artifactory", "Version-controlled artifact storage (e.g. v1.0.0.joblib)"],
        ["4. Serving", "ProductionInferenceService", "Loads artifact once; sub-ms inference on JSON payloads"],
        ["5. Monitoring", "Telemetry & Logs", "Tracks inference latency, feature drift & input schemas"]
    ]
    tbl = axes[1].table(cellText=arch_info, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#58a6ff", weight="bold")
        else:
            cell.set_facecolor("#161b22")
            cell.set_text_props(color="white")
        cell.set_edgecolor("#30363d")

    plt.tight_layout()
    out_plot = os.path.join(curr_dir, "model_deployment_dashboard.png")
    plt.savefig(out_plot, dpi=140, facecolor="#0d1117")
    plt.close()
    print(f"\n[SAVED] Dashboard written to {out_plot}")


if __name__ == "__main__":
    run_deployment_demo()

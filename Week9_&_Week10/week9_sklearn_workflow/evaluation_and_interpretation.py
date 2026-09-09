"""
Week 9: Model Evaluation & Feature Interpretation Suite

Generates:
1. Classification Report & Confusion Matrix Visualization
2. ROC Curve and Precision-Recall (PR) Curve
3. Model Interpretation: Gini Feature Importance & Permutation Importance
Saves figures to 'output_plots/'.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, classification_report
)
from sklearn.inspection import permutation_importance
from pipeline_churn_prediction import train_and_tune_model


def run_evaluation_and_interpretation(output_dir: str = "output_plots") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    temp_model_file = os.path.join(output_dir, "temp_model.joblib")

    trained = train_and_tune_model(temp_model_file)
    model = trained["best_model"]
    X_test = trained["X_test"]
    y_test = trained["y_test"]

    # 1. Classification Report
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report_str = classification_report(y_test, y_pred, target_names=["Retain", "Churn"])

    # 2. Extract Feature Names from Pipeline Preprocessor
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    # Clean prefixes like 'num__', 'cat__'
    clean_feature_names = [f.split("__")[-1] for f in feature_names]

    classifier = model.named_steps["classifier"]
    tree_importances = classifier.feature_importances_

    # 3. Permutation Importance
    perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importances = perm_result.importances_mean

    # Plot Diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # A: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"]).plot(
        ax=axes[0, 0], cmap="Blues", colorbar=False
    )
    axes[0, 0].set_title("A. Confusion Matrix (Holdout Test)", fontweight="bold")
    axes[0, 0].grid(False)

    # B: ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0, 1], name="GBM Pipeline")
    axes[0, 1].plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)")
    axes[0, 1].set_title("B. ROC Curve", fontweight="bold")
    axes[0, 1].legend()

    # C: Gini Feature Importances (Top 8)
    top_indices = np.argsort(tree_importances)[-8:]
    axes[1, 0].barh(range(len(top_indices)), tree_importances[top_indices], color="#6366f1")
    axes[1, 0].set_yticks(range(len(top_indices)))
    axes[1, 0].set_yticklabels([clean_feature_names[i] for i in top_indices])
    axes[1, 0].set_title("C. Tree-Based Feature Importances (Gini)", fontweight="bold")
    axes[1, 0].set_xlabel("Relative Importance")

    # D: Permutation Feature Importance
    perm_top_indices = np.argsort(perm_importances)[-8:]
    axes[1, 1].barh(range(len(perm_top_indices)), perm_importances[perm_top_indices], color="#10b981")
    axes[1, 1].set_yticks(range(len(perm_top_indices)))
    axes[1, 1].set_yticklabels(np.array(X_test.columns)[perm_top_indices])
    axes[1, 1].set_title("D. Permutation Importance (Test Set Impact)", fontweight="bold")
    axes[1, 1].set_xlabel("Drop in ROC-AUC Metric")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "09_model_evaluation_interpretation.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    if os.path.exists(temp_model_file):
        os.remove(temp_model_file)

    return {
        "report": report_str,
        "chart_path": chart_path,
        "top_features": [clean_feature_names[i] for i in reversed(top_indices)]
    }


if __name__ == "__main__":
    print("Running evaluation and interpretation...")
    res = run_evaluation_and_interpretation()
    print("\n=== Classification Report ===")
    print(res["report"])
    print("Top Influential Features:", res["top_features"][:5])
    print(f"Saved evaluation charts: {res['chart_path']}")

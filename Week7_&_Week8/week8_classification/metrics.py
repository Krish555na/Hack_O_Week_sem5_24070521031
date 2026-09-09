"""
Week 8: Classification Metrics from Scratch

Implements core evaluation metrics from mathematical definitions:
1. Confusion Matrix (TP, FP, TN, FN)
2. Accuracy, Precision, Recall / Sensitivity, Specificity, F1-Score
3. ROC Curve coordinates (FPR vs TPR) and Area Under Curve (ROC-AUC)
4. Comprehensive Classification Report generator
"""

import numpy as np
from typing import Dict, Tuple, List, Any


def confusion_matrix_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Computes binary 2x2 confusion matrix elements:
    TP, FP, TN, FN
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes all primary scalar classification metrics."""
    cm = confusion_matrix_scratch(y_true, y_pred)
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Specificity": round(specificity, 4),
        "F1_Score": round(f1, 4),
        "Confusion_Matrix": cm
    }


def roc_curve_and_auc_scratch(y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generates ROC curve points (FPR, TPR) across probability thresholds
    and computes Area Under Curve (AUC) via trapezoidal integration.
    """
    y_true = np.array(y_true, dtype=int)
    y_scores = np.array(y_scores, dtype=float)

    thresholds = np.linspace(1.0, 0.0, n_thresholds)
    fpr_list = []
    tpr_list = []

    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        tpr = tp / pos_count if pos_count > 0 else 0.0
        fpr = fp / neg_count if neg_count > 0 else 0.0

        fpr_list.append(fpr)
        tpr_list.append(tpr)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    # Trapezoidal rule: integrate TPR over FPR
    auc = float(np.trapezoid(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, round(auc, 4)


def classification_report_str(metrics: Dict[str, Any]) -> str:
    """Formats a clean tabular report."""
    cm = metrics["Confusion_Matrix"]
    lines = [
        "================ Classification Performance ================",
        f" Accuracy:    {metrics['Accuracy'] * 100:6.2f}%",
        f" Precision:   {metrics['Precision'] * 100:6.2f}%",
        f" Recall:      {metrics['Recall'] * 100:6.2f}%",
        f" Specificity: {metrics['Specificity'] * 100:6.2f}%",
        f" F1-Score:    {metrics['F1_Score'] * 100:6.2f}%",
        "---------------- Confusion Matrix --------------------------",
        f" [True Neg:  {cm['TN']:4d}]   [False Pos: {cm['FP']:4d}]",
        f" [False Neg: {cm['FN']:4d}]   [True Pos:  {cm['TP']:4d}]",
        "============================================================"
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    y_actual = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    y_predicted = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.4, 0.1, 0.85, 0.2, 0.65, 0.8, 0.15, 0.95, 0.05])

    m = compute_classification_metrics(y_actual, y_predicted)
    print(classification_report_str(m))

    fpr, tpr, auc = roc_curve_and_auc_scratch(y_actual, y_prob)
    print(f"Computed ROC-AUC Score: {auc:.4f}")

"""
Week 8: Automated Unit Tests for Classification Suite (Logistic Regression, KNN, Metrics)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from logistic_regression import LogisticRegressionScratch, MulticlassLogisticRegressionOvR
from knn_classifier import KNNClassifierScratch
from metrics import (
    confusion_matrix_scratch, compute_classification_metrics,
    roc_curve_and_auc_scratch
)
from classification_demo import run_classification_benchmark


class TestClassificationSuite(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 2 Linearly separable clusters
        X0 = np.random.normal(loc=[-2.0, -2.0], scale=0.5, size=(40, 2))
        X1 = np.random.normal(loc=[2.0, 2.0], scale=0.5, size=(40, 2))
        self.X_binary = np.vstack([X0, X1])
        self.y_binary = np.array([0] * 40 + [1] * 40)

    def test_logistic_regression_binary(self):
        clf = LogisticRegressionScratch(lr=0.5, max_iter=1000).fit(self.X_binary, self.y_binary)
        preds = clf.predict(self.X_binary)
        acc = np.mean(preds == self.y_binary)
        self.assertGreaterEqual(acc, 0.95)

        # Probabilities bounded in [0, 1]
        probs = clf.predict_proba(self.X_binary)
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_multiclass_logistic_ovr(self):
        X3 = np.random.normal(loc=[0.0, 4.0], scale=0.5, size=(40, 2))
        X_multi = np.vstack([self.X_binary, X3])
        y_multi = np.array([0] * 40 + [1] * 40 + [2] * 40)

        ovr = MulticlassLogisticRegressionOvR(lr=0.5, max_iter=1000).fit(X_multi, y_multi)
        preds = ovr.predict(X_multi)
        acc = np.mean(preds == y_multi)
        self.assertGreaterEqual(acc, 0.90)

    def test_knn_classifier_metrics(self):
        # Euclidean
        knn_euc = KNNClassifierScratch(n_neighbors=3, distance_metric="euclidean").fit(self.X_binary, self.y_binary)
        preds_euc = knn_euc.predict(self.X_binary)
        self.assertGreaterEqual(np.mean(preds_euc == self.y_binary), 0.95)

        # Manhattan
        knn_man = KNNClassifierScratch(n_neighbors=3, distance_metric="manhattan").fit(self.X_binary, self.y_binary)
        preds_man = knn_man.predict(self.X_binary)
        self.assertGreaterEqual(np.mean(preds_man == self.y_binary), 0.95)

    def test_metrics_evaluation(self):
        y_t = np.array([1, 1, 0, 0, 1])
        y_p = np.array([1, 0, 0, 1, 1])
        # TP=2, FP=1, TN=1, FN=1
        cm = confusion_matrix_scratch(y_t, y_p)
        self.assertEqual(cm, {"TP": 2, "FP": 1, "TN": 1, "FN": 1})

        metrics = compute_classification_metrics(y_t, y_p)
        self.assertEqual(metrics["Accuracy"], 0.6)
        self.assertAlmostEqual(metrics["Precision"], 2 / 3, places=3)
        self.assertAlmostEqual(metrics["Recall"], 2 / 3, places=3)

    def test_roc_auc(self):
        y_t = np.array([0, 0, 1, 1])
        y_s = np.array([0.1, 0.4, 0.35, 0.8])
        _, _, auc = roc_curve_and_auc_scratch(y_t, y_s)
        self.assertGreater(auc, 0.5)

    def test_demo_and_plot(self):
        reps, plot_path = run_classification_benchmark(output_dir="test_plots_w8")
        self.assertIn("KNN (k=15)", reps)
        self.assertTrue(os.path.exists(plot_path))
        os.remove(plot_path)
        if os.path.exists("test_plots_w8"):
            os.rmdir("test_plots_w8")


if __name__ == "__main__":
    unittest.main()

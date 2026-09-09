"""
Week 7: Automated Unit Tests for Linear, Polynomial, Ridge & Lasso Regression
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from linear_regression import (
    LinearRegressionScratch, mean_squared_error,
    root_mean_squared_error, r2_score_scratch
)
from polynomial_regression import generate_polynomial_features, analyze_degree_tradeoff
from regularization_ridge_lasso import RidgeRegressionScratch, LassoRegressionScratch
from regression_demo import run_housing_price_demo


class TestRegressionSuite(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic clean line
        self.X = np.linspace(1, 10, 50).reshape(-1, 1)
        self.y = 2.0 * self.X.squeeze() + 3.0

    def test_linear_regression_solvers(self):
        # Normal equation should recover exact weights
        model_norm = LinearRegressionScratch(solver="normal").fit(self.X, self.y)
        self.assertAlmostEqual(model_norm.theta[0], 3.0, places=4)
        self.assertAlmostEqual(model_norm.theta[1], 2.0, places=4)

        # Standardize for GD
        X_scaled = (self.X - np.mean(self.X)) / np.std(self.X)
        model_gd = LinearRegressionScratch(solver="gd", lr=0.1, max_iter=1500).fit(X_scaled, self.y)
        preds = model_gd.predict(X_scaled)
        self.assertAlmostEqual(r2_score_scratch(self.y, preds), 1.0, places=3)

    def test_metrics_calculation(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        # Errors: [0.5, -0.5, 0.0, -1.0], Squared: [0.25, 0.25, 0.0, 1.0], Mean: 1.5/4 = 0.375
        self.assertEqual(mean_squared_error(y_true, y_pred), 0.375)
        self.assertAlmostEqual(root_mean_squared_error(y_true, y_pred), np.sqrt(0.375))
        self.assertGreater(r2_score_scratch(y_true, y_pred), 0.9)

    def test_polynomial_feature_expansion(self):
        x_raw = np.array([[2.0], [3.0]])
        poly_feat = generate_polynomial_features(x_raw, degree=3)
        # Should be [x, x^2, x^3]
        np.testing.assert_allclose(poly_feat, [[2.0, 4.0, 8.0], [3.0, 9.0, 27.0]])

    def test_ridge_and_lasso_regularization(self):
        # Uninformative feature test
        X_noisy = np.column_stack([self.X, np.random.randn(len(self.X), 3)])
        ridge = RidgeRegressionScratch(alpha=50.0).fit(X_noisy, self.y)
        # Weights should be non-zero but small
        self.assertEqual(len(ridge.theta), 5)

        lasso = LassoRegressionScratch(alpha=1.0, max_iter=2000).fit(X_noisy, self.y)
        # At least one noisy feature weight should be set precisely to 0.0
        self.assertTrue((lasso.theta[2:] == 0.0).any())

    def test_demo_and_plot(self):
        metrics, plot_path = run_housing_price_demo(output_dir="test_plots_w7")
        self.assertIn("OLS", metrics)
        self.assertGreater(metrics["OLS"]["R2"], 0.8)
        self.assertTrue(os.path.exists(plot_path))
        os.remove(plot_path)
        if os.path.exists("test_plots_w7"):
            os.rmdir("test_plots_w7")


if __name__ == "__main__":
    unittest.main()

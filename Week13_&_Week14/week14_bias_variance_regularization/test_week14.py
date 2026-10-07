"""
Week 14: Automated Unit Tests for Bias-Variance, Overfitting/Underfitting Diagnostics & Regularization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from bias_variance_decomposition import (
    PolynomialRegressionScratch,
    perform_bias_variance_decomposition,
    true_function
)
from overfitting_underfitting_diagnostics import (
    PolynomialRegressorGD,
    k_fold_cv
)
from regularization_l1_l2 import (
    RidgeRegressionScratch,
    LassoRegressionScratch,
    ElasticNetScratch
)


class TestBiasVarianceDecomposition(unittest.TestCase):
    """Unit tests for Polynomial model and Bias-Variance analysis."""

    def setUp(self):
        np.random.seed(42)
        self.x = np.linspace(0, 1, 50)
        self.y = true_function(self.x) + np.random.normal(0, 0.1, 50)

    def test_polynomial_fitting(self):
        m = PolynomialRegressionScratch(degree=3).fit(self.x, self.y)
        preds = m.predict(self.x)
        self.assertEqual(len(preds), len(self.y))
        mse = np.mean((preds - self.y)**2)
        self.assertLess(mse, 0.05)

    def test_decomposition_additivity(self):
        """Verify Bias^2 + Variance + Noise matches Total MSE."""
        degrees, bias_sq, variance, noise, total_mse = perform_bias_variance_decomposition(
            n_datasets=25, n_train_samples=20, noise_std=0.2, degrees=[1, 3]
        )
        for b, v, tot in zip(bias_sq, variance, total_mse):
            self.assertAlmostEqual(b + v + noise, tot, places=6)


class TestDiagnosticsAndEarlyStopping(unittest.TestCase):
    """Unit tests for Diagnostics and Early Stopping."""

    def setUp(self):
        np.random.seed(99)
        self.x = np.sort(np.random.uniform(-1, 1, 60))
        self.y = np.sin(np.pi * self.x) + np.random.normal(0, 0.2, 60)

    def test_early_stopping_activation(self):
        x_tr, y_tr = self.x[:45], self.y[:45]
        x_va, y_va = self.x[45:], self.y[45:]

        model = PolynomialRegressorGD(degree=4, learning_rate=0.05, max_epochs=1000, patience=15)
        model.fit(x_tr, y_tr, x_va, y_va)

        self.assertIsNotNone(model.best_epoch)
        self.assertLessEqual(len(model.train_history), 1000)
        self.assertIsNotNone(model.best_weights)

    def test_k_fold_cv(self):
        mean_mse, std_mse = k_fold_cv(self.x, self.y, degree=2, k=4)
        self.assertGreater(mean_mse, 0.0)
        self.assertGreaterEqual(std_mse, 0.0)


class TestRegularizationL1L2(unittest.TestCase):
    """Unit tests for Ridge, Lasso, and ElasticNet."""

    def setUp(self):
        np.random.seed(77)
        self.n_samples = 80
        self.X = np.random.randn(self.n_samples, 5)
        # Only feature 0 and 1 are informative
        self.true_beta = np.array([4.0, -3.0, 0.0, 0.0, 0.0])
        self.y = self.X @ self.true_beta + np.random.normal(0, 0.2, self.n_samples)

    def test_ridge_shrinkage(self):
        ridge_light = RidgeRegressionScratch(l2_lambda=0.1).fit(self.X, self.y)
        ridge_heavy = RidgeRegressionScratch(l2_lambda=50.0).fit(self.X, self.y)

        # Higher lambda must produce smaller coefficient L2 norm
        norm_light = np.linalg.norm(ridge_light.coef_)
        norm_heavy = np.linalg.norm(ridge_heavy.coef_)
        self.assertGreater(norm_light, norm_heavy)

    def test_lasso_sparsity_and_selection(self):
        lasso = LassoRegressionScratch(l1_lambda=8.0, max_iter=800).fit(self.X, self.y)
        # Should drive uninformative features (indices 2, 3, 4) to 0
        zero_coefs = np.sum(np.abs(lasso.coef_[2:]) < 1e-3)
        self.assertGreaterEqual(zero_coefs, 2)
        # Informative features should remain non-zero
        self.assertGreater(abs(lasso.coef_[0]), 1.0)

    def test_elastic_net_fit(self):
        enet = ElasticNetScratch(alpha=5.0, l1_ratio=0.5).fit(self.X, self.y)
        preds = enet.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        self.assertLess(np.mean((preds - self.y)**2), np.var(self.y))


if __name__ == "__main__":
    unittest.main()

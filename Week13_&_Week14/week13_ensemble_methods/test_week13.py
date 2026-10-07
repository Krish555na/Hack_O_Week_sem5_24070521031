"""
Week 13: Automated Unit Tests for Ensemble Methods (Bagging, Random Forest, Boosting, XGBoost & LightGBM)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from bagging_from_scratch import DecisionTreeScratch, BaggingEnsembleScratch
from boosting_xgboost_lightgbm import (
    GradientBoostingScratch,
    XGBoostScratchRegressor,
    XGBoostTreeScratch,
    LightGBMSimulator
)


class TestBaggingAndTrees(unittest.TestCase):
    """Unit tests for Decision Trees, Bagging, and Random Forest."""

    def setUp(self):
        np.random.seed(42)
        self.X_clf = np.random.randn(80, 4)
        self.y_clf = (self.X_clf[:, 0] + self.X_clf[:, 1] > 0).astype(float)

        self.X_reg = np.random.uniform(-2, 2, (70, 2))
        self.y_reg = self.X_reg[:, 0] ** 2 + self.X_reg[:, 1]

    def test_decision_tree_classification(self):
        dt = DecisionTreeScratch(max_depth=4, task="classification")
        dt.fit(self.X_clf, self.y_clf)
        preds = dt.predict(self.X_clf)
        self.assertEqual(len(preds), len(self.y_clf))
        acc = np.mean(preds == self.y_clf)
        self.assertGreater(acc, 0.75)

    def test_decision_tree_regression(self):
        dt = DecisionTreeScratch(max_depth=4, task="regression")
        dt.fit(self.X_reg, self.y_reg)
        preds = dt.predict(self.X_reg)
        self.assertEqual(len(preds), len(self.y_reg))
        mse = np.mean((preds - self.y_reg) ** 2)
        self.assertLess(mse, np.var(self.y_reg))

    def test_bagging_ensemble_classification_and_oob(self):
        bag = BaggingEnsembleScratch(n_estimators=15, max_depth=4, task="classification")
        bag.fit(self.X_clf, self.y_clf)
        preds = bag.predict(self.X_clf)
        self.assertEqual(len(preds), len(self.y_clf))
        self.assertIsNotNone(bag.oob_score_)
        self.assertGreater(bag.oob_score_, 0.60)

    def test_random_forest_feature_subspacing(self):
        rf = BaggingEnsembleScratch(n_estimators=10, max_depth=4, max_features="sqrt", task="classification")
        rf.fit(self.X_clf, self.y_clf)
        preds = rf.predict(self.X_clf)
        self.assertEqual(len(preds), len(self.y_clf))
        # Ensure base estimators exist and trained with sub-features
        self.assertEqual(len(rf.estimators_), 10)
        self.assertEqual(rf.estimators_[0].n_sub_features, 2)


class TestBoostingMechanics(unittest.TestCase):
    """Unit tests for Gradient Boosting, XGBoost math, and LightGBM simulators."""

    def setUp(self):
        np.random.seed(123)
        self.X = np.random.uniform(-3, 3, (60, 2))
        self.y = np.sin(self.X[:, 0]) + 0.2 * self.X[:, 1]

    def test_gradient_boosting_loss_decreases(self):
        gbm = GradientBoostingScratch(n_estimators=15, learning_rate=0.1)
        gbm.fit(self.X, self.y)
        self.assertEqual(len(gbm.trees), 15)
        # Loss must strictly decrease overall
        self.assertLess(gbm.train_loss_history[-1], gbm.train_loss_history[0])

    def test_xgboost_split_gain_calculation(self):
        tree = XGBoostTreeScratch(max_depth=2, l2_reg=1.0, gamma=0.1)
        # With zero gradients, gain should be zero or negative (pruned)
        gain_zero = tree._calc_split_gain(0.0, 5.0, 0.0, 5.0)
        self.assertLessEqual(gain_zero, 0.0)

        # With large conflicting gradients, gain should be strongly positive
        gain_pos = tree._calc_split_gain(10.0, 5.0, -10.0, 5.0)
        self.assertGreater(gain_pos, 0.0)

    def test_xgboost_regressor_fit_and_regularization(self):
        xgb_reg = XGBoostScratchRegressor(n_estimators=15, learning_rate=0.1, max_depth=2, l2_reg=2.0, gamma=0.05)
        xgb_reg.fit(self.X, self.y)
        preds = xgb_reg.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        mse = np.mean((preds - self.y) ** 2)
        self.assertLess(mse, np.var(self.y))
        self.assertLess(xgb_reg.loss_history[-1], xgb_reg.loss_history[0])

    def test_lightgbm_goss_sampling(self):
        g = np.array([-10.0, -8.0, 0.1, 0.2, 0.05, 0.3, 0.15, 0.01])
        indices, weights = LightGBMSimulator.goss_sampling(g, a=0.25, b=0.5)
        self.assertGreater(len(indices), 0)
        self.assertEqual(len(indices), len(weights))
        # Top gradient elements must be retained first
        self.assertIn(0, indices)  # -10 has max abs gradient
        self.assertIn(1, indices)  # -8 has 2nd max abs gradient
        # Amplification weight must be > 1.0 for small gradients
        self.assertGreater(weights[-1], 1.0)

    def test_lightgbm_histogram_binning(self):
        X = np.random.uniform(0, 100, (50, 3))
        X_binned, edges = LightGBMSimulator.histogram_binning(X, max_bins=8)
        self.assertEqual(X_binned.shape, X.shape)
        self.assertTrue(np.all(X_binned >= 0))
        self.assertTrue(np.all(X_binned < 8))


if __name__ == "__main__":
    unittest.main()

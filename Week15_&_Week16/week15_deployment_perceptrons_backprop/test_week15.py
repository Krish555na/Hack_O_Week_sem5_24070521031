"""
Week 15: Automated Unit Tests for Deployment Basics, Perceptrons, Activations & Backpropagation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from model_persistence_deployment import (
    LogisticModelArtifact,
    ModelDeploymentManager,
    ProductionInferenceService
)
from perceptron_and_activations import (
    RosenblattPerceptron,
    Activations
)
from forward_backward_propagation import (
    TwoLayerNNScratch,
    gradient_check
)


class TestModelDeployment(unittest.TestCase):
    """Tests for model persistence and production inference service."""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(80, 3)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)
        self.model = LogisticModelArtifact(feature_names=["f1", "f2", "f3"]).fit(self.X, self.y)
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

    def test_pickle_persistence(self):
        pkl_path = os.path.join(self.test_dir, "temp_test_model.pkl")
        ModelDeploymentManager.save_with_pickle(self.model, pkl_path)
        self.assertTrue(os.path.exists(pkl_path))

        loaded, _ = ModelDeploymentManager.load_with_pickle(pkl_path)
        p_orig = self.model.predict_proba(self.X[:10])
        p_load = loaded.predict_proba(self.X[:10])
        self.assertTrue(np.allclose(p_orig, p_load, atol=1e-10))

        if os.path.exists(pkl_path):
            os.remove(pkl_path)

    def test_joblib_persistence_and_service(self):
        joblib_path = os.path.join(self.test_dir, "temp_test_model.joblib")
        ModelDeploymentManager.save_with_joblib(self.model, joblib_path)
        self.assertTrue(os.path.exists(joblib_path))

        service = ProductionInferenceService(joblib_path)
        resp = service.predict_request({"features": [1.0, 2.0, -1.0]})
        self.assertEqual(resp["status"], 200)
        self.assertIn("prediction", resp)
        self.assertIn("probability", resp)
        self.assertGreaterEqual(resp["probability"], 0.0)
        self.assertLessEqual(resp["probability"], 1.0)

        # Bad payload handling
        bad_resp = service.predict_request({"features": [1.0]})
        self.assertEqual(bad_resp["status"], 422)

        if os.path.exists(joblib_path):
            os.remove(joblib_path)


class TestPerceptronAndActivations(unittest.TestCase):
    """Tests for Perceptron logic gates and activation functions."""

    def test_perceptron_and_or_gates(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_and = np.array([0, 0, 0, 1])
        y_or = np.array([0, 1, 1, 1])

        p_and = RosenblattPerceptron(learning_rate=0.1, max_epochs=40).fit(X, y_and)
        p_or = RosenblattPerceptron(learning_rate=0.1, max_epochs=40).fit(X, y_or)

        self.assertTrue(np.all(p_and.predict(X) == y_and))
        self.assertTrue(np.all(p_or.predict(X) == y_or))

    def test_activation_ranges_and_derivatives(self):
        z = np.array([-2.0, 0.0, 2.0])

        # Sigmoid in (0, 1), peak deriv at 0 is 0.25
        sig = Activations.sigmoid(z)
        self.assertTrue(np.all((sig > 0) & (sig < 1)))
        self.assertAlmostEqual(Activations.sigmoid_derivative(np.array([0.0]))[0], 0.25, places=5)

        # Tanh in (-1, 1), peak deriv at 0 is 1.0
        tanh = Activations.tanh(z)
        self.assertTrue(np.all((tanh > -1) & (tanh < 1)))
        self.assertAlmostEqual(Activations.tanh_derivative(np.array([0.0]))[0], 1.0, places=5)

        # ReLU and LeakyReLU
        self.assertEqual(Activations.relu(np.array([-5.0]))[0], 0.0)
        self.assertEqual(Activations.relu(np.array([5.0]))[0], 5.0)
        self.assertEqual(Activations.leaky_relu(np.array([-10.0]), alpha=0.01)[0], -0.1)

        # Softmax sums to 1.0
        sm = Activations.softmax(np.array([[1.0, 2.0, 3.0]]))
        self.assertAlmostEqual(np.sum(sm), 1.0, places=5)


class TestBackpropagationAndGradients(unittest.TestCase):
    """Tests for 2-layer forward/backward pass and numerical gradient check."""

    def test_gradient_check_precision(self):
        np.random.seed(42)
        X = np.random.randn(5, 2)
        Y = np.array([[1], [0], [1], [0], [1]])
        model = TwoLayerNNScratch(input_dim=2, hidden_dim=3, output_dim=1)

        rel_diff, _, _ = gradient_check(model, X, Y, epsilon=1e-7)
        # Relative error should be extremely small (< 1e-6)
        self.assertLess(rel_diff, 1e-6)

    def test_xor_resolution(self):
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y_xor = np.array([[0], [1], [1], [0]], dtype=float)

        net = TwoLayerNNScratch(input_dim=2, hidden_dim=6, output_dim=1, learning_rate=0.8)
        net.fit(X_xor, y_xor, epochs=1200)

        preds = net.predict(X_xor)
        acc = np.mean(preds == y_xor)
        self.assertEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()

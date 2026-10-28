"""
Week 16: Automated Unit Tests for Loss Functions, Optimizers & Neural Network from Scratch
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from loss_functions import LossFunctions
from optimizers_from_scratch import (
    SGDOptimizer,
    MomentumOptimizer,
    RMSpropOptimizer,
    AdamOptimizer
)
from neural_network_from_scratch import (
    Dense,
    ReLU,
    Sigmoid,
    NeuralNetworkScratch
)


class TestLossFunctions(unittest.TestCase):
    """Tests for MSE, BCE, and CCE loss functions and derivatives."""

    def test_mse_value_and_gradient(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])
        # (0.5^2 + 0 + (-0.5)^2) / (2 * 3) = (0.25 + 0.25) / 6 = 0.5 / 6 = 1/12
        loss = LossFunctions.mse(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.5 / 6.0, places=5)

        grad = LossFunctions.mse_derivative(y_true, y_pred)
        expected_grad = (y_pred - y_true) / 3.0
        self.assertTrue(np.allclose(grad, expected_grad))

    def test_bce_value_and_extreme_penalty(self):
        y_true = np.array([1.0, 0.0])
        y_pred_correct = np.array([0.99, 0.01])
        y_pred_wrong = np.array([0.01, 0.99])

        loss_low = LossFunctions.binary_cross_entropy(y_true, y_pred_correct)
        loss_high = LossFunctions.binary_cross_entropy(y_true, y_pred_wrong)
        self.assertLess(loss_low, 0.1)
        self.assertGreater(loss_high, 4.0)

    def test_categorical_cross_entropy(self):
        Y_true = np.array([[1, 0, 0], [0, 1, 0]])
        Y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        cce = LossFunctions.categorical_cross_entropy(Y_true, Y_pred)
        self.assertGreater(cce, 0.0)
        self.assertLess(cce, 1.0)


class TestOptimizers(unittest.TestCase):
    """Tests for SGD, Momentum, RMSprop, and Adam state and update mechanics."""

    def setUp(self):
        np.random.seed(42)
        self.w = np.array([2.0, -3.0])
        self.grad = np.array([0.5, -1.0])

    def test_sgd_update(self):
        opt = SGDOptimizer(lr=0.1)
        w_new = opt.update(self.w, self.grad)
        expected = self.w - 0.1 * self.grad
        self.assertTrue(np.allclose(w_new, expected))

    def test_momentum_state_tracking(self):
        opt = MomentumOptimizer(lr=0.1, beta=0.9)
        w_new1 = opt.update(self.w, self.grad, key="w1")
        self.assertIn("w1", opt.velocities)
        self.assertTrue(np.allclose(opt.velocities["w1"], 0.1 * self.grad))

    def test_adam_moments_and_bias_correction(self):
        opt = AdamOptimizer(lr=0.01, beta1=0.9, beta2=0.999)
        w_new = opt.update(self.w, self.grad, key="w_adam")
        self.assertEqual(opt.t["w_adam"], 1)
        self.assertIn("w_adam", opt.m)
        self.assertIn("w_adam", opt.v)
        self.assertFalse(np.allclose(w_new, self.w))


class TestNeuralNetworkFramework(unittest.TestCase):
    """Tests for Dense layer, Activations, Network fit, and Serialization."""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(40, 2)
        self.y = (self.X[:, 0] ** 2 + self.X[:, 1] ** 2 < 1.0).astype(int)

    def test_dense_forward_backward_shapes(self):
        layer = Dense(in_features=2, out_features=4)
        out = layer.forward(self.X)
        self.assertEqual(out.shape, (40, 4))

        grad_in = np.ones((40, 4))
        grad_out = layer.backward(grad_in)
        self.assertEqual(grad_out.shape, (40, 2))
        self.assertEqual(layer.grad_weights.shape, (2, 4))
        self.assertEqual(layer.grad_biases.shape, (1, 4))

    def test_network_training_and_accuracy(self):
        net = NeuralNetworkScratch(optimizer="adam", lr=0.05)
        net.add(Dense(2, 8))
        net.add(ReLU())
        net.add(Dense(8, 1))
        net.add(Sigmoid())

        net.fit(self.X, self.y, epochs=150, batch_size=16, verbose=False)
        self.assertLess(net.history["loss"][-1], net.history["loss"][0])
        preds = net.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        acc = np.mean(preds.ravel() == self.y)
        self.assertGreater(acc, 0.70)

    def test_model_persistence(self):
        net = NeuralNetworkScratch(optimizer="adam", lr=0.01)
        net.add(Dense(2, 4))
        net.add(Sigmoid())
        net.fit(self.X, self.y, epochs=5, batch_size=20, verbose=False)

        test_dir = os.path.dirname(os.path.abspath(__file__))
        save_file = os.path.join(test_dir, "test_net_save.joblib")
        net.save(save_file)
        self.assertTrue(os.path.exists(save_file))

        loaded = NeuralNetworkScratch.load(save_file)
        p_orig = net.forward(self.X[:5])
        p_load = loaded.forward(self.X[:5])
        self.assertTrue(np.allclose(p_orig, p_load))

        if os.path.exists(save_file):
            os.remove(save_file)


if __name__ == "__main__":
    unittest.main()

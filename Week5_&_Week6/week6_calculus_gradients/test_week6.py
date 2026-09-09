"""
Week 6: Automated Unit Tests for Calculus, Derivatives & Autograd Engine
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from derivatives_and_gradients import (
    numerical_derivative_1d, sigmoid, sigmoid_derivative,
    relu, relu_derivative, check_gradient
)
from autograd_engine import Scalar, neuron_demo
from gradient_descent_optimizers import (
    loss_function, optimize_vanilla_gd, optimize_adam,
    plot_optimizer_trajectories
)


class TestCalculusAndAutograd(unittest.TestCase):
    def test_numerical_and_analytical_derivatives(self):
        # Test sigmoid derivative at x = 0.5
        x = 0.5
        num_d = numerical_derivative_1d(sigmoid, x, h=1e-5, method="central")
        ana_d = sigmoid_derivative(x)
        self.assertAlmostEqual(num_d, ana_d, places=4)

        # Test relu derivative
        self.assertEqual(relu_derivative(2.5), 1.0)
        self.assertEqual(relu_derivative(-1.5), 0.0)

    def test_gradient_checking(self):
        def f(w): return float(w[0]**3 + 2.0 * w[1]**2)
        def grad_f(w): return np.array([3.0 * w[0]**2, 4.0 * w[1]])

        is_valid, rel_diff = check_gradient(f, grad_f, np.array([2.0, -1.5]))
        self.assertTrue(is_valid)
        self.assertLess(rel_diff, 1e-4)

    def test_autograd_chain_rule(self):
        # Simple computation: z = (a + b) * c
        a = Scalar(3.0)
        b = Scalar(-4.0)
        c = Scalar(2.0)
        d = a + b    # -1.0
        z = d * c    # -2.0

        z.backward()
        self.assertEqual(z.data, -2.0)
        self.assertEqual(d.grad, 2.0)
        self.assertEqual(c.grad, -1.0)
        self.assertEqual(a.grad, 2.0)
        self.assertEqual(b.grad, 2.0)

    def test_neuron_autograd_demo(self):
        res = neuron_demo()
        self.assertGreater(res["output"], 0.0)
        self.assertLess(res["output"], 1.0)
        self.assertIsNotNone(res["grad_w1"])

    def test_optimizers_convergence(self):
        w_start = np.array([4.0, 3.0])
        init_loss = loss_function(w_start)

        traj_adam = optimize_adam(w_start, steps=30)
        final_loss = loss_function(traj_adam[-1])

        self.assertLess(final_loss, init_loss)
        self.assertLess(final_loss, 0.5)

    def test_optimizer_plot(self):
        plot_path = plot_optimizer_trajectories(output_dir="test_plots_w6")
        self.assertTrue(os.path.exists(plot_path))
        os.remove(plot_path)
        if os.path.exists("test_plots_w6"):
            os.rmdir("test_plots_w6")


if __name__ == "__main__":
    unittest.main()

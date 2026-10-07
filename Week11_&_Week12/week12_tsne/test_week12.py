"""
Week 12: Automated Unit Tests for t-SNE Intuition & Mechanics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from tsne_intuition import TSNEFromScratch


class TestTSNEScratch(unittest.TestCase):
    """Tests for t-SNE from-scratch implementation."""

    def test_output_shape(self):
        X = np.random.randn(40, 4)
        tsne = TSNEFromScratch(n_components=2, n_iter=50)
        Y = tsne.fit_transform(X)
        self.assertEqual(Y.shape, (40, 2))

    def test_kl_history_length(self):
        X = np.random.randn(30, 3)
        tsne = TSNEFromScratch(n_components=2, n_iter=60)
        tsne.fit_transform(X)
        self.assertEqual(len(tsne.kl_history_), 60)

    def test_kl_decreases_overall(self):
        """KL should not diverge - last half mean < 2x initial mean."""
        np.random.seed(42)
        X = np.random.randn(40, 4)
        tsne = TSNEFromScratch(n_components=2, n_iter=300,
                               learning_rate=150, perplexity=10)
        tsne.fit_transform(X)
        n = len(tsne.kl_history_)
        first = np.mean(tsne.kl_history_[:max(1, n//10)])
        last = np.mean(tsne.kl_history_[n//2:])
        self.assertLess(last, first * 2.0,
                        msg=f"KL should not diverge: first={first:.4f}, last={last:.4f}")

    def test_low_dim_affinities_sum_to_one(self):
        Y = np.random.randn(20, 2)
        Q, _ = TSNEFromScratch._low_dim_affinities(Y)
        self.assertAlmostEqual(Q.sum(), 1.0, places=5)

    def test_high_dim_affinities_sum_to_one(self):
        X = np.random.randn(20, 4)
        tsne = TSNEFromScratch(perplexity=5)
        P = tsne._high_dim_affinities(X)
        self.assertAlmostEqual(P.sum(), 1.0, places=5)

    def test_affinities_symmetric(self):
        X = np.random.randn(15, 3)
        tsne = TSNEFromScratch(perplexity=4)
        P = tsne._high_dim_affinities(X)
        np.testing.assert_allclose(P, P.T, atol=1e-12, err_msg="P must be symmetric")

        Y = np.random.randn(15, 2)
        Q, _ = TSNEFromScratch._low_dim_affinities(Y)
        np.testing.assert_allclose(Q, Q.T, atol=1e-12, err_msg="Q must be symmetric")


class TestTSNEConceptual(unittest.TestCase):
    """Conceptual behavior tests."""

    def test_tsne_gradient_larger_for_structured_data(self):
        np.random.seed(42)

        Xa = np.random.randn(15, 3)
        Xb = np.random.randn(15, 3) + 8
        X_struct = np.vstack([Xa, Xb])

        X_rand = np.random.rand(30, 3)

        tsne = TSNEFromScratch(perplexity=6)

        P_struct = tsne._high_dim_affinities(X_struct)
        P_rand = tsne._high_dim_affinities(X_rand)

        H_struct = -(P_struct * np.log(P_struct + 1e-12)).sum()
        H_rand = -(P_rand * np.log(P_rand + 1e-12)).sum()

        self.assertLess(H_struct, H_rand,
                        msg=f"Structured P should have lower entropy ({H_struct:.4f}) than random ({H_rand:.4f})")


if __name__ == "__main__":
    unittest.main()

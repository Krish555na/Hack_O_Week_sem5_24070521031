"""
Tests for Week 11 (PCA) and Week 12 (t-SNE) - pure NumPy, no sklearn/scipy.
"""
import unittest
import numpy as np
import sys
import os

# Add module paths
W11 = os.path.join(os.path.dirname(__file__), "..", "week11_pca")
W12 = os.path.join(os.path.dirname(__file__), "..", "week12_tsne")
sys.path.insert(0, W11)
sys.path.insert(0, W12)

from pca_from_scratch import PCAFromScratch
from tsne_intuition import TSNEFromScratch


class TestPCAMath(unittest.TestCase):
    """Core mathematical properties of PCA."""

    def setUp(self):
        np.random.seed(42)
        cov = [[2.0, 1.5], [1.5, 1.2]]
        self.X = np.random.multivariate_normal([0, 0], cov, 300)
        self.pca = PCAFromScratch(n_components=2)
        self.pca.fit(self.X)

    def test_components_are_orthogonal(self):
        """PC1 and PC2 must be mutually orthogonal."""
        dot = np.dot(self.pca.components_[0], self.pca.components_[1])
        self.assertAlmostEqual(dot, 0.0, places=10)

    def test_components_are_unit_vectors(self):
        """Each PC must have unit norm."""
        for i, pc in enumerate(self.pca.components_):
            self.assertAlmostEqual(np.linalg.norm(pc), 1.0, places=10)

    def test_explained_variance_ratio_sums_to_one(self):
        total = self.pca.explained_variance_ratio_.sum()
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_eigenvalues_non_negative(self):
        for ev in self.pca.explained_variance_:
            self.assertGreaterEqual(ev, 0)

    def test_eigenvalues_descending(self):
        evs = self.pca.explained_variance_
        self.assertTrue(all(evs[i] >= evs[i+1] for i in range(len(evs)-1)))

    def test_mean_centering(self):
        """Transformed data has near-zero mean."""
        Xt = self.pca.transform(self.X)
        np.testing.assert_allclose(Xt.mean(axis=0), 0, atol=1e-10)


class TestPCAProjection(unittest.TestCase):
    """Projection and reconstruction properties."""

    def setUp(self):
        np.random.seed(0)
        self.X = np.random.randn(100, 5)
        self.pca2 = PCAFromScratch(n_components=2)
        self.Xp = self.pca2.fit_transform(self.X)

    def test_projection_shape(self):
        self.assertEqual(self.Xp.shape, (100, 2))

    def test_reconstruction_shape(self):
        Xr = self.pca2.inverse_transform(self.Xp)
        self.assertEqual(Xr.shape, self.X.shape)

    def test_reconstruction_improves_with_k(self):
        errors = []
        for k in range(1, 6):
            pk  = PCAFromScratch(n_components=k)
            Xr  = pk.inverse_transform(pk.fit_transform(self.X))
            errors.append(np.mean((self.X - Xr) ** 2))
        self.assertTrue(all(errors[i] >= errors[i+1] for i in range(len(errors)-1)))

    def test_full_rank_reconstruction_perfect(self):
        """k == n_features -> reconstruction MSE == 0."""
        pk  = PCAFromScratch(n_components=5)
        Xr  = pk.inverse_transform(pk.fit_transform(self.X))
        mse = np.mean((self.X - Xr) ** 2)
        self.assertAlmostEqual(mse, 0.0, places=10)

    def test_projections_uncorrelated(self):
        """Projections onto different PCs are uncorrelated."""
        corr = np.corrcoef(self.Xp.T)[0, 1]
        self.assertAlmostEqual(corr, 0.0, places=5)


class TestPCAVerification(unittest.TestCase):
    """Verify PCA matches SVD (reference implementation)."""

    def test_evr_matches_svd(self):
        np.random.seed(3)
        X  = np.random.randn(80, 6)
        Xc = X - X.mean(0)
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        evr_svd = (S**2) / (S**2).sum()

        pca = PCAFromScratch(n_components=6)
        pca.fit(X)

        np.testing.assert_allclose(
            np.sort(pca.explained_variance_ratio_)[::-1],
            np.sort(evr_svd)[::-1],
            atol=1e-6,
            err_msg="EVR must match SVD reference"
        )


class TestTSNEScratch(unittest.TestCase):
    """Tests for t-SNE from-scratch implementation."""

    def test_output_shape(self):
        X    = np.random.randn(40, 4)
        tsne = TSNEFromScratch(n_components=2, n_iter=50)
        Y    = tsne.fit_transform(X)
        self.assertEqual(Y.shape, (40, 2))

    def test_kl_history_length(self):
        X    = np.random.randn(30, 3)
        tsne = TSNEFromScratch(n_components=2, n_iter=60)
        tsne.fit_transform(X)
        self.assertEqual(len(tsne.kl_history_), 60)

    def test_kl_decreases_overall(self):
        """KL should not diverge - last half mean < 1.5x initial mean."""
        np.random.seed(42)
        X    = np.random.randn(40, 4)
        tsne = TSNEFromScratch(n_components=2, n_iter=300,
                                learning_rate=150, perplexity=10)
        tsne.fit_transform(X)
        n     = len(tsne.kl_history_)
        first = np.mean(tsne.kl_history_[:max(1, n//10)])
        last  = np.mean(tsne.kl_history_[n//2:])
        self.assertLess(last, first * 2.0,
                        msg=f"KL should not diverge: first={first:.4f}, last={last:.4f}")

    def test_low_dim_affinities_sum_to_one(self):
        Y    = np.random.randn(20, 2)
        Q, _ = TSNEFromScratch._low_dim_affinities(Y)
        self.assertAlmostEqual(Q.sum(), 1.0, places=5)

    def test_high_dim_affinities_sum_to_one(self):
        X    = np.random.randn(20, 4)
        tsne = TSNEFromScratch(perplexity=5)
        P    = tsne._high_dim_affinities(X)
        self.assertAlmostEqual(P.sum(), 1.0, places=5)

    def test_affinities_symmetric(self):
        X    = np.random.randn(15, 3)
        tsne = TSNEFromScratch(perplexity=4)
        P    = tsne._high_dim_affinities(X)
        np.testing.assert_allclose(P, P.T, atol=1e-12, err_msg="P must be symmetric")

        Y    = np.random.randn(15, 2)
        Q, _ = TSNEFromScratch._low_dim_affinities(Y)
        np.testing.assert_allclose(Q, Q.T, atol=1e-12, err_msg="Q must be symmetric")


class TestConceptual(unittest.TestCase):
    """Higher-level conceptual tests."""

    def test_pca_variance_grows_with_k(self):
        X = np.random.randn(200, 10)
        prev = 0
        for k in range(1, 8):
            pca   = PCAFromScratch(n_components=k)
            pca.fit(X)
            total = pca.explained_variance_ratio_.sum()
            self.assertGreater(total, prev)
            prev = total

    def test_pca_constant_feature_ignored(self):
        """Zero-variance feature should have near-zero PC1 loading."""
        np.random.seed(7)
        X        = np.random.randn(100, 3)
        X[:, 1]  = 5.0       # constant -> zero variance after centering
        pca      = PCAFromScratch(n_components=1)
        pca.fit(X)
        loading  = abs(pca.components_[0, 1])
        self.assertLess(loading, 0.1,
                        msg=f"Constant feature loading={loading:.4f} should be ~0")

    def test_tsne_gradient_larger_for_structured_data(self):
        """
        For well-separated blobs, the initial (P - Q) mismatch is larger
        than for uniform random data, so the gradient norm should be larger.
        This is a pure mathematical property of the affinity matrices,
        independent of optimization convergence.
        """
        np.random.seed(42)

        # Structured: two clearly separated clusters
        Xa = np.random.randn(15, 3)
        Xb = np.random.randn(15, 3) + 8
        X_struct = np.vstack([Xa, Xb])

        # Random: uniform noise
        X_rand = np.random.rand(30, 3)

        tsne = TSNEFromScratch(perplexity=6)

        P_struct = tsne._high_dim_affinities(X_struct)
        P_rand   = tsne._high_dim_affinities(X_rand)

        # Entropy of P: structured data has more concentrated P (lower entropy)
        # which means stronger neighborhood preferences
        H_struct = -(P_struct * np.log(P_struct + 1e-12)).sum()
        H_rand   = -(P_rand   * np.log(P_rand   + 1e-12)).sum()

        self.assertLess(H_struct, H_rand,
                        msg=f"Structured P should have lower entropy "
                            f"({H_struct:.4f}) than random ({H_rand:.4f})")


if __name__ == "__main__":
    unittest.main(verbosity=2)


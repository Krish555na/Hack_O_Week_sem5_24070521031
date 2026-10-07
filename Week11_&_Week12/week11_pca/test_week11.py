"""
Week 11: Automated Unit Tests for Principal Component Analysis (PCA)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from pca_from_scratch import PCAFromScratch


class TestPCAMath(unittest.TestCase):
    """Core mathematical properties of PCA."""

    def setUp(self):
        np.random.seed(42)
        cov = [[2.0, 1.5], [1.5, 1.2]]
        self.X = np.random.multivariate_normal([0, 0], cov, 300)
        self.pca = PCAFromScratch(n_components=2)
        self.pca.fit(self.X)

    def test_components_are_orthogonal(self):
        dot = np.dot(self.pca.components_[0], self.pca.components_[1])
        self.assertAlmostEqual(dot, 0.0, places=10)

    def test_components_are_unit_vectors(self):
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
            pk = PCAFromScratch(n_components=k)
            Xr = pk.inverse_transform(pk.fit_transform(self.X))
            errors.append(np.mean((self.X - Xr) ** 2))
        self.assertTrue(all(errors[i] >= errors[i+1] for i in range(len(errors)-1)))

    def test_full_rank_reconstruction_perfect(self):
        pk = PCAFromScratch(n_components=5)
        Xr = pk.inverse_transform(pk.fit_transform(self.X))
        mse = np.mean((self.X - Xr) ** 2)
        self.assertAlmostEqual(mse, 0.0, places=10)

    def test_projections_uncorrelated(self):
        corr = np.corrcoef(self.Xp.T)[0, 1]
        self.assertAlmostEqual(corr, 0.0, places=5)


class TestPCAVerification(unittest.TestCase):
    """Verify PCA matches SVD (reference implementation)."""

    def test_evr_matches_svd(self):
        np.random.seed(3)
        X = np.random.randn(80, 6)
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


class TestPCAConceptual(unittest.TestCase):
    """Higher-level conceptual tests."""

    def test_pca_variance_grows_with_k(self):
        X = np.random.randn(200, 10)
        prev = 0
        for k in range(1, 8):
            pca = PCAFromScratch(n_components=k)
            pca.fit(X)
            total = pca.explained_variance_ratio_.sum()
            self.assertGreater(total, prev)
            prev = total

    def test_pca_constant_feature_ignored(self):
        np.random.seed(7)
        X = np.random.randn(100, 3)
        X[:, 1] = 5.0
        pca = PCAFromScratch(n_components=1)
        pca.fit(X)
        loading = abs(pca.components_[0, 1])
        self.assertLess(loading, 0.1)


if __name__ == "__main__":
    unittest.main()

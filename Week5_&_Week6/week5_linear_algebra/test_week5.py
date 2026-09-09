"""
Week 5: Automated Unit Tests for Linear Algebra Foundations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from vector_matrix_ops import (
    vector_norm, dot_product_scratch, cosine_similarity,
    angle_between_vectors, matrix_mult_scratch, inverse_2x2
)
from transformations_and_projections import (
    get_rotation_matrix, project_vector_onto_vector,
    gram_schmidt_orthogonalization
)
from eigenvalues_intuition import power_iteration, pca_intuition_demo
from visualize_transformations import visualize_eigen_preservation


class TestLinearAlgebra(unittest.TestCase):
    def test_vector_norm_and_dot(self):
        u = [3.0, 4.0]
        self.assertEqual(vector_norm(u, order=2), 5.0)
        self.assertEqual(vector_norm(u, order=1), 7.0)

        v = [4.0, 3.0]
        self.assertEqual(dot_product_scratch(u, v), 24.0)

        # Cosine similarity between orthogonal vectors should be 0
        u_orth = [1.0, 0.0]
        v_orth = [0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(u_orth, v_orth), 0.0)
        self.assertAlmostEqual(angle_between_vectors(u_orth, v_orth), 90.0)

    def test_matrix_mult_and_inverse(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[2.0, 0.0], [1.0, 2.0]]
        expected = [[4.0, 4.0], [10.0, 8.0]]
        self.assertEqual(matrix_mult_scratch(A, B), expected)

        A_inv = inverse_2x2(A)
        # A * A_inv should equal identity
        I = np.array(A) @ np.array(A_inv)
        np.testing.assert_allclose(I, np.eye(2), atol=1e-7)

    def test_projections_and_gram_schmidt(self):
        v = np.array([4.0, 4.0])
        u = np.array([2.0, 0.0])
        proj = project_vector_onto_vector(v, u)
        np.testing.assert_allclose(proj, [4.0, 0.0])

        # Orthonormal basis
        v1 = np.array([3.0, 1.0])
        v2 = np.array([2.0, 2.0])
        basis = gram_schmidt_orthogonalization([v1, v2])
        self.assertEqual(len(basis), 2)
        # Unit norm check
        self.assertAlmostEqual(np.linalg.norm(basis[0]), 1.0)
        self.assertAlmostEqual(np.linalg.norm(basis[1]), 1.0)
        # Orthogonality check
        self.assertAlmostEqual(np.dot(basis[0], basis[1]), 0.0, places=6)

    def test_power_iteration(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        val, vec = power_iteration(A)
        # Known dominant eigenvalue for [[2,1],[1,2]] is 3.0
        self.assertAlmostEqual(val, 3.0, places=3)
        # A * v should equal lambda * v
        np.testing.assert_allclose(A @ vec, val * vec, atol=1e-3)

    def test_visualization_output(self):
        plot_path = visualize_eigen_preservation(output_dir="test_plots")
        self.assertTrue(os.path.exists(plot_path))
        os.remove(plot_path)
        if os.path.exists("test_plots"):
            os.rmdir("test_plots")


if __name__ == "__main__":
    unittest.main()

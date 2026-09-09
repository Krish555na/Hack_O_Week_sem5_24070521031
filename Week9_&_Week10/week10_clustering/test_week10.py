"""
Week 10: Automated Unit Tests for Clustering Algorithms (K-Means, Agglomerative, DBSCAN)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from kmeans_clustering import KMeansScratch, silhouette_score_scratch
from hierarchical_clustering import AgglomerativeClusteringScratch, plot_dendrogram_scipy
from dbscan_clustering import DBSCANScratch
from clustering_benchmark import run_clustering_benchmark


class TestClusteringSuite(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 3 Well-separated Gaussian clusters
        c1 = np.random.normal(loc=[-5.0, -5.0], scale=0.4, size=(30, 2))
        c2 = np.random.normal(loc=[0.0, 0.0], scale=0.4, size=(30, 2))
        c3 = np.random.normal(loc=[5.0, 5.0], scale=0.4, size=(30, 2))
        self.X = np.vstack([c1, c2, c3])

    def test_kmeans_convergence_and_silhouette(self):
        km = KMeansScratch(n_clusters=3, random_state=42).fit(self.X)
        self.assertEqual(len(km.centroids), 3)
        self.assertEqual(len(np.unique(km.labels_)), 3)

        sil = silhouette_score_scratch(self.X, km.labels_)
        # Separated clusters should yield high silhouette score (> 0.7)
        self.assertGreater(sil, 0.70)

    def test_hierarchical_clustering_linkages(self):
        for link in ["single", "complete", "average"]:
            agg = AgglomerativeClusteringScratch(n_clusters=3, linkage=link).fit(self.X)
            self.assertEqual(len(np.unique(agg.labels_)), 3)
            self.assertEqual(len(agg.labels_), len(self.X))

    def test_dbscan_clustering_and_noise(self):
        # Add obvious outliers
        outliers = np.array([[-15.0, 15.0], [20.0, -20.0]])
        X_with_noise = np.vstack([self.X, outliers])

        db = DBSCANScratch(eps=1.2, min_samples=5).fit(X_with_noise)
        # Outliers should be assigned label -1
        self.assertEqual(db.labels_[-1], -1)
        self.assertEqual(db.labels_[-2], -1)
        # Main clusters should be discovered
        non_noise_labels = set(db.labels_[db.labels_ != -1])
        self.assertGreaterEqual(len(non_noise_labels), 2)

    def test_benchmark_grid_plot(self):
        test_dir = "test_plots_w10"
        plot_path = run_clustering_benchmark(output_dir=test_dir)
        self.assertTrue(os.path.exists(plot_path))
        os.remove(plot_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)

    def test_dendrogram_plot(self):
        dend_path = "test_dendrogram.png"
        plot_dendrogram_scipy(self.X[:20], output_path=dend_path)
        self.assertTrue(os.path.exists(dend_path))
        os.remove(dend_path)


if __name__ == "__main__":
    unittest.main()

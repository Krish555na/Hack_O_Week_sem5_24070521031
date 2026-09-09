"""
Week 4: Automated Unit Tests for NumPy, Pandas & Data Visualization Pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
import pandas as pd

from numpy_operations import (
    demonstrate_array_creation_and_slicing,
    demonstrate_broadcasting,
    benchmark_vectorization
)
from pandas_pipeline import (
    generate_sample_datasets,
    clean_and_impute_data,
    execute_relational_pipeline
)
from visualizations import generate_all_charts


class TestNumPyOperations(unittest.TestCase):
    def test_array_slicing_and_masking(self):
        res = demonstrate_array_creation_and_slicing()
        self.assertEqual(res["grid_2d"].shape, (5, 5))
        self.assertEqual(res["sub_matrix"].shape, (3, 3))
        # Mask: even and > 10
        for val in res["even_gt_10"]:
            self.assertEqual(val % 2, 0)
            self.assertGreater(val, 10)

    def test_broadcasting(self):
        res = demonstrate_broadcasting()
        self.assertEqual(res["outer_sum_shape"], (3, 4))
        # Standardized mean should be close to 0
        np.testing.assert_allclose(res["standardized_mean"], [0.0, 0.0, 0.0], atol=1e-5)
        # Standardized std should be close to 1
        np.testing.assert_allclose(res["standardized_std"], [1.0, 1.0, 1.0], atol=1e-5)

    def test_vectorization_benchmark(self):
        bench = benchmark_vectorization(10_000)
        self.assertTrue(bench["results_match"])
        self.assertGreater(bench["speedup_factor"], 1.0)


class TestPandasPipeline(unittest.TestCase):
    def test_dataset_generation_and_imputation(self):
        c_df, p_df, o_df = generate_sample_datasets()
        self.assertEqual(len(c_df), 12)
        self.assertEqual(len(p_df), 6)
        self.assertEqual(len(o_df), 30)

        # Before cleaning, check for nulls
        self.assertTrue(c_df["credit_score"].isnull().any())

        clean_c, clean_o = clean_and_impute_data(c_df, o_df)
        self.assertFalse(clean_c["credit_score"].isnull().any())
        self.assertFalse(clean_o["discount_pct"].isnull().any())

    def test_relational_aggregations(self):
        pipeline = execute_relational_pipeline()
        full_df = pipeline["full_dataset"]
        self.assertEqual(len(full_df), 30)
        self.assertIn("net_revenue", full_df.columns)
        self.assertTrue((full_df["net_revenue"] >= 0).all())

        tier_summary = pipeline["tier_summary"]
        self.assertTrue("total_spend" in tier_summary.columns)

    def test_chart_generation(self):
        test_dir = "test_output_charts"
        charts = generate_all_charts(output_dir=test_dir)
        self.assertEqual(len(charts), 4)
        for c in charts:
            self.assertTrue(os.path.exists(c))
            self.assertGreater(os.path.getsize(c), 1000)
            os.remove(c)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


if __name__ == "__main__":
    unittest.main()

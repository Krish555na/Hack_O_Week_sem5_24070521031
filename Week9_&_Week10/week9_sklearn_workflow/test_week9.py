"""
Week 9: Automated Unit Tests for Scikit-learn Pipeline & Evaluation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import pandas as pd
from pipeline_churn_prediction import (
    generate_churn_dataset, build_full_pipeline,
    train_and_tune_model, predict_single_customer
)
from evaluation_and_interpretation import run_evaluation_and_interpretation


class TestSklearnPipeline(unittest.TestCase):
    def setUp(self):
        self.df = generate_churn_dataset(n_samples=300)

    def test_dataset_schema(self):
        self.assertEqual(len(self.df), 300)
        self.assertIn("churn", self.df.columns)
        self.assertTrue(self.df["total_charges"].isnull().any())

    def test_pipeline_fit_and_inference(self):
        X = self.df.drop(columns=["churn"])
        y = self.df["churn"]
        pipeline, _, _, _ = build_full_pipeline()
        pipeline.fit(X, y)

        preds = pipeline.predict(X)
        self.assertEqual(len(preds), len(y))

        # Probabilities should sum to 1
        probs = pipeline.predict_proba(X)
        self.assertAlmostEqual(float(probs[0].sum()), 1.0, places=5)

    def test_model_persistence_and_prediction(self):
        model_file = "test_pipeline.joblib"
        _ = train_and_tune_model(save_model_path=model_file)
        self.assertTrue(os.path.exists(model_file))

        sample = {
            "tenure_months": 5.0,
            "monthly_charges": 65.0,
            "total_charges": 325.0,
            "support_tickets": 1,
            "logins_per_month": 15.0,
            "contract_type": "One Year",
            "payment_method": "Credit Card",
            "internet_service": "DSL",
            "paperless_billing": "No"
        }
        res = predict_single_customer(model_file, sample)
        self.assertIn("churn_prediction", res)
        self.assertGreaterEqual(res["churn_probability"], 0.0)
        self.assertLessEqual(res["churn_probability"], 1.0)

        if os.path.exists(model_file):
            os.remove(model_file)

    def test_evaluation_plots(self):
        test_dir = "test_plots_w9"
        res = run_evaluation_and_interpretation(output_dir=test_dir)
        self.assertTrue(os.path.exists(res["chart_path"]))
        os.remove(res["chart_path"])
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


if __name__ == "__main__":
    unittest.main()

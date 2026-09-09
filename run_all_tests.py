"""
Master Test Runner for 10-Week AI/ML & Fullstack Curriculum

Executes all automated test suites across all 5 folders:
- Folder 2 (Weeks 3 & 4): Python Essentials, NumPy, Pandas, DataViz
- Folder 3 (Weeks 5 & 6): Linear Algebra, Calculus & Gradients
- Folder 4 (Weeks 7 & 8): Regression, Classification & Metrics
- Folder 5 (Weeks 9 & 10): Scikit-learn Pipeline & Clustering
"""

import sys
import unittest
import time


def run_curriculum_tests():
    test_modules = [
        "02_python_data_essentials_week3_4.week3_python_essentials.test_week3",
        "02_python_data_essentials_week3_4.week4_numpy_pandas_dataviz.test_week4",
        "03_math_foundations_ml_week5_6.week5_linear_algebra.test_week5",
        "03_math_foundations_ml_week5_6.week6_calculus_gradients.test_week6",
        "04_regression_classification_week7_8.week7_regression.test_week7",
        "04_regression_classification_week7_8.week8_classification.test_week8",
        "05_scikit_learn_clustering_week9_10.week9_sklearn_workflow.test_week9",
        "05_scikit_learn_clustering_week9_10.week10_clustering.test_week10"
    ]

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 70)
    print("[*] Running 10-Week AI/ML & Web Development Full Test Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for mod in test_modules:
        suite.addTests(loader.loadTestsFromName(mod))

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Successful:      {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:        {len(result.failures)}")
    print(f"Errors:          {len(result.errors)}")
    print(f"Elapsed Time:    {elapsed:.2f} seconds")
    print("=" * 70)

    if result.wasSuccessful():
        print("[SUCCESS] ALL CURRICULUM TEST SUITES PASSED PERFECTLY!")
        return 0
    else:
        print("[FAIL] SOME TESTS FAILED. PLEASE REVIEW LOGS.")
        return 1


if __name__ == "__main__":
    sys.exit(run_curriculum_tests())

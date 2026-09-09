"""
Week 9: Scikit-learn Workflow — End-to-End Production ML Pipeline & Model Persistence

Demonstrates best practices in modern scikit-learn workflows:
1. Synthetic generation of realistic customer churn dataset (imbalanced classes, mixed types, nulls)
2. Modular preprocessing with ColumnTransformer:
   - Numerical: SimpleImputer(median) -> StandardScaler
   - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder
3. Composite Pipeline with classification estimator
4. Hyperparameter tuning via GridSearchCV with StratifiedKFold
5. Serialization to disk with joblib and round-trip inference on raw payloads
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def generate_churn_dataset(n_samples: int = 1200) -> pd.DataFrame:
    """Generates realistic customer churn dataset with mixed feature types and missing values."""
    np.random.seed(42)

    tenure = np.random.exponential(scale=24.0, size=n_samples).clip(1, 72)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50.0, size=n_samples)
    total_charges = np.clip(total_charges, 20.0, None)

    support_tickets = np.random.poisson(lam=1.5, size=n_samples)
    logins_per_month = np.random.normal(loc=18.0, scale=5.0, size=n_samples).clip(1, 40)

    contract_type = np.random.choice(["Month-to-Month", "One Year", "Two Year"], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice(["Electronic Check", "Bank Transfer", "Credit Card", "Mailed Check"], size=n_samples)
    internet_service = np.random.choice(["Fiber Optic", "DSL", "No"], size=n_samples, p=[0.45, 0.35, 0.20])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])

    # Inject random missing values (5% rate)
    mask_null_charges = np.random.rand(n_samples) < 0.05
    total_charges[mask_null_charges] = np.nan

    # Target: Churn probability driven by contract type, support tickets, and monthly charges
    logits = (
        -1.8
        + 0.02 * monthly_charges
        - 0.04 * tenure
        + 0.45 * support_tickets
        + (contract_type == "Month-to-Month") * 0.8
        - (contract_type == "Two Year") * 1.2
        + (internet_service == "Fiber Optic") * 0.3
    )
    prob_churn = 1.0 / (1.0 + np.exp(-logits))
    churn = (np.random.rand(n_samples) < prob_churn).astype(int)

    df = pd.DataFrame({
        "tenure_months": np.round(tenure, 1),
        "monthly_charges": np.round(monthly_charges, 2),
        "total_charges": np.round(total_charges, 2),
        "support_tickets": support_tickets,
        "logins_per_month": np.round(logins_per_month, 1),
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "paperless_billing": paperless_billing,
        "churn": churn
    })

    return df


def build_full_pipeline() -> Tuple[Pipeline, Dict[str, list], list, list]:
    """Constructs the preprocessor ColumnTransformer and Model Pipeline."""
    numerical_cols = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "logins_per_month"]
    categorical_cols = ["contract_type", "payment_method", "internet_service", "paperless_billing"]

    # 1. Numerical Pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 2. Categorical Pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 3. Composite Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ]
    )

    # 4. Master Pipeline with Model
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        "classifier__n_estimators": [30, 60],
        "classifier__learning_rate": [0.1],
        "classifier__max_depth": [3]
    }

    return full_pipeline, param_grid, numerical_cols, categorical_cols


def train_and_tune_model(save_model_path: str = "churn_pipeline.joblib") -> Dict[str, Any]:
    """Runs complete end-to-end training, GridSearch CV, and serializes model artifact."""
    df = generate_churn_dataset(n_samples=400)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline, param_grid, num_cols, cat_cols = build_full_pipeline()

    # Stratified 3-Fold Grid Search
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Evaluation on unseen holdout test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "best_params": grid_search.best_params_,
        "best_cv_score_auc": round(grid_search.best_score_, 4),
        "test_roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_f1": round(f1_score(y_test, y_pred), 4)
    }

    # Persist model
    joblib.dump(best_model, save_model_path)

    return {
        "metrics": metrics,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "model_path": save_model_path
    }


def predict_single_customer(model_path: str, raw_customer_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Infers prediction on a single un-preprocessed customer dictionary."""
    model = joblib.load(model_path)
    sample_df = pd.DataFrame([raw_customer_dict])
    churn_prob = model.predict_proba(sample_df)[0, 1]
    prediction = int(model.predict(sample_df)[0])

    return {
        "churn_prediction": "Likely to Churn" if prediction == 1 else "Likely to Retain",
        "churn_probability": round(float(churn_prob), 4)
    }


if __name__ == "__main__":
    print("=== Training Production Scikit-learn Pipeline ===")
    res = train_and_tune_model("test_churn_pipeline.joblib")
    print("Best Hyperparameters:", res["metrics"]["best_params"])
    print(f"5-Fold CV ROC-AUC:    {res['metrics']['best_cv_score_auc']:.4f}")
    print(f"Holdout Test ROC-AUC:  {res['metrics']['test_roc_auc']:.4f}")
    print(f"Holdout Test F1-Score: {res['metrics']['test_f1']:.4f}")

    print("\n=== Single Customer Live Inference ===")
    sample_cust = {
        "tenure_months": 2.0,
        "monthly_charges": 95.50,
        "total_charges": None,  # Testing missing value handling
        "support_tickets": 4,
        "logins_per_month": 8.0,
        "contract_type": "Month-to-Month",
        "payment_method": "Electronic Check",
        "internet_service": "Fiber Optic",
        "paperless_billing": "Yes"
    }
    pred_res = predict_single_customer("test_churn_pipeline.joblib", sample_cust)
    print("Payload:", sample_cust)
    print("Inference Result:", pred_res)

    if os.path.exists("test_churn_pipeline.joblib"):
        os.remove("test_churn_pipeline.joblib")

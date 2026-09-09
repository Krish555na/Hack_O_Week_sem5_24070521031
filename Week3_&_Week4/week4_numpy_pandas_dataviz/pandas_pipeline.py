"""
Week 4: Pandas Pipeline — DataFrames, Cleaning, Merging & GroupBy Aggregations

This module implements a complete, reproducible data pipeline:
1. Synthetic generation of relational business datasets (Customers, Products, Orders)
2. Handling null values via statistical imputation (median for continuous, mode for categorical)
3. Relational joins: Left, Inner, and Outer merges
4. Advanced multi-level GroupBy aggregations (Revenue, Order Counts, Average Order Value)
5. Pivot tables and cross-tabulation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def generate_sample_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates realistic relational DataFrames for learning and experimentation."""
    np.random.seed(42)

    # 1. Customers Table
    customers_df = pd.DataFrame({
        "customer_id": [f"CUST-{100 + i}" for i in range(12)],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona", "George", "Hannah", "Ian", "Julia", "Kevin", "Laura"],
        "tier": ["Gold", "Silver", "Platinum", "Gold", "Bronze", "Silver", "Gold", "Platinum", "Bronze", "Silver", "Gold", "Bronze"],
        "region": ["North", "West", "East", "North", "South", "West", "East", "North", "South", "East", "West", "North"],
        "credit_score": [720, 680, 810, 750, np.nan, 690, 710, 840, 620, np.nan, 770, 650]  # Missing values injected
    })

    # 2. Products Table
    products_df = pd.DataFrame({
        "product_id": [f"PROD-{10 + i}" for i in range(6)],
        "category": ["Electronics", "Electronics", "Books", "Books", "Office", "Office"],
        "product_name": ["Noise-Cancelling Headphones", "4K Monitor", "ML Systems Book", "Deep Learning Textbook", "Ergonomic Chair", "Mechanical Keyboard"],
        "unit_price": [299.99, 449.99, 64.99, 89.99, 219.99, 129.99]
    })

    # 3. Orders Table
    n_orders = 30
    orders_df = pd.DataFrame({
        "order_id": [f"ORD-{1000 + i}" for i in range(n_orders)],
        "customer_id": np.random.choice(customers_df["customer_id"], size=n_orders),
        "product_id": np.random.choice(products_df["product_id"], size=n_orders),
        "quantity": np.random.choice([1, 2, 3, 4, 5], size=n_orders, p=[0.5, 0.25, 0.15, 0.05, 0.05]),
        "discount_pct": np.random.choice([0.0, 0.05, 0.10, 0.15, np.nan], size=n_orders, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    })

    return customers_df, products_df, orders_df


def clean_and_impute_data(customers_df: pd.DataFrame, orders_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cleans missing data and applies statistical imputations."""
    c_df = customers_df.copy()
    o_df = orders_df.copy()

    # Impute missing credit_score with median of their respective tier
    c_df["credit_score"] = c_df.groupby("tier")["credit_score"].transform(
        lambda s: s.fillna(s.median())
    )
    # Fallback to global median if still null
    c_df["credit_score"] = c_df["credit_score"].fillna(c_df["credit_score"].median())

    # Impute missing discount_pct in orders with 0.0 (no discount)
    o_df["discount_pct"] = o_df["discount_pct"].fillna(0.0)

    return c_df, o_df


def execute_relational_pipeline() -> Dict[str, Any]:
    """Executes end-to-end cleaning, joining, and aggregation pipeline."""
    raw_cust, raw_prod, raw_orders = generate_sample_datasets()
    clean_cust, clean_orders = clean_and_impute_data(raw_cust, raw_orders)

    # 1. Merge: Orders -> Products
    orders_with_prod = pd.merge(
        clean_orders,
        raw_prod,
        on="product_id",
        how="inner"
    )

    # Calculate financial line totals
    orders_with_prod["gross_amount"] = orders_with_prod["quantity"] * orders_with_prod["unit_price"]
    orders_with_prod["net_revenue"] = orders_with_prod["gross_amount"] * (1 - orders_with_prod["discount_pct"])

    # 2. Merge: (Orders+Prod) -> Customers
    full_dataset = pd.merge(
        orders_with_prod,
        clean_cust,
        on="customer_id",
        how="left"
    )

    # 3. Multi-Level GroupBy: Performance by Region & Product Category
    region_category_perf = full_dataset.groupby(["region", "category"]).agg(
        total_orders=("order_id", "count"),
        total_quantity=("quantity", "sum"),
        total_revenue=("net_revenue", "sum"),
        avg_order_value=("net_revenue", "mean")
    ).reset_index()

    # 4. Customer Tier Performance Aggregation
    tier_summary = full_dataset.groupby("tier").agg(
        active_customers=("customer_id", "nunique"),
        total_spend=("net_revenue", "sum"),
        avg_spend_per_order=("net_revenue", "mean")
    ).sort_values("total_spend", ascending=False)

    # 5. Pivot Table: Category Revenue across Customer Tiers
    revenue_pivot = pd.pivot_table(
        full_dataset,
        values="net_revenue",
        index="tier",
        columns="category",
        aggfunc="sum",
        fill_value=0.0
    )

    return {
        "full_dataset": full_dataset,
        "region_category_perf": region_category_perf,
        "tier_summary": tier_summary,
        "revenue_pivot": revenue_pivot
    }


if __name__ == "__main__":
    print("=== Running Week 4 Pandas Relational Pipeline ===")
    results = execute_relational_pipeline()
    print("\nMerged Dataset Sample (first 3 rows):")
    print(results["full_dataset"][["order_id", "customer_id", "product_name", "quantity", "net_revenue", "tier"]].head(3))

    print("\n--- Region & Category GroupBy Aggregations ---")
    print(results["region_category_perf"].to_string(index=False))

    print("\n--- Customer Tier Summary ---")
    print(results["tier_summary"])

    print("\n--- Revenue Pivot Table (Tier vs Category) ---")
    print(results["revenue_pivot"].round(2))

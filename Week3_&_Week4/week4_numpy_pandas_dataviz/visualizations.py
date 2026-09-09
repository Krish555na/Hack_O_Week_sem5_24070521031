"""
Week 4: Data Visualization — Matplotlib & Seaborn Charting Suite

Generates publication-quality charts:
1. Distribution Analysis (Histogram + KDE with summary statistics)
2. Correlation Matrix Heatmap (diverging colormap, formatted annotations)
3. Relational Scatter Plot with hue and style encoding
4. Categorical Box & Violin plots with strip overlays
All figures are saved to 'output_charts/' directory.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas_pipeline import execute_relational_pipeline


def configure_visual_style():
    """Sets a clean, modern aesthetic for charts."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "figure.titlesize": 16,
        "figure.dpi": 150
    })


def generate_all_charts(output_dir: str = "output_charts") -> list:
    """Generates and persists all 4 analytical figures."""
    os.makedirs(output_dir, exist_ok=True)
    configure_visual_style()

    pipeline_res = execute_relational_pipeline()
    df = pipeline_res["full_dataset"]
    generated_files = []

    # -------------------------------------------------------------
    # 1. Distribution of Net Order Revenues
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["net_revenue"], kde=True, color="#6366f1", bins=15, ax=ax)
    mean_rev = df["net_revenue"].mean()
    median_rev = df["net_revenue"].median()
    ax.axvline(mean_rev, color="#ef4444", linestyle="--", linewidth=1.8, label=f"Mean: ${mean_rev:.2f}")
    ax.axvline(median_rev, color="#10b981", linestyle="-.", linewidth=1.8, label=f"Median: ${median_rev:.2f}")

    ax.set_title("Distribution of Net Order Revenue", fontweight="bold", pad=12)
    ax.set_xlabel("Net Revenue ($)")
    ax.set_ylabel("Order Count")
    ax.legend()
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "01_revenue_distribution.png")
    fig.savefig(chart1_path)
    plt.close(fig)
    generated_files.append(chart1_path)

    # -------------------------------------------------------------
    # 2. Correlation Matrix Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    num_cols = ["quantity", "discount_pct", "unit_price", "gross_amount", "net_revenue", "credit_score"]
    corr_matrix = df[num_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson Correlation"},
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Multivariate Correlation Heatmap", fontweight="bold", pad=12)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "02_correlation_heatmap.png")
    fig.savefig(chart2_path)
    plt.close(fig)
    generated_files.append(chart2_path)

    # -------------------------------------------------------------
    # 3. Relational Scatter: Unit Price vs Net Revenue by Category
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=df,
        x="unit_price",
        y="net_revenue",
        hue="category",
        size="quantity",
        sizes=(40, 200),
        style="tier",
        palette="bright",
        alpha=0.85,
        ax=ax
    )
    ax.set_title("Unit Price vs. Net Revenue by Product Category & Customer Tier", fontweight="bold", pad=12)
    ax.set_xlabel("Unit Price ($)")
    ax.set_ylabel("Net Order Revenue ($)")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "03_relational_scatter.png")
    fig.savefig(chart3_path)
    plt.close(fig)
    generated_files.append(chart3_path)

    # -------------------------------------------------------------
    # 4. Categorical Boxplot: Revenue Distribution by Customer Tier
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="tier",
        y="net_revenue",
        hue="tier",
        legend=False,
        order=["Bronze", "Silver", "Gold", "Platinum"],
        palette="Set2",
        ax=ax
    )
    sns.stripplot(
        data=df,
        x="tier",
        y="net_revenue",
        order=["Bronze", "Silver", "Gold", "Platinum"],
        color="black",
        alpha=0.5,
        jitter=0.2,
        size=5,
        ax=ax
    )
    ax.set_title("Order Revenue Spread Across Customer Loyalty Tiers", fontweight="bold", pad=12)
    ax.set_xlabel("Customer Loyalty Tier")
    ax.set_ylabel("Net Revenue ($)")
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "04_tier_revenue_boxplot.png")
    fig.savefig(chart4_path)
    plt.close(fig)
    generated_files.append(chart4_path)

    return generated_files


if __name__ == "__main__":
    print("Generating Matplotlib & Seaborn charts...")
    saved = generate_all_charts()
    for p in saved:
        print(f" Saved chart: {p}")

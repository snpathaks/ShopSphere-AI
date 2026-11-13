"""
generate_plots.py - Generate high-resolution figures for EDA and Model Evaluation.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Output directory
fig_dir = Path(__file__).resolve().parent.parent / "notebooks" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# 1. Load data
data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
users = pd.read_csv(data_dir / "users.csv")
products = pd.read_csv(data_dir / "products.csv")
interactions = pd.read_csv(data_dir / "interactions.csv")
interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], utc=True)

# Plot 1: User Activity Distribution
plt.figure(figsize=(8, 4.5))
user_counts = interactions.groupby("user_id").size()
sns.histplot(user_counts, bins=30, kde=True, color="#4C72B0")
plt.axvline(user_counts.median(), color="red", linestyle="--", label=f"Median ({int(user_counts.median())})")
plt.title("Distribution of Interaction Events per User", fontsize=13, fontweight="bold")
plt.xlabel("Number of Interactions")
plt.ylabel("Number of Users")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "eda_user_activity.png", dpi=200)
plt.close()

# Plot 2: Long-tail Item Popularity
plt.figure(figsize=(8, 4.5))
item_counts = interactions.groupby("item_id").size().sort_values(ascending=False).values
plt.plot(range(len(item_counts)), item_counts, color="#C44E52", lw=2)
plt.fill_between(range(len(item_counts)), item_counts, color="#C44E52", alpha=0.3)
plt.title("Product Popularity Distribution (The Long Tail)", fontsize=13, fontweight="bold")
plt.xlabel("Item Rank (Ordered by Popularity)")
plt.ylabel("Interaction Frequency")
plt.tight_layout()
plt.savefig(fig_dir / "eda_item_popularity_long_tail.png", dpi=200)
plt.close()

# Plot 3: Interaction Funnel (View -> Cart -> Purchase)
plt.figure(figsize=(7, 4.5))
type_counts = interactions["interaction_type"].value_counts()
colors = ["#4C72B0", "#DD8452", "#55A868"]
sns.barplot(x=type_counts.index, y=type_counts.values, palette=colors)
for idx, val in enumerate(type_counts.values):
    pct = val / len(interactions) * 100
    plt.text(idx, val + 150, f"{val:,}\n({pct:.1f}%)", ha="center", fontweight="bold", fontsize=10)
plt.title("Interaction Type Breakdown (Implicit Conversion Funnel)", fontsize=13, fontweight="bold")
plt.xlabel("Interaction Type")
plt.ylabel("Total Events")
plt.ylim(0, max(type_counts.values) * 1.15)
plt.tight_layout()
plt.savefig(fig_dir / "eda_interaction_funnel.png", dpi=200)
plt.close()

# Plot 4: Temporal Interaction Volume over 180 Days
plt.figure(figsize=(9, 4.5))
daily_counts = interactions.set_index("timestamp").resample("D").size()
daily_counts.plot(color="#8172B3", lw=1.5)
plt.title("Daily Interaction Volume over 180-Day Observation Window", fontsize=13, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Daily Interactions")
plt.tight_layout()
plt.savefig(fig_dir / "eda_temporal_activity.png", dpi=200)
plt.close()

# Plot 5: Model Comparison (NDCG@10, Recall@10, Precision@10)
eval_path = Path(__file__).resolve().parent.parent / "models" / "evaluation_results.json"
if eval_path.exists():
    with open(eval_path) as f:
        eval_data = json.load(f)
    
    comp_df = pd.DataFrame(eval_data["comparison"])
    comp_k10 = comp_df[comp_df["k"] == 10].copy()
    
    # Model Comparison Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comp_k10))
    width = 0.25
    
    ax.bar(x - width, comp_k10["ndcg@k"], width, label="NDCG@10", color="#4C72B0")
    ax.bar(x, comp_k10["recall@k"], width, label="Recall@10", color="#55A868")
    ax.bar(x + width, comp_k10["precision@k"], width, label="Precision@10", color="#DD8452")
    
    ax.set_ylabel("Metric Score")
    ax.set_title("Recommendation Model Performance on Held-Out Test Set (K=10)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    clean_names = [n.replace("Recommender", "") for n in comp_k10["model_name"]]
    ax.set_xticklabels(clean_names, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison_k10.png", dpi=200)
    plt.close()

    # Plot 6: Catalog Coverage
    plt.figure(figsize=(7, 4.5))
    coverage = eval_data.get("coverage", {})
    cov_df = pd.DataFrame(list(coverage.items()), columns=["Model", "Coverage"])
    cov_df["Coverage Pct"] = cov_df["Coverage"] * 100
    sns.barplot(data=cov_df, x="Model", y="Coverage Pct", palette="Blues_d")
    for idx, row in cov_df.iterrows():
        plt.text(idx, row["Coverage Pct"] + 1.5, f"{row['Coverage Pct']:.1f}%", ha="center", fontweight="bold")
    plt.title("Catalog Coverage (%) Across Recommendation Models", fontsize=13, fontweight="bold")
    plt.ylabel("Catalog Coverage (%)")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(fig_dir / "model_catalog_coverage.png", dpi=200)
    plt.close()

print(f"Generated 6 figures in {fig_dir}")


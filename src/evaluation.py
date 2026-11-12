"""
evaluation.py
Evaluation framework for the E-Commerce Product Recommendation Engine.
"""

import numpy as np
import pandas as pd
from typing import List, Dict


# ---------------------------------------------------------------------------
# Core ranking metrics
# ---------------------------------------------------------------------------

def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """P@K = |recommended[:k] ∩ relevant| / k"""
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """R@K = |recommended[:k] ∩ relevant| / |relevant|"""
    if not relevant or k <= 0:
        return 0.0
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
    DCG@K  = sum(rel_i / log2(i+2)) for i in 0..K-1
    rel_i  = 1 if recommended[i] in relevant else 0
    """
    if not relevant or k <= 0:
        return 0.0

    relevant_set = set(relevant)
    top_k = recommended[:k]

    # DCG
    dcg = sum(
        (1.0 / np.log2(i + 2)) for i, item in enumerate(top_k) if item in relevant_set
    )

    # IDCG – ideal: all relevant items at the top positions
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Model-level evaluation helpers
# ---------------------------------------------------------------------------

def _get_recommendations_for_user(model, user_id: str, train_items: List[str], k: int) -> List[str]:
    """Return top-k recommendations for a user, excluding already-seen train items."""
    try:
        recs = model.recommend(user_id=user_id, n=k + len(train_items))
        # recs may be a list of dicts {'item_id': ..., 'score': ...} or list of str
        if recs and isinstance(recs[0], dict):
            item_ids = [r["item_id"] for r in recs]
        else:
            item_ids = list(recs)
        # Exclude training items
        train_set = set(train_items)
        filtered = [i for i in item_ids if i not in train_set]
        return filtered[:k]
    except Exception:
        return []


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int = 10,
) -> Dict[str, float]:
    """
    For each user in the test set:
      - Get their test items (ground truth relevant items)
      - Get recommendations from model (exclude training items)
      - Compute P@K, R@K, NDCG@K
    Returns dict with mean metrics:
      {"precision@k": ..., "recall@k": ..., "ndcg@k": ..., "k": k}
    Only evaluates users with at least 1 test interaction.
    """
    # Build per-user lookup tables
    test_user_items: Dict[str, List[str]] = (
        test_df.groupby("user_id")["item_id"].apply(list).to_dict()
    )
    train_user_items: Dict[str, List[str]] = (
        train_df.groupby("user_id")["item_id"].apply(list).to_dict()
    )

    precisions, recalls, ndcgs = [], [], []

    for user_id, relevant_items in test_user_items.items():
        if not relevant_items:
            continue
        train_items = train_user_items.get(user_id, [])
        recommended = _get_recommendations_for_user(model, user_id, train_items, k)

        precisions.append(precision_at_k(recommended, relevant_items, k))
        recalls.append(recall_at_k(recommended, relevant_items, k))
        ndcgs.append(ndcg_at_k(recommended, relevant_items, k))

    n = len(precisions)
    if n == 0:
        return {"precision@k": 0.0, "recall@k": 0.0, "ndcg@k": 0.0, "k": k}

    return {
        "precision@k": float(np.mean(precisions)),
        "recall@k": float(np.mean(recalls)),
        "ndcg@k": float(np.mean(ndcgs)),
        "k": k,
    }


# ---------------------------------------------------------------------------
# Catalog coverage
# ---------------------------------------------------------------------------

def catalog_coverage(
    recommendations: Dict[str, List[str]], all_items: List[str]
) -> float:
    """
    Coverage = |unique recommended items| / |all items|
    recommendations: {user_id: [item_id, ...]}
    """
    if not all_items:
        return 0.0
    unique_recommended: set = set()
    for items in recommendations.values():
        unique_recommended.update(items)
    return len(unique_recommended) / len(set(all_items))


# ---------------------------------------------------------------------------
# Cold-start analysis
# ---------------------------------------------------------------------------

def cold_start_analysis(
    model,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int = 10,
) -> Dict[str, Dict]:
    """
    Segment users by training interaction count:
      - cold: 0–5 interactions
      - warm: 6–20 interactions
      - hot : 21+ interactions
    Evaluate each segment separately.
    Returns {"cold": metrics_dict, "warm": metrics_dict, "hot": metrics_dict}
    """
    # Count per-user training interactions
    train_counts = train_df.groupby("user_id").size().to_dict()

    # All users who appear in test set
    test_users = set(test_df["user_id"].unique())

    segments: Dict[str, List[str]] = {"cold": [], "warm": [], "hot": []}
    for user in test_users:
        cnt = train_counts.get(user, 0)
        if cnt <= 5:
            segments["cold"].append(user)
        elif cnt <= 20:
            segments["warm"].append(user)
        else:
            segments["hot"].append(user)

    results: Dict[str, Dict] = {}
    for segment, users in segments.items():
        if not users:
            results[segment] = {
                "precision@k": 0.0,
                "recall@k": 0.0,
                "ndcg@k": 0.0,
                "k": k,
                "n_users": 0,
            }
            continue
        seg_test = test_df[test_df["user_id"].isin(users)]
        seg_train = train_df[train_df["user_id"].isin(users)]
        metrics = evaluate_model(model, seg_test, seg_train, k=k)
        metrics["n_users"] = len(users)
        results[segment] = metrics

    return results


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(
    models: list,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k_values: list = [5, 10, 20],
) -> pd.DataFrame:
    """
    Run evaluate_model for each model at each K value.
    Returns a DataFrame with columns:
      [model_name, k, precision@k, recall@k, ndcg@k]
    """
    rows = []
    for model in models:
        # Determine a display name
        name = getattr(model, "name", None) or type(model).__name__
        for k in k_values:
            metrics = evaluate_model(model, test_df, train_df, k=k)
            rows.append(
                {
                    "model_name": name,
                    "k": k,
                    "precision@k": metrics["precision@k"],
                    "recall@k": metrics["recall@k"],
                    "ndcg@k": metrics["ndcg@k"],
                }
            )

    return pd.DataFrame(rows, columns=["model_name", "k", "precision@k", "recall@k", "ndcg@k"])

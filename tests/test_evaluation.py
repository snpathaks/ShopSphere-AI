"""
test_evaluation.py - Unit tests for ranking metrics, NDCG, catalog coverage, and cold-start analysis.
"""

import pytest
import numpy as np
from src.evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    catalog_coverage,
    evaluate_model,
    cold_start_analysis,
)
from src.models.popularity import PopularityRecommender


def test_precision_at_k():
    rec = ["I1", "I2", "I3", "I4", "I5"]
    rel = ["I1", "I3", "I9"]
    # At K=3: top 3 are [I1, I2, I3], hits are I1, I3 -> 2/3
    assert precision_at_k(rec, rel, k=3) == pytest.approx(2 / 3)
    # At K=5: hits are I1, I3 -> 2/5 = 0.4
    assert precision_at_k(rec, rel, k=5) == pytest.approx(0.4)
    # No overlap
    assert precision_at_k(rec, ["I99"], k=5) == 0.0


def test_recall_at_k():
    rec = ["I1", "I2", "I3", "I4", "I5"]
    rel = ["I1", "I3", "I9"]  # total relevant = 3
    # At K=3: hits are I1, I3 -> 2/3
    assert recall_at_k(rec, rel, k=3) == pytest.approx(2 / 3)
    # At K=1: hit is I1 -> 1/3
    assert recall_at_k(rec, rel, k=1) == pytest.approx(1 / 3)
    # Empty relevant
    assert recall_at_k(rec, [], k=5) == 0.0


def test_ndcg_at_k_ranking_sensitivity():
    # Ideal ranking: relevant items first
    ideal_rec = ["I1", "I2", "I3", "I4"]
    # Sub-optimal ranking: relevant items pushed down
    worse_rec = ["I4", "I3", "I2", "I1"]
    rel = ["I1", "I2"]

    ndcg_ideal = ndcg_at_k(ideal_rec, rel, k=4)
    ndcg_worse = ndcg_at_k(worse_rec, rel, k=4)

    assert ndcg_ideal == pytest.approx(1.0)
    assert ndcg_worse < ndcg_ideal
    assert ndcg_worse > 0.0


def test_catalog_coverage():
    all_items = ["I1", "I2", "I3", "I4", "I5"]
    recs = {
        "U1": ["I1", "I2"],
        "U2": ["I2", "I3"],
    }
    # Unique recs: I1, I2, I3 -> 3 / 5 = 0.6
    assert catalog_coverage(recs, all_items) == pytest.approx(0.6)

    # Empty
    assert catalog_coverage({}, all_items) == 0.0


def test_evaluate_model_pipeline(toy_split_data, toy_products_df, toy_users_df):
    train_df, test_df = toy_split_data
    model = PopularityRecommender()
    model.fit(train_df, toy_products_df, toy_users_df)

    metrics = evaluate_model(model, test_df, train_df, k=5)
    assert "precision@k" in metrics
    assert "recall@k" in metrics
    assert "ndcg@k" in metrics
    assert 0.0 <= metrics["precision@k"] <= 1.0
    assert 0.0 <= metrics["recall@k"] <= 1.0
    assert 0.0 <= metrics["ndcg@k"] <= 1.0


def test_cold_start_analysis(toy_split_data, toy_products_df, toy_users_df):
    train_df, test_df = toy_split_data
    model = PopularityRecommender()
    model.fit(train_df, toy_products_df, toy_users_df)

    analysis = cold_start_analysis(model, test_df, train_df, k=5)
    assert "cold" in analysis
    assert "warm" in analysis
    assert "hot" in analysis


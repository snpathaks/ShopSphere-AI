"""
test_models.py - Unit tests for all recommendation models and cold-start fallback.
"""

import pytest
from src.models.popularity import PopularityRecommender
from src.models.collaborative import ItemItemCF
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender


def test_popularity_recommender_fit_and_recommend(toy_interactions_df, toy_products_df, toy_users_df):
    model = PopularityRecommender()
    model.fit(toy_interactions_df, toy_products_df, toy_users_df)
    assert model.is_fitted

    # Recommend for existing user
    recs = model.recommend("U0001", n=5)
    assert len(recs) <= 5
    assert len(recs) > 0
    assert "item_id" in recs[0]
    assert "score" in recs[0]
    assert "reason" in recs[0]


def test_popularity_handles_new_user(toy_interactions_df, toy_products_df, toy_users_df):
    model = PopularityRecommender()
    model.fit(toy_interactions_df, toy_products_df, toy_users_df)

    # Completely new user
    recs = model.recommend("U_NEW_999", n=5)
    assert len(recs) > 0
    assert all("item_id" in r for r in recs)


def test_item_item_cf_fit_and_recommend(toy_interactions_df, toy_products_df, toy_users_df):
    model = ItemItemCF()
    model.fit(toy_interactions_df, toy_products_df, toy_users_df)
    assert model.is_fitted

    recs = model.recommend("U0001", n=5)
    assert len(recs) <= 5
    assert len(recs) > 0
    assert "item_id" in recs[0]


def test_item_item_cf_cold_start_fallback(toy_interactions_df, toy_products_df, toy_users_df):
    model = ItemItemCF()
    model.fit(toy_interactions_df, toy_products_df, toy_users_df)

    # Brand new user with 0 interactions should trigger fallback to popularity
    recs = model.recommend("U_COLD_999", n=5)
    assert len(recs) > 0
    assert any("Trending" in r["reason"] or "popular" in r["reason"].lower() for r in recs)


def test_content_based_fit_and_recommend(toy_interactions_df, toy_products_df, toy_users_df):
    model = ContentBasedRecommender()
    model.fit(toy_interactions_df, toy_products_df, toy_users_df)
    assert model.is_fitted

    recs = model.recommend("U0001", n=5)
    assert len(recs) > 0
    assert "item_id" in recs[0]


def test_hybrid_fit_and_recommend(toy_interactions_df, toy_products_df, toy_users_df):
    model = HybridRecommender(alpha=0.5, beta=0.3, gamma=0.2)
    model.fit(toy_interactions_df, toy_products_df, toy_users_df)
    assert model.is_fitted

    recs = model.recommend("U0001", n=5)
    assert len(recs) > 0
    assert "score" in recs[0]
    assert "reason" in recs[0]


def test_exclude_seen_logic(toy_interactions_df, toy_products_df, toy_users_df):
    model = PopularityRecommender()
    model.fit(toy_interactions_df, toy_products_df, toy_users_df)

    user_seen = set(toy_interactions_df[toy_interactions_df["user_id"] == "U0001"]["item_id"])
    recs_exclude = model.recommend("U0001", n=10, exclude_seen=True)
    rec_items = [r["item_id"] for r in recs_exclude]

    # None of the recommended items should be in the user's training history
    assert len(set(rec_items).intersection(user_seen)) == 0


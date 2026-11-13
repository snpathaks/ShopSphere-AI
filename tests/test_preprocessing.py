"""
test_preprocessing.py - Unit tests for data preprocessing and split functions.
"""

import pandas as pd
import numpy as np
import pytest
from src.preprocessing import (
    clean_interactions,
    chronological_split,
    build_user_item_matrix,
    get_user_history,
    build_item_features,
)


def test_clean_interactions_deduplication():
    raw_df = pd.DataFrame({
        "interaction_id": ["1", "1", "2"],
        "user_id": ["U1", "U1", "U2"],
        "item_id": ["I1", "I1", "I2"],
        "interaction_type": ["view", "view", "cart"],
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 10:00:00", "2024-01-02 12:00:00"]
    })
    cleaned = clean_interactions(raw_df)
    assert len(cleaned) == 2
    assert pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"])


def test_chronological_split_no_leakage(toy_interactions_df):
    train_df, test_df = chronological_split(toy_interactions_df, test_days=10)
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(test_df) == len(toy_interactions_df)

    # Fundamental leakage check: max train timestamp <= min test timestamp
    max_train_ts = train_df["timestamp"].max()
    min_test_ts = test_df["timestamp"].min()
    assert max_train_ts <= min_test_ts, f"Leakage detected: max train {max_train_ts} > min test {min_test_ts}"


def test_build_user_item_matrix_weights():
    df = pd.DataFrame({
        "user_id": ["U1", "U1", "U1", "U2"],
        "item_id": ["I1", "I1", "I2", "I1"],
        "interaction_type": ["view", "cart", "purchase", "view"],
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01"], utc=True)
    })
    matrix = build_user_item_matrix(df, weight_map={"view": 1, "cart": 3, "purchase": 5})
    # U1 on I1: view(1) + cart(3) = 4
    assert matrix.loc["U1", "I1"] == 4.0
    # U1 on I2: purchase(5) = 5
    assert matrix.loc["U1", "I2"] == 5.0
    # U2 on I1: view(1) = 1
    assert matrix.loc["U2", "I1"] == 1.0
    # U2 on I2: unobserved = 0
    assert matrix.loc["U2", "I2"] == 0.0


def test_get_user_history():
    df = pd.DataFrame({
        "user_id": ["U1", "U1", "U2"],
        "item_id": ["I1", "I2", "I3"],
        "interaction_type": ["view", "cart", "purchase"]
    })
    history_u1 = get_user_history(df, "U1")
    assert set(history_u1) == {"I1", "I2"}
    assert get_user_history(df, "U999") == []


def test_build_item_features(toy_products_df):
    features = build_item_features(toy_products_df)
    assert len(features) == len(toy_products_df)
    assert "price_norm" in features.columns
    # Normalized price should be bounded in [0, 1]
    assert features["price_norm"].min() >= 0.0
    assert features["price_norm"].max() <= 1.0

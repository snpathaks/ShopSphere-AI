"""
conftest.py - Pytest fixtures with lightweight toy data for testing without disk dependencies.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def toy_users_df():
    return pd.DataFrame({
        "user_id": [f"U000{i}" for i in range(1, 11)],
        "age": [25, 34, 45, 22, 29, 52, 38, 27, 41, 30],
        "gender": ["M", "F", "M", "F", "Other", "M", "F", "M", "F", "M"],
        "location": ["US", "CA", "UK", "US", "DE", "FR", "US", "CA", "UK", "US"],
        "signup_date": ["2024-01-01"] * 10,
        "user_segment": ["regular"] * 10
    })


@pytest.fixture
def toy_products_df():
    categories = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports"]
    return pd.DataFrame({
        "item_id": [f"I00{i:02d}" for i in range(1, 21)],
        "name": [f"Test Product {i}" for i in range(1, 21)],
        "category": [categories[i % len(categories)] for i in range(20)],
        "subcategory": [f"Subcat {i % 3}" for i in range(20)],
        "price": [10.0 + i * 5.0 for i in range(20)],
        "description": [f"High quality test product {i} with premium features" for i in range(20)],
        "brand": [f"Brand {i % 4}" for i in range(20)],
        "avg_rating": [4.0 + (i % 10) * 0.1 for i in range(20)],
        "num_reviews": [10 + i * 2 for i in range(20)]
    })


@pytest.fixture
def toy_interactions_df():
    np.random.seed(42)
    user_ids = [f"U000{i}" for i in range(1, 11)]
    item_ids = [f"I00{i:02d}" for i in range(1, 21)]
    types = ["view", "view", "view", "cart", "purchase"]

    records = []
    base_ts = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
    for idx in range(120):
        u = user_ids[idx % len(user_ids)]
        i = item_ids[(idx * 3 + idx // 5) % len(item_ids)]
        itype = types[idx % len(types)]
        ts = base_ts + pd.Timedelta(days=idx // 2, hours=(idx % 24))
        rating = 4.5 if itype == "purchase" else np.nan
        records.append({
            "interaction_id": f"int_{idx:04d}",
            "user_id": u,
            "item_id": i,
            "interaction_type": itype,
            "timestamp": ts,
            "rating": rating,
            "session_id": f"sess_{idx // 4}"
        })
    return pd.DataFrame(records)


@pytest.fixture
def toy_split_data(toy_interactions_df):
    from src.preprocessing import chronological_split
    return chronological_split(toy_interactions_df, test_days=10)

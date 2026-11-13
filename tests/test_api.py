"""
test_api.py - Unit and integration tests for FastAPI application endpoints and ModelLoader.
"""

import pytest
from api.main import root, health, list_models, recommend, similar_items, user_history, submit_feedback, FeedbackRequest
from api.model_loader import ModelLoader
from api.schemas import RecommendationResponse, HealthResponse, ModelInfoResponse


@pytest.fixture(scope="module", autouse=True)
def load_all_models():
    """Ensure models and data are loaded into the singleton ModelLoader."""
    loader = ModelLoader.get_instance()
    loader.load_models()
    return loader


def test_root_endpoint():
    res = root()
    assert isinstance(res, dict)
    assert "message" in res
    assert res["docs"] == "/docs"


def test_health_endpoint():
    res = health()
    assert isinstance(res, HealthResponse)
    assert res.status == "ok"
    assert len(res.models_loaded) >= 1
    assert "hybrid" in res.models_loaded or "popularity" in res.models_loaded


def test_list_models_endpoint():
    res = list_models()
    assert isinstance(res, ModelInfoResponse)
    assert len(res.available_models) >= 1
    assert res.default_model == "hybrid"


def test_recommend_existing_user():
    res = recommend(user_id="U0001", n=5, model="hybrid", exclude_seen=True)
    assert isinstance(res, RecommendationResponse)
    assert res.user_id == "U0001"
    assert len(res.recommendations) <= 5
    assert len(res.recommendations) > 0
    # Every recommendation must have name, category, price, score, reason
    first_item = res.recommendations[0]
    assert first_item.name != ""
    assert first_item.category != ""
    assert first_item.reason != ""
    assert first_item.rank == 1


def test_recommend_cold_start_new_user():
    # Brand new user with 0 history should trigger is_cold_start=True and popularity fallback
    res = recommend(user_id="U_BRAND_NEW_9999", n=5, model="hybrid", exclude_seen=True)
    assert isinstance(res, RecommendationResponse)
    assert res.user_id == "U_BRAND_NEW_9999"
    assert res.is_cold_start is True
    assert len(res.recommendations) > 0


def test_user_history_endpoint():
    res = user_history(user_id="U0001")
    assert isinstance(res, dict)
    assert res["user_id"] == "U0001"
    assert "history" in res


def test_similar_items_endpoint():
    res = similar_items(item_id="I0001", n=3)
    assert isinstance(res, RecommendationResponse)
    assert len(res.recommendations) <= 3


def test_feedback_submission():
    payload = FeedbackRequest(user_id="U0001", item_id="I0001", rating=5)
    res = submit_feedback(payload)
    assert isinstance(res, dict)
    assert res["status"] == "ok"
    assert res["record"]["user_id"] == "U0001"
    assert res["record"]["item_id"] == "I0001"

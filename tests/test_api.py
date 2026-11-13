"""
test_api.py - Unit and integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from api.main import app
from api.model_loader import ModelLoader


@pytest.fixture
def client():
    # Use TestClient with app
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["docs"] == "/docs"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "default_model" in data
    assert "description" in data


def test_feedback_endpoint(client):
    payload = {
        "user_id": "U0001",
        "item_id": "I0001",
        "rating": 5
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["feedback"]["user_id"] == "U0001"


def test_recommend_endpoint_with_mock():
    # Test recommend logic with a mock model attached to loader
    loader = ModelLoader.get_instance()
    mock_model = MagicMock()
    mock_model.recommend.return_value = [
        {"item_id": "I0001", "score": 0.95, "reason": "Test mock reason"}
    ]
    loader._models["mock_test_model"] = mock_model
    loader._user_interaction_counts["U9999"] = 10

    with TestClient(app) as c:
        response = c.get("/recommend/U9999?model=mock_test_model&n=5")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "U9999"
        assert len(data["recommendations"]) >= 1
        assert data["recommendations"][0]["item_id"] == "I0001"

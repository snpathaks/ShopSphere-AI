from pydantic import BaseModel
from typing import List, Optional


class RecommendationItem(BaseModel):
    item_id: str
    name: str
    category: str
    price: float
    score: float
    reason: str
    rank: int


class RecommendationResponse(BaseModel):
    user_id: str
    model_used: str
    recommendations: List[RecommendationItem]
    total_items: int
    user_interaction_count: int
    is_cold_start: bool
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    available_models: List[str]
    default_model: str
    description: dict

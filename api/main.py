import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.model_loader import ModelLoader
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    RecommendationItem,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan: load models on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    loader = ModelLoader.get_instance()
    loader.load_models()
    logger.info("Models loaded on startup.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="E-Commerce Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DESCRIPTIONS = {
    "popularity": "Popularity-based recommender — returns globally popular items. Best for cold-start users.",
    "item_item_cf": "Item-Item Collaborative Filtering — finds similar items based on user co-interaction patterns.",
    "content_based": "Content-Based Filtering — recommends items similar to what the user has interacted with.",
    "hybrid": "Hybrid model — combines collaborative filtering and content-based signals for best results.",
}

DEFAULT_MODEL = "hybrid"

# Feedback file path (project root)
FEEDBACK_FILE = Path(__file__).resolve().parent.parent / "feedback.json"


# ---------------------------------------------------------------------------
# Schemas for request bodies
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    rating: int


# ---------------------------------------------------------------------------
# Helper: safe recommend call
# ---------------------------------------------------------------------------

def _call_recommend(model, user_id: str, n: int, exclude_seen: bool) -> list:
    """
    Call model.recommend() with graceful argument handling.
    Models may accept different signatures; try the richest signature first.
    Returns a list of dicts with at least 'item_id' and 'score'.
    """
    try:
        result = model.recommend(user_id, n=n, exclude_seen=exclude_seen)
    except TypeError:
        try:
            result = model.recommend(user_id, n=n)
        except TypeError:
            result = model.recommend(user_id)

    if result is None:
        return []

    # Normalise to list[dict]
    normalised = []
    for i, rec in enumerate(result):
        if isinstance(rec, dict):
            normalised.append(rec)
        elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
            normalised.append({"item_id": str(rec[0]), "score": float(rec[1])})
        else:
            normalised.append({"item_id": str(rec), "score": 0.0})
    return normalised


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "E-Commerce Recommendation API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    loader = ModelLoader.get_instance()
    return HealthResponse(
        status="ok",
        models_loaded=loader.get_loaded_model_names(),
        version="1.0.0",
    )


@app.get("/models", response_model=ModelInfoResponse)
def list_models():
    loader = ModelLoader.get_instance()
    available = loader.get_loaded_model_names()
    return ModelInfoResponse(
        available_models=available,
        default_model=DEFAULT_MODEL,
        description=MODEL_DESCRIPTIONS,
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: str,
    n: int = Query(10, ge=1, le=100),
    model: str = Query(DEFAULT_MODEL),
    exclude_seen: bool = Query(True),
):
    loader = ModelLoader.get_instance()
    interaction_count = loader.get_user_interaction_count(user_id)
    is_cold_start = interaction_count == 0

    # Choose model — fall back to popularity on cold start or missing model
    model_name = model
    chosen_model = loader.get_model(model_name)

    if is_cold_start or chosen_model is None:
        model_name = "popularity"
        chosen_model = loader.get_model("popularity")

    if chosen_model is None:
        raise HTTPException(status_code=503, detail="No recommendation models available.")

    try:
        raw_recs = _call_recommend(chosen_model, user_id, n, exclude_seen)
    except Exception as exc:
        logger.error(f"Model recommend() failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}")

    enriched = loader.enrich_recommendations(raw_recs)

    items = []
    for rank, rec in enumerate(enriched[:n], start=1):
        item_id = rec.get("item_id", "")
        score = float(rec.get("score", 0.0))
        name = rec.get("name", "Unknown")
        category = rec.get("category", "Unknown")
        price = float(rec.get("price", 0.0))
        reason = rec.get("reason", f"Recommended by {model_name} model")
        items.append(
            RecommendationItem(
                item_id=item_id,
                name=name,
                category=category,
                price=price,
                score=score,
                reason=reason,
                rank=rank,
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        model_used=model_name,
        recommendations=items,
        total_items=len(items),
        user_interaction_count=interaction_count,
        is_cold_start=is_cold_start,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/items/{item_id}/similar", response_model=RecommendationResponse)
def similar_items(
    item_id: str,
    n: int = Query(10, ge=1, le=100),
):
    loader = ModelLoader.get_instance()

    # Verify item exists
    products_df = loader.get_products()
    if products_df is not None and not products_df.empty:
        if item_id not in products_df["item_id"].values:
            raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")

    cb_model = loader.get_model("content_based")
    if cb_model is None:
        raise HTTPException(status_code=503, detail="Content-based model not available.")

    try:
        # Try get_similar_items first; fall back to recommend with item_id as user
        if hasattr(cb_model, "get_similar_items"):
            raw_recs = cb_model.get_similar_items(item_id, n=n)
        else:
            raw_recs = _call_recommend(cb_model, item_id, n, exclude_seen=True)
    except Exception as exc:
        logger.error(f"Similar items failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Similar items retrieval failed: {exc}")

    if not raw_recs:
        raise HTTPException(status_code=404, detail=f"No similar items found for '{item_id}'.")

    enriched = loader.enrich_recommendations(raw_recs)
    items = []
    for rank, rec in enumerate(enriched[:n], start=1):
        items.append(
            RecommendationItem(
                item_id=rec.get("item_id", ""),
                name=rec.get("name", "Unknown"),
                category=rec.get("category", "Unknown"),
                price=float(rec.get("price", 0.0)),
                score=float(rec.get("score", 0.0)),
                reason=rec.get("reason", f"Similar to item {item_id}"),
                rank=rank,
            )
        )

    return RecommendationResponse(
        user_id=item_id,
        model_used="content_based",
        recommendations=items,
        total_items=len(items),
        user_interaction_count=0,
        is_cold_start=False,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/users/{user_id}/history")
def user_history(user_id: str):
    loader = ModelLoader.get_instance()
    interactions_df = loader.get_interactions()
    products_df = loader.get_products()

    if interactions_df is None or interactions_df.empty:
        raise HTTPException(status_code=404, detail=f"No interaction data available.")

    user_interactions = interactions_df[interactions_df["user_id"] == user_id]
    if user_interactions.empty:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found or has no history.")

    # Enrich with product names
    history = []
    if products_df is not None and not products_df.empty:
        products_indexed = products_df.set_index("item_id")
    else:
        products_indexed = None

    for _, row in user_interactions.iterrows():
        item_id = row.get("item_id", "")
        name = "Unknown"
        category = "Unknown"
        price = 0.0
        if products_indexed is not None and item_id in products_indexed.index:
            p = products_indexed.loc[item_id]
            name = p.get("name", "Unknown")
            category = p.get("category", "Unknown")
            price = float(p.get("price", 0.0))

        history.append({
            "item_id": item_id,
            "name": name,
            "category": category,
            "price": price,
            "interaction_type": row.get("interaction_type", ""),
            "timestamp": str(row.get("timestamp", "")),
            "rating": row.get("rating", None),
        })

    return {
        "user_id": user_id,
        "interaction_count": len(history),
        "history": history,
    }


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    record = {
        "user_id": feedback.user_id,
        "item_id": feedback.item_id,
        "rating": feedback.rating,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    # Append to feedback.json
    existing = []
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    existing.append(record)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    return {"status": "ok", "message": "Feedback recorded.", "record": record}

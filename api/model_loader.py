import pickle
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ModelLoader:
    _instance = None
    _models: dict = {}
    _products_df: Optional[pd.DataFrame] = None
    _interactions_df: Optional[pd.DataFrame] = None

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self, models_dir: str = "models", data_dir: str = "data/raw") -> None:
        """Load all saved pkl models and data files."""
        # Resolve paths relative to this file's location (api/ -> project root)
        base_dir = Path(__file__).resolve().parent.parent
        models_path = base_dir / models_dir
        data_path = base_dir / data_dir

        # Load products and interactions CSVs
        products_file = data_path / "products.csv"
        interactions_file = data_path / "interactions.csv"

        if products_file.exists():
            self._products_df = pd.read_csv(products_file)
            logger.info(f"Loaded products: {len(self._products_df)} rows")
        else:
            logger.warning(f"products.csv not found at {products_file}")
            self._products_df = pd.DataFrame(
                columns=["item_id", "name", "category", "subcategory", "price",
                         "description", "brand", "avg_rating", "num_reviews"]
            )

        if interactions_file.exists():
            self._interactions_df = pd.read_csv(interactions_file)
            logger.info(f"Loaded interactions: {len(self._interactions_df)} rows")
        else:
            logger.warning(f"interactions.csv not found at {interactions_file}")
            self._interactions_df = pd.DataFrame(
                columns=["interaction_id", "user_id", "item_id", "interaction_type",
                         "timestamp", "rating", "session_id"]
            )

        # Load pkl model files
        model_files = {
            "popularity": "popularity.pkl",
            "item_item_cf": "item_item_cf.pkl",
            "content_based": "content_based.pkl",
            "hybrid": "hybrid.pkl",
        }

        self._models = {}
        for model_name, filename in model_files.items():
            model_file = models_path / filename
            if model_file.exists():
                try:
                    with open(model_file, "rb") as f:
                        self._models[model_name] = pickle.load(f)
                    logger.info(f"Loaded model: {model_name}")
                except Exception as exc:
                    logger.error(f"Failed to load {model_name}: {exc}")
            else:
                logger.warning(f"Model file not found: {model_file}")

    def get_model(self, model_name: str):
        """Return loaded model by name, or None if not found."""
        return self._models.get(model_name)

    def get_products(self) -> pd.DataFrame:
        """Return products DataFrame."""
        return self._products_df

    def get_interactions(self) -> pd.DataFrame:
        """Return interactions DataFrame."""
        return self._interactions_df

    def get_loaded_model_names(self) -> list:
        """Return list of successfully loaded model names."""
        return list(self._models.keys())

    def get_user_interaction_count(self, user_id: str) -> int:
        """Return number of interactions for a user."""
        if self._interactions_df is None or self._interactions_df.empty:
            return 0
        return int((self._interactions_df["user_id"] == user_id).sum())

    def enrich_recommendations(self, recs: list) -> list:
        """Add name, category, price from products_df to raw recommendation dicts."""
        if self._products_df is None or self._products_df.empty:
            return recs

        products_indexed = self._products_df.set_index("item_id")
        enriched = []
        for rec in recs:
            item_id = rec.get("item_id", "")
            if item_id in products_indexed.index:
                row = products_indexed.loc[item_id]
                rec = {
                    **rec,
                    "name": row.get("name", "Unknown"),
                    "category": row.get("category", "Unknown"),
                    "price": float(row.get("price", 0.0)),
                }
            else:
                rec = {
                    **rec,
                    "name": rec.get("name", "Unknown"),
                    "category": rec.get("category", "Unknown"),
                    "price": float(rec.get("price", 0.0)),
                }
            enriched.append(rec)
        return enriched

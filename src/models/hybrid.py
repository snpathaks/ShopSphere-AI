import numpy as np
import pandas as pd

from .base import BaseRecommender
from .collaborative import ItemItemCF
from .content_based import ContentBasedRecommender
from .popularity import PopularityRecommender


class HybridRecommender(BaseRecommender):
    """Weighted hybrid of collaborative, content-based, and popularity models."""

    COLD_START_THRESHOLD = 5

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        super().__init__("HybridRecommender")
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1."
        self.alpha = alpha  # collaborative weight
        self.beta = beta    # content-based weight
        self.gamma = gamma  # popularity weight

        self._cf = ItemItemCF()
        self._cb = ContentBasedRecommender()
        self._pop = PopularityRecommender()
        self._user_history: dict = {}

    # ------------------------------------------------------------------
    def fit(
        self,
        train_interactions: pd.DataFrame,
        products_df: pd.DataFrame,
        users_df: pd.DataFrame,
    ) -> "HybridRecommender":
        np.random.seed(42)
        self._cf.fit(train_interactions, products_df, users_df)
        self._cb.fit(train_interactions, products_df, users_df)
        self._pop.fit(train_interactions, products_df, users_df)

        self._user_history = (
            train_interactions.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(scores: dict) -> dict:
        """Min-max normalize score dict to [0, 1]."""
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    @staticmethod
    def _recs_to_dict(recs: list) -> dict:
        return {r["item_id"]: r["score"] for r in recs}

    # ------------------------------------------------------------------
    def recommend(self, user_id: str, n: int = 10, exclude_seen: bool = True) -> list:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        seen = self._user_history.get(user_id, set())
        n_seen = len(seen)

        # Adjust weights for cold-start users
        if n_seen < self.COLD_START_THRESHOLD:
            alpha = max(0.0, self.alpha - 0.2)
            beta = max(0.0, self.beta - 0.1)
            gamma = 1.0 - alpha - beta
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma

        # Gather a larger candidate pool from each model
        pool = max(n * 3, 30)
        cf_recs = self._cf.recommend(user_id, n=pool, exclude_seen=exclude_seen)
        cb_recs = self._cb.recommend(user_id, n=pool, exclude_seen=exclude_seen)
        pop_recs = self._pop.recommend(user_id, n=pool, exclude_seen=exclude_seen)

        cf_dict = self._normalize(self._recs_to_dict(cf_recs))
        cb_dict = self._normalize(self._recs_to_dict(cb_recs))
        pop_dict = self._normalize(self._recs_to_dict(pop_recs))

        # Collect all candidate items
        all_items = set(cf_dict) | set(cb_dict) | set(pop_dict)

        blended: dict = {}
        for item_id in all_items:
            score = (
                alpha * cf_dict.get(item_id, 0.0)
                + beta * cb_dict.get(item_id, 0.0)
                + gamma * pop_dict.get(item_id, 0.0)
            )
            blended[item_id] = score

        sorted_items = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:n]
        return [
            {
                "item_id": item_id,
                "score": float(score),
                "reason": "Recommended based on your behavior and similar users",
            }
            for item_id, score in sorted_items
        ]

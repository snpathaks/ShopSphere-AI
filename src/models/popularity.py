import numpy as np
import pandas as pd
from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommender — works for any user including cold-start."""

    WEIGHT_MAP = {"view": 1, "cart": 3, "purchase": 5}

    def __init__(self):
        super().__init__("PopularityRecommender")
        self.popularity_scores: pd.Series = pd.Series(dtype=float)
        self.products_df: pd.DataFrame = pd.DataFrame()
        self.interactions_df: pd.DataFrame = pd.DataFrame()
        self._user_history: dict = {}

    def fit(
        self,
        train_interactions: pd.DataFrame,
        products_df: pd.DataFrame,
        users_df: pd.DataFrame,
    ) -> "PopularityRecommender":
        np.random.seed(42)
        self.products_df = products_df.copy()
        self.interactions_df = train_interactions.copy()

        # Build user history for exclude_seen
        self._user_history = (
            train_interactions.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        # Weighted popularity: weight per interaction type
        df = train_interactions.copy()
        df["weight"] = df["interaction_type"].map(self.WEIGHT_MAP).fillna(1)
        item_scores = df.groupby("item_id")["weight"].sum()

        # Number of unique purchasers per item (for reason string)
        purchases = df[df["interaction_type"] == "purchase"]
        self._purchase_counts = purchases.groupby("item_id")["user_id"].nunique().to_dict()

        self.popularity_scores = item_scores.sort_values(ascending=False)
        self.is_fitted = True
        return self

    def recommend(self, user_id: str, n: int = 10, exclude_seen: bool = True) -> list:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        seen = self._user_history.get(user_id, set()) if exclude_seen else set()
        candidates = self.popularity_scores[~self.popularity_scores.index.isin(seen)]

        top = candidates.head(n)
        results = []
        for item_id, score in top.items():
            purchase_cnt = self._purchase_counts.get(item_id, 0)
            results.append(
                {
                    "item_id": item_id,
                    "score": float(score),
                    "reason": f"Trending: purchased by {purchase_cnt} users",
                }
            )
        return results

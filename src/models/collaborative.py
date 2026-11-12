import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender
from .popularity import PopularityRecommender


class ItemItemCF(BaseRecommender):
    """Item-Item Collaborative Filtering using cosine similarity on user-item matrix columns."""

    WEIGHT_MAP = {"view": 1, "cart": 3, "purchase": 5}
    COLD_START_THRESHOLD = 3

    def __init__(self):
        super().__init__("ItemItemCF")
        self.user_item_matrix: pd.DataFrame = pd.DataFrame()
        self.item_similarity: np.ndarray = np.array([])
        self.products_df: pd.DataFrame = pd.DataFrame()
        self._user_history: dict = {}
        self._fallback: PopularityRecommender = PopularityRecommender()
        self._items: list = []
        self._users: list = []

    # ------------------------------------------------------------------
    def fit(
        self,
        train_interactions: pd.DataFrame,
        products_df: pd.DataFrame,
        users_df: pd.DataFrame,
    ) -> "ItemItemCF":
        np.random.seed(42)
        self.products_df = products_df.copy()

        # Build user-item weighted matrix
        df = train_interactions.copy()
        df["weight"] = df["interaction_type"].map(self.WEIGHT_MAP).fillna(1)
        agg = df.groupby(["user_id", "item_id"])["weight"].sum().reset_index()
        matrix = agg.pivot(index="user_id", columns="item_id", values="weight").fillna(0)

        self.user_item_matrix = matrix
        self._users = list(matrix.index)
        self._items = list(matrix.columns)

        # Build user history
        self._user_history = (
            train_interactions.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        # Sparse item-item cosine similarity (shape: n_items x n_items)
        sparse = csr_matrix(matrix.values.T)  # items as rows
        self.item_similarity = cosine_similarity(sparse)  # (n_items, n_items)

        # Fit fallback
        self._fallback.fit(train_interactions, products_df, users_df)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def _get_item_name(self, item_id: str) -> str:
        if self.products_df.empty:
            return str(item_id)
        row = self.products_df[self.products_df["item_id"] == item_id]
        if row.empty:
            return str(item_id)
        return str(row.iloc[0].get("name", item_id))

    # ------------------------------------------------------------------
    def recommend(self, user_id: str, n: int = 10, exclude_seen: bool = True) -> list:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        seen = self._user_history.get(user_id, set())

        # Cold-start: fall back to popularity for users with <3 interactions
        if len(seen) < self.COLD_START_THRESHOLD:
            return self._fallback.recommend(user_id, n=n, exclude_seen=exclude_seen)

        if user_id not in self._users:
            return self._fallback.recommend(user_id, n=n, exclude_seen=exclude_seen)

        user_idx = self._users.index(user_id)
        user_vector = self.user_item_matrix.iloc[user_idx].values  # shape: (n_items,)

        # Aggregate similarity scores: sum(interaction_weight * similarity_to_item)
        scores = user_vector @ self.item_similarity  # shape: (n_items,)

        # Build score map, exclude seen if needed
        item_scores: dict = {}
        for idx, item_id in enumerate(self._items):
            if exclude_seen and item_id in seen:
                continue
            item_scores[item_id] = float(scores[idx])

        if not item_scores:
            return self._fallback.recommend(user_id, n=n, exclude_seen=exclude_seen)

        # Find the most influential seen item for reason string
        seen_interactions = {
            item: user_vector[self._items.index(item)]
            for item in seen
            if item in self._items
        }
        anchor_item = (
            max(seen_interactions, key=seen_interactions.get)
            if seen_interactions
            else None
        )
        anchor_name = self._get_item_name(anchor_item) if anchor_item else "items you interacted with"

        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [
            {
                "item_id": item_id,
                "score": score,
                "reason": f"Similar to {anchor_name} you interacted with",
            }
            for item_id, score in sorted_items
        ]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ItemItemCF":
        with open(path, "rb") as f:
            return pickle.load(f)

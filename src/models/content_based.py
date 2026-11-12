import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender
from .popularity import PopularityRecommender


class ContentBasedRecommender(BaseRecommender):
    """Content-Based recommender using TF-IDF on item text features."""

    def __init__(self):
        super().__init__("ContentBasedRecommender")
        self.products_df: pd.DataFrame = pd.DataFrame()
        self._tfidf_matrix = None          # shape: (n_items, n_features)
        self._vectorizer: TfidfVectorizer = None
        self._item_ids: list = []
        self._user_history: dict = {}
        self._fallback: PopularityRecommender = PopularityRecommender()
        # category popularity for cold-start
        self._category_popularity: pd.Series = pd.Series(dtype=float)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_text(row: pd.Series) -> str:
        parts = [
            str(row.get("category", "")),
            str(row.get("subcategory", "")),
            str(row.get("description", "")),
            str(row.get("brand", "")),
        ]
        return " ".join(p for p in parts if p and p != "nan")

    # ------------------------------------------------------------------
    def fit(
        self,
        train_interactions: pd.DataFrame,
        products_df: pd.DataFrame,
        users_df: pd.DataFrame,
    ) -> "ContentBasedRecommender":
        np.random.seed(42)
        self.products_df = products_df.copy().reset_index(drop=True)

        # Build TF-IDF matrix
        self.products_df["_text"] = self.products_df.apply(self._build_text, axis=1)
        self._vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self.products_df["_text"])
        self._item_ids = list(self.products_df["item_id"])

        # User history
        self._user_history = (
            train_interactions.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        # Category popularity (for cold-start)
        WEIGHT_MAP = {"view": 1, "cart": 3, "purchase": 5}
        df = train_interactions.copy()
        df["weight"] = df["interaction_type"].map(WEIGHT_MAP).fillna(1)
        item_scores = df.groupby("item_id")["weight"].sum()
        self.products_df["_pop_score"] = self.products_df["item_id"].map(item_scores).fillna(0)
        self._category_popularity = (
            self.products_df.groupby("category")["_pop_score"].sum().sort_values(ascending=False)
        )

        # Fit fallback
        self._fallback.fit(train_interactions, products_df, users_df)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def _get_category(self, item_id: str) -> str:
        row = self.products_df[self.products_df["item_id"] == item_id]
        if row.empty:
            return "products"
        return str(row.iloc[0].get("category", "products"))

    # ------------------------------------------------------------------
    def recommend(self, user_id: str, n: int = 10, exclude_seen: bool = True) -> list:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        seen = self._user_history.get(user_id, set())

        # Cold-start: no history → recommend by category popularity
        if not seen:
            return self._fallback.recommend(user_id, n=n, exclude_seen=exclude_seen)

        # Build user profile: average TF-IDF vectors of seen items
        seen_indices = [
            self._item_ids.index(iid)
            for iid in seen
            if iid in self._item_ids
        ]
        if not seen_indices:
            return self._fallback.recommend(user_id, n=n, exclude_seen=exclude_seen)

        user_profile = self._tfidf_matrix[seen_indices].mean(axis=0)  # (1, n_features)
        # numpy matrix → array
        user_profile = np.asarray(user_profile)

        # Cosine similarity between user profile and all items
        sims = cosine_similarity(user_profile, self._tfidf_matrix).flatten()

        # Determine dominant category from seen items for reason string
        seen_categories = [self._get_category(iid) for iid in seen if iid in self._item_ids]
        dominant_category = (
            pd.Series(seen_categories).value_counts().index[0]
            if seen_categories
            else "your interests"
        )

        # Build results excluding seen items
        results = []
        for idx in np.argsort(sims)[::-1]:
            item_id = self._item_ids[idx]
            if exclude_seen and item_id in seen:
                continue
            results.append(
                {
                    "item_id": item_id,
                    "score": float(sims[idx]),
                    "reason": f"Matches your interest in {dominant_category}",
                }
            )
            if len(results) >= n:
                break
        return results

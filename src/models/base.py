from abc import ABC, abstractmethod
import pandas as pd


class BaseRecommender(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_interactions: pd.DataFrame, products_df: pd.DataFrame, users_df: pd.DataFrame) -> "BaseRecommender":
        pass

    @abstractmethod
    def recommend(self, user_id: str, n: int = 10, exclude_seen: bool = True) -> list:
        """Returns list of dicts: [{"item_id": ..., "score": ..., "reason": ...}]"""
        pass

    def get_name(self) -> str:
        return self.model_name

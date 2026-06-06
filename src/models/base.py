"""Common recommender interface plus shared rating fallbacks."""

from abc import ABC, abstractmethod
from typing import cast

import pandas as pd


class Recommender(ABC):
    """fit/predict/recommend interface with mean-based fallbacks.

    Subclasses call `self._fit_means(train)` in `fit`, implement `predict`,
    and may rely on the default `recommend` (rank candidates by predicted rating,
    excluding items already seen in training).
    """

    def __init__(self) -> None:
        self.global_mean_: float = 3.0
        self.item_means_: dict[str, float] = {}
        self.user_means_: dict[str, float] = {}
        self.user_items_: dict[str, set[str]] = {}

    def _fit_means(self, train: pd.DataFrame) -> None:
        self.global_mean_ = float(train["rating"].mean())
        self.item_means_ = cast(
            dict[str, float],
            train.groupby("parent_asin")["rating"].mean().to_dict(),
        )
        self.user_means_ = cast(
            dict[str, float],
            train.groupby("user_id")["rating"].mean().to_dict(),
        )
        self.user_items_ = cast(
            dict[str, set[str]],
            {
                user: set(items)
                for user, items in train.groupby("user_id")["parent_asin"]
            },
        )

    def _fallback(self, user_id: str, parent_asin: str) -> float:
        if parent_asin in self.item_means_:
            return float(self.item_means_[parent_asin])
        if user_id in self.user_means_:
            return float(self.user_means_[user_id])
        return float(self.global_mean_)

    @staticmethod
    def _clip(rating: float) -> float:
        return float(min(5.0, max(1.0, rating)))

    @abstractmethod
    def fit(self, train: pd.DataFrame, metadata: object = None) -> "Recommender":
        ...

    @abstractmethod
    def predict(self, user_id: str, parent_asin: str) -> float:
        ...

    def recommend(
        self,
        user_id: str,
        k: int,
        candidates: list[str] | None = None,
    ) -> list[str]:
        """Rank candidate items by predicted rating, excluding seen items."""
        seen = self.user_items_.get(user_id, set())
        if candidates is None:
            candidates = list(self.item_means_.keys())
        scored = [
            (item, self.predict(user_id, item))
            for item in candidates
            if item not in seen
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [item for item, _ in scored[:k]]

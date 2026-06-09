"""Sanity baselines: random ranking + popularity ranking.

Both share the Recommender mean-fallback machinery for rating prediction so
RMSE/MAE are comparable to the other models in the table. Only the ranking
behavior differs: Random shuffles candidates under a fixed seed; Popularity
ranks by training-side interaction count.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import pandas as pd

from .base import Recommender


def _stable_user_seed(base_seed: int, user_id: str) -> int:
    """Deterministic per-user seed, stable across processes (NOT Python hash())."""
    digest = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
    return base_seed + (int(digest, 16) & 0xFFFFFFFF)


class RandomRecommender(Recommender):
    """Predicts the global mean; ranks candidates via a seeded shuffle."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.seed = int(seed)

    def fit(self, train: pd.DataFrame, metadata: object = None) -> "RandomRecommender":
        self._fit_means(train)
        return self

    def predict(self, user_id: str, parent_asin: str) -> float:
        return self._clip(self.global_mean_)

    def recommend(
        self,
        user_id: str,
        k: int,
        candidates: list[str] | None = None,
    ) -> list[str]:
        seen = self.user_items_.get(user_id, set())
        if candidates is None:
            candidates = list(self.item_means_.keys())
        pool = [item for item in candidates if item not in seen]
        # Per-user deterministic ordering, stable across processes (hashlib, not hash()).
        rng = np.random.default_rng(_stable_user_seed(self.seed, user_id))
        idx = rng.permutation(len(pool))
        return [pool[int(i)] for i in idx[:k]]


class PopularityRecommender(Recommender):
    """Ranks candidates by training-side interaction count; rating prediction = means."""

    def __init__(self) -> None:
        super().__init__()
        self.popularity_: Counter[str] = Counter()

    def fit(self, train: pd.DataFrame, metadata: object = None) -> "PopularityRecommender":
        self._fit_means(train)
        self.popularity_ = Counter(train["parent_asin"].tolist())
        return self

    def predict(self, user_id: str, parent_asin: str) -> float:
        return self._clip(self._fallback(user_id, parent_asin))

    def recommend(
        self,
        user_id: str,
        k: int,
        candidates: list[str] | None = None,
    ) -> list[str]:
        seen = self.user_items_.get(user_id, set())
        if candidates is None:
            candidates = list(self.item_means_.keys())
        # Sort by (-popularity, parent_asin) for deterministic tie-breaking.
        ranked = sorted(
            (item for item in candidates if item not in seen),
            key=lambda item: (-self.popularity_.get(item, 0), item),
        )
        return ranked[:k]

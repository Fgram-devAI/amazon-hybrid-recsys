"""Content recommender: item_features_v2 = [text ; categories ; numeric ; item sentiment].

Decisions (spec §5):
- The enriched **text embedding uses `title` + `description` only** — NOT the Phase-1
  ``text`` blob (which already concatenates categories). Categories are represented
  explicitly as multi-hot columns, so embedding them again would double-count. Falls
  back to the ``text`` column when title/description are absent (tiny test fixtures).
- **Item sentiment aggregates** (train-only) are appended as extra item-feature
  columns and a **train-only user-generosity offset** is applied in ``predict`` — so the
  sentiment / user / item aggregates from ``review_features.py`` are actually consumed
  by an evaluated model. Aggregates are optional: when their parquet caches are absent
  (tests, or before the offline sentiment job runs) the model uses text+categories+numeric.

Embeddings are **cached** via ``load_or_compute_item_embeddings`` under a dedicated
``cache_dir`` so they are not recomputed every eval run, and never collide with the
Phase-1 content model's (title+description+categories) cache.

Item-feature construction is now delegated to ``src.features.node_features`` so that
GraphSAGE can consume the same matrix; this module owns only the user-generosity
offset path and the similarity-weighted prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import Recommender


class ContentEnrichedRecommender(Recommender):
    def __init__(
        self,
        embedder: object,
        *,
        generic_roots: list[str],
        max_vocab: int,
        min_doc_freq: int,
        cache_dir: object = None,
        review_features_dir: object = None,
        use_item_sentiment: bool = True,
        use_user_offset: bool = True,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.generic_roots = list(generic_roots)
        self.max_vocab = int(max_vocab)
        self.min_doc_freq = int(min_doc_freq)
        self.cache_dir = cache_dir                      # cached title+description embeddings
        self.review_features_dir = review_features_dir  # train-only sentiment aggregates
        self.use_item_sentiment = bool(use_item_sentiment)
        self.use_user_offset = bool(use_user_offset)
        self.features_: np.ndarray | None = None
        self.item_index_: dict[str, int] = {}
        self.category_vocab_: list[str] = []
        self.user_history_: dict[str, list[tuple[str, float]]] = {}
        self.user_offset_: dict[str, float] = {}

    def fit(
        self, train: pd.DataFrame, metadata: object = None
    ) -> "ContentEnrichedRecommender":
        self._fit_means(train)
        if metadata is None:
            raise ValueError("ContentEnrichedRecommender.fit requires metadata")

        from src.features.node_features import build_item_node_features

        features, ids = build_item_node_features(
            metadata,  # type: ignore[arg-type]
            embedder=self.embedder,  # type: ignore[arg-type]
            generic_roots=self.generic_roots,
            max_vocab=self.max_vocab,
            min_doc_freq=self.min_doc_freq,
            cache_dir=Path(str(self.cache_dir)) if self.cache_dir is not None else None,
            review_features_dir=Path(str(self.review_features_dir))
                if self.review_features_dir is not None else None,
            use_item_sentiment=self.use_item_sentiment,
        )
        self.features_ = features
        self.item_index_ = {item: row for row, item in enumerate(ids)}
        self.category_vocab_ = []  # builder owns the vocab; keep attribute for backwards-compat

        self.user_history_ = {}
        for user, group in train.groupby("user_id"):
            self.user_history_[str(user)] = list(
                zip(group["parent_asin"], group["rating"])
            )
        self.user_offset_ = self._user_generosity_offsets()
        return self

    def _aggregate_path(self, name: str) -> Path | None:
        if self.review_features_dir is None:
            return None
        path = Path(str(self.review_features_dir)) / name
        return path if path.exists() else None

    def _user_generosity_offsets(self) -> dict[str, float]:
        """Train-only per-user offset from sentiment/rating gap, falling back to rating bias."""
        if not self.use_user_offset:
            return {}
        path = self._aggregate_path("user_review_aggregates.parquet")
        if path is None:
            return {}
        agg = pd.read_parquet(path)
        if "user_rating_minus_sentiment_gap" in agg.columns:
            gaps = agg["user_rating_minus_sentiment_gap"].astype(float)
            offsets = (gaps - gaps.mean()).clip(-1.0, 1.0)
        elif "user_mean_rating" in agg.columns:
            offsets = (agg["user_mean_rating"] - self.global_mean_).clip(-1.0, 1.0)
        else:
            return {}
        return dict(zip(agg["user_id"].astype(str), offsets.astype(float)))

    def predict(self, user_id: str, parent_asin: str) -> float:
        offset = self.user_offset_.get(user_id, 0.0)   # train-only generosity adjustment
        target_row = self.item_index_.get(parent_asin)
        history = self.user_history_.get(user_id)
        if target_row is None or not history:
            return self._clip(self._fallback(user_id, parent_asin) + offset)

        assert self.features_ is not None
        target = self.features_[target_row]
        weighted_sum = 0.0
        sim_sum = 0.0
        for item, rating in history:
            if item == parent_asin:
                continue
            row = self.item_index_.get(item)
            if row is None:
                continue
            sim = float(self.features_[row] @ target)
            weighted_sum += sim * float(rating)
            sim_sum += abs(sim)

        if sim_sum == 0.0:
            return self._clip(self._fallback(user_id, parent_asin) + offset)
        return self._clip(weighted_sum / sim_sum + offset)

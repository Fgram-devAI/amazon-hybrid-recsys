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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.categories import build_category_features, build_category_vocab
from src.models.embedding import content_hash_for, load_or_compute_item_embeddings

from .base import Recommender

_NUMERIC_COLS = ["price", "average_rating", "rating_number"]
_ITEM_SENTIMENT_COLS = ["item_train_sentiment_mean", "item_rating_minus_sentiment_gap"]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _standardize(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] == 0:
        return matrix
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def _embed_text(meta_df: pd.DataFrame) -> list[str]:
    """Embedding text = title + description (spec §5); fall back to `text` if absent."""
    if "title" in meta_df.columns or "description" in meta_df.columns:
        empty = pd.Series([""] * len(meta_df), index=meta_df.index)
        title = meta_df.get("title", empty).fillna("").astype(str)
        desc = meta_df.get("description", empty).fillna("").astype(str)
        return (title + " " + desc).str.strip().tolist()
    return meta_df["text"].fillna("").tolist()


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
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.generic_roots = list(generic_roots)
        self.max_vocab = int(max_vocab)
        self.min_doc_freq = int(min_doc_freq)
        self.cache_dir = cache_dir                      # cached title+description embeddings
        self.review_features_dir = review_features_dir  # train-only sentiment aggregates
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

        meta_df: pd.DataFrame = metadata  # type: ignore[assignment]
        meta_df = meta_df.drop_duplicates(subset=["parent_asin"], keep="last").reset_index(
            drop=True
        )

        # --- text embedding (title+description), cached when cache_dir is set ---
        emb_df = meta_df.assign(text=_embed_text(meta_df))
        if self.cache_dir is not None:
            content_hash = content_hash_for(emb_df, self.embedder.name)  # type: ignore[union-attr]
            text_emb, ids = load_or_compute_item_embeddings(
                emb_df,
                self.embedder,  # type: ignore[arg-type]
                Path(str(self.cache_dir)),
                content_hash,
            )
        else:
            ids = meta_df["parent_asin"].tolist()
            text_emb = np.asarray(
                self.embedder.encode(emb_df["text"].tolist()),  # type: ignore[union-attr]
                dtype=np.float32,
            )
        self.item_index_ = {item: row for row, item in enumerate(ids)}

        # align metadata to the embedding id order (robust to cache ordering)
        meta_by_id = meta_df.set_index("parent_asin").loc[ids].reset_index()

        self.category_vocab_ = build_category_vocab(
            meta_by_id,
            generic_roots=self.generic_roots,
            max_vocab=self.max_vocab,
            min_doc_freq=self.min_doc_freq,
        )
        category_feats, _ = build_category_features(
            meta_by_id, self.category_vocab_, generic_roots=self.generic_roots
        )
        numeric = _standardize(meta_by_id[_NUMERIC_COLS].to_numpy(dtype=np.float32))
        item_sentiment = self._item_sentiment_features(ids)   # (n, 0) when absent

        self.features_ = _l2_normalize(
            np.hstack([text_emb, category_feats, numeric, item_sentiment]).astype(np.float32)
        )

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

    def _item_sentiment_features(self, ids: list[str]) -> np.ndarray:
        """Train-only item-sentiment columns aligned to ids; (n, 0) when cache absent."""
        path = self._aggregate_path("item_review_aggregates.parquet")
        if path is None:
            return np.zeros((len(ids), 0), dtype=np.float32)
        agg = pd.read_parquet(path).set_index("parent_asin")
        cols = [c for c in _ITEM_SENTIMENT_COLS if c in agg.columns]
        if not cols:
            return np.zeros((len(ids), 0), dtype=np.float32)
        feats = agg.reindex(ids)[cols].to_numpy(dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0)   # cold items -> 0
        return _standardize(feats)

    def _user_generosity_offsets(self) -> dict[str, float]:
        """Train-only per-user offset from sentiment/rating gap, falling back to rating bias."""
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

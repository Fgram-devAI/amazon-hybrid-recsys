"""Content-based recommender: similarity-weighted historical ratings.

First-pass simplification: the item feature is the text embedding (Granite/MiniLM)
concatenated with standardized numeric metadata. Explicit categorical encoding
(categories/store multi-hot) is intentionally OMITTED in this first pass because
those fields are already inside the embedded `text` blob built in preprocessing
(title + description + categories). A later pass may add categorical features if
they prove to help.
"""

import numpy as np

from .base import Recommender
from .embedding import content_hash_for, load_or_compute_item_embeddings

_NUMERIC_COLS = ["price", "average_rating", "rating_number"]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class ContentBasedRecommender(Recommender):
    def __init__(self, embedder: object, cache_dir: object = None) -> None:
        super().__init__()
        self.embedder = embedder
        self.cache_dir = cache_dir   # when set, item embeddings are cached on disk
        self.features_: np.ndarray | None = None  # (n_items, d) L2-normalized
        self.item_index_: dict[str, int] = {}     # parent_asin -> row
        self.user_history_: dict[str, list[tuple[str, float]]] = {}  # user -> list[(item, rating)]

    def fit(self, train: object, metadata: object = None) -> "ContentBasedRecommender":
        import pandas as pd

        train_df = train  # type: ignore[assignment]
        self._fit_means(train_df)  # type: ignore[arg-type]
        if metadata is None:
            raise ValueError("ContentBasedRecommender.fit requires metadata")

        meta_df: pd.DataFrame = metadata  # type: ignore[assignment]
        meta_df = meta_df.drop_duplicates(subset=["parent_asin"], keep="last").reset_index(
            drop=True
        )

        # use the on-disk embedding cache (Task 4) when a cache_dir is provided
        if self.cache_dir is not None:
            content_hash = content_hash_for(meta_df, self.embedder.name)  # type: ignore[union-attr]
            text_emb, ids = load_or_compute_item_embeddings(
                meta_df, self.embedder, self.cache_dir, content_hash  # type: ignore[arg-type]
            )
        else:
            ids = meta_df["parent_asin"].tolist()
            text_emb = np.asarray(
                self.embedder.encode(meta_df["text"].fillna("").tolist()),  # type: ignore[union-attr]
                dtype=np.float32,
            )

        self.item_index_ = {item: row for row, item in enumerate(ids)}

        # align numeric features to the embedding id order (robust to cache order)
        numeric_by_id = meta_df.set_index("parent_asin")[_NUMERIC_COLS]
        numeric = numeric_by_id.loc[ids].to_numpy(dtype=np.float32)
        mean = numeric.mean(axis=0)
        std = numeric.std(axis=0)
        std[std == 0] = 1.0
        numeric = (numeric - mean) / std

        self.features_ = _l2_normalize(
            np.hstack([text_emb, numeric]).astype(np.float32)
        )

        self.user_history_ = {}
        for user, group in train_df.groupby("user_id"):  # type: ignore[union-attr]
            self.user_history_[str(user)] = list(
                zip(group["parent_asin"], group["rating"])
            )
        return self

    def predict(self, user_id: str, parent_asin: str) -> float:
        target_row = self.item_index_.get(parent_asin)
        history = self.user_history_.get(user_id)
        if target_row is None or not history:
            return self._clip(self._fallback(user_id, parent_asin))

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
            sim = float(self.features_[row] @ target)  # cosine (features L2-normalized)
            weighted_sum += sim * float(rating)
            sim_sum += abs(sim)

        if sim_sum == 0.0:
            return self._clip(self._fallback(user_id, parent_asin))
        return self._clip(weighted_sum / sim_sum)

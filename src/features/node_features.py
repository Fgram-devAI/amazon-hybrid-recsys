"""Shared item + user node-feature builders.

Item features (used by ContentEnrichedRecommender AND GraphSAGE item nodes):
    [ text_embedding ; filtered_category_multi_hot ; numeric_metadata ; train_item_sentiment? ]
    L2-normalised row-wise.

User features (used by GraphSAGE user nodes; ContentEnrichedRecommender keeps its
own per-user generosity offset path):
    [ user_mean_rating ; rating_count ; positive_ratio ; generosity_offset ]
    train-only; cold users get zeros.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.categories import build_category_features, build_category_vocab
from src.models.embedding import (
    Embedder,
    content_hash_for,
    load_or_compute_item_embeddings,
)

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
    if "title" in meta_df.columns or "description" in meta_df.columns:
        empty = pd.Series([""] * len(meta_df), index=meta_df.index)
        title = meta_df.get("title", empty).fillna("").astype(str)
        desc = meta_df.get("description", empty).fillna("").astype(str)
        return (title + " " + desc).str.strip().tolist()
    return meta_df["text"].fillna("").tolist()


def _item_sentiment_features(
    ids: list[str],
    review_features_dir: Path | None,
    use_item_sentiment: bool,
) -> np.ndarray:
    if not use_item_sentiment or review_features_dir is None:
        return np.zeros((len(ids), 0), dtype=np.float32)
    path = Path(review_features_dir) / "item_review_aggregates.parquet"
    if not path.exists():
        return np.zeros((len(ids), 0), dtype=np.float32)
    agg = pd.read_parquet(path).set_index("parent_asin")
    cols = [c for c in _ITEM_SENTIMENT_COLS if c in agg.columns]
    if not cols:
        return np.zeros((len(ids), 0), dtype=np.float32)
    feats = agg.reindex(ids)[cols].to_numpy(dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0)
    return _standardize(feats)


def build_item_node_features(
    metadata: pd.DataFrame,
    *,
    embedder: Embedder,
    generic_roots: list[str],
    max_vocab: int,
    min_doc_freq: int,
    cache_dir: Path | None,
    review_features_dir: Path | None,
    use_text: bool = True,
    use_categories: bool = True,
    use_numeric: bool = True,
    use_item_sentiment: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build the item node-feature matrix. Returns (features, ids) aligned by row.

    Feature-group flags remove (not zero) the corresponding columns. Defaults
    preserve today's behavior so ContentEnriched and GraphSAGE-MSE call sites
    are unaffected.
    """
    meta_df = metadata.drop_duplicates(subset=["parent_asin"], keep="last").reset_index(drop=True)

    emb_df = meta_df.assign(text=_embed_text(meta_df))
    if use_text:
        if cache_dir is not None:
            content_hash = content_hash_for(emb_df, embedder.name)
            text_emb, ids = load_or_compute_item_embeddings(
                emb_df, embedder, Path(cache_dir), content_hash,
            )
        else:
            ids = meta_df["parent_asin"].tolist()
            text_emb = np.asarray(embedder.encode(emb_df["text"].tolist()), dtype=np.float32)
    else:
        # Alignment still needs the canonical id order. We skip the embedder
        # call so the run does NOT pay for text encoding when text is off.
        ids = meta_df["parent_asin"].tolist()
        text_emb = np.zeros((len(ids), 0), dtype=np.float32)

    meta_by_id = meta_df.set_index("parent_asin").loc[ids].reset_index()

    if use_categories:
        vocab = build_category_vocab(
            meta_by_id,
            generic_roots=generic_roots,
            max_vocab=max_vocab,
            min_doc_freq=min_doc_freq,
        )
        category_feats, _ = build_category_features(
            meta_by_id, vocab, generic_roots=generic_roots,
        )
    else:
        category_feats = np.zeros((len(ids), 0), dtype=np.float32)

    if use_numeric:
        numeric = _standardize(meta_by_id[_NUMERIC_COLS].to_numpy(dtype=np.float32))
    else:
        numeric = np.zeros((len(ids), 0), dtype=np.float32)

    item_sentiment = _item_sentiment_features(ids, review_features_dir, use_item_sentiment)

    features = _l2_normalize(
        np.hstack([text_emb, category_feats, numeric, item_sentiment]).astype(np.float32)
    )
    return features, ids


def build_user_node_features(
    train: pd.DataFrame,
    *,
    user_ids: list[str],
    review_features_dir: Path | None,
    use_generosity_offset: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Train-only behavioural features per user. Cold users (not in train) get zeros.

    When ``use_generosity_offset=False`` the user-side sentiment-derived offset
    column is skipped AND the parquet under ``review_features_dir`` is NOT read.
    """
    grouped = train.groupby("user_id")
    mean_rating = grouped["rating"].mean()
    count = grouped["rating"].count().astype(float)
    positive_ratio = (
        train.assign(_pos=(train["rating"] >= 4.0).astype(float))
        .groupby("user_id")["_pos"].mean()
    )

    generosity = pd.Series(dtype=float)
    if use_generosity_offset and review_features_dir is not None:
        path = Path(review_features_dir) / "user_review_aggregates.parquet"
        if path.exists():
            agg = pd.read_parquet(path)
            if "user_rating_minus_sentiment_gap" in agg.columns:
                gaps = agg["user_rating_minus_sentiment_gap"].astype(float)
                generosity = pd.Series(
                    (gaps - gaps.mean()).clip(-1.0, 1.0).to_numpy(),
                    index=agg["user_id"].astype(str).to_numpy(),
                )

    rows = []
    for uid in user_ids:
        if uid in mean_rating.index:
            base = [
                float(mean_rating[uid]),
                float(count[uid]),
                float(positive_ratio.get(uid, 0.0)),
            ]
            if use_generosity_offset:
                base.append(float(generosity.get(uid, 0.0)))
            rows.append(base)
        else:
            rows.append([0.0, 0.0, 0.0, 0.0] if use_generosity_offset else [0.0, 0.0, 0.0])
    return np.asarray(rows, dtype=np.float32), list(user_ids)


def _structure_only_item_features(
    item_ids: list[str],
    train: pd.DataFrame,
) -> np.ndarray:
    """Three train-structural columns per item: log-degree, mean rating, positive ratio.

    No text, no categories, no numeric metadata, no sentiment, no graph object.
    Cold items (not in ``train``) get a zero row.
    """
    grouped = train.groupby("parent_asin")
    degree = grouped["rating"].count().astype(float)
    mean_rating = grouped["rating"].mean()
    positive_ratio = (
        train.assign(_pos=(train["rating"] >= 4.0).astype(float))
        .groupby("parent_asin")["_pos"].mean()
    )

    rows = []
    for iid in item_ids:
        if iid in degree.index:
            rows.append([
                float(np.log1p(float(degree[iid]))),
                float(mean_rating[iid]),
                float(positive_ratio.get(iid, 0.0)),
            ])
        else:
            rows.append([0.0, 0.0, 0.0])
    return np.asarray(rows, dtype=np.float32)

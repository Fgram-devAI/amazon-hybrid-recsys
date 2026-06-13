"""Model-agnostic evaluation metrics: RMSE, MAE, and ranking metrics."""

from __future__ import annotations

import math

import numpy as np


def rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def precision_recall_f1_at_k(
    recommended, relevant, k
) -> tuple[float, float, float] | None:
    """Precision/Recall/F1@K for one user. Returns None if no relevant items.

    recommended: item ids ordered most->least recommended.
    relevant: set of relevant item ids for the user.
    """
    relevant = set(relevant)
    if not relevant:
        return None
    top_k = list(recommended)[:k]
    hits = sum(1 for item in top_k if item in relevant)
    precision = hits / k
    recall = hits / len(relevant)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def relevant_items_by_user(test_df, min_rating_relevant) -> dict:
    """Map user_id -> set of test items with rating >= threshold."""
    rel = test_df[test_df["rating"] >= min_rating_relevant]
    return {
        user: set(items)
        for user, items in rel.groupby("user_id")["parent_asin"]
    }


def hit_rate_at_k(recommended, relevant, k) -> float | None:
    """1.0 if any of the top-K items is relevant, else 0.0. None if no relevant items."""
    relevant = set(relevant)
    if not relevant:
        return None
    top_k = list(recommended)[:k]
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def ndcg_at_k(recommended, relevant, k) -> float | None:
    """Binary-relevance NDCG@K. None if no relevant items, 0.0 if no hits in top-K."""
    relevant = set(relevant)
    if not relevant:
        return None
    top_k = list(recommended)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(top_k, start=1)
        if item in relevant
    )
    n_ideal = min(k, len(relevant))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_ideal + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg

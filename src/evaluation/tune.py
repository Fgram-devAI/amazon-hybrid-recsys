"""Alpha tuning for the weighted hybrid on a validation slice.

NEVER tune on the held-out test split. The validation slice is carved
per-user-chronologically from ``train.parquet`` so eval cannot see it either.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import rmse
from src.models.base import Recommender


@dataclass
class TuningResult:
    best_alpha: float
    scores: list[tuple[float, float]]   # [(alpha, rmse), ...]


def split_validation(
    train: pd.DataFrame, *, validation_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user chronological split: last fraction of each user's history -> validation.

    Users with fewer than 2 rows contribute 0 validation rows (they stay fully in train).
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    sorted_df = train.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    val_indices: list[int] = []
    for _, group in sorted_df.groupby("user_id", sort=False):
        n = len(group)
        n_val = int(np.floor(n * validation_fraction))
        if n_val == 0 and n >= 2:
            n_val = 1   # ensure at least one val row when the user has history to spare
        if n_val == 0 or n_val >= n:
            continue
        val_indices.extend(group.index[-n_val:].tolist())
    val_mask = sorted_df.index.isin(val_indices)
    return sorted_df[~val_mask].reset_index(drop=True), sorted_df[val_mask].reset_index(drop=True)


def _cap_users(train: pd.DataFrame, max_users: int | None, seed: int) -> pd.DataFrame:
    """Subsample to at most ``max_users`` distinct users (seeded) — bounds tuning cost."""
    users = pd.unique(train["user_id"])
    if max_users is None or len(users) <= max_users:
        return train
    rng = np.random.default_rng(seed)
    chosen = set(users[rng.permutation(len(users))[:max_users]])
    return train[train["user_id"].isin(chosen)].reset_index(drop=True)


def tune_alpha(
    train: pd.DataFrame,
    *,
    metadata: object,
    grid: list[float],
    hybrid_factory: Callable[[float], Recommender],
    validation_fraction: float,
    seed: int,
    max_users: int | None = None,
    max_val_rows: int | None = None,
) -> TuningResult:
    """Sweep alpha; return the alpha minimizing validation RMSE.

    ``max_users`` / ``max_val_rows`` bound the work so ``--tune-alpha`` stays safe on
    large datasets (Video_Games / Movies_and_TV); both default to no cap.
    """
    train = _cap_users(train, max_users, seed)
    train_only, val = split_validation(
        train, validation_fraction=validation_fraction, seed=seed
    )
    if val.empty:
        raise ValueError("validation slice is empty — increase validation_fraction or train size")
    if max_val_rows is not None and len(val) > max_val_rows:
        val = val.sample(n=max_val_rows, random_state=seed).reset_index(drop=True)

    scores: list[tuple[float, float]] = []
    y_true = val["rating"].to_numpy(dtype=float)
    for alpha in grid:
        model = hybrid_factory(float(alpha)).fit(train_only, metadata)
        y_pred = np.array(
            [model.predict(u, i) for u, i in zip(val["user_id"], val["parent_asin"])],
            dtype=float,
        )
        scores.append((float(alpha), rmse(y_true, y_pred)))
    best_alpha = min(scores, key=lambda pair: pair[1])[0]
    return TuningResult(best_alpha=best_alpha, scores=scores)

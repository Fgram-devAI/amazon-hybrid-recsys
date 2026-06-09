"""Tests for alpha tuning on a validation slice carved from train."""

import pandas as pd
import pytest

from src.evaluation.tune import split_validation, tune_alpha
from src.models.base import Recommender


class ConstantStub(Recommender):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def fit(self, train, metadata=None):
        self._fit_means(train)
        return self

    def predict(self, user_id, parent_asin):
        return self.value


def test_split_validation_carves_per_user_chronological_tail():
    train = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "i2", "rating": 4.0, "timestamp": 2},
            {"user_id": "u1", "parent_asin": "i3", "rating": 3.0, "timestamp": 3},
            {"user_id": "u1", "parent_asin": "i4", "rating": 2.0, "timestamp": 4},
            {"user_id": "u2", "parent_asin": "i1", "rating": 5.0, "timestamp": 5},
            {"user_id": "u2", "parent_asin": "i2", "rating": 4.0, "timestamp": 6},
        ]
    )
    train_only, val = split_validation(train, validation_fraction=0.25, seed=42)
    # u1: 4 rows -> last 1 in val (i4); u2: 2 rows -> 0.25*2=0.5 -> at least 1 in val (i2)
    assert set(val["parent_asin"]) == {"i4", "i2"}
    assert len(train_only) == 4 and len(val) == 2   # u1 keeps 3, u2 keeps 1
    # Train_only + val recombine to original rows (no overlap, no loss)
    assert len(train_only) + len(val) == len(train)


def test_tune_alpha_picks_alpha_that_minimizes_validation_rmse():
    train = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "i2", "rating": 5.0, "timestamp": 2},
        ]
    )

    def factory(_alpha):
        # CF always says 5, content always says 1 -> RMSE is minimal at alpha=1.0 (cf-only)
        from src.models.weighted_hybrid import WeightedHybrid

        return WeightedHybrid(ConstantStub(5.0), ConstantStub(1.0), alpha=_alpha)

    result = tune_alpha(
        train,
        metadata=None,
        grid=[0.0, 0.5, 1.0],
        hybrid_factory=factory,
        validation_fraction=0.5,
        seed=42,
    )
    assert result.best_alpha == pytest.approx(1.0)
    # Best should be at least as good as every other alpha sampled.
    best_score = next(s for a, s in result.scores if a == result.best_alpha)
    for _, score in result.scores:
        assert best_score <= score + 1e-9

"""Tests for the evaluation runner (synthetic models + tiny data)."""

import pandas as pd

from src.evaluation.evaluate import evaluate_models, sample_negatives
from src.models.base import Recommender

TRAIN = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
        {"user_id": "u2", "parent_asin": "i2", "rating": 4.0},
    ]
)
TEST = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i3", "rating": 5.0},
        {"user_id": "u2", "parent_asin": "i4", "rating": 2.0},
    ]
)
META = pd.DataFrame(
    {"parent_asin": ["i1", "i2", "i3", "i4", "i5"], "text": ["a", "b", "c", "d", "e"]}
)


class FixedScore(Recommender):
    def fit(self, train, metadata=None):
        self._fit_means(train)
        return self

    def predict(self, user_id, parent_asin):
        return 5.0  # ranks everything equally; smoke-tests the harness


def test_sample_negatives_is_reproducible_and_excludes_exclude_set():
    import numpy as np

    items = ["i1", "i2", "i3", "i4", "i5"]
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a = sample_negatives(items, exclude={"i1"}, n=2, rng=rng1)
    b = sample_negatives(items, exclude={"i1"}, n=2, rng=rng2)
    assert a == b                      # reproducible under same seed
    assert "i1" not in a               # excludes seen/positive items


def test_evaluate_models_returns_one_row_per_model_with_metric_columns():
    table = evaluate_models(
        {"fixed": FixedScore()},
        TRAIN,
        TEST,
        META,
        k=2,
        min_rating_relevant=4.0,
        num_negatives=2,
        seed=42,
    )
    assert list(table["model"]) == ["fixed"]
    for col in ["dataset", "model", "rmse", "mae",
                "precision_at_k", "recall_at_k", "f1_at_k"]:
        assert col in table.columns
    assert table["rmse"].iloc[0] >= 0.0

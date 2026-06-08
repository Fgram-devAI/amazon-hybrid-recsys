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


def test_evaluate_models_can_cap_ranking_users():
    test = pd.DataFrame(
        [
            {"user_id": f"u{i}", "parent_asin": f"t{i}", "rating": 5.0}
            for i in range(5)
        ]
    )
    train = pd.DataFrame(
        [
            {"user_id": f"u{i}", "parent_asin": f"i{i}", "rating": 4.0}
            for i in range(5)
        ]
    )
    metadata = pd.DataFrame({"parent_asin": [f"i{i}" for i in range(5)]})

    table = evaluate_models(
        {"fixed": FixedScore()},
        train,
        test,
        metadata,
        k=2,
        min_rating_relevant=4.0,
        num_negatives=2,
        seed=42,
        max_eval_users=2,
    )

    assert table["n_eval_users"].iloc[0] == 2
    assert table["max_eval_users"].iloc[0] == 2


def test_build_models_shares_hybrid_component_instances():
    from src.evaluation.evaluate import build_models
    from src.models.embedding import FakeEmbedder

    config = {
        "processed_dir": "data/processed",
        "hybrid": {"alpha": 0.5},
        "models": {"ranking_random_seed": 42},
    }
    models = build_models(config, "tiny", FakeEmbedder(dim=8), no_knn=True)

    # the hybrid reuses the standalone svd/content instances -> fitted once each
    assert models["hybrid"].cf is models["svd"]
    assert models["hybrid"].content is models["content"]


def test_sample_negatives_returns_available_when_exclude_exceeds_catalog():
    import numpy as np

    items = np.asarray(["i1", "i2", "i3"], dtype=object)
    # exclude = 1 catalog item + 3 out-of-catalog positives -> len(exclude) > catalog
    exclude = {"i1", "p1", "p2", "p3"}
    negs = sample_negatives(items, exclude, n=2, rng=np.random.default_rng(42))
    assert set(negs) == {"i2", "i3"}  # available catalog negatives, not []

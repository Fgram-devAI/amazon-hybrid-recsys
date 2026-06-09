"""Tests for random/popularity sanity baselines."""

import pandas as pd

from src.models.baselines import PopularityRecommender, RandomRecommender

_TRAIN = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
        {"user_id": "u1", "parent_asin": "i2", "rating": 4.0},
        {"user_id": "u2", "parent_asin": "i1", "rating": 5.0},
        {"user_id": "u2", "parent_asin": "i3", "rating": 3.0},
        {"user_id": "u3", "parent_asin": "i1", "rating": 5.0},
    ]
)


def test_random_recommender_is_deterministic_under_seed():
    a = RandomRecommender(seed=42).fit(_TRAIN)
    b = RandomRecommender(seed=42).fit(_TRAIN)
    candidates = ["i1", "i2", "i3", "i4"]
    assert a.recommend("u1", k=3, candidates=candidates) == b.recommend(
        "u1", k=3, candidates=candidates
    )


def test_random_recommender_predict_returns_global_mean():
    model = RandomRecommender(seed=42).fit(_TRAIN)
    # rating predictions are mean-based, not random -> deterministic RMSE/MAE
    assert model.predict("u1", "i1") == model.global_mean_


def test_popularity_ranks_by_training_count_only():
    model = PopularityRecommender().fit(_TRAIN)
    # i1 appears 3x in train, i2 1x, i3 1x. u1 already saw i1 + i2 -> top1 is i3.
    assert model.recommend("u1", k=1, candidates=["i1", "i2", "i3"]) == ["i3"]
    # New user u9 has no history -> top should be i1 (most popular)
    assert model.recommend("u9", k=1, candidates=["i1", "i2", "i3"]) == ["i1"]


def test_popularity_predict_falls_back_to_means():
    model = PopularityRecommender().fit(_TRAIN)
    # unseen item -> fallback to user mean -> u1 mean = (5+4)/2 = 4.5
    assert model.predict("u1", "i99") == 4.5

"""Tests for the Surprise-backed CF wrappers (tiny fixture)."""

import pandas as pd

from src.models.cf import KNNRecommender, SVDRecommender

TRAIN = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
        {"user_id": "u1", "parent_asin": "i2", "rating": 4.0},
        {"user_id": "u2", "parent_asin": "i1", "rating": 4.0},
        {"user_id": "u2", "parent_asin": "i2", "rating": 5.0},
        {"user_id": "u3", "parent_asin": "i1", "rating": 3.0},
    ]
)


def test_svd_fits_and_predicts_in_range():
    model = SVDRecommender(random_state=42).fit(TRAIN)
    value = model.predict("u1", "i2")
    assert 1.0 <= value <= 5.0
    # unknown user/item still returns a clipped float (no crash)
    assert 1.0 <= model.predict("u_new", "i_new") <= 5.0


def test_knn_fits_and_predicts_in_range():
    model = KNNRecommender().fit(TRAIN)
    value = model.predict("u1", "i2")
    assert 1.0 <= value <= 5.0
    assert 1.0 <= model.predict("u_new", "i_new") <= 5.0

"""Tests for the Recommender base class (fallbacks, clip, recommend)."""

import pandas as pd

from src.models.base import Recommender

TRAIN = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
        {"user_id": "u1", "parent_asin": "i2", "rating": 3.0},
        {"user_id": "u2", "parent_asin": "i1", "rating": 4.0},
        {"user_id": "u2", "parent_asin": "i3", "rating": 2.0},
    ]
)


class ConstantRecommender(Recommender):
    """Minimal concrete subclass that predicts a fixed value for known pairs."""

    def fit(self, train, metadata=None):
        self._fit_means(train)
        return self

    def predict(self, user_id, parent_asin):
        return self._clip(10.0)  # deliberately out of range to test clipping


def test_predict_is_clipped_to_1_5():
    model = ConstantRecommender().fit(TRAIN)
    assert model.predict("u1", "i1") == 5.0


def test_fallback_uses_item_then_user_then_global_mean():
    model = ConstantRecommender().fit(TRAIN)
    # item mean of i1 = (5+4)/2 = 4.5
    assert model._fallback("u_new", "i1") == 4.5
    # unknown item, known user u1 -> user mean = (5+3)/2 = 4.0
    assert model._fallback("u1", "i_new") == 4.0
    # unknown user and item -> global mean = (5+3+4+2)/4 = 3.5
    assert model._fallback("u_new", "i_new") == 3.5


def test_recommend_excludes_seen_items_and_caps_at_k():
    model = ConstantRecommender().fit(TRAIN)
    # u1 has seen i1, i2; candidates i1,i2,i3 -> only i3 unseen
    recs = model.recommend("u1", k=5, candidates=["i1", "i2", "i3"])
    assert recs == ["i3"]

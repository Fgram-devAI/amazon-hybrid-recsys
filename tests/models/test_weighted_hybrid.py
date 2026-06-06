"""Tests for the weighted hybrid (alpha endpoints)."""

import pandas as pd

from src.models.base import Recommender
from src.models.weighted_hybrid import WeightedHybrid

TRAIN = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
        {"user_id": "u1", "parent_asin": "i2", "rating": 3.0},
    ]
)


class StubRecommender(Recommender):
    """Predicts a constant; used to verify the blend exactly."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def fit(self, train, metadata=None):
        self._fit_means(train)
        return self

    def predict(self, user_id, parent_asin):
        return self.value


def _hybrid(alpha):
    cf = StubRecommender(4.0)
    content = StubRecommender(2.0)
    return WeightedHybrid(cf, content, alpha=alpha).fit(TRAIN)


def test_alpha_endpoints_and_midpoint():
    assert _hybrid(1.0).predict("u1", "i1") == 4.0   # all CF
    assert _hybrid(0.0).predict("u1", "i1") == 2.0   # all content
    assert _hybrid(0.5).predict("u1", "i1") == 3.0   # average

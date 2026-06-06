"""Tests for the content-based recommender (FakeEmbedder, tiny fixture)."""

import pandas as pd

from src.models.content_based import ContentBasedRecommender
from src.models.embedding import FakeEmbedder

# u1 likes RPGs (i1, i2 rated 5). i3 is another RPG; i4 is unrelated.
TRAIN = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
        {"user_id": "u1", "parent_asin": "i2", "rating": 5.0},
        {"user_id": "u2", "parent_asin": "i4", "rating": 4.0},
    ]
)
META = pd.DataFrame(
    [
        {"parent_asin": "i1", "text": "rpg dragon quest", "price": 10.0,
         "average_rating": 4.5, "rating_number": 100},
        {"parent_asin": "i2", "text": "rpg dragon hero", "price": 12.0,
         "average_rating": 4.4, "rating_number": 80},
        {"parent_asin": "i3", "text": "rpg dragon adventure", "price": 11.0,
         "average_rating": 4.6, "rating_number": 90},
        {"parent_asin": "i4", "text": "cooking pasta recipe", "price": 9.0,
         "average_rating": 4.0, "rating_number": 50},
    ]
)


def test_similar_item_scores_higher_than_unrelated():
    model = ContentBasedRecommender(FakeEmbedder(dim=64)).fit(TRAIN, META)
    score_similar = model.predict("u1", "i3")   # RPG, like u1's history
    score_unrelated = model.predict("u1", "i4")  # cooking
    assert score_similar > score_unrelated
    assert 1.0 <= score_similar <= 5.0


def test_unknown_user_falls_back_without_crashing():
    model = ContentBasedRecommender(FakeEmbedder(dim=64)).fit(TRAIN, META)
    value = model.predict("u_new", "i1")
    assert 1.0 <= value <= 5.0

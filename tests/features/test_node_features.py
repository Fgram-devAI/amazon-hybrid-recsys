"""Shared node-feature builder used by ContentEnrichedRecommender + GraphSAGE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.node_features import (
    build_item_node_features,
    build_user_node_features,
)


class _FakeEmbedder:
    name = "fake-embedder-v1"
    device = "cpu"

    def encode(self, texts):
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(texts), 4)).astype(np.float32)


def _toy_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_asin": ["i1", "i2", "i3"],
            "title": ["a", "b", "c"],
            "description": ["d1", "d2", "d3"],
            "categories": [["Action"], ["Comedy"], ["Action", "Comedy"]],
            "price": [10.0, 20.0, 30.0],
            "average_rating": [4.0, 4.5, 3.5],
            "rating_number": [100, 50, 75],
        }
    )


def _toy_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id":    ["u1", "u1", "u2"],
            "parent_asin":["i1", "i2", "i1"],
            "rating":     [5.0,  3.0,  4.0],
            "timestamp":  [1, 2, 3],
        }
    )


def test_build_item_node_features_returns_aligned_matrix():
    embedder = _FakeEmbedder()
    meta = _toy_metadata()

    features, ids = build_item_node_features(
        meta,
        embedder=embedder,
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        cache_dir=None,
        review_features_dir=None,
        use_item_sentiment=False,
    )

    assert features.shape[0] == 3
    assert features.shape[1] > 4   # text 4d + categories + numeric
    assert ids == ["i1", "i2", "i3"]
    # L2-normalised rows
    norms = np.linalg.norm(features, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)


def test_build_user_node_features_uses_train_only_aggregates():
    train = _toy_train()
    users = ["u1", "u2", "u3"]   # u3 is cold

    features, user_ids = build_user_node_features(
        train,
        user_ids=users,
        review_features_dir=None,
    )

    assert user_ids == users
    assert features.shape == (3, 4)   # [mean_rating, count, positive_ratio, generosity_offset]
    # u1 has 2 ratings (5,3) -> mean 4, count 2, positive_ratio 0.5
    assert features[0, 0] == pytest.approx(4.0)
    assert features[0, 1] == pytest.approx(2.0)
    assert features[0, 2] == pytest.approx(0.5)
    # cold user u3 -> zeros
    np.testing.assert_array_equal(features[2], np.zeros(4))

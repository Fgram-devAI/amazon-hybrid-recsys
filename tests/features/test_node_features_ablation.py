"""Feature-group ablation tests for GraphSAGE-BPR.

These tests pin the contract that:
1. The default-args call to ``build_item_node_features`` returns the SAME matrix
   as the explicit-True call (regression guard for ContentEnriched and
   GraphSAGE-MSE which keep default args).
2. Each named ``feature_set`` produces the expected matrix column count
   (added in later tasks).
3. Disabling the sentiment-derived feature groups means NO parquet read happens
   under ``review_features_dir`` (leakage guard, added in a later task).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.node_features import (
    build_item_node_features,
    build_user_node_features,
    _structure_only_item_features,
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
            "parent_asin": ["i1", "i2", "i3", "i4"],
            "title":       ["a", "b", "c", "d"],
            "description": ["d1", "d2", "d3", "d4"],
            "categories":  [["Action"], ["Comedy"], ["Action", "Comedy"], ["Sports"]],
            "price":          [10.0, 20.0, 30.0, 15.0],
            "average_rating": [4.0,  4.5,  3.5,  4.2],
            "rating_number":  [100,  50,   75,   25],
        }
    )


def _toy_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id":     ["u1", "u1", "u2", "u2", "u3"],
            "parent_asin": ["i1", "i2", "i1", "i3", "i4"],
            "rating":      [5.0,  3.0,  4.0,  5.0,  4.0],
            "timestamp":   [1, 2, 3, 4, 5],
        }
    )


def test_default_args_match_explicit_true_for_items():
    """Pins ContentEnriched + GraphSAGE-MSE behavior. Both keep default args."""
    embedder = _FakeEmbedder()
    meta = _toy_metadata()

    features_default, ids_default = build_item_node_features(
        meta,
        embedder=embedder,
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        cache_dir=None,
        review_features_dir=None,
    )
    features_explicit, ids_explicit = build_item_node_features(
        meta,
        embedder=embedder,
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        cache_dir=None,
        review_features_dir=None,
        use_text=True,
        use_categories=True,
        use_numeric=True,
        use_item_sentiment=True,
    )

    assert ids_default == ids_explicit
    np.testing.assert_array_equal(features_default, features_explicit)


def test_default_args_match_explicit_true_for_users():
    """Pins GraphSAGE-MSE user-feature behavior. Defaults must equal explicit True."""
    train = _toy_train()
    user_ids = ["u1", "u2", "u3", "u4_cold"]

    feats_default, ids_default = build_user_node_features(
        train,
        user_ids=user_ids,
        review_features_dir=None,
    )
    feats_explicit, ids_explicit = build_user_node_features(
        train,
        user_ids=user_ids,
        review_features_dir=None,
        use_generosity_offset=True,
    )

    assert ids_default == ids_explicit
    np.testing.assert_array_equal(feats_default, feats_explicit)


TEXT_DIM = 4  # _FakeEmbedder produces 4-dim outputs
NUMERIC_DIM = 3  # ["price", "average_rating", "rating_number"]
SENTIMENT_DIM = 0  # no review_features_dir provided -> 0 columns


def _build_items(use_text, use_categories, use_numeric, use_item_sentiment):
    return build_item_node_features(
        _toy_metadata(),
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        cache_dir=None,
        review_features_dir=None,
        use_text=use_text,
        use_categories=use_categories,
        use_numeric=use_numeric,
        use_item_sentiment=use_item_sentiment,
    )


def _expected_full_dim(features) -> int:
    # text + categories + numeric (+ sentiment which is 0 here)
    cat_dim = features.shape[1] - TEXT_DIM - NUMERIC_DIM - SENTIMENT_DIM
    return TEXT_DIM + cat_dim + NUMERIC_DIM + SENTIMENT_DIM


def test_full_feature_set_dim():
    features, _ = _build_items(True, True, True, True)
    assert features.shape[1] == _expected_full_dim(features)
    assert features.shape[1] > TEXT_DIM + NUMERIC_DIM  # categories present


def test_no_text_drops_only_text_columns():
    full_features, _ = _build_items(True, True, True, True)
    notext_features, _ = _build_items(False, True, True, True)
    assert notext_features.shape[1] == full_features.shape[1] - TEXT_DIM
    assert notext_features.shape[0] == full_features.shape[0]


def test_no_sentiment_drops_only_sentiment_columns():
    # With review_features_dir=None there are 0 sentiment columns, so
    # no_sentiment and full have the SAME shape. The assertion is that the
    # flag is accepted and that the result is dim-equivalent in this case;
    # the leakage guard test in Task 4 covers behavior when the dir exists.
    full_features, _ = _build_items(True, True, True, True)
    nosent_features, _ = _build_items(True, True, True, False)
    assert nosent_features.shape[1] == full_features.shape[1] - SENTIMENT_DIM


def test_metadata_only_drops_text_and_sentiment():
    full_features, _ = _build_items(True, True, True, True)
    meta_features, _ = _build_items(False, True, True, False)
    assert meta_features.shape[1] == full_features.shape[1] - TEXT_DIM - SENTIMENT_DIM


def test_structure_only_returns_3_columns():
    """Three train-structural columns: [log_degree, mean_rating, positive_ratio]."""
    train = _toy_train()
    item_ids = ["i1", "i2", "i3", "i4"]

    features = _structure_only_item_features(item_ids, train)

    assert features.shape == (4, 3)
    assert features.dtype == np.float32


def test_structure_only_cold_items_get_zero_row():
    train = _toy_train()
    item_ids = ["i1", "i_cold"]  # i_cold not in train

    features = _structure_only_item_features(item_ids, train)

    assert features.shape == (2, 3)
    assert np.allclose(features[1], 0.0)


def test_structure_only_log_degree_is_log1p_count():
    train = _toy_train()
    item_ids = ["i1", "i2"]  # i1 has 2 interactions, i2 has 1

    features = _structure_only_item_features(item_ids, train)

    # First column is the log-degree (log1p of train interaction count).
    assert features[0, 0] == np.float32(np.log1p(2.0))
    assert features[1, 0] == np.float32(np.log1p(1.0))


def test_no_sentiment_does_not_read_item_aggregates(tmp_path, monkeypatch):
    """`use_item_sentiment=False` must NOT call pd.read_parquet on the aggregates path."""
    aggregates = tmp_path / "item_review_aggregates.parquet"
    pd.DataFrame(
        {
            "parent_asin": ["i1", "i2", "i3"],
            "item_train_sentiment_mean": [0.1, 0.2, 0.3],
            "item_rating_minus_sentiment_gap": [0.0, 0.1, -0.1],
        }
    ).to_parquet(aggregates)

    forbidden_reads: list[str] = []
    real_read_parquet = pd.read_parquet

    def guarded(path, *args, **kwargs):
        if str(aggregates) in str(path):
            forbidden_reads.append(str(path))
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)

    # Sentiment OFF: must NOT read the item aggregates parquet.
    build_item_node_features(
        _toy_metadata(),
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        cache_dir=None,
        review_features_dir=tmp_path,
        use_item_sentiment=False,
    )
    assert forbidden_reads == [], (
        f"use_item_sentiment=False must not read aggregates, got: {forbidden_reads}"
    )

    # Sentiment ON: MUST read the item aggregates parquet (sanity check).
    build_item_node_features(
        _toy_metadata(),
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        cache_dir=None,
        review_features_dir=tmp_path,
        use_item_sentiment=True,
    )
    assert any(str(aggregates) in p for p in forbidden_reads), (
        "use_item_sentiment=True should read the item aggregates parquet"
    )


def test_no_generosity_offset_does_not_read_user_aggregates(tmp_path, monkeypatch):
    """`use_generosity_offset=False` must NOT call pd.read_parquet on user aggregates."""
    aggregates = tmp_path / "user_review_aggregates.parquet"
    pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "user_rating_minus_sentiment_gap": [0.1, -0.2, 0.0],
        }
    ).to_parquet(aggregates)

    forbidden_reads: list[str] = []
    real_read_parquet = pd.read_parquet

    def guarded(path, *args, **kwargs):
        if str(aggregates) in str(path):
            forbidden_reads.append(str(path))
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)

    build_user_node_features(
        _toy_train(),
        user_ids=["u1", "u2", "u3"],
        review_features_dir=tmp_path,
        use_generosity_offset=False,
    )
    assert forbidden_reads == [], (
        f"use_generosity_offset=False must not read user aggregates, got: {forbidden_reads}"
    )

    build_user_node_features(
        _toy_train(),
        user_ids=["u1", "u2", "u3"],
        review_features_dir=tmp_path,
        use_generosity_offset=True,
    )
    assert any(str(aggregates) in p for p in forbidden_reads), (
        "use_generosity_offset=True should read the user aggregates parquet"
    )


def test_no_sentiment_drops_two_item_aggregate_columns(tmp_path):
    """When aggregate parquet exists, no_sentiment must remove the 2 sentiment columns."""
    pd.DataFrame(
        {
            "parent_asin": ["i1", "i2", "i3", "i4"],
            "item_train_sentiment_mean": [0.1, 0.2, 0.3, 0.4],
            "item_rating_minus_sentiment_gap": [0.0, 0.1, -0.1, 0.2],
        }
    ).to_parquet(tmp_path / "item_review_aggregates.parquet")

    full_features, _ = build_item_node_features(
        _toy_metadata(),
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        cache_dir=None,
        review_features_dir=tmp_path,
        use_item_sentiment=True,
    )
    no_sent_features, _ = build_item_node_features(
        _toy_metadata(),
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        cache_dir=None,
        review_features_dir=tmp_path,
        use_item_sentiment=False,
    )

    assert no_sent_features.shape[1] == full_features.shape[1] - 2


def test_graphsage_bpr_rejects_unknown_feature_set():
    """Constructor validation should reject unknown feature_set values."""
    import pytest

    from src.models.graphsage_bpr import GraphSAGEBPRRecommender

    with pytest.raises(ValueError, match="feature_set must be one of"):
        GraphSAGEBPRRecommender(
            embedder=_FakeEmbedder(),
            generic_roots=["Movies & TV"],
            max_vocab=8,
            min_doc_freq=1,
            feature_set="not_a_real_set",
        )


def test_graphsage_bpr_metadata_only_wires_smaller_feature_matrix():
    """A non-full feature_set must reach GraphSAGE-BPR's prepared node matrix."""
    from src.models.graphsage_bpr import GraphSAGEBPRRecommender

    train = _toy_train()
    meta = _toy_metadata()

    full = GraphSAGEBPRRecommender(
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        feature_set="full",
        epochs=1,
        batch_size=4,
        progress=False,
    ).prepare_for_checkpoint(train, meta)
    metadata_only = GraphSAGEBPRRecommender(
        embedder=_FakeEmbedder(),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        feature_set="metadata_only",
        epochs=1,
        batch_size=4,
        progress=False,
    ).prepare_for_checkpoint(train, meta)

    assert full._x is not None
    assert metadata_only._x is not None
    assert metadata_only._x.shape[0] == full._x.shape[0]
    assert metadata_only._x.shape[1] < full._x.shape[1]

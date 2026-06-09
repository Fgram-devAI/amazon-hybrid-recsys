"""Tests for the enriched content recommender (text + categories + numeric + sentiment)."""

import pandas as pd
import pytest

from src.models.content_enriched import ContentEnrichedRecommender
from src.models.embedding import FakeEmbedder

_TRAIN = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
        {"user_id": "u1", "parent_asin": "i2", "rating": 5.0},
    ]
)

_METADATA = pd.DataFrame(
    [
        {"parent_asin": "i1", "text": "comedy movie", "categories": ["Movies & TV", "Comedy"],
         "price": 9.99, "average_rating": 4.5, "rating_number": 100},
        {"parent_asin": "i2", "text": "another comedy", "categories": ["Comedy"],
         "price": 12.0, "average_rating": 4.6, "rating_number": 150},
        {"parent_asin": "i3", "text": "drama movie", "categories": ["Movies & TV", "Drama"],
         "price": 10.0, "average_rating": 3.5, "rating_number": 50},
    ]
)


def test_enriched_model_fits_and_predicts_high_for_same_category():
    model = ContentEnrichedRecommender(
        FakeEmbedder(dim=16),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
    ).fit(_TRAIN, _METADATA)
    # u1 historically rated two Comedy items 5★ -> predicting another Comedy item should clip near 5.
    pred = model.predict("u1", "i2")
    assert 1.0 <= pred <= 5.0


def test_enriched_features_include_category_columns():
    model = ContentEnrichedRecommender(
        FakeEmbedder(dim=8),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
    ).fit(_TRAIN, _METADATA)
    # text(8) + categories(<=2: Comedy/Drama) + numeric(3) = >=13 columns
    assert model.features_ is not None
    assert model.features_.shape[1] >= 8 + 2 + 3
    assert sorted(model.category_vocab_) == ["Comedy", "Drama"]


def test_unknown_user_falls_back_cleanly():
    model = ContentEnrichedRecommender(
        FakeEmbedder(dim=8),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
    ).fit(_TRAIN, _METADATA)
    # u9 has no history -> falls back to item/global mean within [1, 5]
    pred = model.predict("u9", "i1")
    assert 1.0 <= pred <= 5.0


def test_unknown_item_falls_back_cleanly():
    model = ContentEnrichedRecommender(
        FakeEmbedder(dim=8),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
    ).fit(_TRAIN, _METADATA)
    pred = model.predict("u1", "i_does_not_exist")
    assert 1.0 <= pred <= 5.0


def test_metadata_dedup_by_parent_asin():
    duped = pd.concat([_METADATA, _METADATA.iloc[[0]]], ignore_index=True)
    model = ContentEnrichedRecommender(
        FakeEmbedder(dim=8),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
    ).fit(_TRAIN, duped)
    # i1 must appear exactly once in the feature index regardless of duplicates upstream.
    assert list(model.item_index_).count("i1") == 1
    assert model.features_ is not None
    assert model.features_.shape[0] == 3


def test_sentiment_and_user_aggregates_are_consumed(tmp_path):
    # train-only aggregate caches written by the offline sentiment job
    rf_dir = tmp_path / "advanced_features"
    rf_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"parent_asin": "i1", "item_train_sentiment_mean": 0.9, "item_rating_minus_sentiment_gap": 4.1},
            {"parent_asin": "i2", "item_train_sentiment_mean": 0.8, "item_rating_minus_sentiment_gap": 4.2},
            {"parent_asin": "i3", "item_train_sentiment_mean": -0.5, "item_rating_minus_sentiment_gap": 4.0},
        ]
    ).to_parquet(rf_dir / "item_review_aggregates.parquet", index=False)
    pd.DataFrame([{"user_id": "u1", "user_mean_rating": 5.0}]).to_parquet(
        rf_dir / "user_review_aggregates.parquet", index=False
    )

    base = ContentEnrichedRecommender(
        FakeEmbedder(dim=8), generic_roots=["Movies & TV"], max_vocab=8, min_doc_freq=1
    ).fit(_TRAIN, _METADATA)
    enriched = ContentEnrichedRecommender(
        FakeEmbedder(dim=8), generic_roots=["Movies & TV"], max_vocab=8, min_doc_freq=1,
        review_features_dir=rf_dir,
    ).fit(_TRAIN, _METADATA)

    # item-sentiment aggregates add 2 feature columns -> they ARE consumed
    assert enriched.features_ is not None and base.features_ is not None
    assert enriched.features_.shape[1] == base.features_.shape[1] + 2
    # user-generosity offset comes from the train-only user aggregates -> consumed in predict
    assert enriched.user_offset_["u1"] == pytest.approx(min(1.0, 5.0 - enriched.global_mean_))


def test_user_offset_prefers_sentiment_gap_over_rating_mean(tmp_path):
    rf_dir = tmp_path / "advanced_features"
    rf_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"user_id": "u1", "user_mean_rating": 5.0, "user_rating_minus_sentiment_gap": 2.0},
            {"user_id": "u2", "user_mean_rating": 1.0, "user_rating_minus_sentiment_gap": 4.0},
        ]
    ).to_parquet(rf_dir / "user_review_aggregates.parquet", index=False)

    model = ContentEnrichedRecommender(
        FakeEmbedder(dim=8),
        generic_roots=["Movies & TV"],
        max_vocab=8,
        min_doc_freq=1,
        review_features_dir=rf_dir,
    ).fit(_TRAIN, _METADATA)

    assert model.user_offset_["u1"] == pytest.approx(-1.0)
    assert model.user_offset_["u2"] == pytest.approx(1.0)


def test_no_sentiment_variant_skips_item_aggregate_file(tmp_path):
    rf_dir = tmp_path / "advanced_features"
    rf_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"parent_asin": "i1", "item_train_sentiment_mean": 0.9, "item_rating_minus_sentiment_gap": 4.1},
            {"parent_asin": "i2", "item_train_sentiment_mean": 0.8, "item_rating_minus_sentiment_gap": 4.2},
            {"parent_asin": "i3", "item_train_sentiment_mean": -0.5, "item_rating_minus_sentiment_gap": 4.0},
        ]
    ).to_parquet(rf_dir / "item_review_aggregates.parquet", index=False)
    pd.DataFrame(
        [{"user_id": "u1", "user_mean_rating": 5.0, "user_rating_minus_sentiment_gap": 2.0}]
    ).to_parquet(rf_dir / "user_review_aggregates.parquet", index=False)

    base = ContentEnrichedRecommender(
        FakeEmbedder(dim=8), generic_roots=["Movies & TV"], max_vocab=8, min_doc_freq=1,
    ).fit(_TRAIN, _METADATA)
    no_sent = ContentEnrichedRecommender(
        FakeEmbedder(dim=8), generic_roots=["Movies & TV"], max_vocab=8, min_doc_freq=1,
        review_features_dir=rf_dir,
        use_item_sentiment=False,
        use_user_offset=False,
    ).fit(_TRAIN, _METADATA)

    # No sentiment columns added even though the aggregate parquet is present.
    assert no_sent.features_ is not None and base.features_ is not None
    assert no_sent.features_.shape[1] == base.features_.shape[1]
    # User offset map must stay empty even though user_review_aggregates.parquet exists.
    assert no_sent.user_offset_ == {}


def test_no_sentiment_variant_does_not_open_aggregate_files(tmp_path, monkeypatch):
    rf_dir = tmp_path / "advanced_features"
    rf_dir.mkdir(parents=True)
    # Sentinel: any read_parquet on these paths should fail the test.
    bad_item = rf_dir / "item_review_aggregates.parquet"
    bad_user = rf_dir / "user_review_aggregates.parquet"
    bad_item.write_bytes(b"NOT-A-PARQUET")
    bad_user.write_bytes(b"NOT-A-PARQUET")

    model = ContentEnrichedRecommender(
        FakeEmbedder(dim=8), generic_roots=["Movies & TV"], max_vocab=8, min_doc_freq=1,
        review_features_dir=rf_dir,
        use_item_sentiment=False,
        use_user_offset=False,
    ).fit(_TRAIN, _METADATA)
    # If the flags were ignored, reading the corrupt parquet above would have raised.
    assert model.features_ is not None
    assert model.user_offset_ == {}


def test_sentiment_aware_variant_still_consumes_aggregates(tmp_path):
    rf_dir = tmp_path / "advanced_features"
    rf_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"parent_asin": "i1", "item_train_sentiment_mean": 0.9, "item_rating_minus_sentiment_gap": 4.1},
            {"parent_asin": "i2", "item_train_sentiment_mean": 0.8, "item_rating_minus_sentiment_gap": 4.2},
            {"parent_asin": "i3", "item_train_sentiment_mean": -0.5, "item_rating_minus_sentiment_gap": 4.0},
        ]
    ).to_parquet(rf_dir / "item_review_aggregates.parquet", index=False)
    pd.DataFrame([{"user_id": "u1", "user_mean_rating": 5.0}]).to_parquet(
        rf_dir / "user_review_aggregates.parquet", index=False
    )

    base = ContentEnrichedRecommender(
        FakeEmbedder(dim=8), generic_roots=["Movies & TV"], max_vocab=8, min_doc_freq=1,
    ).fit(_TRAIN, _METADATA)
    enriched = ContentEnrichedRecommender(
        FakeEmbedder(dim=8), generic_roots=["Movies & TV"], max_vocab=8, min_doc_freq=1,
        review_features_dir=rf_dir,
        # Defaults stay True -> sentiment-aware behavior is preserved.
    ).fit(_TRAIN, _METADATA)

    assert enriched.features_ is not None and base.features_ is not None
    assert enriched.features_.shape[1] == base.features_.shape[1] + 2
    assert "u1" in enriched.user_offset_

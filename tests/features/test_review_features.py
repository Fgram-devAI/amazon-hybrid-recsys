"""Tests for user/item train-only review aggregates."""

import pandas as pd
import pytest

from src.features.review_features import (
    item_review_aggregates,
    user_review_aggregates,
)

_TRAIN = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 1},
        {"user_id": "u1", "parent_asin": "i2", "rating": 3.0, "timestamp": 2},
        {"user_id": "u2", "parent_asin": "i1", "rating": 4.0, "timestamp": 3},
    ]
)

_SENTIMENT = pd.DataFrame(
    [
        {"user_id": "u1", "parent_asin": "i1", "timestamp": 1,
         "sentiment_score": 1.0, "sentiment_label": "POSITIVE",
         "positive_prob": 1.0, "negative_prob": 0.0},
        {"user_id": "u1", "parent_asin": "i2", "timestamp": 2,
         "sentiment_score": -1.0, "sentiment_label": "NEGATIVE",
         "positive_prob": 0.0, "negative_prob": 1.0},
        {"user_id": "u2", "parent_asin": "i1", "timestamp": 3,
         "sentiment_score": 0.5, "sentiment_label": "POSITIVE",
         "positive_prob": 0.75, "negative_prob": 0.25},
    ]
)


def test_user_aggregates_match_hand_computed():
    df = user_review_aggregates(_TRAIN, _SENTIMENT)
    by_user = df.set_index("user_id")
    # u1: ratings [5,3] -> mean 4, std 1; sentiment [1,-1] -> mean 0, std 1; gap mean(rating - 3*sent) ... use spec form
    assert by_user.loc["u1", "user_mean_rating"] == pytest.approx(4.0)
    assert by_user.loc["u1", "user_review_sentiment_mean"] == pytest.approx(0.0)
    assert by_user.loc["u1", "user_rating_minus_sentiment_gap"] == pytest.approx(4.0)
    # u2 single row -> std is 0
    assert by_user.loc["u2", "user_rating_std"] == pytest.approx(0.0)


def test_item_aggregates_match_hand_computed():
    df = item_review_aggregates(_TRAIN, _SENTIMENT)
    by_item = df.set_index("parent_asin")
    # i1: ratings [5,4], sentiment [1.0, 0.5]
    assert by_item.loc["i1", "item_train_sentiment_mean"] == pytest.approx(0.75)
    assert by_item.loc["i1", "item_train_sentiment_count"] == 2
    assert by_item.loc["i1", "item_rating_minus_sentiment_gap"] == pytest.approx(4.5 - 0.75)


def test_aggregates_use_train_only_keys_not_extras():
    # Sentiment cache also contains a stray (u3, i9) row that is NOT in train.
    stray = _SENTIMENT.copy()
    stray = pd.concat(
        [
            stray,
            pd.DataFrame(
                [{"user_id": "u3", "parent_asin": "i9", "timestamp": 99,
                  "sentiment_score": 1.0, "sentiment_label": "POSITIVE",
                  "positive_prob": 1.0, "negative_prob": 0.0}]
            ),
        ],
        ignore_index=True,
    )
    df = user_review_aggregates(_TRAIN, stray)
    assert set(df["user_id"]) == {"u1", "u2"}   # u3 dropped — not in train


def test_duplicate_sentiment_keys_do_not_duplicate_train_rows():
    duplicated = pd.concat([_SENTIMENT, _SENTIMENT.iloc[[0]]], ignore_index=True)
    df = item_review_aggregates(_TRAIN, duplicated)
    by_item = df.set_index("parent_asin")
    assert by_item.loc["i1", "item_train_sentiment_count"] == 2


def test_cold_user_returns_no_row():
    # No interactions for u9 in train -> aggregates omit u9 entirely.
    df = user_review_aggregates(_TRAIN, _SENTIMENT)
    assert "u9" not in set(df["user_id"])


def test_item_aggregates_handle_missing_sentiment():
    # i3 has training interactions but no sentiment row -> still present with NaN-safe defaults.
    train_with_i3 = pd.concat(
        [
            _TRAIN,
            pd.DataFrame(
                [{"user_id": "u2", "parent_asin": "i3", "rating": 2.0, "timestamp": 4}]
            ),
        ],
        ignore_index=True,
    )
    df = item_review_aggregates(train_with_i3, _SENTIMENT)
    by_item = df.set_index("parent_asin")
    assert by_item.loc["i3", "item_train_sentiment_count"] == 0
    # No reviewed sentiment -> mean defaults to 0.0, gap = mean_rating - 0
    assert by_item.loc["i3", "item_train_sentiment_mean"] == pytest.approx(0.0)
    assert by_item.loc["i3", "item_rating_minus_sentiment_gap"] == pytest.approx(2.0)

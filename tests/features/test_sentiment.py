"""Tests for sentiment protocol + fake model + train-only filtering."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.features.sentiment import (
    FakeSentimentModel,
    score_train_reviews,
)


def test_fake_model_is_deterministic_and_in_range():
    model = FakeSentimentModel()
    scores_a = model.score(["love this game", "terrible product", ""])
    scores_b = model.score(["love this game", "terrible product", ""])
    assert scores_a == scores_b
    for row in scores_a:
        assert row["sentiment_label"] in ("POSITIVE", "NEGATIVE", "NEUTRAL")
        assert -1.0 <= row["sentiment_score"] <= 1.0
        assert 0.0 <= row["positive_prob"] <= 1.0
        assert 0.0 <= row["negative_prob"] <= 1.0


def test_score_train_reviews_filters_to_train_only(tmp_path: Path):
    train = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 100},
            {"user_id": "u1", "parent_asin": "i2", "rating": 4.0, "timestamp": 200},
        ]
    )
    raw_reviews = [
        {"user_id": "u1", "parent_asin": "i1", "timestamp": 100, "text": "love it"},
        {"user_id": "u1", "parent_asin": "i2", "timestamp": 200, "text": "fine"},
        # held-out test row — must NOT be scored
        {"user_id": "u1", "parent_asin": "i3", "timestamp": 300, "text": "test row"},
    ]
    out_dir = tmp_path / "advanced_features"
    score_train_reviews(
        train=train,
        raw_reviews=iter(raw_reviews),
        model=FakeSentimentModel(),
        out_dir=out_dir,
        dataset="tiny",
        text_column="text",
    )
    scored = pd.read_parquet(out_dir / "train_sentiment.parquet")
    keys = set(zip(scored["user_id"], scored["parent_asin"], scored["timestamp"]))
    assert keys == {("u1", "i1", 100), ("u1", "i2", 200)}
    meta = json.loads((out_dir / "train_sentiment_meta.json").read_text())
    assert meta["dataset"] == "tiny"
    assert meta["row_count"] == 2
    assert meta["text_column"] == "text"
    assert meta["model_name"] == FakeSentimentModel().name


def test_empty_text_falls_back_to_neutral():
    model = FakeSentimentModel()
    [row] = model.score([""])
    assert row["sentiment_label"] == "NEUTRAL"
    assert row["sentiment_score"] == 0.0


def test_score_train_reviews_rejects_non_positive_max_rows(tmp_path: Path):
    with pytest.raises(ValueError, match="max_rows must be positive"):
        score_train_reviews(
            train=pd.DataFrame(
                [{"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 1}]
            ),
            raw_reviews=[],
            model=FakeSentimentModel(),
            out_dir=tmp_path,
            dataset="tiny",
            max_rows=0,
        )

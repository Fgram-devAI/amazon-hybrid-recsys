"""Tests for interaction-matrix preprocessing."""

import pandas as pd

from src.data.interactions import load_interactions


def test_load_interactions_drops_invalid_and_counts_raw():
    records = [
        {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 1},
        {"user_id": "u2", "parent_asin": "i2", "rating": 4.0},  # no timestamp: kept
        {"parent_asin": "i3", "rating": 3.0},                   # no user: dropped
        {"user_id": "u4", "parent_asin": "i4"},                 # no rating: dropped
    ]

    df, raw_count = load_interactions(iter(records))

    assert raw_count == 4
    assert len(df) == 2
    assert set(df["user_id"]) == {"u1", "u2"}
    assert df["rating"].tolist() == [5.0, 4.0]


from src.data.interactions import deduplicate_interactions


def test_deduplicate_keeps_latest_interaction_per_user_item():
    df = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 3.0, "timestamp": 10},
            {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 20},
            {"user_id": "u1", "parent_asin": "i2", "rating": 4.0, "timestamp": 15},
        ]
    )

    out = deduplicate_interactions(df)

    assert len(out) == 2
    kept = out[(out["user_id"] == "u1") & (out["parent_asin"] == "i1")]
    assert kept["rating"].iloc[0] == 5.0  # latest by timestamp

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


from src.data.interactions import apply_k_core


def test_apply_k_core_is_iterative_and_cascades():
    # Removing u2/u3 (1 interaction each) drops i1/i2 below k, which then drops
    # u1. A single pass would wrongly leave rows; iteration empties the set.
    df = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "i2", "rating": 5.0, "timestamp": 2},
            {"user_id": "u1", "parent_asin": "i3", "rating": 5.0, "timestamp": 3},
            {"user_id": "u2", "parent_asin": "i1", "rating": 5.0, "timestamp": 4},
            {"user_id": "u3", "parent_asin": "i2", "rating": 5.0, "timestamp": 5},
        ]
    )

    assert len(apply_k_core(df, k=2)) == 0


def test_apply_k_core_keeps_a_set_that_already_satisfies_k():
    df = pd.DataFrame(
        [
            {"user_id": "a", "parent_asin": "x", "rating": 5.0, "timestamp": 1},
            {"user_id": "a", "parent_asin": "y", "rating": 5.0, "timestamp": 2},
            {"user_id": "b", "parent_asin": "x", "rating": 5.0, "timestamp": 3},
            {"user_id": "b", "parent_asin": "y", "rating": 5.0, "timestamp": 4},
        ]
    )

    assert len(apply_k_core(df, k=2)) == 4

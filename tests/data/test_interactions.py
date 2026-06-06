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


from src.data.interactions import split_per_user


def test_split_per_user_holds_out_latest_no_leakage_reproducible():
    rows = []
    for user in ["u1", "u2"]:
        for t in range(1, 6):  # 5 interactions each, increasing timestamp
            rows.append(
                {"user_id": user, "parent_asin": f"{user}_i{t}",
                 "rating": 5.0, "timestamp": t}
            )
    df = pd.DataFrame(rows)

    train, test = split_per_user(df, test_size=0.2, random_seed=42)

    # 0.2 * 5 = 1 held out per user (the latest), 4 train each
    assert len(train) == 8 and len(test) == 2
    assert set(test["parent_asin"]) == {"u1_i5", "u2_i5"}  # chronological holdout
    # every test user appears in train
    assert set(test["user_id"]) <= set(train["user_id"])
    # no (user, item) pair leaks across train/test
    train_pairs = set(zip(train["user_id"], train["parent_asin"]))
    test_pairs = set(zip(test["user_id"], test["parent_asin"]))
    assert train_pairs.isdisjoint(test_pairs)
    # reproducible
    train2, test2 = split_per_user(df, test_size=0.2, random_seed=42)
    assert test2.equals(test)

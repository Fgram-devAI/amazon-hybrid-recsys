"""Tests for interaction-matrix preprocessing."""

import pytest
import pandas as pd

from src.data.interactions import (
    apply_k_core,
    deduplicate_interactions,
    load_interactions,
    split_leave_last_out,
    split_per_user,
)


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


def test_split_per_user_handles_empty_input():
    empty = pd.DataFrame(columns=["user_id", "parent_asin", "rating", "timestamp"])

    train, test = split_per_user(empty, test_size=0.2, random_seed=42)

    assert len(train) == 0 and len(test) == 0


def test_deduplicate_keeps_earliest_when_policy_first():
    df = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 3.0, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 9},
        ]
    )
    out = deduplicate_interactions(df, policy="first")
    assert len(out) == 1
    assert out.iloc[0]["rating"] == 3.0


def test_deduplicate_keeps_latest_by_default_unchanged():
    df = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 3.0, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 9},
        ]
    )
    out = deduplicate_interactions(df)  # default "latest"
    assert len(out) == 1
    assert out.iloc[0]["rating"] == 5.0


def test_deduplicate_invalid_policy_raises():
    df = pd.DataFrame(
        [{"user_id": "u1", "parent_asin": "i1", "rating": 3.0, "timestamp": 1}]
    )
    with pytest.raises(ValueError):
        deduplicate_interactions(df, policy="middle")


def _user_history(user, n):
    return pd.DataFrame(
        [
            {"user_id": user, "parent_asin": f"i{n}_{i}", "rating": 5.0, "timestamp": i}
            for i in range(n)
        ]
    )


def test_split_leave_last_out_five_interactions_gives_3_1_1():
    df = _user_history("u1", 5)
    train, val, test = split_leave_last_out(df)
    assert list(train["parent_asin"]) == ["i5_0", "i5_1", "i5_2"]
    assert list(val["parent_asin"]) == ["i5_3"]
    assert list(test["parent_asin"]) == ["i5_4"]


def test_split_leave_last_out_two_interactions_gives_train_one_test_one_no_val():
    df = _user_history("u1", 2)
    train, val, test = split_leave_last_out(df)
    assert list(train["parent_asin"]) == ["i2_0"]
    assert val.empty
    assert list(test["parent_asin"]) == ["i2_1"]


def test_split_leave_last_out_one_interaction_goes_to_train_only():
    df = _user_history("u1", 1)
    train, val, test = split_leave_last_out(df)
    assert len(train) == 1
    assert val.empty
    assert test.empty


def test_split_leave_last_out_multiple_users_independent():
    df = pd.concat([_user_history("u1", 5), _user_history("u2", 3)], ignore_index=True)
    train, val, test = split_leave_last_out(df)
    # u1: train=3, val=1, test=1; u2: train=1, val=1, test=1
    assert (train["user_id"] == "u1").sum() == 3
    assert (val["user_id"] == "u1").sum() == 1
    assert (test["user_id"] == "u1").sum() == 1
    assert (train["user_id"] == "u2").sum() == 1
    assert (val["user_id"] == "u2").sum() == 1
    assert (test["user_id"] == "u2").sum() == 1

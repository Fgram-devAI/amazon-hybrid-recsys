"""Tests for evaluation metrics."""

import math

import pandas as pd
import pytest

from src.evaluation.metrics import (
    hit_rate_at_k,
    mae,
    ndcg_at_k,
    precision_recall_f1_at_k,
    relevant_items_by_user,
    rmse,
)


def test_rmse_and_mae_match_hand_computed():
    y_true = [5.0, 3.0, 4.0]
    y_pred = [4.0, 3.0, 2.0]
    # errors: 1, 0, 2 -> RMSE = sqrt((1+0+4)/3) = sqrt(5/3); MAE = 3/3 = 1
    assert rmse(y_true, y_pred) == pytest.approx((5 / 3) ** 0.5)
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_precision_recall_f1_at_k_hand_computed():
    recommended = ["a", "b", "c", "d"]
    relevant = {"b", "d", "x"}
    # top-2 = [a, b]; hits = 1 -> P = 1/2, R = 1/3, F1 = 0.4
    result = precision_recall_f1_at_k(recommended, relevant, k=2)
    assert result is not None
    p, r, f = result
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(1 / 3)
    assert f == pytest.approx(0.4)


def test_no_relevant_items_returns_none_not_nan():
    assert precision_recall_f1_at_k(["a", "b"], set(), k=2) is None


def test_zero_hits_returns_zero_f1_not_nan():
    # common in real sampled ranking: none of the top-K are relevant
    result = precision_recall_f1_at_k(["a", "b"], {"x", "y"}, k=2)
    assert result == (0.0, 0.0, 0.0)


def test_relevant_items_by_user_thresholds_on_rating():
    test_df = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 5.0},
            {"user_id": "u1", "parent_asin": "i2", "rating": 3.0},  # below threshold
            {"user_id": "u2", "parent_asin": "i3", "rating": 4.0},
        ]
    )
    rel = relevant_items_by_user(test_df, min_rating_relevant=4.0)
    assert rel == {"u1": {"i1"}, "u2": {"i3"}}


# ---------------------------------------------------------------------------
# hit_rate_at_k / ndcg_at_k
# ---------------------------------------------------------------------------


def test_hit_rate_at_k_one_when_any_top_k_hit():
    assert hit_rate_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], {"a"}, k=1) == 1.0


def test_hit_rate_at_k_zero_when_no_top_k_hit():
    assert hit_rate_at_k(["a", "b", "c"], {"x", "y"}, k=3) == 0.0


def test_hit_rate_at_k_none_when_no_relevant_items():
    assert hit_rate_at_k(["a", "b"], set(), k=2) is None


def test_ndcg_at_k_perfect_ranking_is_one():
    # all relevant items at top -> dcg == ideal_dcg
    assert ndcg_at_k(["a", "b", "c", "d"], {"a", "b"}, k=4) == pytest.approx(1.0)


def test_ndcg_at_k_handcomputed_one_hit_at_rank_2():
    # relevant set = {b}; recommended=[a,b,c,d], K=4
    # dcg = 1 / log2(2 + 1) = 1 / log2(3)
    # ideal_dcg = 1 / log2(1 + 1) = 1 (one relevant item, ideal rank 1)
    # ndcg = 1 / log2(3)
    expected = 1.0 / math.log2(3)
    assert ndcg_at_k(["a", "b", "c", "d"], {"b"}, k=4) == pytest.approx(expected)


def test_ndcg_at_k_no_hits_returns_zero():
    assert ndcg_at_k(["a", "b"], {"x", "y"}, k=2) == 0.0


def test_ndcg_at_k_none_when_no_relevant_items():
    assert ndcg_at_k(["a", "b"], set(), k=2) is None

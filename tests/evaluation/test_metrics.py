"""Tests for evaluation metrics."""

import math

import pandas as pd
import pytest

from src.evaluation.metrics import (
    aggregate_metric_bundle,
    compute_user_metric_bundle,
    hit_rate_at_k,
    mae,
    ndcg_at_k,
    oracle_hit_rate_at_k,
    oracle_ndcg_at_k,
    oracle_precision_recall_f1_at_k,
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


# ---------------------------------------------------------------------------
# oracle ceiling functions
# ---------------------------------------------------------------------------


def test_oracle_pr_f1_single_relevant_item_k10():
    # 1 relevant, K=10: oracle_hits = 1; P = 1/10, R = 1/1, F1 = 2*0.1/(1.1)
    result = oracle_precision_recall_f1_at_k(relevant_count=1, k=10)
    assert result is not None
    p, r, f = result
    assert p == pytest.approx(0.1)
    assert r == pytest.approx(1.0)
    assert f == pytest.approx(2 * 0.1 * 1.0 / (0.1 + 1.0))


def test_oracle_pr_f1_more_relevant_than_k():
    # 15 relevant, K=10: oracle_hits = 10; P = 1.0, R = 10/15, F1 from harmonic mean
    result = oracle_precision_recall_f1_at_k(relevant_count=15, k=10)
    assert result is not None
    p, r, f = result
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(10 / 15)
    assert f == pytest.approx(2 * 1.0 * (10 / 15) / (1.0 + 10 / 15))


def test_oracle_pr_f1_returns_none_when_no_relevant():
    assert oracle_precision_recall_f1_at_k(relevant_count=0, k=10) is None


def test_oracle_hit_rate_sentinel_one_when_relevant_exists():
    assert oracle_hit_rate_at_k(relevant_count=1) == 1.0
    assert oracle_hit_rate_at_k(relevant_count=42) == 1.0


def test_oracle_hit_rate_none_when_no_relevant():
    assert oracle_hit_rate_at_k(relevant_count=0) is None


def test_oracle_ndcg_sentinel_one_when_relevant_exists():
    assert oracle_ndcg_at_k(relevant_count=1) == 1.0


def test_oracle_ndcg_none_when_no_relevant():
    assert oracle_ndcg_at_k(relevant_count=0) is None


# ---------------------------------------------------------------------------
# compute_user_metric_bundle
# ---------------------------------------------------------------------------


def test_compute_user_metric_bundle_one_hit_at_rank_2():
    # recommended=[a,b,c,d], relevant={b}, K=4
    # P = 1/4, R = 1/1 = 1.0, F1 = 2*0.25*1/(1.25) = 0.4
    # HR = 1.0
    # NDCG = (1/log2(3)) / 1.0
    # oracle: relevant_count=1 -> oracle_P=1/4, oracle_R=1, oracle_F1 = 2*0.25*1/1.25 = 0.4
    bundle = compute_user_metric_bundle(["a", "b", "c", "d"], {"b"}, k=4)
    assert bundle is not None
    assert bundle["precision_at_k"] == pytest.approx(0.25)
    assert bundle["recall_at_k"] == pytest.approx(1.0)
    assert bundle["f1_at_k"] == pytest.approx(0.4)
    assert bundle["hit_rate_at_k"] == 1.0
    assert bundle["ndcg_at_k"] == pytest.approx(1.0 / math.log2(3))
    assert bundle["oracle_precision_at_k"] == pytest.approx(0.25)
    assert bundle["oracle_recall_at_k"] == pytest.approx(1.0)
    assert bundle["oracle_f1_at_k"] == pytest.approx(0.4)
    assert bundle["oracle_hit_rate_at_k"] == 1.0
    assert bundle["oracle_ndcg_at_k"] == 1.0


def test_compute_user_metric_bundle_returns_none_when_no_relevant():
    assert compute_user_metric_bundle(["a", "b"], set(), k=2) is None


def test_compute_user_metric_bundle_zero_hits_records_zeros():
    bundle = compute_user_metric_bundle(["a", "b"], {"x", "y"}, k=2)
    assert bundle is not None
    assert bundle["precision_at_k"] == 0.0
    assert bundle["recall_at_k"] == 0.0
    assert bundle["f1_at_k"] == 0.0
    assert bundle["hit_rate_at_k"] == 0.0
    assert bundle["ndcg_at_k"] == 0.0
    # oracle is still the ceiling for this user (2 relevant items, K=2 -> P=1, R=1)
    assert bundle["oracle_precision_at_k"] == pytest.approx(1.0)
    assert bundle["oracle_recall_at_k"] == pytest.approx(1.0)
    assert bundle["oracle_f1_at_k"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# aggregate_metric_bundle
# ---------------------------------------------------------------------------


def test_aggregate_metric_bundle_means_and_oracle_ratios():
    # Two users; one perfect at rank 1, one half-hit at rank 2
    # User A: recommended=[a,b], relevant={a}, K=2
    #   P=0.5, R=1.0, F1=2/3, HR=1, NDCG=1, oracle P=0.5/R=1/F1=2/3/HR=1/NDCG=1
    # User B: recommended=[a,b], relevant={b}, K=2
    #   P=0.5, R=1.0, F1=2/3, HR=1, NDCG=1/log2(3), oracle same as A
    bundle_a = compute_user_metric_bundle(["a", "b"], {"a"}, k=2)
    bundle_b = compute_user_metric_bundle(["a", "b"], {"b"}, k=2)
    assert bundle_a is not None and bundle_b is not None

    agg = aggregate_metric_bundle([bundle_a, bundle_b], k=2)

    assert agg["n_eval_users"] == 2
    assert agg["precision_at_k"] == pytest.approx(0.5)
    assert agg["recall_at_k"] == pytest.approx(1.0)
    assert agg["f1_at_k"] == pytest.approx(2 / 3)
    assert agg["hit_rate_at_k"] == pytest.approx(1.0)
    # NDCG mean = (1 + 1/log2(3)) / 2
    assert agg["ndcg_at_k"] == pytest.approx((1.0 + 1.0 / math.log2(3)) / 2)
    # oracle means equal each user's ceiling
    assert agg["oracle_precision_at_k"] == pytest.approx(0.5)
    assert agg["oracle_recall_at_k"] == pytest.approx(1.0)
    assert agg["oracle_f1_at_k"] == pytest.approx(2 / 3)
    assert agg["oracle_hit_rate_at_k"] == 1.0
    assert agg["oracle_ndcg_at_k"] == 1.0
    # ratios: mean(P)/mean(oracle_P), etc. — all equal 1.0 here
    assert agg["precision_oracle_ratio_at_k"] == pytest.approx(1.0)
    assert agg["recall_oracle_ratio_at_k"] == pytest.approx(1.0)
    assert agg["f1_oracle_ratio_at_k"] == pytest.approx(1.0)


def test_aggregate_metric_bundle_empty_input_returns_zero_users():
    agg = aggregate_metric_bundle([], k=10)
    assert agg["n_eval_users"] == 0
    assert agg["precision_at_k"] is None
    assert agg["recall_at_k"] is None
    assert agg["f1_at_k"] is None
    assert agg["hit_rate_at_k"] is None
    assert agg["ndcg_at_k"] is None
    assert agg["oracle_precision_at_k"] is None
    assert agg["precision_oracle_ratio_at_k"] is None


def test_aggregate_metric_bundle_oracle_ratio_uses_mean_over_mean():
    # Build a bundle where per-user ratios differ from mean/mean,
    # so we can show the implementation chose mean(P)/mean(oP), not mean(P/oP).
    # User A: P=0.10, oracle_P=0.10 -> per-user ratio = 1.0
    # User B: P=0.00, oracle_P=1.00 -> per-user ratio = 0.0
    # mean(P)=0.05, mean(oP)=0.55 -> mean/mean = 0.0909... ;  mean per-user ratio = 0.5
    bundle_a = {
        "precision_at_k": 0.10, "recall_at_k": 0.10, "f1_at_k": 0.10,
        "hit_rate_at_k": 1.0, "ndcg_at_k": 1.0,
        "oracle_precision_at_k": 0.10, "oracle_recall_at_k": 0.10,
        "oracle_f1_at_k": 0.10, "oracle_hit_rate_at_k": 1.0, "oracle_ndcg_at_k": 1.0,
    }
    bundle_b = {
        "precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0,
        "hit_rate_at_k": 0.0, "ndcg_at_k": 0.0,
        "oracle_precision_at_k": 1.0, "oracle_recall_at_k": 1.0,
        "oracle_f1_at_k": 1.0, "oracle_hit_rate_at_k": 1.0, "oracle_ndcg_at_k": 1.0,
    }
    agg = aggregate_metric_bundle([bundle_a, bundle_b], k=10)
    assert agg["precision_oracle_ratio_at_k"] == pytest.approx(0.05 / 0.55)
    assert agg["precision_oracle_ratio_at_k"] != pytest.approx(0.5)

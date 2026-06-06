"""Tests for evaluation metrics."""

import pandas as pd
import pytest

from src.evaluation.metrics import (
    mae,
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

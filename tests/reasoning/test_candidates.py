"""Unit tests for score normalization and weighted hybrid fusion."""
from __future__ import annotations

import math

import pytest

from src.reasoning.candidates import (
    min_max_normalize,
    weighted_fuse,
    redistribute_weights,
)


def test_min_max_normalize_basic():
    out = min_max_normalize([1.0, 3.0, 5.0])
    assert out == [0.0, 0.5, 1.0]


def test_min_max_normalize_constant_returns_zeros():
    out = min_max_normalize([2.0, 2.0, 2.0])
    assert out == [0.0, 0.0, 0.0]


def test_min_max_normalize_handles_none():
    out = min_max_normalize([1.0, None, 5.0])
    assert out[0] == 0.0
    assert out[1] is None
    assert out[2] == 1.0


def test_min_max_normalize_empty():
    assert min_max_normalize([]) == []


def test_redistribute_weights_drops_missing_and_renormalizes():
    weights = {"graph": 0.5, "svd": 0.25, "semantic": 0.15, "popularity": 0.10}
    out = redistribute_weights(weights, available={"graph", "popularity"})
    assert pytest.approx(out["graph"], rel=1e-6) == 0.5 / 0.6
    assert pytest.approx(out["popularity"], rel=1e-6) == 0.10 / 0.6
    assert "svd" not in out
    assert "semantic" not in out


def test_redistribute_weights_all_missing_raises():
    with pytest.raises(ValueError):
        redistribute_weights({"graph": 1.0}, available=set())


def test_weighted_fuse_combines_sources():
    rows = [
        {"parent_asin": "A", "graph": 1.0, "svd": 0.5, "popularity": 0.2},
        {"parent_asin": "B", "graph": 0.0, "svd": 1.0, "popularity": 0.8},
    ]
    weights = {"graph": 0.5, "svd": 0.3, "popularity": 0.2}
    fused = weighted_fuse(rows, weights=weights)
    a = next(r for r in fused if r["parent_asin"] == "A")
    expected = 0.5 * 1.0 + 0.3 * 0.5 + 0.2 * 0.2
    assert math.isclose(a["hybrid_score"], expected, rel_tol=1e-6)
    assert set(a["score_sources"]) == {"graph", "svd", "popularity"}


def test_weighted_fuse_skips_missing_per_row():
    rows = [
        {"parent_asin": "A", "graph": 1.0, "svd": None},
        {"parent_asin": "B", "graph": None, "svd": 0.4},
    ]
    weights = {"graph": 0.6, "svd": 0.4}
    fused = weighted_fuse(rows, weights=weights)
    a = next(r for r in fused if r["parent_asin"] == "A")
    b = next(r for r in fused if r["parent_asin"] == "B")
    # A has only graph (1.0): hybrid normalized over available weight only
    assert math.isclose(a["hybrid_score"], 1.0, rel_tol=1e-6)
    assert a["score_sources"] == ["graph"]
    # B has only svd (0.4)
    assert math.isclose(b["hybrid_score"], 0.4, rel_tol=1e-6)
    assert b["score_sources"] == ["svd"]

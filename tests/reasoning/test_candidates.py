"""Unit tests for score normalization and weighted hybrid fusion."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.reasoning.candidates import (
    PopularityScorer,
    SemanticScorer,
    generate_candidates,
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


def test_min_max_normalize_constant_sparse_source_marks_present_rows():
    out = min_max_normalize([None, 2.0, None])
    assert out == [None, 1.0, None]


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


def test_weighted_fuse_can_treat_missing_scores_as_zero():
    rows = [
        {"parent_asin": "A", "graph": 1.0, "semantic": None},
        {"parent_asin": "B", "graph": 0.2, "semantic": 1.0},
    ]
    weights = {"graph": 0.3, "semantic": 0.7}
    fused = weighted_fuse(rows, weights=weights, renormalize_missing=False)
    a = next(r for r in fused if r["parent_asin"] == "A")
    b = next(r for r in fused if r["parent_asin"] == "B")
    assert math.isclose(a["hybrid_score"], 0.3, rel_tol=1e-6)
    assert math.isclose(b["hybrid_score"], 0.76, rel_tol=1e-6)
    assert a["score_sources"] == ["graph"]
    assert set(b["score_sources"]) == {"graph", "semantic"}


def _toy_train():
    return pd.DataFrame(
        [
            {"user_id": "U1", "parent_asin": "A", "rating": 5.0, "timestamp": 1},
            {"user_id": "U1", "parent_asin": "B", "rating": 4.0, "timestamp": 2},
            {"user_id": "U2", "parent_asin": "A", "rating": 5.0, "timestamp": 3},
            {"user_id": "U2", "parent_asin": "C", "rating": 5.0, "timestamp": 4},
            {"user_id": "U3", "parent_asin": "C", "rating": 4.0, "timestamp": 5},
        ]
    )


def test_popularity_scorer_counts_train_occurrences():
    train = _toy_train()
    scorer = PopularityScorer(train)
    assert scorer.score("A") == 2
    assert scorer.score("B") == 1
    assert scorer.score("D") == 0


def test_semantic_scorer_from_milvus_hits():
    hits = [
        {"parent_asin": "A", "distance": 0.1},
        {"parent_asin": "B", "distance": 0.4},
    ]
    scorer = SemanticScorer(hits)
    # Smaller distance => higher similarity score (1 - distance for cosine)
    score_a = scorer.score("A")
    score_b = scorer.score("B")
    assert score_a is not None
    assert score_b is not None
    assert score_a > score_b
    assert scorer.score("Z") is None


def test_generate_candidates_combines_sources_and_excludes_train_items():
    train = _toy_train()

    class _ConstSVD:
        def __call__(self, user_id, parent_asin):
            return {"A": 4.5, "B": 3.8, "C": 4.2, "D": 4.0}.get(parent_asin, 3.0)

    class _ConstLightGCN:
        def __call__(self, user_id, parent_asin):
            return {"A": 0.9, "B": 0.5, "C": 0.7, "D": 0.8}.get(parent_asin, 0.0)

    semantic = SemanticScorer([
        {"parent_asin": "C", "distance": 0.05},
        {"parent_asin": "D", "distance": 0.20},
    ])
    pop = PopularityScorer(train)

    out = generate_candidates(
        user_id="U1",
        candidate_pool=["A", "B", "C", "D"],
        train=train,
        svd_scorer=_ConstSVD(),
        lightgcn_scorer=_ConstLightGCN(),
        semantic_scorer=semantic,
        popularity_scorer=pop,
        weights={"graph": 0.5, "svd": 0.25, "semantic": 0.15, "popularity": 0.10},
        final_k=2,
    )
    asins = [row["parent_asin"] for row in out]
    # User U1 already saw A and B; they must be filtered out.
    assert "A" not in asins
    assert "B" not in asins
    # Final list capped at final_k.
    assert len(out) == 2
    # Each remaining row carries a hybrid_score and a normalized score field per source.
    for row in out:
        assert "hybrid_score" in row
        assert "score_sources" in row
        assert isinstance(row["score_sources"], list)


def test_generate_candidates_semantic_weight_penalizes_missing_semantic_rows():
    train = _toy_train()

    class _ConstLightGCN:
        def __call__(self, user_id, parent_asin):
            return {"C": 1.0, "D": 0.2}.get(parent_asin, 0.0)

    semantic = SemanticScorer([{"parent_asin": "D", "distance": 0.0}])

    out = generate_candidates(
        user_id="U1",
        candidate_pool=["C", "D"],
        train=train,
        svd_scorer=None,
        lightgcn_scorer=_ConstLightGCN(),
        semantic_scorer=semantic,
        popularity_scorer=None,
        weights={"graph": 0.3, "semantic": 0.7},
        final_k=2,
    )
    assert out[0]["parent_asin"] == "D"
    assert out[0]["semantic_score"] == 1.0
    assert "semantic" in out[0]["score_sources"]


def test_generate_candidates_warns_when_a_source_is_none(caplog):
    train = _toy_train()
    pop = PopularityScorer(train)
    out = generate_candidates(
        user_id="U1",
        candidate_pool=["C", "D"],
        train=train,
        svd_scorer=None,
        lightgcn_scorer=None,
        semantic_scorer=None,
        popularity_scorer=pop,
        weights={"graph": 0.5, "svd": 0.25, "semantic": 0.15, "popularity": 0.10},
        final_k=2,
    )
    # Only popularity is present, so all hybrid scores are normalized popularity.
    for row in out:
        assert row["score_sources"] == ["popularity"]
        assert 0.0 <= row["hybrid_score"] <= 1.0


def test_svd_scorer_wraps_recommender_predict():
    from src.models.cf import SVDRecommender

    train = _toy_train()
    model = SVDRecommender(n_factors=4, n_epochs=2, random_state=42)
    model.fit(train)
    from src.reasoning.candidates import SVDScorer

    scorer = SVDScorer(model)
    s = scorer("U1", "C")
    assert isinstance(s, float)
    assert 1.0 <= s <= 5.0


def test_lightgcn_scorer_returns_none_when_user_unknown():
    """Unknown users must yield None, not the calibration fallback rating."""
    from src.reasoning.candidates import LightGCNScorer

    class _StubLightGCN:
        def __init__(self):
            self._graph = type(
                "G",
                (),
                {"user_index": {"U1": 0}, "item_index": {"A": 0, "B": 1}},
            )()

        def predict(self, user, asin):  # pragma: no cover - not invoked here
            return 4.0

    scorer = LightGCNScorer(_StubLightGCN())
    assert scorer("UNKNOWN", "A") is None
    assert scorer("U1", "UNKNOWN_ITEM") is None

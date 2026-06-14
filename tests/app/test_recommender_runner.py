"""Tests for the app-facing recommender runner.

All tests use tiny in-memory DataFrames; nothing touches Milvus / Neo4j / Groq.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from app.recommender_runner import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    EMPTY_ROW_FIELDS,
    KNN_SIM_NAMES,
    LocalArtifacts,
    _metadata_lookup,
    _normalize_categories,
    build_candidate_pool,
    load_local_artifacts,
    representative_users,
    run_knn,
    run_popularity,
    run_svd,
    seen_items_for_user,
    user_history_rows,
)


def test_normalize_categories_handles_list() -> None:
    assert _normalize_categories(["RPG", "Indie"]) == ["RPG", "Indie"]


def test_normalize_categories_handles_pipe_string() -> None:
    assert _normalize_categories("RPG|Indie|Adventure") == ["RPG", "Indie", "Adventure"]


def test_normalize_categories_handles_comma_string() -> None:
    assert _normalize_categories("RPG, Indie ,Adventure") == ["RPG", "Indie", "Adventure"]


def test_normalize_categories_handles_plain_single_string() -> None:
    assert _normalize_categories("Indie") == ["Indie"]


def test_normalize_categories_handles_none_empty_and_nan() -> None:
    assert _normalize_categories(None) == []
    assert _normalize_categories(math.nan) == []
    assert _normalize_categories("") == []


def test_metadata_lookup_dedupes_parent_asin_first_wins() -> None:
    df = pd.DataFrame(
        [
            {"parent_asin": "A", "title": "First A", "categories": "RPG"},
            {"parent_asin": "A", "title": "Dup A", "categories": "Strategy"},
            {"parent_asin": "B", "title": "B", "categories": ["Indie", "Casual"]},
        ]
    )
    lookup = _metadata_lookup(df)
    assert lookup["A"]["title"] == "First A"
    assert lookup["A"]["categories"] == ["RPG"]
    assert lookup["B"]["categories"] == ["Indie", "Casual"]


def _write_train(parquet_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "A", "rating": 5.0, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "B", "rating": 4.0, "timestamp": 2},
            {"user_id": "u2", "parent_asin": "A", "rating": 4.0, "timestamp": 3},
            {"user_id": "u2", "parent_asin": "C", "rating": 5.0, "timestamp": 4},
            {"user_id": "u3", "parent_asin": "B", "rating": 5.0, "timestamp": 5},
            {"user_id": "u3", "parent_asin": "D", "rating": 5.0, "timestamp": 6},
        ]
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path)


def _write_metadata(parquet_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"parent_asin": "A", "title": "Game A", "categories": "RPG"},
            {"parent_asin": "B", "title": "Game B", "categories": "Strategy"},
            {"parent_asin": "C", "title": "Game C", "categories": "RPG|Indie"},
            {"parent_asin": "D", "title": "Game D", "categories": "Indie"},
        ]
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path)


def test_load_local_artifacts_returns_dataframes(tmp_path: Path) -> None:
    _write_train(tmp_path / "video_games" / "train.parquet")
    _write_metadata(tmp_path / "video_games" / "metadata.parquet")
    artifacts = load_local_artifacts(processed_dir=tmp_path, dataset="video_games")

    assert isinstance(artifacts, LocalArtifacts)
    assert artifacts.available is True
    assert artifacts.missing_reasons == []
    assert len(artifacts.train) == 6
    assert len(artifacts.metadata) == 4


def test_load_local_artifacts_marks_missing_files(tmp_path: Path) -> None:
    artifacts = load_local_artifacts(processed_dir=tmp_path, dataset="video_games")

    assert artifacts.available is False
    assert any("train.parquet" in r for r in artifacts.missing_reasons)
    assert any("metadata.parquet" in r for r in artifacts.missing_reasons)
    assert artifacts.train.empty
    assert artifacts.metadata.empty


def test_load_local_artifacts_handles_only_metadata_present(tmp_path: Path) -> None:
    _write_metadata(tmp_path / "video_games" / "metadata.parquet")
    artifacts = load_local_artifacts(processed_dir=tmp_path, dataset="video_games")

    assert artifacts.available is False
    assert any("train.parquet" in r for r in artifacts.missing_reasons)
    assert artifacts.metadata.shape[0] == 4


def _toy_train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "A", "rating": 5.0, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "B", "rating": 4.0, "timestamp": 2},
            {"user_id": "u2", "parent_asin": "A", "rating": 4.0, "timestamp": 3},
            {"user_id": "u2", "parent_asin": "C", "rating": 5.0, "timestamp": 4},
            {"user_id": "u3", "parent_asin": "B", "rating": 5.0, "timestamp": 5},
            {"user_id": "u3", "parent_asin": "D", "rating": 5.0, "timestamp": 6},
        ]
    )


def _toy_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"parent_asin": "A", "title": "Game A", "categories": "RPG"},
            {"parent_asin": "B", "title": "Game B", "categories": "Strategy"},
            {"parent_asin": "C", "title": "Game C", "categories": "RPG|Indie"},
            {"parent_asin": "D", "title": "Game D", "categories": "Indie"},
        ]
    )


def test_representative_users_returns_stable_top_n_by_interaction_count() -> None:
    users = representative_users(_toy_train(), limit=2)
    # All three users have 2 interactions; deterministic tie-break by user_id.
    assert users == ["u1", "u2"]


def test_representative_users_respects_min_interactions_filter() -> None:
    train = _toy_train().drop(index=[5])  # u3 now has 1 interaction
    users = representative_users(train, limit=5, min_interactions=2)
    assert users == ["u1", "u2"]


def test_seen_items_for_user_includes_train_items_only() -> None:
    seen = seen_items_for_user(_toy_train(), user_id="u1")
    assert seen == {"A", "B"}


def test_seen_items_for_user_handles_unknown_user() -> None:
    seen = seen_items_for_user(_toy_train(), user_id="ghost")
    assert seen == set()


def test_user_history_rows_joins_metadata_with_parsed_categories() -> None:
    rows = user_history_rows(
        _toy_train(),
        _toy_metadata(),
        user_id="u1",
        min_rating=4.0,
        limit=5,
    )
    # newest-first by timestamp: u1 has B@ts=2 then A@ts=1
    assert [r["parent_asin"] for r in rows] == ["B", "A"]
    a_row = next(r for r in rows if r["parent_asin"] == "A")
    assert a_row["title"] == "Game A"
    assert a_row["categories"] == ["RPG"]
    assert a_row["rating"] == 5.0


def test_default_candidate_pool_size_matches_spec() -> None:
    assert DEFAULT_CANDIDATE_POOL_SIZE == 500


def test_build_candidate_pool_excludes_seen_and_caps_size() -> None:
    train = _toy_train()
    seen = {"A", "B"}
    pool = build_candidate_pool(
        train=train,
        metadata=_toy_metadata(),
        seen=seen,
        pool_size=10,
        seed_asin=None,
    )
    assert "A" not in pool and "B" not in pool
    assert len(pool) <= 10


def test_build_candidate_pool_promotes_same_category_items_when_seed_given() -> None:
    train = _toy_train()
    metadata = pd.DataFrame(
        [
            {"parent_asin": "A", "title": "A", "categories": "RPG"},
            {"parent_asin": "B", "title": "B", "categories": "Strategy"},
            {"parent_asin": "C", "title": "C", "categories": "RPG|Indie"},
            {"parent_asin": "D", "title": "D", "categories": "Indie"},
        ]
    )
    pool = build_candidate_pool(
        train=train,
        metadata=metadata,
        seen={"A"},
        pool_size=10,
        seed_asin="C",
    )
    # The seed itself must be excluded; same-category items should appear before
    # unrelated items.
    assert "C" not in pool
    assert pool.index("D") < pool.index("B")  # D shares Indie with seed C


def test_build_candidate_pool_handles_unknown_seed_gracefully() -> None:
    pool = build_candidate_pool(
        train=_toy_train(),
        metadata=_toy_metadata(),
        seen=set(),
        pool_size=5,
        seed_asin="ZZZ",
    )
    assert isinstance(pool, list)
    assert len(pool) > 0


def test_empty_row_fields_match_spec_contract() -> None:
    required = {
        "rank", "parent_asin", "title", "categories", "method",
        "hybrid_score", "svd_score", "knn_score", "lightgcn_score",
        "semantic_score", "popularity_score", "score_sources", "already_seen",
    }
    assert required == set(EMPTY_ROW_FIELDS)


def test_run_popularity_returns_normalized_rows() -> None:
    rows = run_popularity(
        train=_toy_train(),
        metadata=_toy_metadata(),
        user_id="u1",
        top_k=3,
    )
    asins = [r["parent_asin"] for r in rows]
    assert "A" not in asins and "B" not in asins
    assert set(EMPTY_ROW_FIELDS).issubset(rows[0].keys())
    assert rows[0]["method"] == "popularity"
    assert rows[0]["already_seen"] is False
    assert rows[0]["popularity_score"] is not None
    assert rows[0]["score_sources"] == ["popularity"]
    assert rows[0]["svd_score"] is None
    assert rows[0]["lightgcn_score"] is None
    assert rows[0]["title"]


def test_run_popularity_returns_empty_when_no_unseen_items() -> None:
    train = _toy_train()
    rows = run_popularity(
        train=train[train["user_id"] == "u1"],
        metadata=_toy_metadata(),
        user_id="u1",
        top_k=5,
    )
    assert rows == []


class _FakeSVD:
    """Deterministic predict: rating depends only on parent_asin tail char."""

    def fit(self, train):
        self._fit_called = True
        return self

    def predict(self, user_id, parent_asin):
        return float(ord(str(parent_asin)[-1]) % 5) + 1.0


def test_run_svd_uses_injected_factory_and_excludes_seen() -> None:
    rows = run_svd(
        train=_toy_train(),
        metadata=_toy_metadata(),
        user_id="u1",
        top_k=3,
        candidate_pool_size=10,
        svd_factory=_FakeSVD,
    )
    asins = [r["parent_asin"] for r in rows]
    assert "A" not in asins and "B" not in asins
    assert rows[0]["method"] == "svd"
    assert rows[0]["svd_score"] is not None
    assert rows[0]["popularity_score"] is None
    assert rows[0]["score_sources"] == ["svd"]


class _FakeKNN:
    def __init__(self, sim_name: str) -> None:
        self.sim_name = sim_name

    def fit(self, train):
        return self

    def predict(self, user_id, parent_asin):
        # Deterministic per-asin score.
        return float(ord(str(parent_asin)[-1]) % 5) + 1.0


def test_knn_sim_names_match_spec() -> None:
    assert KNN_SIM_NAMES == ("cosine", "pearson", "msd")


@pytest.mark.parametrize("sim_name", ["cosine", "pearson", "msd"])
def test_run_knn_returns_rows_for_each_sim_name(sim_name: str) -> None:
    rows = run_knn(
        train=_toy_train(),
        metadata=_toy_metadata(),
        user_id="u1",
        top_k=2,
        sim_name=sim_name,
        candidate_pool_size=10,
        knn_factory=_FakeKNN,
    )
    asins = [r["parent_asin"] for r in rows]
    assert "A" not in asins and "B" not in asins
    assert rows[0]["method"] == f"knn_{sim_name}"
    assert rows[0]["knn_score"] is not None
    assert rows[0]["score_sources"] == ["knn"]


def test_run_knn_rejects_unknown_sim_name() -> None:
    with pytest.raises(ValueError):
        run_knn(
            train=_toy_train(),
            metadata=_toy_metadata(),
            user_id="u1",
            top_k=2,
            sim_name="bogus",
            knn_factory=_FakeKNN,
        )

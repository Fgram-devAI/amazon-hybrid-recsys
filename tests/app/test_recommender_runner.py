"""Tests for the app-facing recommender runner.

All tests use tiny in-memory DataFrames; nothing touches Milvus / Neo4j / Groq.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from app.recommender_runner import (
    LocalArtifacts,
    _metadata_lookup,
    _normalize_categories,
    load_local_artifacts,
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

"""Tests for the app-facing recommender runner.

All tests use tiny in-memory DataFrames; nothing touches Milvus / Neo4j / Groq.
"""
from __future__ import annotations

import math

import pandas as pd

from app.recommender_runner import (
    _metadata_lookup,
    _normalize_categories,
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

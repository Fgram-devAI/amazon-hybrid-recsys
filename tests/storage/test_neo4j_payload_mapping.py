"""Tests for src.storage.neo4j_payloads — train-only leakage rule + filtering."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.storage.neo4j_payloads import (
    build_co_rating_rows,
    build_item_category_rows,
    build_item_rows,
    build_rating_rows,
)


def test_build_item_rows_emits_one_row_per_unique_parent_asin():
    metadata = pd.DataFrame(
        {
            "parent_asin": ["a", "b", "a"],
            "title": ["t_a1", "t_b", "t_a2"],
            "store": ["s", None, "s"],
            "price": [1.0, None, 1.0],
            "average_rating": [4.0, None, 4.0],
            "rating_number": [10, None, 10],
        }
    )
    rows = build_item_rows(metadata)
    asins = sorted(r["parent_asin"] for r in rows)
    assert asins == ["a", "b"]
    by_asin = {r["parent_asin"]: r for r in rows}
    assert by_asin["b"]["store"] is None
    assert by_asin["b"]["price"] is None


def test_build_item_category_rows_drops_generic_roots():
    metadata = pd.DataFrame(
        {
            "parent_asin": ["a", "b"],
            "categories": [["Video Games", "Nintendo Switch"], ["Video Games"]],
        }
    )
    rows = build_item_category_rows(metadata, generic_roots=["Video Games"])
    pairs = sorted((r["parent_asin"], r["category"]) for r in rows)
    assert pairs == [("a", "Nintendo Switch")]


def test_build_rating_rows_uses_train_only_and_stamps_split_train():
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "parent_asin": ["a", "b", "a"],
            "rating": [5.0, 4.0, 3.0],
            "timestamp": [1, 2, 3],
        }
    )
    rows = build_rating_rows(train)
    assert len(rows) == 3
    assert all(r["split"] == "train" for r in rows)
    assert {r["user_id"] for r in rows} == {"u1", "u2"}


def test_build_co_rating_rows_reads_report_and_caps_max_edges(tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "edges": [
                    {"a": "x", "b": "y", "weight_count": 10, "weight_jaccard": 0.5},
                    {"a": "y", "b": "z", "weight_count": 4, "weight_jaccard": 0.3},
                    {"a": "x", "b": "z", "weight_count": 2, "weight_jaccard": 0.1},
                ]
            }
        )
    )
    rows = build_co_rating_rows(report, max_edges=2)
    assert len(rows) == 2
    assert rows[0]["weight_count"] >= rows[1]["weight_count"]


def test_build_co_rating_rows_returns_empty_when_report_missing(tmp_path: Path):
    assert build_co_rating_rows(tmp_path / "missing.json", max_edges=10) == []


def test_build_rating_rows_optional_limit_truncates():
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "parent_asin": ["a", "b", "c"],
            "rating": [5.0, 4.0, 3.0],
            "timestamp": [1, 2, 3],
        }
    )
    rows = build_rating_rows(train, limit=2)
    assert len(rows) == 2

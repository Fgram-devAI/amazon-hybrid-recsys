"""Tests for src.storage.artifacts.build_vector_payload."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.storage.artifacts import build_vector_payload


def _embeddings(n: int, dim: int) -> np.ndarray:
    return np.arange(n * dim, dtype="float32").reshape(n, dim)


def test_build_vector_payload_aligns_metadata_by_parent_asin():
    item_ids = ["a", "b", "c"]
    emb = _embeddings(3, 4)
    metadata = pd.DataFrame(
        {
            "parent_asin": ["c", "a", "b"],
            "title": ["title_c", "title_a", "title_b"],
            "categories": [["Action"], ["Comedy"], ["Drama"]],
            "store": ["sc", "sa", None],
            "price": [9.99, None, 4.50],
            "average_rating": [4.2, 4.6, None],
            "rating_number": [120, 80, None],
        }
    )

    rows = build_vector_payload(item_ids, emb, metadata)
    assert len(rows) == 3
    assert rows[0]["id"] == 0
    assert rows[0]["parent_asin"] == "a"
    assert rows[0]["title"] == "title_a"
    assert rows[0]["categories"] == "Comedy"
    assert rows[0]["price"] is None
    assert rows[0]["average_rating"] == 4.6
    assert isinstance(rows[0]["vector"], list)
    assert len(rows[0]["vector"]) == 4


def test_build_vector_payload_handles_missing_metadata_row():
    item_ids = ["a", "b"]
    emb = _embeddings(2, 3)
    metadata = pd.DataFrame(
        {
            "parent_asin": ["a"],
            "title": ["title_a"],
            "categories": [["X"]],
            "store": ["s"],
            "price": [1.0],
            "average_rating": [3.0],
            "rating_number": [10],
        }
    )

    rows = build_vector_payload(item_ids, emb, metadata)
    assert rows[1]["parent_asin"] == "b"
    assert rows[1]["title"] == ""
    assert rows[1]["categories"] == ""
    assert rows[1]["store"] is None
    assert rows[1]["price"] is None
    assert rows[1]["average_rating"] is None
    assert rows[1]["rating_number"] is None


def test_build_vector_payload_serializes_category_list_as_pipe_joined_string():
    item_ids = ["a"]
    emb = _embeddings(1, 2)
    metadata = pd.DataFrame(
        {
            "parent_asin": ["a"],
            "title": ["t"],
            "categories": [["Action", "Adventure"]],
            "store": [None],
            "price": [None],
            "average_rating": [None],
            "rating_number": [None],
        }
    )
    rows = build_vector_payload(item_ids, emb, metadata)
    assert rows[0]["categories"] == "Action|Adventure"

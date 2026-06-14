"""Cypher-shape unit tests: assert the queries Neo4jStore would run.

These tests do NOT require a running Neo4j. They use a fake driver that
captures the Cypher query string and parameters, so we can assert query
shape without network access.
"""
from __future__ import annotations

from typing import cast

from neo4j import Driver

from src.storage.neo4j_client import Neo4jStore


class _FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)


class _FakeSession:
    def __init__(self, captured):
        self.captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.captured.append((query, params))
        # Replay deterministic fake rows so the methods can return.
        if "IN_CATEGORY" in query and "parent_asin IN" in query:
            return _FakeResult(
                [
                    {"parent_asin": "A", "category": "RPG"},
                    {"parent_asin": "A", "category": "Adventure"},
                    {"parent_asin": "B", "category": "Strategy"},
                ]
            )
        if "CO_RATED_WITH" in query:
            return _FakeResult(
                [
                    {"parent_asin": "X", "weight_count": 9, "weight_jaccard": 0.42},
                ]
            )
        return _FakeResult([])


class _FakeDriver:
    def __init__(self):
        self.captured: list[tuple[str, dict]] = []

    def session(self):
        return _FakeSession(self.captured)

    def close(self):
        pass


def _store_with_fake_driver() -> tuple[Neo4jStore, _FakeDriver]:
    store = Neo4jStore.__new__(Neo4jStore)
    fake = _FakeDriver()
    store._driver = cast(Driver, fake)
    return store, fake


def test_fetch_categories_for_items_query_shape():
    store, fake = _store_with_fake_driver()
    out = store.fetch_categories_for_items(["A", "B"])
    captured = fake.captured
    assert len(captured) == 1
    query, params = captured[0]
    assert "MATCH (i:Item)" in query
    assert "IN_CATEGORY" in query
    assert "parent_asin IN" in query
    assert params["parent_asins"] == ["A", "B"]
    # Output groups categories by parent_asin
    assert out["A"] == ["RPG", "Adventure"]
    assert out["B"] == ["Strategy"]


def test_fetch_categories_for_items_empty_input_no_query():
    store, fake = _store_with_fake_driver()
    out = store.fetch_categories_for_items([])
    assert out == {}
    assert fake.captured == []


def test_fetch_co_rated_neighbors_query_shape():
    store, fake = _store_with_fake_driver()
    out = store.fetch_co_rated_neighbors("A", top_k=5)
    captured = fake.captured
    assert len(captured) == 1
    query, params = captured[0]
    assert "CO_RATED_WITH" in query
    assert params["parent_asin"] == "A"
    assert params["top_k"] == 5
    assert out == [{"parent_asin": "X", "weight_count": 9, "weight_jaccard": 0.42}]

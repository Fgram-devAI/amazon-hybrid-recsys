"""Evidence assembly: deterministic payload shape, no live Milvus/Neo4j calls."""
from __future__ import annotations

import pandas as pd

from src.reasoning.evidence import (
    MilvusEvidenceFetcher,
    Neo4jEvidenceFetcher,
    assemble_evidence_payload,
)
from src.reasoning.schemas import RecommendationEvidence


class _FakeMilvusStore:
    def __init__(self, hits):
        self._hits = hits
        self.last_query = None

    def has_collection(self, name) -> bool:
        return True

    def search(self, name, query_vector, top_k, output_fields):
        self.last_query = {"name": name, "top_k": top_k, "fields": output_fields}
        return self._hits[:top_k]

    def close(self):
        pass


class _FakeNeo4jStore:
    def __init__(self, top_items, category_map, co_rated=None):
        self._top_items = top_items
        self._category_map = category_map
        self._co_rated = co_rated or {}

    def fetch_top_items_for_user(self, user_id, top_k):
        return list(self._top_items[:top_k])

    def fetch_categories_for_items(self, parent_asins):
        return {a: self._category_map.get(a, []) for a in parent_asins}

    def fetch_co_rated_neighbors(self, parent_asin, top_k):
        return list(self._co_rated.get(parent_asin, []))[:top_k]


def test_milvus_evidence_fetcher_returns_compact_fields():
    hits = [
        {
            "id": 1,
            "parent_asin": "A",
            "title": "Game A",
            "categories": "RPG|Adventure",
            "store": "Nintendo",
            "average_rating": 4.5,
            "rating_number": 1000,
            "distance": 0.13,
        }
    ]
    store = _FakeMilvusStore(hits)
    fetcher = MilvusEvidenceFetcher(store=store, collection_name="amazon_video_games_items")
    out = fetcher.semantic_neighbors(query_vector=[0.1] * 4, top_k=3)
    assert out[0]["parent_asin"] == "A"
    assert out[0]["title"] == "Game A"
    assert out[0]["categories"] == ["RPG", "Adventure"]
    assert out[0]["distance"] == 0.13


def test_milvus_evidence_fetcher_handles_nested_entity_payload():
    """Real pymilvus search responses nest payload under 'entity'."""
    hits = [
        {
            "id": 1,
            "distance": 0.07,
            "entity": {
                "parent_asin": "B",
                "title": "Game B",
                "categories": ["RPG"],
                "store": "Sony",
                "average_rating": 4.0,
                "rating_number": 500,
            },
        }
    ]
    store = _FakeMilvusStore(hits)
    fetcher = MilvusEvidenceFetcher(store=store, collection_name="amazon_x")
    out = fetcher.semantic_neighbors(query_vector=[0.0] * 4, top_k=1)
    assert out[0]["parent_asin"] == "B"
    assert out[0]["title"] == "Game B"
    assert out[0]["categories"] == ["RPG"]
    assert out[0]["distance"] == 0.07


def test_milvus_evidence_fetcher_returns_empty_when_collection_missing():
    class _NoColl(_FakeMilvusStore):
        def has_collection(self, name) -> bool:
            return False

    store = _NoColl([])
    fetcher = MilvusEvidenceFetcher(store=store, collection_name="amazon_x")
    assert fetcher.semantic_neighbors(query_vector=[0.0], top_k=5) == []


def test_neo4j_evidence_fetcher_pulls_user_history_with_categories():
    fake = _FakeNeo4jStore(
        top_items=[
            {"parent_asin": "H1", "title": "Hist 1", "rating": 5.0, "categories": ["RPG"]},
            {"parent_asin": "H2", "title": "Hist 2", "rating": 5.0, "categories": ["Strategy"]},
        ],
        category_map={},
    )
    fetcher = Neo4jEvidenceFetcher(store=fake)
    out = fetcher.user_history("U1", top_k=2)
    assert [row["parent_asin"] for row in out] == ["H1", "H2"]
    assert out[0]["categories"] == ["RPG"]


def test_assemble_evidence_payload_includes_overlap_and_marks_availability():
    metadata = pd.DataFrame(
        [
            {"parent_asin": "C1", "title": "Cand 1"},
            {"parent_asin": "C2", "title": "Cand 2"},
        ]
    )
    candidates = [
        {
            "parent_asin": "C1",
            "hybrid_score": 0.8,
            "lightgcn_score": 0.7,
            "svd_score": 4.2,
            "semantic_score": 0.6,
            "score_sources": ["graph", "svd", "semantic"],
        },
        {
            "parent_asin": "C2",
            "hybrid_score": 0.5,
            "lightgcn_score": 0.4,
            "svd_score": 3.8,
            "semantic_score": None,
            "score_sources": ["graph", "svd"],
        },
    ]
    semantic_neighbors = [
        {"parent_asin": "C1", "title": "Cand 1", "distance": 0.10, "categories": ["RPG"]},
    ]
    user_history = [
        {"parent_asin": "H1", "title": "Hist 1", "rating": 5.0, "categories": ["RPG"]},
    ]
    candidate_categories = {"C1": ["RPG", "Adventure"], "C2": ["Strategy"]}

    payload = assemble_evidence_payload(
        candidates=candidates,
        metadata=metadata,
        user_history=user_history,
        semantic_neighbors=semantic_neighbors,
        candidate_categories=candidate_categories,
        graph_evidence_available=True,
    )
    assert all(isinstance(p, RecommendationEvidence) for p in payload)
    by_asin = {p.candidate.parent_asin: p for p in payload}
    assert by_asin["C1"].candidate.categories == ["RPG", "Adventure"]
    assert by_asin["C1"].user_evidence.category_overlap == ["RPG"]
    assert by_asin["C2"].user_evidence.category_overlap == []
    assert len(by_asin["C1"].retrieval_evidence.semantic_neighbors) == 1
    assert by_asin["C2"].retrieval_evidence.semantic_neighbors == []
    # Score channels propagate through to typed scores object.
    assert by_asin["C1"].candidate.scores.hybrid == 0.8
    assert by_asin["C1"].candidate.scores.semantic == 0.6
    assert by_asin["C2"].candidate.scores.semantic is None


def test_assemble_evidence_payload_marks_graph_unavailable():
    metadata = pd.DataFrame([{"parent_asin": "C1", "title": "Cand 1"}])
    payload = assemble_evidence_payload(
        candidates=[
            {
                "parent_asin": "C1",
                "hybrid_score": 0.8,
                "score_sources": ["svd"],
            }
        ],
        metadata=metadata,
        user_history=[],
        semantic_neighbors=[],
        candidate_categories={},
        graph_evidence_available=False,
    )
    assert payload[0].user_evidence.graph_evidence_available is False

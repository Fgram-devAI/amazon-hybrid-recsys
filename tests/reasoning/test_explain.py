"""End-to-end orchestration test with all external sources faked."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.reasoning.explain import _display_payload, run_explain


class _FakeEmbedder:
    name = "fake-test"
    device = "cpu"

    def encode(self, texts):
        return np.zeros((len(texts), 4), dtype="float32") + 0.1


class _FakeMilvus:
    def __init__(self):
        self.last_vector = None

    def has_collection(self, name):
        return True

    def search(self, name, query_vector, top_k, output_fields):
        self.last_vector = query_vector
        return [
            {
                "id": 1,
                "parent_asin": "C1",
                "title": "Game C1",
                "categories": "RPG|Adventure",
                "distance": 0.1,
            }
        ]

    def close(self):
        pass


class _FakeNeo4j:
    def fetch_top_items_for_user(self, user_id, top_k):
        return [
            {"parent_asin": "H1", "title": "Hist 1", "rating": 5.0, "categories": ["RPG"]},
        ]

    def fetch_categories_for_items(self, parent_asins):
        return {"C1": ["RPG", "Adventure"], "C2": ["Strategy"]}

    def fetch_co_rated_neighbors(self, parent_asin, top_k):
        return []

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _toy_train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "U1", "parent_asin": "H1", "rating": 5.0, "timestamp": 1},
            {"user_id": "U2", "parent_asin": "C1", "rating": 5.0, "timestamp": 2},
            {"user_id": "U3", "parent_asin": "C2", "rating": 4.0, "timestamp": 3},
        ]
    )


def _toy_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"parent_asin": "C1", "title": "Game C1", "categories": ["RPG"]},
            {"parent_asin": "C2", "title": "Game C2", "categories": ["Strategy"]},
            {"parent_asin": "H1", "title": "Hist 1", "categories": ["RPG"]},
        ]
    )


def test_run_explain_dry_run_returns_payload_and_skips_llm(tmp_path: Path) -> None:
    milvus = _FakeMilvus()
    out = run_explain(
        user_id="U1",
        query="open world rpg",
        candidate_k=10,
        final_k=2,
        weights={"graph": 0.0, "svd": 0.0, "semantic": 0.5, "popularity": 0.5},
        train=_toy_train(),
        metadata=_toy_metadata(),
        embedder=_FakeEmbedder(),
        profile_query_vector=None,
        milvus_store=milvus,
        milvus_collection="amazon_video_games_items",
        neo4j_store=_FakeNeo4j(),
        svd_scorer=None,
        lightgcn_scorer=None,
        llm_config=None,
        dry_run=True,
        output_path=None,
    )
    assert out["mode"] == "dry_run"
    assert "candidates" in out
    assert "evidence_payloads" in out
    assert "prompt" in out
    # User U1 already saw H1 in train — H1 must not be a candidate.
    for row in out["candidates"]:
        assert row["parent_asin"] != "H1"
    # evidence_payloads in the return dict are dumped (JSON-ready dicts), not raw schema instances.
    assert all(isinstance(p, dict) for p in out["evidence_payloads"])
    if out["evidence_payloads"]:
        assert "candidate" in out["evidence_payloads"][0]
    assert out["semantic_source"] == "free_text_query"
    assert milvus.last_vector is not None
    assert np.allclose(milvus.last_vector, [0.1, 0.1, 0.1, 0.1])


def test_run_explain_writes_output_when_path_provided(tmp_path: Path) -> None:
    out_path = tmp_path / "llm_outputs" / "U1.json"
    run_explain(
        user_id="U1",
        query=None,
        candidate_k=10,
        final_k=2,
        weights={"semantic": 0.5, "popularity": 0.5},
        train=_toy_train(),
        metadata=_toy_metadata(),
        embedder=_FakeEmbedder(),
        profile_query_vector=None,
        milvus_store=_FakeMilvus(),
        milvus_collection="amazon_video_games_items",
        neo4j_store=_FakeNeo4j(),
        svd_scorer=None,
        lightgcn_scorer=None,
        llm_config=None,
        dry_run=True,
        output_path=out_path,
    )
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved["mode"] == "dry_run"


def test_run_explain_uses_profile_vector_when_query_is_absent() -> None:
    milvus = _FakeMilvus()
    out = run_explain(
        user_id="U1",
        query=None,
        candidate_k=10,
        final_k=2,
        weights={"semantic": 0.5, "popularity": 0.5},
        train=_toy_train(),
        metadata=_toy_metadata(),
        embedder=None,
        profile_query_vector=[0.3, 0.4, 0.5, 0.6],
        milvus_store=milvus,
        milvus_collection="amazon_video_games_items",
        neo4j_store=_FakeNeo4j(),
        svd_scorer=None,
        lightgcn_scorer=None,
        llm_config=None,
        dry_run=True,
        output_path=None,
    )
    assert out["semantic_source"] == "user_high_rated_item_profile"
    assert milvus.last_vector == [0.3, 0.4, 0.5, 0.6]
    assert out["effective_weights"] == {"semantic": 0.5, "popularity": 0.5}


def test_display_payload_is_compact_for_dry_run_by_default() -> None:
    result = {
        "mode": "dry_run",
        "effective_weights": {"popularity": 1.0},
        "semantic_source": "user_high_rated_item_profile",
        "candidates": [
            {
                "parent_asin": "C1",
                "hybrid_score": 0.9,
                "lightgcn_score": 4.8,
                "svd_score": 4.2,
                "semantic_score": None,
                "popularity_score": 10.0,
                "score_sources": ["graph", "svd"],
            }
        ],
        "llm": {"mode": "dry_run"},
        "evidence_payloads": [
            {
                "candidate": {
                    "parent_asin": "C1",
                    "title": "Game C1",
                    "categories": ["RPG"],
                },
                "user_evidence": {
                    "high_rated_items": [{"parent_asin": "H1", "title": "Hist 1"}],
                    "category_overlap": ["RPG"],
                    "graph_evidence_available": True,
                },
            }
        ],
        "prompt": {"system": "S", "user": "U"},
    }
    payload = _display_payload(result)
    assert "evidence_payloads" not in payload
    assert "prompt" not in payload
    assert payload["semantic_source"] == "user_high_rated_item_profile"
    assert payload["inspection_note"].startswith("Compact dry-run")
    assert payload["candidates"][0]["title"] == "Game C1"
    assert payload["candidates"][0]["category_overlap"] == ["RPG"]
    assert payload["user_profile"]["high_rated_items"][0]["parent_asin"] == "H1"


def test_display_payload_full_json_returns_complete_result() -> None:
    result = {
        "mode": "dry_run",
        "effective_weights": {"popularity": 1.0},
        "semantic_source": "user_high_rated_item_profile",
        "candidates": [{"parent_asin": "C1"}],
        "llm": {"mode": "dry_run"},
        "evidence_payloads": [{"candidate": {"parent_asin": "C1"}}],
        "prompt": {"system": "S", "user": "U"},
    }
    assert _display_payload(result, full_json=True) == result


def test_display_payload_keeps_live_output_concise() -> None:
    result = {
        "mode": "live",
        "effective_weights": {"popularity": 1.0},
        "semantic_source": None,
        "candidates": [{"parent_asin": "C1"}],
        "llm": {"mode": "live", "text": "{}"},
        "evidence_payloads": [{"candidate": {"parent_asin": "C1"}}],
        "prompt": {"system": "S", "user": "U"},
    }
    payload = _display_payload(result)
    assert "evidence_payloads" not in payload
    assert "prompt" not in payload

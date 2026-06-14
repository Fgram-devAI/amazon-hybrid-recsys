"""End-to-end orchestration test with all external sources faked."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.reasoning.explain import run_explain


class _FakeEmbedder:
    name = "fake-test"
    device = "cpu"

    def encode(self, texts):
        return np.zeros((len(texts), 4), dtype="float32") + 0.1


class _FakeMilvus:
    def has_collection(self, name):
        return True

    def search(self, name, query_vector, top_k, output_fields):
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
    out = run_explain(
        user_id="U1",
        query="open world rpg",
        candidate_k=10,
        final_k=2,
        weights={"graph": 0.0, "svd": 0.0, "semantic": 0.5, "popularity": 0.5},
        train=_toy_train(),
        metadata=_toy_metadata(),
        embedder=_FakeEmbedder(),
        milvus_store=_FakeMilvus(),
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

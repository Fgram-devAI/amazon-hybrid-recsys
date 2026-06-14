"""Pure string-builder tests for the prompt template."""
from __future__ import annotations

from src.reasoning.prompts import build_prompt
from src.reasoning.schemas import RecommendationEvidence


def _sample_payload() -> list[RecommendationEvidence]:
    raw = {
        "candidate": {
            "parent_asin": "C1",
            "title": "Game C1",
            "categories": ["RPG"],
            "scores": {"hybrid": 0.8, "lightgcn": 0.7, "svd": 4.2, "semantic": 0.6},
            "score_sources": ["graph", "svd", "semantic"],
        },
        "user_evidence": {
            "high_rated_items": [
                {"parent_asin": "H1", "title": "Hist 1", "rating": 5.0, "categories": ["RPG"]},
            ],
            "category_overlap": ["RPG"],
            "graph_evidence_available": True,
        },
        "retrieval_evidence": {
            "semantic_neighbors": [
                {"parent_asin": "C1", "title": "Game C1", "distance": 0.1}
            ],
            "graph_paths": [],
        },
    }
    return [RecommendationEvidence.model_validate(raw)]


def test_prompt_includes_system_and_user_keys():
    out = build_prompt(
        user_id="U1",
        query=None,
        evidence_payloads=_sample_payload(),
        effective_weights={"graph": 0.5, "svd": 0.25, "semantic": 0.15, "popularity": 0.10},
    )
    assert "system" in out
    assert "user" in out
    assert isinstance(out["system"], str) and out["system"]
    assert isinstance(out["user"], str) and out["user"]


def test_prompt_contains_leakage_guard_phrasing():
    out = build_prompt(
        user_id="U1",
        query=None,
        evidence_payloads=_sample_payload(),
        effective_weights={"graph": 1.0},
    )
    text = out["system"] + out["user"]
    assert "only the provided evidence" in text.lower()
    assert "never claim" in text.lower()
    assert "invent" in text.lower() or "fabricat" in text.lower()


def test_prompt_includes_evidence_json_and_score_sources():
    payload = _sample_payload()
    out = build_prompt(
        user_id="U1",
        query="open world rpg",
        evidence_payloads=payload,
        effective_weights={"graph": 0.5, "svd": 0.5},
    )
    assert "U1" in out["user"]
    assert "open world rpg" in out["user"]
    assert "C1" in out["user"]
    assert "evidence_used" in out["system"].lower() or "evidence_used" in out["user"]


def test_prompt_includes_semantic_source():
    out = build_prompt(
        user_id="U1",
        query=None,
        semantic_source="user_high_rated_item_profile",
        evidence_payloads=_sample_payload(),
        effective_weights={"semantic": 1.0},
    )
    assert "Semantic retrieval source: user_high_rated_item_profile" in out["user"]


def test_prompt_marks_graph_unavailable():
    payload = _sample_payload()
    # Mutate the schema instance into one with graph_evidence_available=False.
    raw = payload[0].model_dump(mode="json")
    raw["user_evidence"]["graph_evidence_available"] = False
    payload = [RecommendationEvidence.model_validate(raw)]
    out = build_prompt(
        user_id="U1",
        query=None,
        evidence_payloads=payload,
        effective_weights={"graph": 1.0},
    )
    assert "graph evidence" in out["system"].lower() or "graph evidence" in out["user"].lower()


def test_prompt_lists_effective_weights():
    out = build_prompt(
        user_id="U1",
        query=None,
        evidence_payloads=_sample_payload(),
        effective_weights={"graph": 0.75, "svd": 0.25},
    )
    assert "0.75" in out["user"]
    assert "0.25" in out["user"]

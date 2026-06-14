"""Unit tests for the Pydantic evidence + LLM response contracts."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.reasoning.schemas import (
    CandidateEvidence,  # noqa: F401 — public-contract symbol used by downstream tasks
    CandidateScores,  # noqa: F401 — public-contract symbol used by downstream tasks
    LLMRecommendation,  # noqa: F401 — public-contract symbol used by downstream tasks
    LLMRecommendationResponse,
    RecommendationEvidence,
    SemanticNeighbor,
    UserEvidence,
    UserHistoryItem,
    RetrievalEvidence,
)


def _sample_evidence_dict():
    return {
        "candidate": {
            "parent_asin": "C1",
            "title": "Game C1",
            "categories": ["RPG", "Adventure"],
            "scores": {
                "hybrid": 0.82,
                "lightgcn": 0.76,
                "svd": 4.12,
                "semantic": 0.68,
                "popularity": 14.0,
            },
            "score_sources": ["graph", "svd", "semantic", "popularity"],
        },
        "user_evidence": {
            "high_rated_items": [
                {
                    "parent_asin": "H1",
                    "title": "Hist 1",
                    "rating": 5.0,
                    "categories": ["RPG"],
                }
            ],
            "category_overlap": ["RPG"],
            "graph_evidence_available": True,
        },
        "retrieval_evidence": {
            "semantic_neighbors": [
                {"parent_asin": "C1", "title": "Game C1", "distance": 0.13}
            ],
            "graph_paths": [],
        },
    }


def test_valid_evidence_payload_validates():
    payload = RecommendationEvidence.model_validate(_sample_evidence_dict())
    assert payload.candidate.parent_asin == "C1"
    assert payload.candidate.scores.hybrid == 0.82
    assert payload.user_evidence.high_rated_items[0].rating == 5.0


def test_missing_optional_scores_are_accepted():
    data = _sample_evidence_dict()
    data["candidate"]["scores"] = {"hybrid": 0.5}  # only hybrid
    payload = RecommendationEvidence.model_validate(data)
    assert payload.candidate.scores.hybrid == 0.5
    assert payload.candidate.scores.lightgcn is None
    assert payload.candidate.scores.svd is None
    assert payload.candidate.scores.semantic is None
    assert payload.candidate.scores.popularity is None


def test_missing_required_field_is_rejected():
    data = _sample_evidence_dict()
    del data["candidate"]["parent_asin"]
    with pytest.raises(ValidationError):
        RecommendationEvidence.model_validate(data)


def test_bad_confidence_value_is_rejected():
    bad = {
        "summary": "ok",
        "recommendations": [
            {
                "parent_asin": "C1",
                "title": "Game C1",
                "why": "because",
                "evidence_used": ["lightgcn_score"],
                "confidence": "ultra-high",  # invalid
            }
        ],
        "caveats": [],
    }
    with pytest.raises(ValidationError):
        LLMRecommendationResponse.model_validate(bad)


def test_valid_llm_response_validates():
    good = {
        "summary": "ok",
        "recommendations": [
            {
                "parent_asin": "C1",
                "title": "Game C1",
                "why": "matches user RPG history",
                "evidence_used": ["graph_category_overlap", "lightgcn_score"],
                "confidence": "medium",
            }
        ],
        "caveats": ["evidence is sparse"],
    }
    parsed = LLMRecommendationResponse.model_validate(good)
    assert parsed.recommendations[0].confidence == "medium"
    assert parsed.caveats == ["evidence is sparse"]


def test_model_dump_json_mode_is_prompt_ready():
    payload = RecommendationEvidence.model_validate(_sample_evidence_dict())
    dumped = payload.model_dump(mode="json")
    # Round-trip through JSON to confirm the dump is plain JSON-serializable.
    re_loaded = json.loads(json.dumps(dumped))
    assert re_loaded["candidate"]["parent_asin"] == "C1"
    assert re_loaded["retrieval_evidence"]["graph_paths"] == []


def test_llm_response_model_validate_json_round_trip():
    text = json.dumps({
        "summary": "concise",
        "recommendations": [
            {
                "parent_asin": "C1",
                "title": "Game C1",
                "why": "fits the user",
                "evidence_used": ["lightgcn_score"],
                "confidence": "high",
            }
        ],
        "caveats": [],
    })
    parsed = LLMRecommendationResponse.model_validate_json(text)
    assert parsed.summary == "concise"
    assert parsed.recommendations[0].parent_asin == "C1"


def test_semantic_neighbor_optional_fields():
    SemanticNeighbor.model_validate({"parent_asin": "X"})  # title + distance both optional


def test_user_history_item_categories_default_to_empty_list():
    item = UserHistoryItem.model_validate({"parent_asin": "X", "rating": 5.0})
    assert item.categories == []


def test_user_evidence_default_graph_unavailable():
    ue = UserEvidence.model_validate({})
    assert ue.high_rated_items == []
    assert ue.category_overlap == []
    assert ue.graph_evidence_available is False


def test_retrieval_evidence_defaults_to_empty_lists():
    re = RetrievalEvidence.model_validate({})
    assert re.semantic_neighbors == []
    assert re.graph_paths == []

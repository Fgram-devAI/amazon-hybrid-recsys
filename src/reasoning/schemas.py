"""Pydantic v2 schemas for the reasoning evidence + LLM response contract.

These models are the boundary between raw dicts coming out of Milvus/Neo4j and
the prompt builder / LLM client. Earlier modules (candidates, evidence) build
raw dicts; this module turns them into typed objects before prompt rendering.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CandidateScores(BaseModel):
    """Per-source scores attached to a candidate item. Hybrid is required; rest optional."""

    model_config = ConfigDict(extra="forbid")

    hybrid: float
    lightgcn: float | None = None
    svd: float | None = None
    semantic: float | None = None
    popularity: float | None = None


class CandidateEvidence(BaseModel):
    """One ranked candidate item with its categories and per-source scores."""

    model_config = ConfigDict(extra="forbid")

    parent_asin: str
    title: str = ""
    categories: list[str] = Field(default_factory=list)
    scores: CandidateScores
    score_sources: list[str] = Field(default_factory=list)


class UserHistoryItem(BaseModel):
    """A train-only item the user rated highly, used as evidence."""

    model_config = ConfigDict(extra="forbid")

    parent_asin: str
    title: str = ""
    rating: float | None = None
    categories: list[str] = Field(default_factory=list)


class UserEvidence(BaseModel):
    """Per-candidate user-side evidence pulled from the graph."""

    model_config = ConfigDict(extra="forbid")

    high_rated_items: list[UserHistoryItem] = Field(default_factory=list)
    category_overlap: list[str] = Field(default_factory=list)
    graph_evidence_available: bool = False


class SemanticNeighbor(BaseModel):
    """One Milvus neighbor for a candidate; distance is cosine distance."""

    model_config = ConfigDict(extra="forbid")

    parent_asin: str
    title: str = ""
    distance: float | None = None


class RetrievalEvidence(BaseModel):
    """Retrieval-store-derived evidence attached to a candidate."""

    model_config = ConfigDict(extra="forbid")

    semantic_neighbors: list[SemanticNeighbor] = Field(default_factory=list)
    graph_paths: list[str] = Field(default_factory=list)


class RecommendationEvidence(BaseModel):
    """The full evidence object for one candidate, as fed to the prompt."""

    model_config = ConfigDict(extra="forbid")

    candidate: CandidateEvidence
    user_evidence: UserEvidence = Field(default_factory=UserEvidence)
    retrieval_evidence: RetrievalEvidence = Field(default_factory=RetrievalEvidence)


Confidence = Literal["low", "medium", "high"]


class LLMRecommendation(BaseModel):
    """One LLM-explained recommendation in the response payload."""

    model_config = ConfigDict(extra="forbid")

    parent_asin: str
    title: str
    why: str
    evidence_used: list[str] = Field(default_factory=list)
    confidence: Confidence


class LLMRecommendationResponse(BaseModel):
    """Top-level schema for the LLM's structured JSON response."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    recommendations: list[LLMRecommendation] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

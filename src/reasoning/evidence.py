"""Milvus + Neo4j evidence retrieval and per-candidate evidence assembly.

Pure data assembly: no LLM call, no embedding computation. Callers pass an
already-encoded query vector for semantic search and already-built Neo4j /
Milvus client wrappers. This keeps tests deterministic without network access.

The output of ``assemble_evidence_payload`` is a list of Pydantic
``RecommendationEvidence`` instances (see ``src/reasoning/schemas.py``).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from src.reasoning.schemas import (
    CandidateEvidence,
    CandidateScores,
    RecommendationEvidence,
    RetrievalEvidence,
    SemanticNeighbor,
    UserEvidence,
    UserHistoryItem,
)


_MILVUS_OUTPUT_FIELDS = [
    "id",
    "parent_asin",
    "title",
    "categories",
    "store",
    "average_rating",
    "rating_number",
]


def _split_categories(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    return [part for part in str(value).split("|") if part]


class _MilvusStoreLike(Protocol):
    def has_collection(self, name: str) -> bool: ...

    def search(
        self,
        name: str,
        query_vector: list[float],
        top_k: int,
        output_fields: list[str],
    ) -> list[dict[str, Any]]: ...


@dataclass
class MilvusEvidenceFetcher:
    store: _MilvusStoreLike
    collection_name: str

    def semantic_neighbors(
        self, *, query_vector: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """Return compact neighbor dicts. Handles both flat and entity-nested hit shapes."""
        if not self.store.has_collection(self.collection_name):
            return []
        raw = self.store.search(
            self.collection_name,
            query_vector,
            top_k=top_k,
            output_fields=_MILVUS_OUTPUT_FIELDS,
        )
        out: list[dict[str, Any]] = []
        for hit in raw:
            entity_obj = hit.get("entity")
            entity = entity_obj if isinstance(entity_obj, dict) else hit
            out.append(
                {
                    "parent_asin": str(entity.get("parent_asin", "")),
                    "title": entity.get("title") or "",
                    "categories": _split_categories(entity.get("categories")),
                    "store": entity.get("store"),
                    "average_rating": entity.get("average_rating"),
                    "rating_number": entity.get("rating_number"),
                    "distance": hit.get("distance"),
                }
            )
        return out


class _Neo4jStoreLike(Protocol):
    def fetch_top_items_for_user(
        self, user_id: str, top_k: int
    ) -> list[dict[str, Any]]: ...

    def fetch_categories_for_items(
        self, parent_asins: list[str]
    ) -> dict[str, list[str]]: ...

    def fetch_co_rated_neighbors(
        self, parent_asin: str, top_k: int
    ) -> list[dict[str, Any]]: ...


@dataclass
class Neo4jEvidenceFetcher:
    store: _Neo4jStoreLike

    def user_history(self, user_id: str, *, top_k: int) -> list[dict[str, Any]]:
        return self.store.fetch_top_items_for_user(user_id, top_k=top_k)

    def categories_for(self, parent_asins: Iterable[str]) -> dict[str, list[str]]:
        return self.store.fetch_categories_for_items(list(parent_asins))

    def co_rated_neighbors(
        self, parent_asin: str, *, top_k: int
    ) -> list[dict[str, Any]]:
        return self.store.fetch_co_rated_neighbors(parent_asin, top_k=top_k)


def _title_for(metadata: pd.DataFrame, parent_asin: str) -> str:
    if metadata.empty:
        return ""
    matches = metadata.loc[metadata["parent_asin"].astype(str) == str(parent_asin), "title"]
    if matches.empty:
        return ""
    value = matches.iloc[0]
    return "" if pd.isna(value) else str(value)


def _build_user_history_items(rows: list[dict[str, Any]]) -> list[UserHistoryItem]:
    items: list[UserHistoryItem] = []
    for row in rows:
        items.append(
            UserHistoryItem(
                parent_asin=str(row["parent_asin"]),
                title=str(row.get("title") or ""),
                rating=None if row.get("rating") is None else float(row["rating"]),
                categories=[str(c) for c in (row.get("categories") or [])],
            )
        )
    return items


def assemble_evidence_payload(
    *,
    candidates: list[dict[str, Any]],
    metadata: pd.DataFrame,
    user_history: list[dict[str, Any]],
    semantic_neighbors: list[dict[str, Any]],
    candidate_categories: dict[str, list[str]],
    graph_evidence_available: bool,
) -> list[RecommendationEvidence]:
    """Build per-candidate Pydantic evidence (spec §6.3, typed)."""
    user_category_set: set[str] = set()
    for item in user_history:
        for cat in item.get("categories") or []:
            if cat:
                user_category_set.add(str(cat))

    history_items = _build_user_history_items(user_history)

    payloads: list[RecommendationEvidence] = []
    for cand in candidates:
        asin = str(cand["parent_asin"])
        cats = candidate_categories.get(asin) or []
        overlap = sorted(c for c in cats if c in user_category_set)
        cand_neighbors = [
            SemanticNeighbor(
                parent_asin=str(n["parent_asin"]),
                title=str(n.get("title") or ""),
                distance=None if n.get("distance") is None else float(n["distance"]),
            )
            for n in semantic_neighbors
            if n.get("parent_asin") == asin
        ]

        scores = CandidateScores(
            hybrid=float(cand.get("hybrid_score", 0.0) or 0.0),
            lightgcn=None if cand.get("lightgcn_score") is None else float(cand["lightgcn_score"]),
            svd=None if cand.get("svd_score") is None else float(cand["svd_score"]),
            semantic=None if cand.get("semantic_score") is None else float(cand["semantic_score"]),
            popularity=None if cand.get("popularity_score") is None else float(cand["popularity_score"]),
        )

        candidate_obj = CandidateEvidence(
            parent_asin=asin,
            title=_title_for(metadata, asin),
            categories=list(cats),
            scores=scores,
            score_sources=list(cand.get("score_sources", [])),
        )
        user_evidence = UserEvidence(
            high_rated_items=history_items,
            category_overlap=overlap,
            graph_evidence_available=graph_evidence_available,
        )
        retrieval_evidence = RetrievalEvidence(
            semantic_neighbors=cand_neighbors,
            graph_paths=[],
        )

        payloads.append(
            RecommendationEvidence(
                candidate=candidate_obj,
                user_evidence=user_evidence,
                retrieval_evidence=retrieval_evidence,
            )
        )
    return payloads

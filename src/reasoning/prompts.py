"""Prompt assembly. Pure string builder — no LLM call, no I/O.

Accepts ``list[RecommendationEvidence]`` and serializes each item via
``model_dump(mode="json")`` so the prompt never depends on arbitrary dict keys.
"""
from __future__ import annotations

import json

from src.reasoning.schemas import RecommendationEvidence

_MAX_TITLE_CHARS = 160
_MAX_CATEGORIES = 6
_MAX_HISTORY_ITEMS = 5
_MAX_SEMANTIC_NEIGHBORS = 3

_SYSTEM_PROMPT = """You are a recommendation explanation assistant.

Rules:
- Use ONLY the provided evidence. Do not invent product facts.
- Never claim a user rated an item unless it appears in the evidence under
  user_evidence.high_rated_items.
- Distinguish model score evidence (lightgcn / svd / hybrid scores) from
  retrieval evidence (graph_category_overlap / semantic_neighbors).
- If graph_evidence_available is false, say graph evidence is unavailable.
- Say when evidence is weak.
- Return raw JSON only. Do not wrap it in Markdown fences.
- Return concise, presentation-friendly JSON with this schema:
  {
    "summary": str,
    "recommendations": [
      {
        "parent_asin": str,
        "title": str,
        "why": str,
        "evidence_used": [str],
        "confidence": "low" | "medium" | "high"
      }
    ],
    "caveats": [str]
  }
"""


def _short_text(value: str, *, max_chars: int = _MAX_TITLE_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _compact_evidence_payloads(
    evidence_payloads: list[RecommendationEvidence],
) -> list[dict]:
    compact: list[dict] = []
    for item in evidence_payloads:
        dumped = item.model_dump(mode="json")
        candidate = dumped["candidate"]
        user_evidence = dumped.get("user_evidence") or {}
        retrieval = dumped.get("retrieval_evidence") or {}
        compact.append(
            {
                "candidate": {
                    "parent_asin": candidate.get("parent_asin"),
                    "title": _short_text(candidate.get("title", "")),
                    "categories": list(candidate.get("categories") or [])[:_MAX_CATEGORIES],
                    "scores": candidate.get("scores") or {},
                    "score_sources": candidate.get("score_sources") or [],
                },
                "user_evidence": {
                    "high_rated_items": [
                        {
                            "parent_asin": hist.get("parent_asin"),
                            "title": _short_text(hist.get("title", "")),
                            "rating": hist.get("rating"),
                            "categories": list(hist.get("categories") or [])[:_MAX_CATEGORIES],
                        }
                        for hist in (user_evidence.get("high_rated_items") or [])[
                            :_MAX_HISTORY_ITEMS
                        ]
                    ],
                    "category_overlap": list(user_evidence.get("category_overlap") or [])[
                        :_MAX_CATEGORIES
                    ],
                    "graph_evidence_available": bool(
                        user_evidence.get("graph_evidence_available")
                    ),
                },
                "retrieval_evidence": {
                    "semantic_neighbors": [
                        {
                            "parent_asin": neighbor.get("parent_asin"),
                            "title": _short_text(neighbor.get("title", "")),
                            "distance": neighbor.get("distance"),
                        }
                        for neighbor in (retrieval.get("semantic_neighbors") or [])[
                            :_MAX_SEMANTIC_NEIGHBORS
                        ]
                    ],
                },
            }
        )
    return compact


def build_prompt(
    *,
    user_id: str,
    query: str | None,
    semantic_source: str | None = None,
    evidence_payloads: list[RecommendationEvidence],
    effective_weights: dict[str, float],
) -> dict[str, str]:
    """Return {'system': ..., 'user': ...} suitable for an OpenAI-compatible chat call."""
    weights_line = ", ".join(
        f"{name}={weight:.2f}" for name, weight in sorted(effective_weights.items())
    )
    query_line = (
        f"Free-text intent: {query}\n" if query else "Free-text intent: <none>\n"
    )
    semantic_line = (
        f"Semantic retrieval source: {semantic_source}\n"
        if semantic_source
        else "Semantic retrieval source: <none>\n"
    )
    evidence_json = json.dumps(
        _compact_evidence_payloads(evidence_payloads),
        separators=(",", ":"),
    )

    user_prompt = (
        f"User id: {user_id}\n"
        f"{query_line}"
        f"{semantic_line}"
        f"Effective hybrid weights: {weights_line}\n\n"
        "Evidence (structured JSON; do not add fields):\n"
        f"{evidence_json}\n\n"
        "Task: produce the JSON object described in the system prompt. "
        "For each recommendation, list which evidence keys you used "
        "(e.g. lightgcn_score, svd_score, semantic_search, "
        "graph_category_overlap)."
    )
    return {"system": _SYSTEM_PROMPT, "user": user_prompt}

"""Candidate generation primitives: score normalization and weighted fusion.

Per-source scorers (SVD, LightGCN, popularity, semantic) and the top-level
``generate_candidates`` orchestrator live in later tasks of the same module.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def min_max_normalize(values: list[float | None]) -> list[float | None]:
    """Linear min-max scale to [0, 1]. Constant sequences -> all zeros. None preserved."""
    present = [v for v in values if v is not None]
    if not present:
        return [None for _ in values]
    lo = min(present)
    hi = max(present)
    if hi == lo:
        return [0.0 if v is not None else None for v in values]
    span = hi - lo
    return [None if v is None else (v - lo) / span for v in values]


def redistribute_weights(
    weights: Mapping[str, float], *, available: set[str]
) -> dict[str, float]:
    """Drop unavailable sources and renormalize remaining weights to sum to 1.0."""
    kept = {k: float(v) for k, v in weights.items() if k in available and v > 0}
    total = sum(kept.values())
    if total <= 0:
        raise ValueError(
            f"No available score sources in {sorted(available)} match weight keys {sorted(weights)}"
        )
    return {k: v / total for k, v in kept.items()}


def weighted_fuse(
    rows: Iterable[Mapping[str, Any]], *, weights: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Combine per-source normalized scores into a hybrid score.

    Each row's hybrid_score uses only the sources actually present on that row;
    the weight denominator is renormalized accordingly. Missing values (None)
    are skipped and the source key is omitted from ``score_sources``.
    """
    fused: list[dict[str, Any]] = []
    for row in rows:
        used_weights = {k: float(weights[k]) for k in weights if row.get(k) is not None}
        total_w = sum(used_weights.values())
        if total_w == 0:
            hybrid = 0.0
            sources: list[str] = []
        else:
            hybrid = sum((row[k] * (w / total_w)) for k, w in used_weights.items())
            sources = sorted(used_weights)
        out = dict(row)
        out["hybrid_score"] = float(hybrid)
        out["score_sources"] = sources
        fused.append(out)
    return fused

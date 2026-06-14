"""Candidate generation primitives: score normalization and weighted fusion.

Per-source scorers (SVD, LightGCN, popularity, semantic) and the top-level
``generate_candidates`` orchestrator live in later tasks of the same module.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


def min_max_normalize(values: list[float | None]) -> list[float | None]:
    """Linear min-max scale to [0, 1]. Constant sequences -> all zeros. None preserved."""
    present = [v for v in values if v is not None]
    if not present:
        return [None for _ in values]
    lo = min(present)
    hi = max(present)
    if hi == lo:
        if any(v is None for v in values):
            return [None if v is None else 1.0 for v in values]
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
    rows: Iterable[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    renormalize_missing: bool = True,
) -> list[dict[str, Any]]:
    """Combine per-source normalized scores into a hybrid score.

    Missing values (None) are skipped and the source key is omitted from
    ``score_sources``. When ``renormalize_missing`` is false, missing row-level
    scores contribute 0 so high weights on semantic/profile evidence can change
    ranking instead of being redistributed away per candidate.
    """
    fused: list[dict[str, Any]] = []
    for row in rows:
        used_weights = {k: float(weights[k]) for k in weights if row.get(k) is not None}
        total_w = sum(used_weights.values()) if renormalize_missing else sum(weights.values())
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


@dataclass
class PopularityScorer:
    """Train-side interaction count per item. Score is the raw count."""

    train: pd.DataFrame

    def __post_init__(self) -> None:
        self._counts = Counter(self.train["parent_asin"].astype(str).tolist())

    def score(self, parent_asin: str) -> float:
        return float(self._counts.get(str(parent_asin), 0))


@dataclass
class SemanticScorer:
    """Wraps a Milvus hit list. Similarity = 1.0 - cosine distance, clipped to >= 0."""

    hits: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self._by_asin: dict[str, float] = {}
        for hit in self.hits:
            asin = str(hit.get("parent_asin", ""))
            if not asin:
                continue
            distance = hit.get("distance")
            if distance is None:
                continue
            sim = max(0.0, 1.0 - float(distance))
            # Keep the closest (highest similarity) occurrence per parent_asin.
            self._by_asin[asin] = max(self._by_asin.get(asin, 0.0), sim)

    def score(self, parent_asin: str) -> float | None:
        return self._by_asin.get(str(parent_asin))


# Scorer callables for SVD / LightGCN are kept as plain Callables to avoid
# pulling heavy model deps into this module's import surface.
SvdScorer = Callable[[str, str], float | None]
LightGCNScorerCallable = Callable[[str, str], float | None]


def generate_candidates(
    *,
    user_id: str,
    candidate_pool: list[str],
    train: pd.DataFrame,
    svd_scorer: SvdScorer | None,
    lightgcn_scorer: LightGCNScorerCallable | None,
    semantic_scorer: SemanticScorer | None,
    popularity_scorer: PopularityScorer | None,
    weights: Mapping[str, float],
    final_k: int,
) -> list[dict[str, Any]]:
    """Build the hybrid candidate table, excluding the user's train items."""
    seen = set(
        train.loc[train["user_id"].astype(str) == str(user_id), "parent_asin"]
        .astype(str)
        .tolist()
    )
    fresh = [a for a in candidate_pool if a not in seen]

    available: set[str] = set()
    if lightgcn_scorer is not None:
        available.add("graph")
    if svd_scorer is not None:
        available.add("svd")
    if semantic_scorer is not None:
        available.add("semantic")
    if popularity_scorer is not None:
        available.add("popularity")
    effective_weights = redistribute_weights(weights, available=available)

    raw_rows: list[dict[str, Any]] = []
    raw_graph: list[float | None] = []
    raw_svd: list[float | None] = []
    raw_sem: list[float | None] = []
    raw_pop: list[float | None] = []
    for asin in fresh:
        raw_graph.append(lightgcn_scorer(user_id, asin) if lightgcn_scorer else None)
        raw_svd.append(svd_scorer(user_id, asin) if svd_scorer else None)
        raw_sem.append(semantic_scorer.score(asin) if semantic_scorer else None)
        raw_pop.append(popularity_scorer.score(asin) if popularity_scorer else None)
        raw_rows.append({"parent_asin": asin})

    norm_graph = min_max_normalize(raw_graph)
    norm_svd = min_max_normalize(raw_svd)
    norm_sem = min_max_normalize(raw_sem)
    norm_pop = min_max_normalize(raw_pop)

    for i, row in enumerate(raw_rows):
        # Keep raw + normalized; raw is helpful for human inspection.
        row["lightgcn_score"] = raw_graph[i]
        row["svd_score"] = raw_svd[i]
        row["semantic_score"] = raw_sem[i]
        row["popularity_score"] = raw_pop[i]
        if "graph" in effective_weights:
            row["graph"] = norm_graph[i]
        if "svd" in effective_weights:
            row["svd"] = norm_svd[i]
        if "semantic" in effective_weights:
            row["semantic"] = norm_sem[i]
        if "popularity" in effective_weights:
            row["popularity"] = norm_pop[i]

    fused = weighted_fuse(raw_rows, weights=effective_weights, renormalize_missing=False)
    fused.sort(key=lambda r: r["hybrid_score"], reverse=True)
    return fused[:final_k]


class _LightGCNLike(Protocol):
    _graph: Any

    def predict(self, user_id: str, parent_asin: str, /) -> float: ...


class LightGCNScorer:
    """Thin wrapper: returns model.predict only when both user and item are known.

    LightGCN's own predict() silently falls back to global/item means for
    unknown users/items; that would pollute the graph score channel, so the
    scorer guards on the model's user/item indices and returns None for misses.
    """

    def __init__(self, model: _LightGCNLike) -> None:
        self._model = model

    def __call__(self, user_id: str, parent_asin: str) -> float | None:
        graph = getattr(self._model, "_graph", None)
        if graph is None:
            return None
        if user_id not in graph.user_index or parent_asin not in graph.item_index:
            return None
        return float(self._model.predict(user_id, parent_asin))


class _SVDLike(Protocol):
    def predict(self, user_id: str, parent_asin: str, /) -> float: ...


class SVDScorer:
    """Wraps SVDRecommender.predict. Returns the raw rating in [1, 5]."""

    def __init__(self, model: _SVDLike) -> None:
        self._model = model

    def __call__(self, user_id: str, parent_asin: str) -> float | None:
        return float(self._model.predict(user_id, parent_asin))


def load_lightgcn_from_checkpoint(
    checkpoint_path: Path,
    train: pd.DataFrame,
    config: dict[str, Any],
) -> "LightGCNScorer":
    """Mirror evaluate_lightgcn_checkpoint.py: prepare inference state, then load weights."""
    from src.models.lightgcn import LightGCNRecommender

    graph_cfg = config.get("graph", {})
    model = LightGCNRecommender(
        embedding_dim=int(graph_cfg.get("embedding_dim", 64)),
        n_layers=int(graph_cfg.get("n_layers", 2)),
        epochs=int(graph_cfg.get("epochs", 10)),
        lr=float(graph_cfg.get("lr", 0.005)),
        weight_decay=float(graph_cfg.get("weight_decay", 0.0)),
        num_negatives=int(graph_cfg.get("num_negatives", 1)),
        batch_size=int(graph_cfg.get("batch_size", 1024)),
        seed=int(graph_cfg.get("seed", 42)),
        device=str(graph_cfg.get("device", "auto")),
        min_rating_positive=float(graph_cfg.get("min_rating_positive", 4.0)),
        validation_fraction=float(graph_cfg.get("validation_fraction", 0.1)),
        progress=False,
    )
    model.prepare_for_checkpoint(train)
    model.load_checkpoint(checkpoint_path)
    return LightGCNScorer(model)

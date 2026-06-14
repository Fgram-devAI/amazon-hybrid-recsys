"""Pure-Python builders for Neo4j payload rows; no driver/Cypher dependency."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.categories import filter_categories


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _opt_int(value: Any) -> int | None:
    f = _opt_float(value)
    return None if f is None else int(f)


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def build_item_rows(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    """One row per unique parent_asin; the first occurrence wins on collisions."""
    if metadata.empty:
        return []
    deduped = metadata.drop_duplicates(subset="parent_asin", keep="first")
    rows: list[dict[str, Any]] = []
    for record in deduped.to_dict(orient="records"):
        rows.append(
            {
                "parent_asin": str(record["parent_asin"]),
                "title": _opt_str(record.get("title")) or "",
                "store": _opt_str(record.get("store")),
                "price": _opt_float(record.get("price")),
                "average_rating": _opt_float(record.get("average_rating")),
                "rating_number": _opt_int(record.get("rating_number")),
            }
        )
    return rows


def build_item_category_rows(
    metadata: pd.DataFrame, *, generic_roots: list[str]
) -> list[dict[str, Any]]:
    """(parent_asin, category) pairs after filtering generic roots."""
    if metadata.empty:
        return []
    deduped = metadata.drop_duplicates(subset="parent_asin", keep="first")
    rows: list[dict[str, Any]] = []
    for record in deduped.to_dict(orient="records"):
        cats = filter_categories(record.get("categories"), generic_roots)
        for cat in cats:
            rows.append({"parent_asin": str(record["parent_asin"]), "category": cat})
    return rows


def build_rating_rows(
    train: pd.DataFrame, *, limit: int | None = None
) -> list[dict[str, Any]]:
    """Train-only RATED edge payloads. Test edges are NEVER ingested as graph evidence."""
    df = train if limit is None else train.head(limit)
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        rows.append(
            {
                "user_id": str(record["user_id"]),
                "parent_asin": str(record["parent_asin"]),
                "rating": float(record["rating"]),
                "timestamp": int(record["timestamp"])
                if record.get("timestamp") is not None
                and not (
                    isinstance(record["timestamp"], float)
                    and math.isnan(record["timestamp"])
                )
                else None,
                "split": "train",
            }
        )
    return rows


def build_co_rating_rows(
    report_path: Path, *, max_edges: int
) -> list[dict[str, Any]]:
    """Read item-item edges from a graph-analysis report; cap by weight_count descending."""
    if not report_path.exists():
        return []
    data = json.loads(report_path.read_text())
    edges = data.get("edges", [])
    edges = sorted(edges, key=lambda e: e.get("weight_count", 0), reverse=True)
    out: list[dict[str, Any]] = []
    for edge in edges[:max_edges]:
        out.append(
            {
                "a": str(edge["a"]),
                "b": str(edge["b"]),
                "weight_count": int(edge.get("weight_count", 0)),
                "weight_jaccard": float(edge.get("weight_jaccard", 0.0)),
            }
        )
    return out

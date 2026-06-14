"""Load vector artifacts and build Milvus row payloads aligned by parent_asin."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VectorArtifacts:
    item_ids: list[str]
    embeddings: np.ndarray  # shape (n, dim), float32
    dim: int
    source_path: Path
    meta: dict[str, Any]


def load_vector_artifacts(
    *,
    processed_dir: Path,
    dataset_key: str,
    embedding_subdir: str,
) -> VectorArtifacts:
    """Load item embeddings + ids, preferring the advanced-features cache."""
    base = Path(processed_dir) / dataset_key
    primary = base / embedding_subdir
    fallback = base / "embeddings"

    chosen: Path | None = None
    for candidate in (primary, fallback):
        if (candidate / "item_emb.npy").exists() and (candidate / "item_ids.json").exists():
            chosen = candidate
            break
    if chosen is None:
        raise FileNotFoundError(
            "No item_emb.npy / item_ids.json found under either of:\n"
            f"  {primary}\n  {fallback}"
        )

    embeddings = np.load(chosen / "item_emb.npy")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")
    item_ids = json.loads((chosen / "item_ids.json").read_text())
    if not isinstance(item_ids, list):
        raise ValueError(f"item_ids.json at {chosen} is not a JSON list")
    if len(item_ids) != embeddings.shape[0]:
        raise ValueError(
            f"item_ids length ({len(item_ids)}) does not match "
            f"embeddings rows ({embeddings.shape[0]}) at {chosen}"
        )

    meta_path = chosen / "embedding_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta_dim = meta.get("dim")
    if meta_dim is not None and int(meta_dim) != int(embeddings.shape[1]):
        raise ValueError(
            f"embedding_meta.json dim ({int(meta_dim)}) does not match "
            f"embeddings columns ({int(embeddings.shape[1])}) at {chosen}"
        )
    return VectorArtifacts(
        item_ids=[str(x) for x in item_ids],
        embeddings=embeddings,
        dim=int(embeddings.shape[1]),
        source_path=chosen,
        meta=meta,
    )


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


def _categories_to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return "|".join(str(v) for v in value if v not in (None, ""))
    return str(value)


# Field caps mirror the VARCHAR max_length values in MilvusLiteStore.create_collection.
# Real Amazon titles can exceed 512 chars; truncate defensively before insert.
_MAX_TITLE_CHARS = 2048
_MAX_CATEGORIES_CHARS = 2048
_MAX_STORE_CHARS = 512


def build_vector_payload(
    item_ids: list[str],
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build Milvus rows aligned to ``item_ids`` order by ``parent_asin``."""
    meta_by_asin = (
        metadata.drop_duplicates(subset="parent_asin").set_index("parent_asin")
        if not metadata.empty
        else metadata
    )

    rows: list[dict[str, Any]] = []
    for idx, asin in enumerate(item_ids):
        if not metadata.empty and asin in meta_by_asin.index:
            m = meta_by_asin.loc[asin]
            title = (_opt_str(m.get("title")) or "")[:_MAX_TITLE_CHARS]
            categories = _categories_to_string(m.get("categories"))[:_MAX_CATEGORIES_CHARS]
            store_raw = _opt_str(m.get("store"))
            store = None if store_raw is None else store_raw[:_MAX_STORE_CHARS]
            price = _opt_float(m.get("price"))
            average_rating = _opt_float(m.get("average_rating"))
            rating_number = _opt_int(m.get("rating_number"))
        else:
            title = ""
            categories = ""
            store = None
            price = None
            average_rating = None
            rating_number = None
        rows.append(
            {
                "id": idx,
                "parent_asin": asin,
                "title": title,
                "categories": categories,
                "store": store,
                "price": price,
                "average_rating": average_rating,
                "rating_number": rating_number,
                "vector": embeddings[idx].tolist(),
            }
        )
    return rows

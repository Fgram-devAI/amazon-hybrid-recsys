"""Filtered category features for item content.

Drops generic dataset-root labels ("Movies & TV", "Video Games", ...) and keeps
informative genre/subgenre/community labels. All vocabulary is built from
training-side metadata; this module never reads test rows.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

_ACRONYMS = {"tv", "dvd", "cd", "pc", "rpg", "4k", "uhd", "vhs", "vr"}


def normalize_category(value: object) -> str | None:
    """Normalize one category label. Returns None for empty/None inputs."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    def _normalize_token(token: str) -> str:
        lowered = token.casefold()
        if lowered in _ACRONYMS:
            return lowered.upper()
        return token[:1].upper() + token[1:].lower()

    # Normalize whitespace and readable casing while preserving common acronyms.
    return " ".join(_normalize_token(part) for part in text.split())


def filter_categories(
    raw: Sequence[object] | None,
    generic_roots: Sequence[str],
) -> list[str]:
    """Normalize, drop generic roots + empties, deduplicate (order-preserving)."""
    if not raw:
        return []
    generic = {
        norm.casefold()
        for g in generic_roots
        if (norm := normalize_category(g)) is not None
    }
    values: Sequence[object]
    if isinstance(raw, str):
        norm_raw = normalize_category(raw)
        if norm_raw is None:
            return []
        key_raw = norm_raw.casefold()
        for root in sorted(generic, key=len, reverse=True):
            prefix = f"{root} "
            if key_raw.startswith(prefix):
                norm_raw = norm_raw[len(prefix) :].strip()
                break
        values = [norm_raw]
    else:
        values = raw
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        norm = normalize_category(value)
        if norm is None:
            continue
        key = norm.casefold()
        if key in generic or key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def _normalized_per_row(
    metadata_df: pd.DataFrame, generic_roots: list[str]
) -> pd.Series:
    """For each metadata row, return its filtered category list (already deduped)."""
    return metadata_df["categories"].apply(
        lambda raw: filter_categories(raw, generic_roots)
    )


def build_category_vocab(
    metadata_df: pd.DataFrame,
    *,
    generic_roots: list[str],
    max_vocab: int,
    min_doc_freq: int,
) -> list[str]:
    """Vocabulary of informative categories, descending by document frequency.

    Deterministic: ties on frequency are broken alphabetically. Categories with
    document frequency below ``min_doc_freq`` are excluded.
    """
    counts: Counter[str] = Counter()
    for filtered in _normalized_per_row(metadata_df, generic_roots):
        counts.update(filtered)
    ranked = sorted(
        ((cat, c) for cat, c in counts.items() if c >= min_doc_freq),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return [cat for cat, _ in ranked[:max_vocab]]


def build_category_features(
    metadata_df: pd.DataFrame,
    vocab: list[str],
    *,
    generic_roots: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Multi-hot category matrix aligned to metadata row order.

    Returns ``(features, parent_asins)``. ``features`` is float32 (n_items, |vocab|).
    """
    index = {cat: col for col, cat in enumerate(vocab)}
    rows = metadata_df["parent_asin"].tolist()
    feats = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for row, filtered in enumerate(_normalized_per_row(metadata_df, generic_roots)):
        for cat in filtered:
            col = index.get(cat)
            if col is not None:
                feats[row, col] = 1.0
    return feats, rows


def save_category_artifacts(
    *,
    out_dir: Path | str,
    vocab: list[str],
    features: np.ndarray,
    ids: list[str],
    dataset: str,
    meta_row_count: int,
    version: int = 1,
) -> None:
    """Persist vocab/features/ids/meta into the advanced_features cache directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "category_vocab.json").write_text(json.dumps(vocab))
    (out / "item_feature_ids.json").write_text(json.dumps(ids))
    np.save(out / "item_features_v2.npy", features)
    (out / "item_features_v2_meta.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "vocab_size": len(vocab),
                "row_count": meta_row_count,
                "version": version,
            }
        )
    )

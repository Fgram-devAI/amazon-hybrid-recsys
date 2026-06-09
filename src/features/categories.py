"""Filtered category features for item content.

Drops generic dataset-root labels ("Movies & TV", "Video Games", ...) and keeps
informative genre/subgenre/community labels. All vocabulary is built from
training-side metadata; this module never reads test rows.
"""

from __future__ import annotations

from collections.abc import Sequence

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
    seen: set[str] = set()
    out: list[str] = []
    for value in raw:
        norm = normalize_category(value)
        if norm is None:
            continue
        key = norm.casefold()
        if key in generic or key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out

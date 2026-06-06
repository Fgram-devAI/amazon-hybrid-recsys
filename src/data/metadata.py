"""Prepare item metadata for content features (text blob + numeric + missing flags).

No stopword removal / stemming / lemmatization: transformer embeddings want
natural text. Only minimal cleanup (HTML strip + whitespace normalization).
"""

import html
import re

import pandas as pd

_HTML_TAG = re.compile(r"<[^>]+>")


def _join_text(value):
    """Coerce a text field that may be None, a string, or a list of strings."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _clean_text(text):
    """Strip HTML tags/entities and collapse whitespace; preserve natural wording."""
    without_tags = _HTML_TAG.sub(" ", text)
    return " ".join(html.unescape(without_tags).split())


def _parse_price(value):
    """Return (price_float, is_missing). Unparseable/empty -> (0.0, True)."""
    if value is None:
        return 0.0, True
    cleaned = str(value).strip().replace("$", "").replace(",", "")
    if not cleaned:
        return 0.0, True
    try:
        return float(cleaned), False
    except ValueError:
        return 0.0, True


def prepare_metadata(records):
    """Build a per-item metadata DataFrame keyed by parent_asin."""
    rows = []
    for r in records:
        if r.get("parent_asin") is None:
            continue
        title = _join_text(r.get("title"))
        description = _join_text(r.get("description"))
        categories = _join_text(r.get("categories"))
        text = _clean_text(" ".join([title, description, categories]))

        price, price_missing = _parse_price(r.get("price"))
        avg = r.get("average_rating")
        rating_number = r.get("rating_number")

        rows.append(
            {
                "parent_asin": r["parent_asin"],
                "title": title,
                "description": description,
                "text": text,
                "categories": categories,
                "store": _join_text(r.get("store")),
                "price": price,
                "price_missing": price_missing,
                "average_rating": float(avg) if avg is not None else 0.0,
                "average_rating_missing": avg is None,
                "rating_number": int(rating_number) if rating_number is not None else 0,
                "rating_number_missing": rating_number is None,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)

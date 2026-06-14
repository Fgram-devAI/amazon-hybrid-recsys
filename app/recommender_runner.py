"""App-facing service module: load local artifacts and run recommendations.

Keeps the Streamlit script in app/streamlit_app.py free of model logic. Every
public function takes pandas DataFrames or already-constructed Pydantic objects;
nothing here touches Streamlit globals or session state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


def _normalize_categories(value: object) -> list[str]:
    """Coerce processed-metadata category cell into a clean list[str].

    Real processed metadata stores ``categories`` as a joined string
    (see ``src/data/metadata.py``); tests use Python lists. Accept either,
    plus pipe- or comma-delimited strings.
    """
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return [str(c).strip() for c in value if c is not None and str(c).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _metadata_lookup(metadata: pd.DataFrame) -> dict[str, dict]:
    """Dedupe metadata on parent_asin (first wins) and return {asin: {title, categories}}."""
    if metadata.empty:
        return {}
    deduped = metadata.drop_duplicates(subset="parent_asin", keep="first")
    out: dict[str, dict] = {}
    for _, row in deduped.iterrows():
        asin = str(row["parent_asin"])
        title_raw = row.get("title")
        title = "" if pd.isna(title_raw) else str(title_raw)
        out[asin] = {
            "title": title,
            "categories": _normalize_categories(row.get("categories")),
        }
    return out


@dataclass
class LocalArtifacts:
    """Result of attempting to load required local train/metadata parquets."""

    dataset: str
    train: pd.DataFrame
    metadata: pd.DataFrame
    available: bool
    missing_reasons: list[str] = field(default_factory=list)


def load_local_artifacts(
    *, processed_dir: Path, dataset: str
) -> LocalArtifacts:
    """Load ``train.parquet`` + ``metadata.parquet`` for the dataset.

    Returns ``LocalArtifacts`` with ``available=False`` and per-file
    ``missing_reasons`` when any required file is missing; files that exist are
    still loaded so callers can use partial data for diagnostics. Callers render
    a friendly warning in the Streamlit UI instead of crashing.
    """
    base = Path(processed_dir) / dataset
    train_path = base / "train.parquet"
    metadata_path = base / "metadata.parquet"

    missing: list[str] = []
    train = (
        pd.read_parquet(train_path) if train_path.is_file() else pd.DataFrame()
    )
    if not train_path.is_file():
        missing.append(f"missing: {train_path}")

    metadata = (
        pd.read_parquet(metadata_path)
        if metadata_path.is_file()
        else pd.DataFrame()
    )
    if not metadata_path.is_file():
        missing.append(f"missing: {metadata_path}")

    return LocalArtifacts(
        dataset=dataset,
        train=train,
        metadata=metadata,
        available=not missing,
        missing_reasons=missing,
    )


def representative_users(
    train: pd.DataFrame,
    *,
    limit: int = 25,
    min_interactions: int = 2,
) -> list[str]:
    """Return up to ``limit`` user ids sorted by interaction count then lex order."""
    if train.empty:
        return []
    counts = train["user_id"].astype(str).value_counts()
    counts = counts[counts >= int(min_interactions)]
    if counts.empty:
        return []
    ranked = counts.reset_index()
    ranked.columns = ["user_id", "count"]
    ranked = ranked.sort_values(
        ["count", "user_id"], ascending=[False, True], kind="mergesort"
    )
    return ranked["user_id"].head(int(limit)).tolist()


def seen_items_for_user(train: pd.DataFrame, *, user_id: str) -> set[str]:
    """Set of parent_asins the user has rated in train (used to exclude later)."""
    if train.empty:
        return set()
    mask = train["user_id"].astype(str) == str(user_id)
    return set(train.loc[mask, "parent_asin"].astype(str).tolist())


def user_history_rows(
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    user_id: str,
    min_rating: float = 4.0,
    limit: int = 10,
) -> list[dict]:
    """High-rated train items for the user, joined with title/categories."""
    if train.empty:
        return []
    mask = (train["user_id"].astype(str) == str(user_id)) & (
        train["rating"].astype(float) >= float(min_rating)
    )
    history = train.loc[mask].copy()
    if history.empty:
        return []
    if "timestamp" in history.columns:
        history = history.sort_values("timestamp", ascending=False, kind="mergesort")
    history = history.head(int(limit))

    meta = _metadata_lookup(metadata)
    rows: list[dict] = []
    for _, row in history.iterrows():
        asin = str(row["parent_asin"])
        meta_entry = meta.get(asin, {"title": "", "categories": []})
        rows.append(
            {
                "parent_asin": asin,
                "title": meta_entry["title"],
                "categories": meta_entry["categories"],
                "rating": float(row["rating"]),
            }
        )
    return rows

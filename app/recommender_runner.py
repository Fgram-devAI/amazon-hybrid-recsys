"""App-facing service module: load local artifacts and run recommendations.

Keeps the Streamlit script in app/streamlit_app.py free of model logic. Every
public function takes pandas DataFrames or already-constructed Pydantic objects;
nothing here touches Streamlit globals or session state.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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


DEFAULT_CANDIDATE_POOL_SIZE = 500


def build_candidate_pool(
    *,
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    seen: set[str],
    pool_size: int,
    seed_asin: str | None = None,
) -> list[str]:
    """Bounded candidate pool with seed-category expansion.

    Order: same-category items first (when seed is provided and in metadata),
    then popular train items, then any remaining metadata items. The seed asin
    and items in ``seen`` are always excluded. Result is capped at ``pool_size``.
    """
    meta_lookup = _metadata_lookup(metadata)
    exclude: set[str] = set(seen)
    if seed_asin:
        exclude.add(str(seed_asin))

    seed_neighbors: list[str] = []
    if seed_asin and seed_asin in meta_lookup:
        seed_cats = set(meta_lookup[seed_asin]["categories"])
        if seed_cats:
            for asin, entry in meta_lookup.items():
                if asin in exclude:
                    continue
                if seed_cats.intersection(entry["categories"]):
                    seed_neighbors.append(asin)

    if not train.empty:
        popular = (
            train.loc[~train["parent_asin"].astype(str).isin(exclude), "parent_asin"]
            .astype(str)
            .value_counts()
            .head(pool_size)
            .index
            .tolist()
        )
    else:
        popular = []

    meta_asins = [a for a in meta_lookup.keys() if a not in exclude]

    combined = list(dict.fromkeys(seed_neighbors + popular + meta_asins))
    return combined[: max(int(pool_size), 1)]


EMPTY_ROW_FIELDS: tuple[str, ...] = (
    "rank",
    "parent_asin",
    "title",
    "categories",
    "method",
    "hybrid_score",
    "svd_score",
    "knn_score",
    "lightgcn_score",
    "semantic_score",
    "popularity_score",
    "score_sources",
    "already_seen",
)


def _empty_row() -> dict:
    return {k: None for k in EMPTY_ROW_FIELDS}


def _decorate_row(
    row: dict,
    *,
    rank: int,
    asin: str,
    meta_lookup: dict[str, dict],
) -> dict:
    meta_entry = meta_lookup.get(asin, {"title": "", "categories": []})
    row.update(
        {
            "rank": rank,
            "parent_asin": asin,
            "title": meta_entry["title"],
            "categories": meta_entry["categories"],
            "already_seen": False,
        }
    )
    return row


def run_fitted_recommender(
    *,
    model,
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    user_id: str,
    top_k: int,
    method: str,
    score_field: str,
    score_sources: list[str],
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    seed_asin: str | None = None,
) -> list[dict]:
    """Rank a bounded unseen pool with an already-fitted recommender."""
    if train.empty:
        return []
    seen = seen_items_for_user(train, user_id=user_id)
    pool = build_candidate_pool(
        train=train,
        metadata=metadata,
        seen=seen,
        pool_size=candidate_pool_size,
        seed_asin=seed_asin,
    )
    if not pool:
        return []

    meta = _metadata_lookup(metadata)
    scored = [(asin, float(model.predict(user_id, asin))) for asin in pool]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    rows: list[dict] = []
    for rank, (asin, score) in enumerate(scored[: int(top_k)], start=1):
        row = _empty_row()
        row.update(
            {
                "method": method,
                score_field: float(score),
                "score_sources": list(score_sources),
            }
        )
        _decorate_row(row, rank=rank, asin=str(asin), meta_lookup=meta)
        rows.append(row)
    return rows


def run_popularity(
    *,
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    user_id: str,
    top_k: int,
    seed_asin: str | None = None,
) -> list[dict]:
    """Rank unseen items by train interaction count."""
    if train.empty:
        return []
    seen = seen_items_for_user(train, user_id=user_id)
    exclude = set(seen)
    if seed_asin:
        exclude.add(str(seed_asin))
    counts = (
        train.loc[~train["parent_asin"].astype(str).isin(exclude), "parent_asin"]
        .astype(str)
        .value_counts()
    )
    if counts.empty:
        return []
    meta = _metadata_lookup(metadata)
    rows: list[dict] = []
    for rank, (asin, count) in enumerate(counts.head(int(top_k)).items(), start=1):
        row = _empty_row()
        row.update(
            {
                "method": "popularity",
                "popularity_score": float(count),
                "score_sources": ["popularity"],
            }
        )
        _decorate_row(row, rank=rank, asin=str(asin), meta_lookup=meta)
        rows.append(row)
    return rows


def _default_svd_factory():
    from src.models.cf import SVDRecommender

    return SVDRecommender(n_factors=32, n_epochs=10, random_state=42)


def run_svd(
    *,
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    user_id: str,
    top_k: int,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    seed_asin: str | None = None,
    svd_factory: Callable | None = None,
) -> list[dict]:
    """Fit SVD locally on ``train``, rank unseen items by predicted rating."""
    if train.empty:
        return []
    factory = svd_factory or _default_svd_factory
    seen = seen_items_for_user(train, user_id=user_id)
    pool = build_candidate_pool(
        train=train,
        metadata=metadata,
        seen=seen,
        pool_size=candidate_pool_size,
        seed_asin=seed_asin,
    )
    if not pool:
        return []

    model = factory()
    model.fit(train)

    meta = _metadata_lookup(metadata)
    scored = [(asin, float(model.predict(user_id, asin))) for asin in pool]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    rows: list[dict] = []
    for rank, (asin, score) in enumerate(scored[: int(top_k)], start=1):
        row = _empty_row()
        row.update(
            {
                "method": "svd",
                "svd_score": float(score),
                "score_sources": ["svd"],
            }
        )
        _decorate_row(row, rank=rank, asin=str(asin), meta_lookup=meta)
        rows.append(row)
    return rows


KNN_SIM_NAMES: tuple[str, ...] = ("cosine", "pearson", "msd")


def _default_knn_factory(sim_name: str):
    from src.models.cf import KNNRecommender

    return KNNRecommender(k=40, sim_name=sim_name, user_based=False)


def run_knn(
    *,
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    user_id: str,
    top_k: int,
    sim_name: str,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    seed_asin: str | None = None,
    knn_factory: Callable | None = None,
) -> list[dict]:
    """Fit Item-KNN with the requested similarity, rank unseen items by predicted rating."""
    if sim_name not in KNN_SIM_NAMES:
        raise ValueError(
            f"sim_name={sim_name!r} not in {KNN_SIM_NAMES}; "
            "use 'cosine', 'pearson', or 'msd'."
        )
    if train.empty:
        return []
    factory = knn_factory or _default_knn_factory
    seen = seen_items_for_user(train, user_id=user_id)
    pool = build_candidate_pool(
        train=train,
        metadata=metadata,
        seen=seen,
        pool_size=candidate_pool_size,
        seed_asin=seed_asin,
    )
    if not pool:
        return []

    model = factory(sim_name)
    model.fit(train)

    meta = _metadata_lookup(metadata)
    scored = [(asin, float(model.predict(user_id, asin))) for asin in pool]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    rows: list[dict] = []
    for rank, (asin, score) in enumerate(scored[: int(top_k)], start=1):
        row = _empty_row()
        row.update(
            {
                "method": f"knn_{sim_name}",
                "knn_score": float(score),
                "score_sources": ["knn"],
            }
        )
        _decorate_row(row, rank=rank, asin=str(asin), meta_lookup=meta)
        rows.append(row)
    return rows


@dataclass
class LightGCNResult:
    rows: list[dict]
    warning: str | None = None


def run_lightgcn_checkpoint(
    *,
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    user_id: str,
    top_k: int,
    checkpoint_path: Path,
    loader: Callable | None = None,
    config: dict | None = None,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    seed_asin: str | None = None,
) -> LightGCNResult:
    """Score unseen items with a LightGCN checkpoint.

    Returns a warning string instead of raising when the checkpoint is missing,
    the loader fails, or no pool item is known to the model. Callers display
    the warning in the Streamlit UI; never crash on missing artifacts.
    """
    if not Path(checkpoint_path).is_file():
        return LightGCNResult(
            rows=[],
            warning=(
                f"LightGCN checkpoint not found at {checkpoint_path}. Train one with "
                "`./.venv/bin/python -m src.evaluation.evaluate --graph-only "
                "--only-model lightgcn --train-only` (see README)."
            ),
        )
    if loader is None:
        from src.reasoning.candidates import load_lightgcn_from_checkpoint as _real_loader

        loader = _real_loader

    try:
        scorer = loader(Path(checkpoint_path), train, config or {})
    except Exception as exc:
        return LightGCNResult(rows=[], warning=f"Failed to load LightGCN checkpoint: {exc}")

    seen = seen_items_for_user(train, user_id=user_id)
    pool = build_candidate_pool(
        train=train,
        metadata=metadata,
        seen=seen,
        pool_size=candidate_pool_size,
        seed_asin=seed_asin,
    )
    if not pool:
        return LightGCNResult(rows=[], warning=None)

    meta = _metadata_lookup(metadata)
    scored: list[tuple[str, float]] = []
    for asin in pool:
        score = scorer(user_id, asin)
        if score is None:
            continue
        scored.append((asin, float(score)))

    if not scored:
        return LightGCNResult(
            rows=[],
            warning=(
                "LightGCN checkpoint loaded but no items in the candidate pool "
                "were known to the model."
            ),
        )

    scored.sort(key=lambda pair: pair[1], reverse=True)
    rows: list[dict] = []
    for rank, (asin, score) in enumerate(scored[: int(top_k)], start=1):
        row = _empty_row()
        row.update(
            {
                "method": "lightgcn",
                "lightgcn_score": float(score),
                "score_sources": ["lightgcn"],
            }
        )
        _decorate_row(row, rank=rank, asin=str(asin), meta_lookup=meta)
        rows.append(row)
    return LightGCNResult(rows=rows, warning=None)


LLM_EXTRA_FIELDS = (
    "semantic_source",
    "llm_summary",
    "llm_why",
    "llm_confidence",
    "validation_error",
)


@dataclass
class LLMResult:
    rows: list[dict]
    warning: str | None = None
    mode: str = "dry_run"
    raw_result: dict | None = None


def _llm_row_base() -> dict:
    base = _empty_row()
    for key in LLM_EXTRA_FIELDS:
        base[key] = None
    return base


def _llm_per_asin_explanation(parsed: dict | None) -> dict[str, dict]:
    if not parsed:
        return {}
    summary = str(parsed.get("summary") or "")
    out: dict[str, dict] = {}
    for rec in parsed.get("recommendations") or []:
        asin = str(rec.get("parent_asin", ""))
        if not asin:
            continue
        out[asin] = {
            "summary": summary,
            "why": str(rec.get("why") or ""),
            "confidence": str(rec.get("confidence") or ""),
        }
    return out


def _parse_explain_stdout(stdout: str) -> dict:
    """Extract the JSON payload from ``explain.py`` stdout.

    Native libraries (Torch, FAISS, Milvus Lite, OpenMP) can print warnings or
    runtime messages to stdout before the actual JSON. Try a strict parse first;
    if it fails, slice from the first ``{`` to the last ``}`` and parse that.
    """
    text = stdout.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError(
            f"explain.py JSON output is not a dict (got {type(payload).__name__})"
        )
    return payload


def _default_subprocess_runner(args, env, cwd):
    completed = subprocess.run(
        args,
        env=env,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def run_llm_hybrid(
    *,
    dataset: str,
    user_id: str,
    top_k: int,
    mode: str,
    query: str | None,
    dry_run: bool,
    metadata: pd.DataFrame,
    seen: set[str],
    subprocess_runner: Callable | None = None,
    project_root: Path | None = None,
    weights: str | None = None,
    config_path: str = "config/config.yaml",
) -> LLMResult:
    """Invoke ``src.reasoning.explain`` as a subprocess with --full-json and
    normalize its candidates into the shared row schema.

    Subprocess isolation: ``KMP_DUPLICATE_LIB_OK=TRUE`` is set so SVD +
    FAISS/Milvus Lite + Torch never share the Streamlit process's OpenMP
    runtime, which previously triggered duplicate-runtime crashes on macOS.
    """
    if mode not in ("profile", "query"):
        raise ValueError(f"mode={mode!r} must be 'profile' or 'query'")
    if mode == "query" and (query is None or not str(query).strip()):
        return LLMResult(
            rows=[],
            warning="Query mode requires a non-empty free-text query.",
            mode="dry_run" if dry_run else "live",
        )

    runner = subprocess_runner or _default_subprocess_runner
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]

    args: list[str] = [
        sys.executable,
        "-m",
        "src.reasoning.explain",
        "--dataset",
        str(dataset),
        "--user-id",
        str(user_id),
        "--top-k",
        str(int(top_k)),
        "--config",
        config_path,
        "--full-json",
    ]
    if mode == "query" and query:
        args += ["--query", query]
    if weights:
        args += ["--weights", weights]
    if dry_run:
        args.append("--dry-run")

    env = dict(os.environ)
    # Explicit (not setdefault) so the isolation guarantee holds even when the
    # parent process already has a differently-cased value such as 'True'.
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    try:
        rc, stdout, stderr = runner(args, env, root)
    except Exception as exc:
        return LLMResult(
            rows=[],
            warning=f"LLM hybrid subprocess failed to launch: {exc}",
            mode="dry_run" if dry_run else "live",
        )

    if rc != 0:
        return LLMResult(
            rows=[],
            warning=(stderr or f"explain.py exited with code {rc}").strip(),
            mode="dry_run" if dry_run else "live",
        )

    try:
        payload = _parse_explain_stdout(stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        return LLMResult(
            rows=[],
            warning=f"Could not parse explain.py JSON output: {exc}",
            mode="dry_run" if dry_run else "live",
        )

    method = f"llm_hybrid_{mode}"
    meta_lookup = _metadata_lookup(metadata)
    candidates = payload.get("candidates") or []
    evidence_payloads = payload.get("evidence_payloads") or []
    evidence_by_asin = {
        str(ev.get("candidate", {}).get("parent_asin", "")): ev
        for ev in evidence_payloads
    }
    parsed = (payload.get("llm") or {}).get("parsed_json")
    validation_error = (payload.get("llm") or {}).get("validation_error")
    explanation_by_asin = _llm_per_asin_explanation(parsed)

    rows: list[dict] = []
    for rank, cand in enumerate(candidates[: int(top_k)], start=1):
        asin = str(cand.get("parent_asin", ""))
        meta_entry = meta_lookup.get(asin)
        # Prefer real metadata; fall back to evidence_payload candidate fields.
        ev_candidate = evidence_by_asin.get(asin, {}).get("candidate", {})
        title = (meta_entry or {}).get("title") or ev_candidate.get("title") or cand.get("title", "")
        if meta_entry and meta_entry.get("categories"):
            categories = meta_entry["categories"]
        elif ev_candidate.get("categories"):
            categories = list(ev_candidate["categories"])
        else:
            categories = _normalize_categories(cand.get("categories"))

        row = _llm_row_base()
        row.update(
            {
                "rank": rank,
                "parent_asin": asin,
                "title": title,
                "categories": categories,
                "method": method,
                "hybrid_score": cand.get("hybrid_score"),
                "svd_score": cand.get("svd_score"),
                "lightgcn_score": cand.get("lightgcn_score"),
                "semantic_score": cand.get("semantic_score"),
                "popularity_score": cand.get("popularity_score"),
                "score_sources": list(cand.get("score_sources") or []),
                "already_seen": asin in seen,
                "semantic_source": payload.get("semantic_source"),
                "validation_error": validation_error,
            }
        )
        if asin in explanation_by_asin:
            row["llm_summary"] = explanation_by_asin[asin]["summary"]
            row["llm_why"] = explanation_by_asin[asin]["why"]
            row["llm_confidence"] = explanation_by_asin[asin]["confidence"]
        rows.append(row)

    return LLMResult(
        rows=rows,
        warning=None,
        mode=str(payload.get("mode") or ("dry_run" if dry_run else "live")),
        raw_result=payload,
    )

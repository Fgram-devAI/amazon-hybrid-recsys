"""CLI: explain hybrid recommendations using LightGCN + SVD + Milvus + Neo4j.

``run_explain`` is the unit-testable orchestrator; ``main`` is the argparse
entry point that wires real models, real stores, and the configured LLM
provider together.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.data.config import load_config
from src.reasoning.candidates import (
    LightGCNScorer,
    PopularityScorer,
    SemanticScorer,
    SVDScorer,
    generate_candidates,
    load_lightgcn_from_checkpoint,
    redistribute_weights,
)
from src.reasoning.evidence import (
    MilvusEvidenceFetcher,
    Neo4jEvidenceFetcher,
    assemble_evidence_payload,
)
from src.reasoning.llm_client import (
    ChatCompletionAdapter,
    LLMConfig,
    MissingApiKeyError,
    build_adapter_from_config,
    call_llm,
)
from src.reasoning.prompts import build_prompt
from src.reasoning.schemas import RecommendationEvidence
from src.storage.config import (
    load_storage_config,
    milvus_lite_path,
    neo4j_credentials,
    vector_collection_name,
)
from src.storage.artifacts import load_vector_artifacts
from src.storage.milvus_lite import MilvusLiteStore
from src.storage.neo4j_client import Neo4jStore


logger = logging.getLogger("reasoning.explain")


def _parse_weights_override(raw: str | None, defaults: dict[str, float]) -> dict[str, float]:
    if not raw:
        return dict(defaults)
    out: dict[str, float] = {}
    for pair in raw.split(","):
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            continue
        out[key] = float(value)
    return out


class _UnusedAdapter:
    """Placeholder so call_llm has an object even though it returns before using it."""

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
    ) -> str:  # pragma: no cover - dry-run only
        raise MissingApiKeyError("LLM adapter is not configured (dry-run path).")


def _candidate_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    by_asin = {
        ev["candidate"]["parent_asin"]: ev for ev in result.get("evidence_payloads", [])
    }
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(result.get("candidates", []), start=1):
        asin = str(row["parent_asin"])
        evidence = by_asin.get(asin, {})
        candidate = evidence.get("candidate", {})
        user_evidence = evidence.get("user_evidence", {})
        rows.append(
            {
                "rank": rank,
                "parent_asin": asin,
                "title": candidate.get("title", ""),
                "categories": candidate.get("categories", []),
                "hybrid_score": row.get("hybrid_score"),
                "raw_scores": {
                    "lightgcn": row.get("lightgcn_score"),
                    "svd": row.get("svd_score"),
                    "semantic": row.get("semantic_score"),
                    "popularity": row.get("popularity_score"),
                },
                "score_sources": row.get("score_sources", []),
                "category_overlap": user_evidence.get("category_overlap", []),
            }
        )
    return rows


def _user_profile_summary(result: dict[str, Any]) -> dict[str, Any]:
    evidence = result.get("evidence_payloads") or []
    if not evidence:
        return {"high_rated_items": [], "graph_evidence_available": False}
    first = evidence[0].get("user_evidence", {})
    return {
        "graph_evidence_available": first.get("graph_evidence_available", False),
        "high_rated_items": first.get("high_rated_items", []),
    }


def _display_payload(result: dict[str, Any], *, full_json: bool = False) -> dict[str, Any]:
    """Return the JSON payload printed to stdout."""
    if full_json:
        return result
    payload = {
        "mode": result["mode"],
        "effective_weights": result["effective_weights"],
        "semantic_source": result.get("semantic_source"),
        "candidates": _candidate_summary(result),
        "user_profile": _user_profile_summary(result),
        "llm": result["llm"],
    }
    if result["mode"] == "dry_run":
        payload["inspection_note"] = (
            "Compact dry-run view. Use --full-json to print prompt/evidence, "
            "or --output to save the full payload."
        )
    return payload


def _build_user_profile_query_vector(
    *,
    train: pd.DataFrame,
    processed_dir: Path,
    dataset: str,
    embedding_subdir: str,
    user_id: str,
    min_rating: float = 4.0,
) -> list[float] | None:
    user_rows = train.loc[
        (train["user_id"].astype(str) == str(user_id))
        & (train["rating"].astype(float) >= float(min_rating)),
        "parent_asin",
    ]
    liked_ids = set(user_rows.astype(str).tolist())
    if not liked_ids:
        return None

    artifacts = load_vector_artifacts(
        processed_dir=processed_dir,
        dataset_key=dataset,
        embedding_subdir=embedding_subdir,
    )
    index = {asin: i for i, asin in enumerate(artifacts.item_ids)}
    rows = [index[asin] for asin in liked_ids if asin in index]
    if not rows:
        return None

    profile = artifacts.embeddings[rows].mean(axis=0).astype("float32")
    norm = float(np.linalg.norm(profile))
    if norm <= 0:
        return None
    return (profile / norm).tolist()


def run_explain(
    *,
    user_id: str,
    query: str | None,
    candidate_k: int,
    final_k: int,
    weights: dict[str, float],
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    embedder: Any | None,
    profile_query_vector: list[float] | None,
    milvus_store: Any | None,
    milvus_collection: str,
    neo4j_store: Any | None,
    svd_scorer: SVDScorer | None,
    lightgcn_scorer: LightGCNScorer | None,
    llm_config: LLMConfig | None,
    dry_run: bool,
    output_path: Path | None,
) -> dict[str, Any]:
    """Pure-Python orchestration; all I/O dependencies are injected."""

    # Semantic neighbors (requires both an embedder and a Milvus collection).
    semantic_scorer: SemanticScorer | None = None
    semantic_hits: list[dict[str, Any]] = []
    semantic_source: str | None = None
    if milvus_store is not None and (embedder is not None or profile_query_vector is not None):
        fetcher = MilvusEvidenceFetcher(
            store=milvus_store, collection_name=milvus_collection
        )
        text = query or ""
        if text:
            if embedder is not None:
                query_vec = embedder.encode([text])[0].astype("float32").tolist()
                semantic_hits = fetcher.semantic_neighbors(query_vector=query_vec, top_k=candidate_k)
                semantic_source = "free_text_query"
        elif profile_query_vector is not None:
            semantic_hits = fetcher.semantic_neighbors(
                query_vector=profile_query_vector, top_k=candidate_k
            )
            semantic_source = "user_high_rated_item_profile"
        if semantic_hits:
            semantic_scorer = SemanticScorer(semantic_hits)

    # Candidate pool: semantic hits first, then popular train items, then metadata
    # fallback. Order matters because we cap the pool before scoring; semantic
    # query hits must not be appended after large metadata lists and truncated away.
    semantic_asins = [str(h["parent_asin"]) for h in semantic_hits if h.get("parent_asin")]
    seen_by_user = set(
        train.loc[train["user_id"].astype(str) == str(user_id), "parent_asin"]
        .astype(str)
        .tolist()
    )
    popularity_asins = (
        train.loc[~train["parent_asin"].astype(str).isin(seen_by_user), "parent_asin"]
        .astype(str)
        .value_counts()
        .head(candidate_k * 4)
        .index
        .tolist()
    )
    metadata_asins = metadata["parent_asin"].astype(str).head(candidate_k * 4).tolist()
    pool = list(dict.fromkeys(semantic_asins + popularity_asins + metadata_asins))
    if not pool:
        pool = list(metadata["parent_asin"].astype(str).tolist())
    pool = pool[: max(candidate_k * 4, candidate_k)]

    popularity_scorer = PopularityScorer(train)

    candidates = generate_candidates(
        user_id=user_id,
        candidate_pool=pool,
        train=train,
        svd_scorer=svd_scorer,
        lightgcn_scorer=lightgcn_scorer,
        semantic_scorer=semantic_scorer,
        popularity_scorer=popularity_scorer,
        weights=weights,
        final_k=final_k,
    )

    # Graph evidence (optional, with graceful degradation if first query fails).
    graph_available = neo4j_store is not None
    user_history: list[dict[str, Any]] = []
    candidate_categories: dict[str, list[str]] = {}
    if neo4j_store is not None:
        try:
            n4 = Neo4jEvidenceFetcher(store=neo4j_store)
            user_history = n4.user_history(user_id, top_k=10)
            candidate_categories = n4.categories_for(
                [c["parent_asin"] for c in candidates]
            )
        except Exception as exc:
            logger.warning("[reasoning] Neo4j evidence unavailable during query: %s", exc)
            graph_available = False
            user_history = []
            candidate_categories = {}

    evidence_payloads: list[RecommendationEvidence] = assemble_evidence_payload(
        candidates=candidates,
        metadata=metadata,
        user_history=user_history,
        semantic_neighbors=semantic_hits,
        candidate_categories=candidate_categories,
        graph_evidence_available=graph_available,
    )

    available = {
        name
        for name in ("graph", "svd", "semantic", "popularity")
        if (
            (name == "graph" and lightgcn_scorer is not None)
            or (name == "svd" and svd_scorer is not None)
            or (name == "semantic" and semantic_scorer is not None)
            or (name == "popularity")
        )
    }
    effective_weights = redistribute_weights(weights, available=available)

    prompt = build_prompt(
        user_id=user_id,
        query=query,
        semantic_source=semantic_source,
        evidence_payloads=evidence_payloads,
        effective_weights=effective_weights,
    )

    # LLM call (optional).
    llm_result: dict[str, Any] = {"mode": "skipped"}
    if llm_config is not None:
        adapter: ChatCompletionAdapter
        if dry_run:
            adapter = cast(ChatCompletionAdapter, _UnusedAdapter())
        else:
            adapter = build_adapter_from_config(llm_config)
        llm_result = call_llm(
            adapter=adapter,
            prompt=prompt,
            config=llm_config,
            dry_run=dry_run,
        )

    # Serialize evidence to JSON-ready dicts for downstream callers / file write.
    evidence_dumps = [ev.model_dump(mode="json") for ev in evidence_payloads]

    out: dict[str, Any] = {
        "mode": "dry_run" if dry_run else llm_result.get("mode", "live"),
        "user_id": user_id,
        "query": query,
        "semantic_source": semantic_source,
        "effective_weights": effective_weights,
        "candidates": candidates,
        "evidence_payloads": evidence_dumps,
        "prompt": prompt,
        "llm": llm_result,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, default=str, indent=2))

    return out


# --------------------------- argparse CLI ---------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain hybrid recommendations.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--query", default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--top-k", dest="final_k", type=int, default=None)
    parser.add_argument("--weights", default=None, help="graph=0.5,svd=0.25,...")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="Print the full prompt/evidence payload instead of the compact summary.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    args = _parse_args(argv)
    config = load_config(args.config)
    rcfg = config.get("reasoning", {})
    dataset = args.dataset
    candidate_k = int(args.candidate_k or rcfg.get("candidate_k", 50))
    final_k = int(args.final_k or rcfg.get("final_k", 5))
    weights = _parse_weights_override(args.weights, rcfg.get("weights", {}))

    processed = Path(config["processed_dir"]) / dataset
    train = pd.read_parquet(processed / "train.parquet")
    metadata = pd.read_parquet(processed / "metadata.parquet")
    logger.info("[reasoning] loaded train rows=%d items=%d", len(train), len(metadata))

    # ---- Milvus ----
    milvus_store: MilvusLiteStore | None = None
    milvus_collection = vector_collection_name(config, dataset)
    try:
        milvus_store = MilvusLiteStore(milvus_lite_path(config))
        if milvus_store.has_collection(milvus_collection):
            logger.info("[reasoning] Milvus collection found: %s", milvus_collection)
        else:
            logger.warning(
                "[reasoning] Milvus collection missing (%s). Run: "
                "./.venv/bin/python -m src.storage.ingest_vectors --dataset %s --reset",
                milvus_collection, dataset,
            )
            milvus_store.close()
            milvus_store = None
    except Exception as exc:
        logger.warning("[reasoning] Milvus unavailable: %s", exc)
        milvus_store = None

    # ---- Semantic profile/query vector ----
    embedder = None
    profile_query_vector: list[float] | None = None
    if milvus_store is not None and args.query:
        try:
            from src.models.embedding import build_embedder

            embedder = build_embedder(config)
        except Exception as exc:
            logger.warning("[reasoning] embedder unavailable: %s", exc)
    elif milvus_store is not None:
        try:
            storage_cfg = load_storage_config(config)
            profile_query_vector = _build_user_profile_query_vector(
                train=train,
                processed_dir=Path(config["processed_dir"]),
                dataset=dataset,
                embedding_subdir=str(storage_cfg["vector_embedding_dir"]),
                user_id=args.user_id,
                min_rating=float(config.get("graph", {}).get("min_rating_positive", 4.0)),
            )
            if profile_query_vector is not None:
                logger.info(
                    "[reasoning] semantic profile vector built from high-rated train items"
                )
            else:
                logger.info("[reasoning] semantic profile vector unavailable for user")
        except Exception as exc:
            logger.warning("[reasoning] semantic profile vector unavailable: %s", exc)

    # ---- Neo4j ----
    neo4j_store: Neo4jStore | None = None
    try:
        creds = neo4j_credentials(config)
        neo4j_store = Neo4jStore(creds)
        logger.info("[reasoning] Neo4j available: %s", creds.uri)
    except Exception as exc:
        logger.warning("[reasoning] Neo4j unavailable: %s", exc)
        neo4j_store = None

    # ---- LightGCN ----
    lightgcn_scorer = None
    ckpt = rcfg.get("lightgcn_checkpoint")
    if ckpt:
        ckpt_path = Path(ckpt)
        if ckpt_path.exists():
            try:
                lightgcn_scorer = load_lightgcn_from_checkpoint(ckpt_path, train, config)
                logger.info("[reasoning] LightGCN checkpoint loaded: %s", ckpt_path)
            except Exception as exc:
                logger.warning("[reasoning] LightGCN load failed: %s", exc)
        else:
            logger.warning("[reasoning] LightGCN checkpoint missing: %s", ckpt_path)

    # ---- SVD ----
    svd_scorer = None
    try:
        from src.models.cf import SVDRecommender

        svd = SVDRecommender(n_factors=64, n_epochs=10, random_state=42)
        svd.fit(train)
        svd_scorer = SVDScorer(svd)
        logger.info("[reasoning] SVD fitted on %d rows", len(train))
    except Exception as exc:
        logger.warning("[reasoning] SVD unavailable: %s", exc)

    # ---- LLM config ----
    llm_block = config.get("llm")
    llm_config: LLMConfig | None = None
    if llm_block:
        api_key = os.environ.get(llm_block.get("api_key_env", "GROQ_API_KEY"))
        llm_config = LLMConfig(
            provider=str(llm_block.get("provider", "groq")),
            model=str(llm_block.get("model", "llama-3.3-70b-versatile")),
            api_key=api_key,
            base_url=str(llm_block.get("base_url", "https://api.groq.com/openai/v1")),
            temperature=float(llm_block.get("temperature", 0.2)),
            max_tokens=int(llm_block.get("max_tokens", 700)),
            timeout_seconds=int(llm_block.get("timeout_seconds", 30)),
        )
        if args.dry_run:
            logger.info("[reasoning] dry-run: no LLM call")
        elif api_key:
            logger.info("[reasoning] LLM provider %s ready", llm_config.provider)
        else:
            logger.warning(
                "[reasoning] LLM provider %s requires env %s — running dry-run instead",
                llm_config.provider, llm_block.get("api_key_env", "GROQ_API_KEY"),
            )

    output_path = Path(args.output) if args.output else None

    try:
        result = run_explain(
            user_id=args.user_id,
            query=args.query,
            candidate_k=candidate_k,
            final_k=final_k,
            weights=weights,
            train=train,
            metadata=metadata,
            embedder=embedder,
            profile_query_vector=profile_query_vector,
            milvus_store=milvus_store,
            milvus_collection=milvus_collection,
            neo4j_store=neo4j_store,
            svd_scorer=svd_scorer,
            lightgcn_scorer=lightgcn_scorer,
            llm_config=llm_config,
            dry_run=bool(args.dry_run or (llm_config is not None and not llm_config.api_key)),
            output_path=output_path,
        )
    finally:
        if milvus_store is not None:
            milvus_store.close()
        if neo4j_store is not None:
            neo4j_store.close()

    print(json.dumps(_display_payload(result, full_json=bool(args.full_json)), default=str, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

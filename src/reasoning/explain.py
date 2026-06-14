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
    milvus_lite_path,
    neo4j_credentials,
    vector_collection_name,
)
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
    if milvus_store is not None and embedder is not None:
        fetcher = MilvusEvidenceFetcher(
            store=milvus_store, collection_name=milvus_collection
        )
        text = query or ""
        if text:
            query_vec = embedder.encode([text])[0].astype("float32").tolist()
            semantic_hits = fetcher.semantic_neighbors(query_vector=query_vec, top_k=candidate_k)
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

    # ---- Embedder ----
    embedder = None
    if milvus_store is not None and args.query:
        try:
            from src.models.embedding import build_embedder

            embedder = build_embedder(config)
        except Exception as exc:
            logger.warning("[reasoning] embedder unavailable: %s", exc)

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

    print(json.dumps(
        {
            "mode": result["mode"],
            "effective_weights": result["effective_weights"],
            "candidates": result["candidates"],
            "llm": result["llm"],
        },
        default=str,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())

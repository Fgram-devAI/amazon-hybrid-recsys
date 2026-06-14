"""CLI: ingest train.parquet + metadata.parquet (+ optional co-rating report) into Neo4j.

Example:

    ./.venv/bin/python -m src.storage.ingest_neo4j --dataset video_games --reset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.data.config import load_config
from src.storage.config import load_storage_config, neo4j_credentials
from src.storage.neo4j_client import Neo4jStore
from src.storage.neo4j_payloads import (
    build_co_rating_rows,
    build_item_category_rows,
    build_item_rows,
    build_rating_rows,
)

logger = logging.getLogger("storage.ingest_neo4j")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest train graph into Neo4j.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="MATCH (n) DETACH DELETE n before ingestion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap train rating rows ingested (smoke runs).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    args = _parse_args(argv)

    config = load_config(args.config)
    storage = load_storage_config(config)
    batch_size = int(storage["neo4j_batch_size"])
    processed_dir = Path(config["processed_dir"])
    dataset_dir = processed_dir / args.dataset

    train = pd.read_parquet(dataset_dir / "train.parquet")
    metadata = pd.read_parquet(dataset_dir / "metadata.parquet")
    logger.info(
        "Loaded %d train rows and %d metadata rows for %s",
        len(train),
        len(metadata),
        args.dataset,
    )

    generic_roots = list(config["advanced_features"]["generic_category_roots"])
    item_rows = build_item_rows(metadata)
    category_rows = build_item_category_rows(metadata, generic_roots=generic_roots)
    rating_rows = build_rating_rows(train, limit=args.limit)

    co_cfg = storage["co_rating_edges"]
    co_rows: list[dict] = []
    if co_cfg["enabled"]:
        report_path = dataset_dir / co_cfg["source_report"]
        co_rows = build_co_rating_rows(report_path, max_edges=int(co_cfg["max_edges"]))
        if not co_rows:
            logger.warning(
                "Co-rating edges enabled but no exportable edges found at %s; skipping",
                report_path,
            )

    credentials = neo4j_credentials(config)
    with Neo4jStore(credentials) as store:
        if args.reset:
            store.reset_database()
            logger.info("Reset database (DETACH DELETE on all nodes)")
        store.ensure_constraints()
        logger.info("Ensured uniqueness constraints")

        n_items = store.upsert_items(item_rows, batch_size=batch_size)
        logger.info("Upserted %d Item nodes", n_items)

        n_cats = store.upsert_categories_and_links(category_rows, batch_size=batch_size)
        logger.info("Upserted %d IN_CATEGORY edges", n_cats)

        n_ratings = store.upsert_users_and_ratings(rating_rows, batch_size=batch_size)
        logger.info("Upserted %d RATED edges (train-only)", n_ratings)

        if co_rows:
            n_co = store.upsert_co_rated(co_rows, batch_size=batch_size)
            logger.info("Upserted %d CO_RATED_WITH edges", n_co)

    return 0


if __name__ == "__main__":
    sys.exit(main())

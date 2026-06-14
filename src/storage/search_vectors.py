"""CLI: encode a query string and run a top-K semantic search on Milvus Lite.

Example:

    ./.venv/bin/python -m src.storage.search_vectors \
        --dataset video_games \
        --query "open world fantasy role playing game" \
        --top-k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.data.config import load_config
from src.models.embedding import build_embedder
from src.storage.config import milvus_lite_path, vector_collection_name
from src.storage.milvus_lite import MilvusLiteStore

logger = logging.getLogger("storage.search_vectors")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-K semantic search on Milvus Lite.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--query", required=True, help="Free-text query.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    config = load_config(args.config)
    collection = vector_collection_name(config, args.dataset)

    embedder = build_embedder(config)
    query_vec = embedder.encode([args.query])[0].astype("float32").tolist()
    logger.info("Encoded query (dim=%d) with %s", len(query_vec), embedder.name)

    store = MilvusLiteStore(milvus_lite_path(config))
    try:
        if not store.has_collection(collection):
            raise SystemExit(
                f"Collection {collection} not found. Run ingest_vectors --reset first."
            )
        hits = store.search(
            collection,
            query_vec,
            top_k=args.top_k,
            output_fields=[
                "id",
                "parent_asin",
                "title",
                "categories",
                "store",
                "price",
                "average_rating",
                "rating_number",
            ],
        )
    finally:
        store.close()

    print(json.dumps(hits, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

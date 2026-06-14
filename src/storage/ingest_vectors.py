"""CLI: load item embeddings + metadata and ingest them into Milvus Lite.

Example:

    ./.venv/bin/python -m src.storage.ingest_vectors --dataset video_games --reset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.config import load_config
from src.storage.artifacts import build_vector_payload, load_vector_artifacts
from src.storage.config import load_storage_config, milvus_lite_path, vector_collection_name
from src.storage.milvus_lite import MilvusLiteStore

logger = logging.getLogger("storage.ingest_vectors")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest item embeddings into Milvus Lite.")
    parser.add_argument("--dataset", required=True, help="Dataset key (e.g. video_games).")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection before inserting.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Rows per Milvus insert batch."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    config = load_config(args.config)
    storage = load_storage_config(config)
    processed_dir = Path(config["processed_dir"])
    dataset_key = args.dataset

    artifacts = load_vector_artifacts(
        processed_dir=processed_dir,
        dataset_key=dataset_key,
        embedding_subdir=storage["vector_embedding_dir"],
    )
    logger.info(
        "Loaded %d embeddings (dim=%d) from %s",
        len(artifacts.item_ids),
        artifacts.dim,
        artifacts.source_path,
    )

    metadata_path = processed_dir / dataset_key / "metadata.parquet"
    metadata = (
        pd.read_parquet(metadata_path)
        if metadata_path.exists()
        else pd.DataFrame(columns=["parent_asin"])
    )
    logger.info("Loaded %d metadata rows from %s", len(metadata), metadata_path)

    rows = build_vector_payload(artifacts.item_ids, artifacts.embeddings, metadata)

    collection = vector_collection_name(config, dataset_key)
    store = MilvusLiteStore(milvus_lite_path(config))
    try:
        if args.reset:
            store.drop_collection(collection)
        if not store.has_collection(collection):
            store.create_collection(collection, dim=artifacts.dim)
            logger.info("Created collection %s (dim=%d)", collection, artifacts.dim)
        inserted = store.insert_rows(collection, rows, batch_size=args.batch_size)
        logger.info("Inserted %d vectors into %s", inserted, collection)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

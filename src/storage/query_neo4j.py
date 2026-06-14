"""CLI: smoke query Neo4j for a known user — top items + category neighbors.

Example:

    ./.venv/bin/python -m src.storage.query_neo4j \\
        --dataset video_games \\
        --user-id AHKLZ... \\
        --top-k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

from src.data.config import load_config
from src.storage.config import neo4j_credentials
from src.storage.neo4j_client import Neo4jStore

logger = logging.getLogger("storage.query_neo4j")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke graph queries against Neo4j.")
    parser.add_argument("--dataset", required=True, help="Dataset key (for logging only).")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    args = _parse_args(argv)

    config = load_config(args.config)
    credentials = neo4j_credentials(config)

    with Neo4jStore(credentials) as store:
        rows = store.fetch_top_items_for_user(args.user_id, top_k=args.top_k)
    logger.info(
        "Fetched %d items for user_id=%s (dataset=%s)",
        len(rows),
        args.user_id,
        args.dataset,
    )
    print(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

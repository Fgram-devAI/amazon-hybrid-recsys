"""Helpers for the storage layer config block (see config/config.yaml#storage)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Neo4jCredentials:
    uri: str
    user: str
    password: str


def load_storage_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the ``storage`` sub-dict. Raises KeyError if the block is missing."""
    return config["storage"]


def milvus_lite_path(config: dict[str, Any]) -> Path:
    """Resolve the Milvus Lite DB file path relative to the project root."""
    raw = load_storage_config(config)["milvus_lite_path"]
    return Path(raw).expanduser().resolve()


def vector_collection_name(config: dict[str, Any], dataset_key: str) -> str:
    """Compose the per-dataset Milvus collection name."""
    prefix = load_storage_config(config)["vector_collection_prefix"]
    return f"{prefix}_{dataset_key}_items"


def neo4j_credentials(config: dict[str, Any]) -> Neo4jCredentials:
    """Build Neo4j connection credentials, reading the password from the env."""
    block = load_storage_config(config)
    env_var = block["neo4j_password_env"]
    password = os.environ.get(env_var)
    if not password:
        raise RuntimeError(
            f"Environment variable {env_var} is not set; copy .env.example to .env "
            "and set the Neo4j password."
        )
    return Neo4jCredentials(uri=block["neo4j_uri"], user=block["neo4j_user"], password=password)

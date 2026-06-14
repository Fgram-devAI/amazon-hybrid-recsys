"""Thin wrapper around pymilvus.MilvusClient for the local Milvus Lite file backend."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pymilvus import DataType, MilvusClient


class MilvusLiteStore:
    """Manage a single Milvus Lite collection of item vectors with metadata payload."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = MilvusClient(uri=str(db_path))

    def has_collection(self, name: str) -> bool:
        return self._client.has_collection(collection_name=name)

    def drop_collection(self, name: str) -> None:
        if self.has_collection(name):
            self._client.drop_collection(collection_name=name)

    def create_collection(self, name: str, dim: int) -> None:
        """Create a collection with a fixed schema: id PK + scalar fields + vector."""
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("parent_asin", DataType.VARCHAR, max_length=64)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("categories", DataType.VARCHAR, max_length=1024)
        schema.add_field("store", DataType.VARCHAR, max_length=256, nullable=True)
        schema.add_field("price", DataType.FLOAT, nullable=True)
        schema.add_field("average_rating", DataType.FLOAT, nullable=True)
        schema.add_field("rating_number", DataType.INT64, nullable=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)

        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

        self._client.create_collection(
            collection_name=name, schema=schema, index_params=index_params
        )

    def insert_rows(self, name: str, rows: Iterable[dict[str, Any]], *, batch_size: int = 1000) -> int:
        """Insert rows in batches; return total inserted count."""
        rows = list(rows)
        total = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            self._client.insert(collection_name=name, data=batch)
            total += len(batch)
        return total

    def search(
        self,
        name: str,
        query_vector: list[float],
        top_k: int,
        output_fields: list[str],
    ) -> list[dict[str, Any]]:
        """Top-K cosine search, returning the requested metadata fields per hit."""
        result = self._client.search(
            collection_name=name,
            data=[query_vector],
            limit=top_k,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE"},
        )
        hits = result[0] if result else []
        return [dict(hit) for hit in hits]

    def close(self) -> None:
        self._client.close()

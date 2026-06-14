"""Thin wrapper around the official neo4j driver: constraints + batched UNWIND writes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, LiteralString

from neo4j import GraphDatabase

from src.storage.config import Neo4jCredentials

_CONSTRAINT_STATEMENTS = (
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
    "CREATE CONSTRAINT item_parent_asin IF NOT EXISTS FOR (i:Item) REQUIRE i.parent_asin IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
)


class Neo4jStore:
    """Connection + helpers for the train-only user-item-category graph."""

    def __init__(self, credentials: Neo4jCredentials):
        self._driver = GraphDatabase.driver(
            credentials.uri, auth=(credentials.user, credentials.password)
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def ensure_constraints(self) -> None:
        with self._driver.session() as session:
            for stmt in _CONSTRAINT_STATEMENTS:
                session.run(stmt).consume()

    def reset_database(self, *, batch_size: int = 500) -> None:
        with self._driver.session() as session:
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT $batch_size "
                    "DETACH DELETE n RETURN count(n) AS deleted",
                    batch_size=batch_size,
                )
                deleted = result.single(strict=True)["deleted"]
                if deleted == 0:
                    break

    def _write_batches(
        self, query: LiteralString, rows: Iterable[dict[str, Any]], batch_size: int
    ) -> int:
        rows = list(rows)
        total = 0
        with self._driver.session() as session:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                session.run(query, rows=batch).consume()
                total += len(batch)
        return total

    def upsert_items(self, rows: list[dict[str, Any]], *, batch_size: int) -> int:
        query = (
            "UNWIND $rows AS row "
            "MERGE (i:Item {parent_asin: row.parent_asin}) "
            "SET i.title = row.title, i.store = row.store, "
            "    i.price = row.price, i.average_rating = row.average_rating, "
            "    i.rating_number = row.rating_number"
        )
        return self._write_batches(query, rows, batch_size)

    def upsert_categories_and_links(
        self, rows: list[dict[str, Any]], *, batch_size: int
    ) -> int:
        query = (
            "UNWIND $rows AS row "
            "MERGE (c:Category {name: row.category}) "
            "WITH c, row "
            "MATCH (i:Item {parent_asin: row.parent_asin}) "
            "MERGE (i)-[:IN_CATEGORY]->(c)"
        )
        return self._write_batches(query, rows, batch_size)

    def upsert_users_and_ratings(
        self, rows: list[dict[str, Any]], *, batch_size: int
    ) -> int:
        query = (
            "UNWIND $rows AS row "
            "MERGE (u:User {user_id: row.user_id}) "
            "WITH u, row "
            "MATCH (i:Item {parent_asin: row.parent_asin}) "
            "MERGE (u)-[r:RATED]->(i) "
            "SET r.rating = row.rating, r.timestamp = row.timestamp, r.split = 'train'"
        )
        return self._write_batches(query, rows, batch_size)

    def upsert_co_rated(self, rows: list[dict[str, Any]], *, batch_size: int) -> int:
        query = (
            "UNWIND $rows AS row "
            "MATCH (a:Item {parent_asin: row.a}) "
            "MATCH (b:Item {parent_asin: row.b}) "
            "MERGE (a)-[e:CO_RATED_WITH]->(b) "
            "SET e.weight_count = row.weight_count, e.weight_jaccard = row.weight_jaccard"
        )
        return self._write_batches(query, rows, batch_size)

    def fetch_top_items_for_user(self, user_id: str, top_k: int) -> list[dict[str, Any]]:
        query = (
            "MATCH (u:User {user_id: $user_id})-[r:RATED]->(i:Item) "
            "OPTIONAL MATCH (i)-[:IN_CATEGORY]->(c:Category) "
            "RETURN i.parent_asin AS parent_asin, i.title AS title, r.rating AS rating, "
            "       collect(DISTINCT c.name) AS categories "
            "ORDER BY r.rating DESC LIMIT $top_k"
        )
        with self._driver.session() as session:
            result = session.run(query, user_id=user_id, top_k=top_k)
            return [dict(record) for record in result]

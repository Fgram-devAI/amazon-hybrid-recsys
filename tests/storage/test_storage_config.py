"""Tests for src.storage.config helpers."""

from __future__ import annotations

import pytest

from src.data.config import load_config
from src.storage import config as storage_config


def test_load_storage_config_returns_storage_block():
    cfg = load_config("config/config.yaml")
    block = storage_config.load_storage_config(cfg)
    assert block["vector_backend"] == "milvus_lite"
    assert block["milvus_lite_path"] == "data/storage/milvus_assignment.db"
    assert block["vector_collection_prefix"] == "amazon"
    assert block["vector_embedding_dir"] == "advanced_features/title_desc_embeddings"
    assert block["neo4j_uri"] == "bolt://localhost:7687"
    assert block["neo4j_user"] == "neo4j"
    assert block["neo4j_password_env"] == "NEO4J_PASSWORD"
    assert block["neo4j_batch_size"] == 5000
    assert block["co_rating_edges"]["enabled"] is False
    assert block["co_rating_edges"]["max_edges"] == 50000


def test_milvus_lite_path_returns_absolute_path(tmp_path):
    cfg = {"storage": {"milvus_lite_path": "data/storage/foo.db"}}
    path = storage_config.milvus_lite_path(cfg)
    assert path.name == "foo.db"
    assert path.parent.name == "storage"


def test_vector_collection_name_uses_prefix_and_dataset_key():
    cfg = {"storage": {"vector_collection_prefix": "amazon"}}
    assert storage_config.vector_collection_name(cfg, "video_games") == "amazon_video_games_items"


def test_neo4j_credentials_reads_password_from_env(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "s3cret")
    cfg = {
        "storage": {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password_env": "NEO4J_PASSWORD",
        }
    }
    creds = storage_config.neo4j_credentials(cfg)
    assert creds.uri == "bolt://localhost:7687"
    assert creds.user == "neo4j"
    assert creds.password == "s3cret"


def test_neo4j_credentials_raises_when_env_unset(monkeypatch):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    cfg = {
        "storage": {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password_env": "NEO4J_PASSWORD",
        }
    }
    with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
        storage_config.neo4j_credentials(cfg)

"""Tests for the embedder interface and on-disk cache (FakeEmbedder only)."""

import json

import numpy as np
import pandas as pd

from src.models.embedding import (
    FakeEmbedder,
    load_or_compute_item_embeddings,
)


def test_fake_embedder_is_deterministic_and_reflects_word_overlap():
    emb = FakeEmbedder(dim=32)
    a, b, c = emb.encode(["rpg dragon quest", "rpg dragon hero", "cooking pasta recipe"])
    # deterministic
    assert np.array_equal(a, emb.encode(["rpg dragon quest"])[0])

    def cos(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    # shared words -> higher cosine than unrelated text
    assert cos(a, b) > cos(a, c)


def test_cache_writes_emb_ids_and_meta_then_reloads(tmp_path):
    meta_df = pd.DataFrame(
        {"parent_asin": ["A1", "A2"], "text": ["hello world", "foo bar"]}
    )
    embedder = FakeEmbedder(dim=8)
    cache = tmp_path / "embeddings"

    emb1, ids1 = load_or_compute_item_embeddings(meta_df, embedder, cache, content_hash="h1")

    assert emb1.shape == (2, 8)
    assert ids1 == ["A1", "A2"]
    assert (cache / "item_emb.npy").exists()
    assert json.loads((cache / "item_ids.json").read_text()) == ["A1", "A2"]
    meta = json.loads((cache / "embedding_meta.json").read_text())
    assert meta["model_name"] == "fake"
    assert meta["dim"] == 8
    assert meta["metadata_row_count"] == 2
    assert meta["content_hash"] == "h1"
    assert meta["version"] == 1

    # second call with same hash reloads from disk (identical array)
    emb2, ids2 = load_or_compute_item_embeddings(meta_df, embedder, cache, content_hash="h1")
    assert np.array_equal(emb1, emb2) and ids1 == ids2


def test_cache_recomputes_when_content_hash_changes(tmp_path):
    cache = tmp_path / "embeddings"
    embedder = FakeEmbedder(dim=8)

    m1 = pd.DataFrame({"parent_asin": ["A1"], "text": ["hello world"]})
    emb1, _ = load_or_compute_item_embeddings(m1, embedder, cache, content_hash="h1")

    # different content + different hash must RECOMPUTE, not serve the stale array
    m2 = pd.DataFrame({"parent_asin": ["A1"], "text": ["totally different text"]})
    emb2, _ = load_or_compute_item_embeddings(m2, embedder, cache, content_hash="h2")

    assert not np.array_equal(emb1, emb2)
    meta = json.loads((cache / "embedding_meta.json").read_text())
    assert meta["content_hash"] == "h2"

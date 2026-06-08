"""Tests for standalone embedding materialization/preview helpers."""

import json

import numpy as np
import pandas as pd

from src.models.embed import build_embeddings, write_embedding_preview


def test_write_embedding_preview_creates_small_csv(tmp_path):
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    item_ids = ["A1", "A2", "A3"]

    out_path = write_embedding_preview(
        embeddings,
        item_ids,
        tmp_path,
        rows=2,
        dims=3,
    )

    preview = pd.read_csv(out_path)
    assert list(preview.columns) == ["parent_asin", "dim_0", "dim_1", "dim_2"]
    assert preview["parent_asin"].tolist() == ["A1", "A2"]


def test_build_embeddings_uses_processed_metadata_and_cache(tmp_path):
    processed = tmp_path / "processed"
    dataset_dir = processed / "tiny"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "parent_asin": ["A1", "A2"],
            "text": ["rpg dragon", "cooking pasta"],
        }
    ).to_parquet(dataset_dir / "metadata.parquet", index=False)

    config = {
        "processed_dir": str(processed),
        "models": {
            "embedding_model": "unused",
            "embedding_fallback": "unused",
        },
    }

    from src.models import embed as embed_module
    from src.models.embedding import FakeEmbedder

    original = embed_module.build_embedder
    embed_module.build_embedder = lambda _config: FakeEmbedder(dim=8)
    try:
        embeddings, item_ids, cache_dir = build_embeddings(config, "tiny")
    finally:
        embed_module.build_embedder = original

    assert embeddings.shape == (2, 8)
    assert item_ids == ["A1", "A2"]
    assert cache_dir == dataset_dir / "embeddings"
    meta = json.loads((cache_dir / "embedding_meta.json").read_text())
    assert meta["model_name"] == "fake"

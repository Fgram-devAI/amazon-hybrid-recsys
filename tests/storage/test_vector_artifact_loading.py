"""Tests for src.storage.artifacts.load_vector_artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.storage.artifacts import load_vector_artifacts


def _write_artifacts(dir_path: Path, n_items: int, dim: int) -> list[str]:
    dir_path.mkdir(parents=True, exist_ok=True)
    item_ids = [f"asin_{i:04d}" for i in range(n_items)]
    embeddings = np.random.RandomState(0).standard_normal((n_items, dim)).astype("float32")
    np.save(dir_path / "item_emb.npy", embeddings)
    (dir_path / "item_ids.json").write_text(json.dumps(item_ids))
    (dir_path / "embedding_meta.json").write_text(
        json.dumps({"dim": dim, "model_name": "fake-embedder", "row_count": n_items})
    )
    return item_ids


def test_load_vector_artifacts_prefers_advanced_features_path(tmp_path: Path):
    advanced = tmp_path / "video_games" / "advanced_features" / "title_desc_embeddings"
    _write_artifacts(advanced, n_items=5, dim=8)
    fallback = tmp_path / "video_games" / "embeddings"
    _write_artifacts(fallback, n_items=3, dim=4)

    arts = load_vector_artifacts(
        processed_dir=tmp_path,
        dataset_key="video_games",
        embedding_subdir="advanced_features/title_desc_embeddings",
    )
    assert arts.embeddings.shape == (5, 8)
    assert len(arts.item_ids) == 5
    assert arts.dim == 8
    assert arts.source_path.name == "title_desc_embeddings"


def test_load_vector_artifacts_falls_back_to_embeddings_dir(tmp_path: Path):
    fallback = tmp_path / "video_games" / "embeddings"
    _write_artifacts(fallback, n_items=3, dim=4)

    arts = load_vector_artifacts(
        processed_dir=tmp_path,
        dataset_key="video_games",
        embedding_subdir="advanced_features/title_desc_embeddings",
    )
    assert arts.embeddings.shape == (3, 4)
    assert arts.source_path.name == "embeddings"


def test_load_vector_artifacts_raises_when_lengths_mismatch(tmp_path: Path):
    advanced = tmp_path / "video_games" / "advanced_features" / "title_desc_embeddings"
    advanced.mkdir(parents=True, exist_ok=True)
    np.save(advanced / "item_emb.npy", np.zeros((4, 8), dtype="float32"))
    (advanced / "item_ids.json").write_text(json.dumps(["a", "b", "c"]))

    with pytest.raises(ValueError, match="item_ids"):
        load_vector_artifacts(
            processed_dir=tmp_path,
            dataset_key="video_games",
            embedding_subdir="advanced_features/title_desc_embeddings",
        )


def test_load_vector_artifacts_raises_when_no_path_exists(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="item_emb.npy"):
        load_vector_artifacts(
            processed_dir=tmp_path,
            dataset_key="video_games",
            embedding_subdir="advanced_features/title_desc_embeddings",
        )


def test_load_vector_artifacts_raises_when_meta_dim_mismatches(tmp_path: Path):
    advanced = tmp_path / "video_games" / "advanced_features" / "title_desc_embeddings"
    _write_artifacts(advanced, n_items=3, dim=8)
    # Overwrite embedding_meta.json with a wrong dim.
    (advanced / "embedding_meta.json").write_text(
        json.dumps({"dim": 16, "model_name": "fake-embedder", "row_count": 3})
    )

    with pytest.raises(ValueError, match="dim"):
        load_vector_artifacts(
            processed_dir=tmp_path,
            dataset_key="video_games",
            embedding_subdir="advanced_features/title_desc_embeddings",
        )

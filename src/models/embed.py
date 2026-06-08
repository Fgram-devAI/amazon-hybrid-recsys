"""CLI helpers to materialize and inspect item text embeddings separately.

This lets us verify the embedding cache before running the full model
evaluation, instead of creating embeddings as a side effect of evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import load_config
from src.models.embedding import (
    build_embedder,
    content_hash_for,
    load_or_compute_item_embeddings,
)


def _processed_paths(config: dict, dataset: str) -> tuple[Path, Path]:
    base = Path(config["processed_dir"]) / dataset
    return base / "metadata.parquet", base / "embeddings"


def build_embeddings(config: dict, dataset: str) -> tuple[np.ndarray, list[str], Path]:
    """Materialize or load cached embeddings for a processed dataset."""
    metadata_path, cache_dir = _processed_paths(config, dataset)
    metadata = pd.read_parquet(metadata_path)
    metadata = metadata.drop_duplicates(subset=["parent_asin"], keep="last").reset_index(
        drop=True
    )

    embedder = build_embedder(config)
    content_hash = content_hash_for(metadata, embedder.name)
    embeddings, item_ids = load_or_compute_item_embeddings(
        metadata, embedder, cache_dir, content_hash
    )
    return embeddings, item_ids, cache_dir


def write_embedding_preview(
    embeddings: np.ndarray,
    item_ids: list[str],
    cache_dir: Path,
    *,
    rows: int = 10,
    dims: int = 20,
) -> Path:
    """Write a small CSV preview of the first embedding rows/dimensions."""
    n_rows = min(rows, len(item_ids), embeddings.shape[0])
    n_dims = min(dims, embeddings.shape[1])
    preview = pd.DataFrame(
        embeddings[:n_rows, :n_dims],
        columns=[f"dim_{idx}" for idx in range(n_dims)],
    )
    preview.insert(0, "parent_asin", item_ids[:n_rows])
    out_path = cache_dir / "embedding_preview.csv"
    preview.to_csv(out_path, index=False)
    return out_path


def main(argv=None):
    """CLI: build/load embeddings and optionally write a small CSV preview."""
    import argparse

    parser = argparse.ArgumentParser(description="Build or inspect item embeddings.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", required=True, help="processed dataset key")
    parser.add_argument("--preview", action="store_true", help="write embedding_preview.csv")
    parser.add_argument("--preview-rows", type=int, default=10)
    parser.add_argument("--preview-dims", type=int, default=20)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    embeddings, item_ids, cache_dir = build_embeddings(config, args.dataset)

    meta_path = cache_dir / "embedding_meta.json"
    cache_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    print(f"dataset: {args.dataset}")
    print(f"cache:   {cache_dir}")
    print(f"shape:   {embeddings.shape}")
    print(f"dtype:   {embeddings.dtype}")
    print(f"items:   {len(item_ids)}")
    print(f"model:   {cache_meta.get('model_name')}")
    if item_ids:
        print(f"first:   {item_ids[0]} -> {embeddings[0, :5].tolist()}")

    if args.preview:
        out_path = write_embedding_preview(
            embeddings,
            item_ids,
            cache_dir,
            rows=args.preview_rows,
            dims=args.preview_dims,
        )
        print(f"preview: {out_path}")


if __name__ == "__main__":
    main()

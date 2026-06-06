"""Resolve Amazon Reviews 2023 download URLs and local file paths for a category.

The McAuley Lab serves raw files as gzipped JSONL:
  {base}/review_categories/{Category}.jsonl.gz       -> reviews
  {base}/meta_categories/meta_{Category}.jsonl.gz    -> item metadata
"""

from pathlib import Path


def review_url(base_url: str, category: str) -> str:
    """URL of the gzipped JSONL review file for a category."""
    return f"{base_url.rstrip('/')}/review_categories/{category}.jsonl.gz"


def meta_url(base_url: str, category: str) -> str:
    """URL of the gzipped JSONL item-metadata file for a category."""
    return f"{base_url.rstrip('/')}/meta_categories/meta_{category}.jsonl.gz"


def raw_paths(raw_dir: str, category: str) -> tuple[Path, Path]:
    """Local (review_path, meta_path) where the raw files are stored."""
    base = Path(raw_dir)
    return base / f"{category}.jsonl.gz", base / f"meta_{category}.jsonl.gz"


def resolve_existing(path) -> Path:
    """Return the file that actually exists on disk for a raw path.

    Prefers the given (``.jsonl.gz``) path, but falls back to the uncompressed
    ``.jsonl`` variant when the gzip is absent (e.g. the user decompressed it).
    """
    p = Path(path)
    if p.exists():
        return p
    if p.suffix == ".gz":
        plain = p.with_suffix("")  # Foo.jsonl.gz -> Foo.jsonl
        if plain.exists():
            return plain
    return p

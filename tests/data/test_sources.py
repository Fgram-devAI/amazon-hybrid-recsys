"""Tests for resolving Amazon Reviews 2023 download URLs and local paths."""

from pathlib import Path

from src.data.sources import review_url, meta_url, raw_paths, resolve_existing

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"


def test_resolves_review_meta_urls_and_local_paths():
    # trailing slash on the base is normalised
    assert review_url(BASE + "/", "Video_Games") == (
        f"{BASE}/review_categories/Video_Games.jsonl.gz"
    )
    assert meta_url(BASE, "Video_Games") == (
        f"{BASE}/meta_categories/meta_Video_Games.jsonl.gz"
    )
    assert raw_paths("data/raw", "Video_Games") == (
        Path("data/raw/Video_Games.jsonl.gz"),
        Path("data/raw/meta_Video_Games.jsonl.gz"),
    )


def test_resolve_existing_falls_back_to_uncompressed_jsonl(tmp_path):
    gz = tmp_path / "Digital_Music.jsonl.gz"  # does not exist (user unzipped)
    plain = tmp_path / "Digital_Music.jsonl"
    plain.write_text("{}\n")

    # falls back to the uncompressed file when the .gz is absent
    assert resolve_existing(gz) == plain
    # but prefers the .gz when it is present
    gz.write_text("x")
    assert resolve_existing(gz) == gz

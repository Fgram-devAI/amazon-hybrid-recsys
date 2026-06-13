"""Tests that preprocess_dataset honors preprocessing.split_protocol and dedup_policy."""

import json

import pandas as pd
import pytest

from src.data.preprocess import preprocess_dataset


def _write_review_jsonl(path, rows):
    import gzip
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_meta_jsonl(path, items):
    import gzip
    with gzip.open(path, "wt") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


@pytest.fixture
def tiny_dataset(tmp_path):
    """Build a tiny raw dataset on disk + a config dict pointing at it."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    review_path = raw_dir / "Video_Games.jsonl.gz"
    meta_path = raw_dir / "meta_Video_Games.jsonl.gz"

    # 3 users x 5 items each (k_core=2 survives: every user has >=2 items,
    # every item has >=2 users). 5-core would not survive this tiny fixture.
    rows = []
    for u in range(3):
        for t in range(5):
            rows.append(
                {
                    "user_id": f"u{u}",
                    "parent_asin": f"i{t}",
                    "rating": 5.0,
                    "timestamp": t,
                    "text": "x",
                }
            )
    _write_review_jsonl(review_path, rows)
    _write_meta_jsonl(
        meta_path,
        [{"parent_asin": f"i{t}", "title": f"T{t}", "description": "d"} for t in range(5)],
    )

    config = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "active_dataset": "video_games",
        "datasets": {"video_games": {"category": "Video_Games", "k_core": 2}},
        "preprocessing": {
            "k_core": 2,
            "min_rating_relevant": 4.0,
            "test_size": 0.2,
            "random_seed": 42,
            "split_protocol": "per_user_chronological_80_20",
            "dedup_policy": "latest",
        },
    }
    return config, processed_dir


def test_preprocess_writes_split_protocol_into_eda_summary(tiny_dataset):
    config, processed_dir = tiny_dataset
    preprocess_dataset(config)
    summary = json.loads((processed_dir / "video_games" / "eda_summary.json").read_text())
    assert summary["split_protocol"] == "per_user_chronological_80_20"
    assert summary["dedup_policy"] == "latest"


def test_preprocess_leave_last_out_routes_to_sibling_dir_and_writes_validation(
    tiny_dataset,
):
    config, processed_dir = tiny_dataset
    config["preprocessing"]["split_protocol"] = "leave_last_out"
    preprocess_dataset(config)
    out = processed_dir / "video_games__leave_last_out"
    assert (out / "train.parquet").exists()
    assert (out / "validation.parquet").exists()
    assert (out / "test.parquet").exists()
    summary = json.loads((out / "eda_summary.json").read_text())
    assert summary["split_protocol"] == "leave_last_out"
    # 3 users x 5 items, leave-last-out -> validation has 3 interactions
    # (one second-to-last interaction per user). Confirm the field is present
    # and the train/test caveat is recorded.
    assert summary["validation_interactions"] == 3
    assert summary["train_test_counts_exclude_validation"] is True

    # Default 80/20 artifacts MUST NOT be overwritten
    assert not (processed_dir / "video_games" / "train.parquet").exists()


def test_preprocess_dedup_policy_first_keeps_earlier_duplicate(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    rows = []
    # User with one duplicate (user_id, parent_asin) pair; t=0 earlier, t=9 later
    for u in range(3):
        for t in range(5):
            rows.append({"user_id": f"u{u}", "parent_asin": f"i{t}",
                         "rating": 3.0 if t == 0 else 5.0,
                         "timestamp": t, "text": "x"})
    # Duplicate row for u0/i0 at later timestamp with different rating
    rows.append({"user_id": "u0", "parent_asin": "i0", "rating": 5.0,
                 "timestamp": 99, "text": "x"})

    _write_review_jsonl(raw_dir / "Video_Games.jsonl.gz", rows)
    _write_meta_jsonl(raw_dir / "meta_Video_Games.jsonl.gz",
                      [{"parent_asin": f"i{t}", "title": f"T{t}",
                        "description": "d"} for t in range(5)])

    config = {
        "raw_dir": str(raw_dir), "processed_dir": str(processed_dir),
        "active_dataset": "video_games",
        "datasets": {"video_games": {"category": "Video_Games", "k_core": 2}},
        "preprocessing": {
            "k_core": 2, "min_rating_relevant": 4.0, "test_size": 0.2,
            "random_seed": 42, "split_protocol": "per_user_chronological_80_20",
            "dedup_policy": "first",
        },
    }
    preprocess_dataset(config)
    interactions = pd.read_parquet(processed_dir / "video_games" / "interactions.parquet")
    u0_i0 = interactions[(interactions["user_id"] == "u0") & (interactions["parent_asin"] == "i0")]
    assert len(u0_i0) == 1
    assert u0_i0.iloc[0]["rating"] == 3.0  # earliest, not 5.0

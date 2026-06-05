"""Test the end-to-end fetch orchestration (config -> downloaded files)."""

from pathlib import Path

from src.data.fetch import fetch_dataset

from ._fakes import FakeResponse, FakeSession


def test_fetches_review_and_meta_from_correct_urls(tmp_path):
    cfg = {
        "amazon_base_url": "https://base",
        "raw_dir": str(tmp_path / "raw"),
        "active_dataset": "video_games",
        "datasets": {"video_games": {"category": "Video_Games"}},
    }
    session = FakeSession(FakeResponse([b"payload"]))

    review_path, meta_path = fetch_dataset(cfg, session=session, progress=False)

    assert review_path == Path(cfg["raw_dir"]) / "Video_Games.jsonl.gz"
    assert meta_path == Path(cfg["raw_dir"]) / "meta_Video_Games.jsonl.gz"
    assert review_path.exists() and meta_path.exists()
    assert session.calls == [
        "https://base/review_categories/Video_Games.jsonl.gz",
        "https://base/meta_categories/meta_Video_Games.jsonl.gz",
    ]

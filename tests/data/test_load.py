"""Tests for streaming records out of gzipped JSONL files."""

import gzip
import json

from src.data.load import read_jsonl_gz


def test_streams_records_and_honours_limit(tmp_path):
    path = tmp_path / "reviews.jsonl.gz"
    rows = [{"rating": 5.0, "asin": "A1"}, {"rating": 3.0, "asin": "B2"}]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    assert list(read_jsonl_gz(path)) == rows
    assert list(read_jsonl_gz(path, limit=1)) == rows[:1]

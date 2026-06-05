"""Stream records out of gzipped JSONL files (memory-safe for large categories)."""

import gzip
import json
from collections.abc import Iterator
from pathlib import Path


def read_jsonl_gz(path, limit: int | None = None) -> Iterator[dict]:
    """Yield each non-blank line of a ``.jsonl.gz`` file as a parsed dict.

    Lazy: reads one line at a time, so multi-million-row files never fully load
    into memory. ``limit`` stops after that many records (useful for sampling).
    """
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        count = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                return

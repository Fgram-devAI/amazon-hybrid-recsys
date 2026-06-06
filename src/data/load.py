"""Stream records out of JSONL files, gzipped or plain (memory-safe for big files)."""

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

_GZIP_MAGIC = b"\x1f\x8b"


def _open_text(path):
    """Open a JSONL file as text, transparently handling gzip or plain content.

    Detection is by magic bytes, not extension, so a file works whether it is
    ``.jsonl.gz`` or an unzipped ``.jsonl``.
    """
    p = Path(path)
    with open(p, "rb") as raw:
        is_gzip = raw.read(2) == _GZIP_MAGIC
    return gzip.open(p, "rt", encoding="utf-8") if is_gzip else open(
        p, "rt", encoding="utf-8"
    )


def read_jsonl_gz(path, limit: int | None = None) -> Iterator[dict]:
    """Yield each non-blank line of a JSONL file (gzipped or plain) as a parsed dict.

    Lazy: reads one line at a time, so multi-million-row files never fully load
    into memory. ``limit`` stops after that many records (useful for sampling).
    """
    with _open_text(path) as fh:
        count = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                return

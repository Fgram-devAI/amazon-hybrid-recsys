"""Stream a remote file to disk with caching, atomic writes, and a progress bar.

The HTTP session is injectable so the downloader can be tested without network
access. In production a ``requests.Session`` is created on demand.
"""

from pathlib import Path

from tqdm import tqdm

CHUNK_SIZE = 1 << 20  # 1 MiB


def download_file(url, dest, *, session=None, force=False, progress=True, timeout=60):
    """Download ``url`` to ``dest``, skipping if already present (unless ``force``).

    Writes to a temporary ``.part`` file and renames on success, so a failed or
    interrupted download never leaves a half-written destination in place.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        return dest

    if session is None:  # pragma: no cover - exercised only against the live server
        import requests

        session = requests.Session()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")

    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        bar = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            disable=not progress,
        )
        try:
            with tmp.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    fh.write(chunk)
                    bar.update(len(chunk))
        finally:
            bar.close()

    tmp.replace(dest)
    return dest

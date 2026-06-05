"""Network-free fakes for exercising the downloader and fetch orchestration."""


class FakeResponse:
    def __init__(self, chunks, ok=True):
        self._chunks = chunks
        self._ok = ok
        self.headers = {"content-length": str(sum(len(c) for c in chunks))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("HTTP error")

    def iter_content(self, chunk_size):
        yield from self._chunks


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, stream=False, timeout=None):
        self.calls.append(url)
        return self.response

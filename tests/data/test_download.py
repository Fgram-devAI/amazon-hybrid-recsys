"""Tests for the streaming downloader: it fetches once and caches thereafter."""

from src.data.download import download_file

from ._fakes import FakeResponse, FakeSession


def test_downloads_streamed_content_to_destination(tmp_path):
    dest = tmp_path / "nested" / "Video_Games.jsonl.gz"
    session = FakeSession(FakeResponse([b"hello ", b"world"]))

    result = download_file("http://x/file.gz", dest, session=session, progress=False)

    assert result == dest
    assert dest.read_bytes() == b"hello world"
    assert session.calls == ["http://x/file.gz"]


def test_skips_download_when_file_already_exists(tmp_path):
    dest = tmp_path / "f.gz"
    dest.write_bytes(b"existing")
    session = FakeSession(FakeResponse([b"new"]))

    download_file("http://x", dest, session=session, progress=False)

    assert dest.read_bytes() == b"existing"
    assert session.calls == []  # cached -> no network call

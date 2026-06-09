"""Item-text embedders (injectable) plus an on-disk embedding cache.

- FakeEmbedder: deterministic, hash-of-words vectors for tests (no downloads).
- SentenceTransformerEmbedder: wraps Granite (primary) or MiniLM (fallback).
- load_or_compute_item_embeddings: cache keyed by model + content hash.
"""

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class FakeEmbedder:
    """Deterministic bag-of-hashed-words embedding; shared words -> similar vectors."""

    name = "fake"

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in str(text).lower().split():
                bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                out[i, bucket] += 1.0
        return out


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model (Granite primary, MiniLM fallback)."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 256,
        max_seq_length: int = 256,
        max_chars: int = 2000,
    ):
        from sentence_transformers import SentenceTransformer

        self.name = f"{model_name}|seq={max_seq_length}|max_chars={max_chars}"
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_chars = max_chars
        self._model = SentenceTransformer(model_name, device=device)
        self._model.max_seq_length = max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        prepared = [str(text)[: self.max_chars] for text in texts]
        vecs = self._model.encode(
            prepared,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )
        return np.asarray(vecs, dtype=np.float32)


def build_embedder(config: dict) -> Embedder:
    """Build the preferred embedder, degrading model then device on failure.

    Tries the primary model on the configured device, then the fallback model,
    then retries both on CPU — so a device that is unavailable on the current
    machine (e.g. "mps" on non-Apple hardware) does not break embedding.
    """
    models = config.get("models", {})
    primary = models.get("embedding_model", "ibm-granite/granite-embedding-97m-multilingual-r2")
    fallback = models.get("embedding_fallback", "sentence-transformers/all-MiniLM-L6-v2")
    device = models.get("embedding_device", "cpu")
    kwargs = {
        "batch_size": int(models.get("embedding_batch_size", 256)),
        "max_seq_length": int(models.get("embedding_max_seq_length", 256)),
        "max_chars": int(models.get("embedding_max_chars", 2000)),
    }

    candidates = [(primary, device), (fallback, device), (primary, "cpu"), (fallback, "cpu")]
    seen = set()
    last_error = None
    for model_name, dev in candidates:
        if (model_name, dev) in seen:
            continue
        seen.add((model_name, dev))
        try:
            return SentenceTransformerEmbedder(model_name, device=dev, **kwargs)
        except Exception as error:
            last_error = error
    raise RuntimeError("could not build any embedder") from last_error


def content_hash_for(metadata_df, model_name: str) -> str:
    """Stable hash of the embedded text + model name (cache invalidation)."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for text in metadata_df["text"].fillna("").tolist():
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
    return h.hexdigest()


def load_or_compute_item_embeddings(
    metadata_df,
    embedder: Embedder,
    cache_dir: Path | str,
    content_hash: str,
    *,
    progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings (n,d) float32, item_ids list). Cache on disk."""
    cache_dir = Path(cache_dir)
    emb_path = cache_dir / "item_emb.npy"
    ids_path = cache_dir / "item_ids.json"
    meta_path = cache_dir / "embedding_meta.json"

    if emb_path.exists() and ids_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("content_hash") == content_hash and meta.get("model_name") == embedder.name:
            if progress:
                print(
                    f"[embeddings] cache hit: {emb_path} "
                    f"({meta.get('metadata_row_count')} items, dim={meta.get('dim')})",
                    flush=True,
                )
            return np.load(emb_path), json.loads(ids_path.read_text())

    ids = metadata_df["parent_asin"].tolist()
    texts = metadata_df["text"].fillna("").tolist()
    if progress:
        print(
            f"[embeddings] computing {len(ids)} item embeddings with {embedder.name} "
            f"-> {cache_dir}",
            flush=True,
        )
    emb = np.asarray(embedder.encode(texts), dtype=np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    ids_path.write_text(json.dumps(ids))
    meta_path.write_text(
        json.dumps(
            {
                "model_name": embedder.name,
                "dim": int(emb.shape[1]),
                "metadata_row_count": len(ids),
                "content_hash": content_hash,
                "version": 1,
            }
        )
    )
    if progress:
        print(
            f"[embeddings] wrote {emb.shape[0]} embeddings "
            f"(dim={emb.shape[1]}) -> {emb_path}",
            flush=True,
        )
    return emb, ids

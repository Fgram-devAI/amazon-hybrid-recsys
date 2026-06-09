"""Sentiment model protocol, deterministic fake, and train-only scoring helper.

The HuggingFace wrapper is intentionally NOT imported at module load — tests must
never trigger a download. Build it explicitly via ``build_sentiment_model`` only
when a real run wants it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

_POSITIVE_WORDS = {"love", "great", "awesome", "good", "excellent", "amazing", "fun"}
_NEGATIVE_WORDS = {"hate", "terrible", "bad", "awful", "boring", "broken", "worst"}


class SentimentModel(Protocol):
    name: str

    def score(self, texts: list[str]) -> list[dict]:
        ...


class FakeSentimentModel:
    """Lexicon-based deterministic sentiment for tests; never downloads anything."""

    name = "fake-sentiment-v1"

    def score(self, texts: list[str]) -> list[dict]:
        out = []
        for text in texts:
            tokens = str(text).lower().split()
            if not tokens:
                out.append(
                    {
                        "sentiment_label": "NEUTRAL",
                        "sentiment_score": 0.0,
                        "positive_prob": 0.5,
                        "negative_prob": 0.5,
                    }
                )
                continue
            pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
            neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
            total = pos + neg
            if total == 0:
                out.append(
                    {
                        "sentiment_label": "NEUTRAL",
                        "sentiment_score": 0.0,
                        "positive_prob": 0.5,
                        "negative_prob": 0.5,
                    }
                )
                continue
            positive_prob = pos / total
            negative_prob = neg / total
            score = positive_prob - negative_prob   # in [-1, 1]
            label = "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL"
            out.append(
                {
                    "sentiment_label": label,
                    "sentiment_score": float(score),
                    "positive_prob": float(positive_prob),
                    "negative_prob": float(negative_prob),
                }
            )
        return out


def score_train_reviews(
    *,
    train: pd.DataFrame,
    raw_reviews: Iterable[dict],
    model: SentimentModel,
    out_dir: Path | str,
    dataset: str,
    text_column: str = "text",
    batch_size: int = 32,
    max_chars: int = 1000,
    max_rows: int | None = None,
) -> Path:
    """Score sentiment on raw reviews whose (user, item, ts) is in train; cache to disk.

    Filters by ``(user_id, parent_asin, timestamp)`` so held-out test reviews are
    NEVER scored. Returns the parquet path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_keys = set(
        zip(train["user_id"], train["parent_asin"], train["timestamp"])
    )

    batched_texts: list[str] = []
    batched_keys: list[tuple[str, str, int]] = []
    rows: list[dict] = []

    def _flush():
        if not batched_texts:
            return
        scored = model.score(batched_texts)
        for (user, item, ts), result in zip(batched_keys, scored):
            rows.append(
                {
                    "user_id": user,
                    "parent_asin": item,
                    "timestamp": ts,
                    **result,
                }
            )
        batched_texts.clear()
        batched_keys.clear()

    for review in raw_reviews:
        key = (review.get("user_id"), review.get("parent_asin"), review.get("timestamp"))
        if key not in train_keys:
            continue
        text = str(review.get(text_column) or "")[:max_chars]
        batched_texts.append(text)
        batched_keys.append(key)  # type: ignore[arg-type]
        if len(batched_texts) >= batch_size:
            _flush()
        if max_rows is not None and len(rows) + len(batched_texts) >= max_rows:
            _flush()
            break
    _flush()

    parquet_path = out / "train_sentiment.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    meta = {
        "dataset": dataset,
        "model_name": model.name,
        "split": "train",
        "text_column": text_column,
        "row_count": len(rows),
        "train_row_count": len(train_keys),
        "max_chars": max_chars,
        "version": 1,
        "key_hash": hashlib.sha256(
            json.dumps(sorted(map(list, train_keys))).encode("utf-8")
        ).hexdigest(),
    }
    (out / "train_sentiment_meta.json").write_text(json.dumps(meta))
    return parquet_path


def build_sentiment_model(config: dict) -> SentimentModel:
    """Construct the real HF sentiment model. Never call this from tests."""
    from transformers import pipeline  # local import keeps tests free of HF

    model_name = config["advanced_features"]["sentiment_model"]
    pipe = pipeline("sentiment-analysis", model=model_name, truncation=True)  # type: ignore[call-overload]

    class _HFSentimentModel:
        # populated below so `name` is the exact configured model id
        name: str = model_name

        def score(self, texts: list[str]) -> list[dict]:
            outputs = pipe(texts, batch_size=int(config["advanced_features"]["sentiment_batch_size"]))
            results = []
            for raw in outputs:
                label = str(raw["label"]).upper()
                prob = float(raw["score"])
                positive_prob = prob if label.startswith("POS") else 1.0 - prob
                negative_prob = 1.0 - positive_prob
                score = positive_prob - negative_prob
                results.append(
                    {
                        "sentiment_label": "POSITIVE" if score > 0 else "NEGATIVE",
                        "sentiment_score": float(np.clip(score, -1.0, 1.0)),
                        "positive_prob": float(positive_prob),
                        "negative_prob": float(negative_prob),
                    }
                )
            return results

    return _HFSentimentModel()  # type: ignore[return-value]

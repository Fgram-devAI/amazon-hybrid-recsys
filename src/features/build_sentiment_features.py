"""CLI: score sentiment over train reviews and build user/item aggregates.

Usage:
    ./.venv/bin/python -m src.features.build_sentiment_features \
        --dataset video_games [--fake] [--max-rows 50000]

--fake uses FakeSentimentModel (no HF download); without it the real HF model
configured in advanced_features.sentiment_model is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.config import load_config
from src.data.load import read_jsonl_gz
from src.data.sources import raw_paths, resolve_existing
from src.features.review_features import (
    item_review_aggregates,
    user_review_aggregates,
)
from src.features.sentiment import (
    FakeSentimentModel,
    build_sentiment_model,
    score_train_reviews,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fake", action="store_true", help="use FakeSentimentModel")
    parser.add_argument("--max-rows", type=int, help="cap rows scored (dev only)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    category = config["datasets"][args.dataset]["category"]
    review_gz, _meta_gz = raw_paths(config["raw_dir"], category)
    raw_path = resolve_existing(review_gz)   # falls back to .jsonl if .gz absent
    processed = Path(config["processed_dir"]) / args.dataset
    out_dir = processed / "advanced_features"
    print(f"[{args.dataset}] loading train split from {processed / 'train.parquet'}", flush=True)
    train = pd.read_parquet(processed / "train.parquet")

    model_name = (
        FakeSentimentModel.name
        if args.fake
        else config["advanced_features"]["sentiment_model"]
    )
    print(f"[{args.dataset}] building sentiment model: {model_name}", flush=True)
    model = FakeSentimentModel() if args.fake else build_sentiment_model(config)
    print(f"[{args.dataset}] sentiment device: {model.device}", flush=True)
    max_rows = (
        args.max_rows
        if args.max_rows is not None
        else config["advanced_features"].get("sentiment_max_rows")
    )
    cap = "all train rows" if max_rows is None else str(max_rows)
    print(
        f"[{args.dataset}] scoring train review text from {raw_path} "
        f"(cap={cap}, batch_size={config['advanced_features']['sentiment_batch_size']})",
        flush=True,
    )
    score_train_reviews(
        train=train,
        raw_reviews=read_jsonl_gz(raw_path),
        model=model,
        out_dir=out_dir,
        dataset=args.dataset,
        text_column="text",
        batch_size=int(config["advanced_features"]["sentiment_batch_size"]),
        max_chars=int(config["advanced_features"]["sentiment_max_chars"]),
        max_rows=max_rows,
        progress=True,
    )
    print(f"[{args.dataset}] building user/item review aggregates", flush=True)
    sentiment = pd.read_parquet(out_dir / "train_sentiment.parquet")

    user_review_aggregates(train, sentiment).to_parquet(
        out_dir / "user_review_aggregates.parquet", index=False
    )
    item_review_aggregates(train, sentiment).to_parquet(
        out_dir / "item_review_aggregates.parquet", index=False
    )
    print(f"[{args.dataset}] sentiment + aggregates -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

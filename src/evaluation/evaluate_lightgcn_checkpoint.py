"""Evaluate a stored LightGCN checkpoint without retraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.config import load_config
from src.evaluation.evaluate import _load_processed, sample_negatives
from src.evaluation.metrics import (
    mae,
    precision_recall_f1_at_k,
    relevant_items_by_user,
    rmse,
)
from src.models.lightgcn import LightGCNRecommender


def evaluate_fitted_lightgcn(
    model: LightGCNRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    dataset: str,
    k: int,
    min_rating_relevant: float,
    num_negatives: int,
    seed: int,
    max_eval_users: int | None,
    max_test_rows: int | None,
    progress: bool,
) -> pd.DataFrame:
    rating_test = test
    if max_test_rows is not None and len(test) > max_test_rows:
        rating_test = test.sample(n=max_test_rows, random_state=seed).reset_index(drop=True)

    y_true = rating_test["rating"].to_numpy(dtype=float)
    pairs = zip(rating_test["user_id"], rating_test["parent_asin"])
    y_pred = np.array([
        model.predict(u, i)
        for u, i in tqdm(
            pairs,
            total=len(rating_test),
            desc=f"[{dataset}] lightgcn checkpoint ratings",
            unit="row",
            disable=not progress,
        )
    ])

    relevant = relevant_items_by_user(test, min_rating_relevant)
    if max_eval_users is not None and len(relevant) > max_eval_users:
        rng = np.random.default_rng(seed)
        users = list(relevant)
        chosen = rng.choice(len(users), size=max_eval_users, replace=False)
        relevant = {users[int(idx)]: relevant[users[int(idx)]] for idx in chosen}

    all_items = np.asarray(pd.unique(train["parent_asin"]), dtype=object)
    user_train_items = {
        user: set(items) for user, items in train.groupby("user_id")["parent_asin"]
    }

    precisions, recalls, f1s = [], [], []
    rng = np.random.default_rng(seed)
    for user, rel in tqdm(
        relevant.items(),
        total=len(relevant),
        desc=f"[{dataset}] lightgcn checkpoint ranking",
        unit="user",
        disable=not progress,
    ):
        exclude = user_train_items.get(user, set()) | rel
        negatives = sample_negatives(all_items, exclude, num_negatives, rng)
        ranked = model.recommend(user, k, candidates=list(rel) + negatives)
        result = precision_recall_f1_at_k(ranked, rel, k)
        if result is None:
            continue
        p, r, f = result
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return pd.DataFrame([{
        "dataset": dataset,
        "model": "lightgcn_checkpoint",
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "precision_at_k": float(np.mean(precisions)) if precisions else None,
        "recall_at_k": float(np.mean(recalls)) if recalls else None,
        "f1_at_k": float(np.mean(f1s)) if f1s else None,
        "n_eval_users": len(precisions),
        "max_eval_users": max_eval_users,
        "max_test_rows": max_test_rows,
    }])


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a saved LightGCN checkpoint.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", help="defaults to graph_checkpoints/lightgcn.pt")
    parser.add_argument("--output", help="defaults to metrics_lightgcn_checkpoint.json")
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    train, test, _ = _load_processed(config["processed_dir"], args.dataset)
    processed = Path(config["processed_dir"]) / args.dataset
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        processed / "graph_checkpoints" / "lightgcn.pt"
    )
    output = Path(args.output) if args.output else (
        processed / "metrics_lightgcn_checkpoint.json"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    progress = not args.quiet
    mc = config.get("models", {})
    gc = config.get("graph", {})

    model = LightGCNRecommender(
        embedding_dim=int(gc.get("embedding_dim", 64)),
        n_layers=int(gc.get("n_layers", 2)),
        epochs=int(gc.get("epochs", 10)),
        lr=float(gc.get("lr", 0.005)),
        weight_decay=float(gc.get("weight_decay", 0.0)),
        num_negatives=int(gc.get("num_negatives", 1)),
        batch_size=int(gc.get("batch_size", 1024)),
        seed=int(gc.get("seed", 42)),
        device=str(gc.get("device", "auto")),
        min_rating_positive=float(gc.get("min_rating_positive", 4.0)),
        validation_fraction=float(gc.get("validation_fraction", 0.1)),
        progress=progress,
    )
    if progress:
        print(f"[{args.dataset}] preparing LightGCN inference state", flush=True)
    model.prepare_for_checkpoint(train)
    model.load_checkpoint(checkpoint)
    if progress:
        print(f"[{args.dataset}] loaded checkpoint -> {checkpoint}", flush=True)

    table = evaluate_fitted_lightgcn(
        model,
        train,
        test,
        dataset=args.dataset,
        k=config["evaluation"]["k"],
        min_rating_relevant=config["preprocessing"]["min_rating_relevant"],
        num_negatives=mc.get("ranking_num_negatives", 100),
        seed=mc.get("ranking_random_seed", 42),
        max_eval_users=args.max_eval_users,
        max_test_rows=args.max_test_rows,
        progress=progress,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table.to_json(orient="records", indent=2))
    print(table.to_string(index=False))
    print(f"\nMetrics -> {output}")


if __name__ == "__main__":
    main()

"""Evaluate a stored GraphSAGE checkpoint without retraining.

This is a recovery/dev utility for long graph runs: it rebuilds the train-only
graph and node features, loads saved GraphSAGE weights, caches final node
embeddings once, then computes the standard RMSE/MAE + sampled ranking metrics.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.config import load_config
from src.evaluation._audit_shared import (
    compute_checkpoint_audit_metrics,
    resolve_split_protocol,
)
from src.evaluation.evaluate import _load_processed, sample_negatives
from src.evaluation.metrics import (
    mae,
    relevant_items_by_user,
    rmse,
)
from src.models.embedding import build_embedder
from src.models.graphsage import GraphSAGERecommender


def evaluate_fitted_graphsage(
    model: GraphSAGERecommender,
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
    split_protocol: str = "per_user_chronological_80_20",
) -> pd.DataFrame:
    rating_test = test
    if max_test_rows is not None and len(test) > max_test_rows:
        rating_test = test.sample(n=max_test_rows, random_state=seed).reset_index(drop=True)

    if progress:
        print(f"[{dataset}] checkpoint ratings: {len(rating_test):,} rows", flush=True)
    y_true = rating_test["rating"].to_numpy(dtype=float)
    pairs = zip(rating_test["user_id"], rating_test["parent_asin"])
    y_pred = np.array([
        model.predict(u, i)
        for u, i in tqdm(
            pairs,
            total=len(rating_test),
            desc=f"[{dataset}] graphsage checkpoint ratings",
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

    per_user_data = []
    rng = np.random.default_rng(seed)
    if progress:
        print(
            f"[{dataset}] checkpoint ranking users: {len(relevant)} "
            f"(cap={max_eval_users}, negatives/user={num_negatives})",
            flush=True,
        )
    for user, rel in tqdm(
        relevant.items(),
        total=len(relevant),
        desc=f"[{dataset}] graphsage checkpoint ranking",
        unit="user",
        disable=not progress,
    ):
        exclude = user_train_items.get(user, set()) | rel
        negatives = sample_negatives(all_items, exclude, num_negatives, rng)
        ranked = model.recommend(user, k, candidates=list(rel) + negatives)
        if not rel:
            continue
        per_user_data.append({"ranked": ranked, "relevant": rel})

    audit = compute_checkpoint_audit_metrics(per_user_data, k=k, split_protocol=split_protocol)
    metrics: dict = {
        "dataset": dataset,
        "model": "graphsage_checkpoint",
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
    metrics.update(audit)
    metrics["max_eval_users"] = max_eval_users
    metrics["max_test_rows"] = max_test_rows
    return pd.DataFrame([metrics])


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a saved GraphSAGE checkpoint.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", help="defaults to graph_checkpoints/graphsage.pt")
    parser.add_argument("--output", help="defaults to metrics_graphsage_checkpoint.json")
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    train, test, metadata = _load_processed(config["processed_dir"], args.dataset)
    processed = Path(config["processed_dir"]) / args.dataset
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        processed / "graph_checkpoints" / "graphsage.pt"
    )
    output = Path(args.output) if args.output else (
        processed / "metrics_graphsage_checkpoint.json"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    progress = not args.quiet
    mc = config.get("models", {})
    gc = config.get("graph", {})
    af = config.get("advanced_features", {})
    af_dir = processed / "advanced_features"

    embedder = build_embedder(config)
    model = GraphSAGERecommender(
        embedder=embedder,
        generic_roots=af.get("generic_category_roots", []),
        max_vocab=int(af.get("category_vocab_max", 256)),
        min_doc_freq=int(af.get("category_min_doc_freq", 5)),
        hidden_dim=int(gc.get("embedding_dim", 64)),
        n_layers=int(gc.get("n_layers", 2)),
        epochs=int(gc.get("epochs", 10)),
        lr=float(gc.get("lr", 0.005)),
        weight_decay=float(gc.get("weight_decay", 0.0)),
        batch_size=int(gc.get("batch_size", 1024)),
        seed=int(gc.get("seed", 42)),
        device=str(gc.get("device", "auto")),
        cache_dir=af_dir / "title_desc_embeddings",
        review_features_dir=af_dir,
        progress=progress,
    )

    start = perf_counter()
    if progress:
        print(f"[{args.dataset}] preparing GraphSAGE inference state", flush=True)
    model.prepare_for_checkpoint(train, metadata)
    model.load_checkpoint(checkpoint)
    if model._final_embeddings is None:
        model.cache_final_embeddings()
    if progress:
        print(
            f"[{args.dataset}] loaded checkpoint in {perf_counter() - start:.1f}s -> {checkpoint}",
            flush=True,
        )

    split_protocol = resolve_split_protocol(
        config["processed_dir"], args.dataset, config["evaluation"]
    )
    table = evaluate_fitted_graphsage(
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
        split_protocol=split_protocol,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table.to_json(orient="records", indent=2))
    print(table.to_string(index=False))
    print(f"\nMetrics -> {output}")


if __name__ == "__main__":
    main()

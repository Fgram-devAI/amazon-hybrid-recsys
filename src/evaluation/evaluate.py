"""Evaluation runner: fit models, compute RMSE/MAE + sampled-negative ranking."""

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import mae, precision_recall_f1_at_k, relevant_items_by_user, rmse


def sample_negatives(all_items, exclude, n, rng):
    """Sample up to n items not in `exclude` (deterministic given rng)."""
    pool = [item for item in all_items if item not in exclude]
    if len(pool) <= n:
        return list(pool)
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]


def evaluate_models(models, train, test, metadata, *, k, min_rating_relevant,
                    num_negatives, seed, dataset="(fixture)"):
    """Fit each model and return a metrics DataFrame (one row per model)."""
    for model in models.values():
        model.fit(train, metadata)

    relevant = relevant_items_by_user(test, min_rating_relevant)
    all_items = list(pd.unique(train["parent_asin"]))
    user_train_items = {
        user: set(items) for user, items in train.groupby("user_id")["parent_asin"]
    }

    rows = []
    for name, model in models.items():
        y_true = test["rating"].to_numpy(dtype=float)
        y_pred = np.array(
            [model.predict(u, i) for u, i in zip(test["user_id"], test["parent_asin"])]
        )

        precisions, recalls, f1s = [], [], []
        rng = np.random.default_rng(seed)  # reset per model -> same candidate sets
        for user, rel in relevant.items():
            exclude = user_train_items.get(user, set()) | rel
            negatives = sample_negatives(all_items, exclude, num_negatives, rng)
            candidates = list(rel) + negatives
            ranked = model.recommend(user, k, candidates=candidates)
            result = precision_recall_f1_at_k(ranked, rel, k)
            if result is None:
                continue
            p, r, f = result
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        rows.append(
            {
                "dataset": dataset,
                "model": name,
                "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "precision_at_k": float(np.mean(precisions)) if precisions else None,
                "recall_at_k": float(np.mean(recalls)) if recalls else None,
                "f1_at_k": float(np.mean(f1s)) if f1s else None,
                "n_eval_users": len(precisions),
            }
        )
    return pd.DataFrame(rows)


def _load_processed(processed_dir, dataset):
    base = Path(processed_dir) / dataset
    return (
        pd.read_parquet(base / "train.parquet"),
        pd.read_parquet(base / "test.parquet"),
        pd.read_parquet(base / "metadata.parquet"),
    )


def main(argv=None):
    """CLI: run the model comparison for a processed dataset."""
    import argparse

    from src.data.config import load_config
    from src.models.cf import KNNRecommender, SVDRecommender
    from src.models.content_based import ContentBasedRecommender
    from src.models.embedding import build_embedder
    from src.models.weighted_hybrid import WeightedHybrid

    parser = argparse.ArgumentParser(description="Evaluate recommender models on a dataset.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", required=True, help="processed dataset key")
    parser.add_argument("--no-knn", action="store_true", help="skip Item-KNN (memory-bound)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    train, test, metadata = _load_processed(config["processed_dir"], args.dataset)

    mc = config.get("models", {})
    embedder = build_embedder(config)
    cache_dir = Path(config["processed_dir"]) / args.dataset / "embeddings"
    content = ContentBasedRecommender(embedder, cache_dir=cache_dir)
    svd = SVDRecommender(random_state=mc.get("ranking_random_seed", 42))

    models = {"content": content, "svd": svd}
    if not args.no_knn:
        models["item_knn"] = KNNRecommender()
    models["hybrid"] = WeightedHybrid(
        SVDRecommender(random_state=mc.get("ranking_random_seed", 42)),
        ContentBasedRecommender(embedder, cache_dir=cache_dir),
        alpha=config["hybrid"]["alpha"],
    )

    table = evaluate_models(
        models,
        train,
        test,
        metadata,
        k=config["evaluation"]["k"],
        min_rating_relevant=config["preprocessing"]["min_rating_relevant"],
        num_negatives=mc.get("ranking_num_negatives", 100),
        seed=mc.get("ranking_random_seed", 42),
        dataset=args.dataset,
    )

    out_dir = Path(config["processed_dir"]) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(table.to_json(orient="records", indent=2))
    print(table.to_string(index=False))
    print(f"\nMetrics -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

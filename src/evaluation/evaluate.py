"""Evaluation runner: fit models, compute RMSE/MAE + sampled-negative ranking."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .metrics import mae, precision_recall_f1_at_k, relevant_items_by_user, rmse


def sample_negatives(all_items, exclude, n, rng):
    """Sample up to n items not in `exclude` without scanning the catalog per user."""
    exclude = set(exclude)
    if n <= 0:
        return []

    n_items = len(all_items)
    sampled = []
    sampled_set = set()

    # Rejection sampling is fast when the catalog dwarfs the exclusions. Skip it
    # when exclusions dominate (which can happen when `exclude` holds held-out
    # positives that are not in the catalog) and go straight to the scan.
    if len(exclude) < n_items:
        attempts = 0
        max_attempts = max(100, n * 20)
        while len(sampled) < n and attempts < max_attempts:
            batch_size = min(n_items, max(32, (n - len(sampled)) * 3))
            indices = rng.integers(0, n_items, size=batch_size)
            for idx in indices:
                item = all_items[int(idx)]
                if item in exclude or item in sampled_set:
                    continue
                sampled.append(item)
                sampled_set.add(item)
                if len(sampled) == n:
                    return sampled
            attempts += 1

    # Deterministic fallback: tiny catalogs, dominant exclusions, or rejection
    # that did not converge. Returns as many as are available, up to n.
    for item in all_items:
        if len(sampled) == n:
            break
        if item in exclude or item in sampled_set:
            continue
        sampled.append(item)
        sampled_set.add(item)
    return sampled


def evaluate_models(models, train, test, metadata, *, k, min_rating_relevant,
                    num_negatives, seed, dataset="(fixture)", max_eval_users=None,
                    max_test_rows=None, progress=False):
    """Fit each model and return a metrics DataFrame (one row per model)."""
    for name, model in models.items():
        fit_start = perf_counter()
        if progress:
            print(f"[{dataset}] fitting {name} ...", flush=True)
        model.fit(train, metadata)
        if progress:
            print(f"[{dataset}] fitted {name} in {perf_counter() - fit_start:.1f}s", flush=True)

    rating_test = test
    if max_test_rows is not None and len(test) > max_test_rows:
        rating_test = test.sample(n=max_test_rows, random_state=seed).reset_index(drop=True)

    relevant = relevant_items_by_user(test, min_rating_relevant)
    if max_eval_users is not None and len(relevant) > max_eval_users:
        rng = np.random.default_rng(seed)
        users = list(relevant)
        chosen = rng.choice(len(users), size=max_eval_users, replace=False)
        relevant = {users[int(idx)]: relevant[users[int(idx)]] for idx in chosen}
    if progress:
        print(
            f"[{dataset}] ranking users: {len(relevant)} "
            f"(cap={max_eval_users}, negatives/user={num_negatives})",
            flush=True,
        )

    all_items = np.asarray(pd.unique(train["parent_asin"]), dtype=object)
    user_train_items = {
        user: set(items) for user, items in train.groupby("user_id")["parent_asin"]
    }

    rows = []
    for name, model in models.items():
        if progress:
            print(f"[{dataset}] predicting ratings for {name} on {len(rating_test):,} rows ...", flush=True)
        rating_start = perf_counter()
        y_true = rating_test["rating"].to_numpy(dtype=float)
        y_pred = np.array(
            [
                model.predict(u, i)
                for u, i in zip(rating_test["user_id"], rating_test["parent_asin"])
            ]
        )
        if progress:
            print(
                f"[{dataset}] rating metrics for {name} done in "
                f"{perf_counter() - rating_start:.1f}s",
                flush=True,
            )

        precisions, recalls, f1s = [], [], []
        rng = np.random.default_rng(seed)  # reset per model -> same candidate sets
        if progress:
            print(f"[{dataset}] ranking candidates for {name} ...", flush=True)
        ranking_start = perf_counter()
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
        if progress:
            print(
                f"[{dataset}] ranking metrics for {name} done in "
                f"{perf_counter() - ranking_start:.1f}s",
                flush=True,
            )

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
                "max_eval_users": max_eval_users,
                "max_test_rows": max_test_rows,
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


def build_models(
    config,
    dataset,
    embedder,
    *,
    no_knn=False,
    advanced=False,
    alpha=None,
):
    """Construct the model set, sharing component instances with the hybrids.

    When ``advanced`` is True, also registers random + popularity baselines, the
    enriched content recommender, and a calibrated hybrid that reuses the svd +
    enriched-content components.
    """
    from src.models.baselines import PopularityRecommender, RandomRecommender
    from src.models.calibrated_hybrid import CalibratedHybrid
    from src.models.cf import KNNRecommender, SVDRecommender
    from src.models.content_based import ContentBasedRecommender
    from src.models.content_enriched import ContentEnrichedRecommender
    from src.models.weighted_hybrid import WeightedHybrid

    mc = config.get("models", {})
    af = config.get("advanced_features", {})
    cache_dir = Path(config["processed_dir"]) / dataset / "embeddings"
    content = ContentBasedRecommender(embedder, cache_dir=cache_dir)
    svd = SVDRecommender(random_state=mc.get("ranking_random_seed", 42))

    blend_alpha = float(alpha if alpha is not None else config["hybrid"]["alpha"])
    models = {"content": content, "svd": svd}
    if not no_knn:
        models["item_knn"] = KNNRecommender()
    models["hybrid"] = WeightedHybrid(svd, content, alpha=blend_alpha)

    if advanced:
        seed = int(mc.get("ranking_random_seed", 42))
        af_dir = Path(config["processed_dir"]) / dataset / "advanced_features"
        models["random"] = RandomRecommender(seed=seed)
        models["popularity"] = PopularityRecommender()
        content_enriched = ContentEnrichedRecommender(
            embedder,
            generic_roots=af.get("generic_category_roots", []),
            max_vocab=int(af.get("category_vocab_max", 256)),
            min_doc_freq=int(af.get("category_min_doc_freq", 5)),
            # dedicated cache for the title+description embeddings (NOT the Phase-1 cache)
            cache_dir=af_dir / "title_desc_embeddings",
            # train-only sentiment/user/item aggregates (consumed if the offline job ran)
            review_features_dir=af_dir,
        )
        models["content_enriched"] = content_enriched
        models["calibrated_hybrid"] = CalibratedHybrid(
            svd,
            content_enriched,
            alpha=blend_alpha,
            calibrate=True,
            calibration_max_rows=config.get("hybrid", {})
            .get("tuning", {})
            .get("calibration_max_rows"),
            random_state=seed,
        )
    return models


def main(argv=None):
    """CLI: run the model comparison for a processed dataset."""
    import argparse

    from src.data.config import load_config
    from src.models.embedding import build_embedder

    parser = argparse.ArgumentParser(description="Evaluate recommender models on a dataset.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", required=True, help="processed dataset key")
    parser.add_argument("--no-knn", action="store_true", help="skip Item-KNN (memory-bound)")
    parser.add_argument("--max-eval-users", type=int, help="cap ranking eval users for large runs")
    parser.add_argument("--max-test-rows", type=int, help="cap rating rows for dev RMSE/MAE runs")
    parser.add_argument("--quiet", action="store_true", help="suppress progress logs")
    parser.add_argument("--advanced", action="store_true",
                        help="add random/popularity baselines + enriched content + calibrated hybrid")
    parser.add_argument("--alpha", type=float,
                        help="override hybrid blend alpha for this run")
    parser.add_argument(
        "--tune-alpha", action="store_true",
        help="sweep hybrid.tuning.grid on a validation slice carved from train; pick best alpha",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    train, test, metadata = _load_processed(config["processed_dir"], args.dataset)
    if not args.quiet:
        print(
            f"[{args.dataset}] loaded train={len(train):,}, test={len(test):,}, "
            f"metadata={len(metadata):,}",
            flush=True,
        )

    mc = config.get("models", {})
    embedder = build_embedder(config)

    chosen_alpha = args.alpha
    if args.tune_alpha:
        from src.evaluation.tune import tune_alpha as _tune
        from src.models.content_based import ContentBasedRecommender
        from src.models.cf import SVDRecommender
        from src.models.weighted_hybrid import WeightedHybrid

        tuning = config["hybrid"].get("tuning", {})
        cache_dir = Path(config["processed_dir"]) / args.dataset / "embeddings"

        def _factory(alpha_val: float):
            return WeightedHybrid(
                SVDRecommender(random_state=mc.get("ranking_random_seed", 42)),
                ContentBasedRecommender(embedder, cache_dir=cache_dir),
                alpha=alpha_val,
            )

        result = _tune(
            train,
            metadata=metadata,
            grid=tuning.get("grid", [0.0, 0.25, 0.5, 0.75, 1.0]),
            hybrid_factory=_factory,
            validation_fraction=float(tuning.get("validation_fraction", 0.1)),
            seed=int(tuning.get("random_seed", 42)),
            max_users=tuning.get("max_users"),       # cap users for tuning (large-data safe)
            max_val_rows=tuning.get("max_val_rows"),  # cap validation rows scored per alpha
        )
        chosen_alpha = result.best_alpha
        if not args.quiet:
            print(f"[{args.dataset}] tuned alpha={chosen_alpha} from scores={result.scores}", flush=True)

    models = build_models(
        config, args.dataset, embedder,
        no_knn=args.no_knn, advanced=args.advanced, alpha=chosen_alpha,
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
        max_eval_users=args.max_eval_users,
        max_test_rows=args.max_test_rows,
        progress=not args.quiet,
    )

    out_dir = Path(config["processed_dir"]) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(table.to_json(orient="records", indent=2))
    if not args.quiet:
        print(f"[{args.dataset}] wrote metrics.json", flush=True)
    print(table.to_string(index=False))
    print(f"\nMetrics -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

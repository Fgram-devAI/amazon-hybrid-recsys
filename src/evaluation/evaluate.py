"""Evaluation runner: fit models, compute RMSE/MAE + sampled-negative ranking."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics import (
    aggregate_metric_bundle,
    compute_user_metric_bundle,
    mae,
    relevant_items_by_user,
    rmse,
)


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
                    max_test_rows=None, progress=False, checkpoint_dir=None,
                    metrics_path=None, checkpoint_tag=None, train_only=False,
                    split_protocol="per_user_chronological_80_20"):
    """Fit each model and return a metrics DataFrame (one row per model)."""
    rating_test = test
    relevant = {}
    all_items = np.asarray([], dtype=object)
    user_train_items = {}
    if not train_only:
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
        fit_start = perf_counter()
        if progress:
            print(f"[{dataset}] fitting {name} ...", flush=True)
        model.fit(train, metadata)
        if progress:
            print(f"[{dataset}] fitted {name} in {perf_counter() - fit_start:.1f}s", flush=True)
        checkpoint_path = None
        if checkpoint_dir is not None and name in {"lightgcn", "graphsage", "graphsage_bpr"}:
            suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
            path = Path(checkpoint_dir) / f"{name}{suffix}.pt"
            model.save_checkpoint(path)
            checkpoint_path = str(path)
            if progress:
                print(f"[{dataset}] checkpointed {name} -> {path}", flush=True)
        if train_only:
            rows.append({
                "dataset": dataset,
                "model": name,
                "checkpoint_path": checkpoint_path,
            })
            continue

        if progress:
            print(f"[{dataset}] predicting ratings for {name} on {len(rating_test):,} rows ...", flush=True)
        rating_start = perf_counter()
        y_true = rating_test["rating"].to_numpy(dtype=float)
        rating_pairs = zip(rating_test["user_id"], rating_test["parent_asin"])
        y_pred = np.array(
            [
                model.predict(u, i)
                for u, i in tqdm(
                    rating_pairs,
                    total=len(rating_test),
                    desc=f"[{dataset}] {name} ratings",
                    unit="row",
                    disable=not progress,
                )
            ]
        )
        if progress:
            print(
                f"[{dataset}] rating metrics for {name} done in "
                f"{perf_counter() - rating_start:.1f}s",
                flush=True,
            )

        per_user_bundles = []
        rng = np.random.default_rng(seed)  # reset per model -> same candidate sets
        if progress:
            print(f"[{dataset}] ranking candidates for {name} ...", flush=True)
        ranking_start = perf_counter()
        for user, rel in tqdm(
            relevant.items(),
            total=len(relevant),
            desc=f"[{dataset}] {name} ranking",
            unit="user",
            disable=not progress,
        ):
            exclude = user_train_items.get(user, set()) | rel
            negatives = sample_negatives(all_items, exclude, num_negatives, rng)
            candidates = list(rel) + negatives
            ranked = model.recommend(user, k, candidates=candidates)
            bundle = compute_user_metric_bundle(ranked, rel, k)
            if bundle is None:
                continue
            per_user_bundles.append(bundle)
        if progress:
            print(
                f"[{dataset}] ranking metrics for {name} done in "
                f"{perf_counter() - ranking_start:.1f}s",
                flush=True,
            )

        aggregated = aggregate_metric_bundle(per_user_bundles, k=k)
        row = {
            "dataset": dataset,
            "model": name,
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "split_protocol": split_protocol,
            "max_eval_users": max_eval_users,
            "max_test_rows": max_test_rows,
        }
        row.update(aggregated)  # adds precision_at_k, recall_at_k, f1_at_k, hit_rate_at_k,
                                # ndcg_at_k, oracle_*, *_oracle_ratio_at_k, k, n_eval_users
        rows.append(row)
        if metrics_path is not None:
            path = Path(metrics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(pd.DataFrame(rows).to_json(orient="records", indent=2))
            if progress:
                print(f"[{dataset}] wrote partial metrics -> {path}", flush=True)
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
    progress=False,
    include_ablation=False,
    graph=False,
    graph_only=False,
    graph_feature_set=None,
):
    """Construct the model set, sharing component instances with the hybrids.

    When ``advanced`` is True, also registers random + popularity baselines, the
    enriched content recommender, and a calibrated hybrid that reuses the svd +
    enriched-content components. When ``include_ablation`` is also True, the
    legacy ``content_enriched`` row is replaced by explicit
    ``content_enriched_with_sentiment`` and ``content_enriched_no_sentiment``
    variants that share the embedder + cache_dir but differ only in whether
    item-sentiment columns and the user-generosity offset are consumed -- so the
    sentiment contribution can be measured head-to-head.
    """
    from src.models.baselines import PopularityRecommender, RandomRecommender
    from src.models.calibrated_hybrid import CalibratedHybrid
    from src.models.cf import KNNRecommender, SVDRecommender
    from src.models.content_based import ContentBasedRecommender
    from src.models.content_enriched import ContentEnrichedRecommender
    from src.models.weighted_hybrid import WeightedHybrid

    mc = config.get("models", {})
    af = config.get("advanced_features", {})
    models = {}
    content = None
    svd = None
    blend_alpha = float(alpha if alpha is not None else config["hybrid"]["alpha"])

    if not graph_only:
        cache_dir = Path(config["processed_dir"]) / dataset / "embeddings"
        content = ContentBasedRecommender(embedder, cache_dir=cache_dir)
        svd = SVDRecommender(random_state=mc.get("ranking_random_seed", 42))

        models = {"content": content, "svd": svd}
        if not no_knn:
            models["item_knn"] = KNNRecommender()
        models["hybrid"] = WeightedHybrid(svd, content, alpha=blend_alpha)

    if advanced and not graph_only:
        assert svd is not None
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
        if not include_ablation:
            models["content_enriched"] = content_enriched
        else:
            # Ablation mode: register explicit names only. Omit the legacy
            # `content_enriched` row so the same sentiment-aware model is not
            # fitted/scored twice in the same evaluation table.
            models["content_enriched_with_sentiment"] = content_enriched
            models["content_enriched_no_sentiment"] = ContentEnrichedRecommender(
                embedder,
                generic_roots=af.get("generic_category_roots", []),
                max_vocab=int(af.get("category_vocab_max", 256)),
                min_doc_freq=int(af.get("category_min_doc_freq", 5)),
                cache_dir=af_dir / "title_desc_embeddings",
                review_features_dir=af_dir,
                use_item_sentiment=False,
                use_user_offset=False,
            )
        models["calibrated_hybrid"] = CalibratedHybrid(
            svd,
            content_enriched,
            alpha=blend_alpha,
            calibrate=True,
            calibration_max_rows=config.get("hybrid", {})
            .get("tuning", {})
            .get("calibration_max_rows"),
            random_state=seed,
            progress=progress,
        )

    if graph:
        from src.models.graphsage import GraphSAGERecommender
        from src.models.graphsage_bpr import GraphSAGEBPRRecommender
        from src.models.lightgcn import LightGCNRecommender

        gc = config.get("graph", {})
        af_dir = Path(config["processed_dir"]) / dataset / "advanced_features"
        models["lightgcn"] = LightGCNRecommender(
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
        models["graphsage"] = GraphSAGERecommender(
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
        models["graphsage_bpr"] = GraphSAGEBPRRecommender(
            embedder=embedder,
            generic_roots=af.get("generic_category_roots", []),
            max_vocab=int(af.get("category_vocab_max", 256)),
            min_doc_freq=int(af.get("category_min_doc_freq", 5)),
            hidden_dim=int(gc.get("embedding_dim", 64)),
            n_layers=int(gc.get("n_layers", 2)),
            epochs=int(gc.get("epochs", 10)),
            lr=float(gc.get("lr", 0.005)),
            weight_decay=float(gc.get("weight_decay", 0.0)),
            num_negatives=int(gc.get("num_negatives", 1)),
            batch_size=int(gc.get("batch_size", 1024)),
            seed=int(gc.get("seed", 42)),
            device=str(gc.get("device", "auto")),
            cache_dir=af_dir / "title_desc_embeddings",
            review_features_dir=af_dir,
            validation_fraction=float(gc.get("validation_fraction", 0.1)),
            progress=progress,
            feature_set=graph_feature_set or "full",
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
    parser.add_argument(
        "--include-ablation", action="store_true",
        help="register content_enriched_with_sentiment and content_enriched_no_sentiment for direct comparison",
    )
    parser.add_argument(
        "--graph", action="store_true",
        help="register lightgcn + graphsage (requires torch_geometric)",
    )
    parser.add_argument(
        "--graph-only", action="store_true",
        help="evaluate only lightgcn + graphsage; implies --graph and skips baseline/advanced models",
    )
    parser.add_argument(
        "--checkpoint-tag",
        help="append a tag to graph checkpoint filenames, e.g. 20ep -> lightgcn_20ep.pt",
    )
    parser.add_argument(
        "--only-model",
        help="fit/evaluate only one registered model, e.g. lightgcn",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="fit selected model(s), save graph checkpoints, and skip metric evaluation",
    )
    parser.add_argument(
        "--graph-epochs", type=int,
        help="override graph.epochs for this run, e.g. 20",
    )
    parser.add_argument(
        "--graph-num-negatives", type=int,
        help="override graph.num_negatives for this run, e.g. 4 for LightGCN BPR",
    )
    parser.add_argument(
        "--graph-weight-decay", type=float,
        help="override graph.weight_decay for this run, e.g. 1e-5",
    )
    parser.add_argument(
        "--graph-feature-set",
        choices=["full", "no_text", "no_sentiment", "metadata_only", "structure_only"],
        help="GraphSAGE-BPR item-node feature composition; no effect on lightgcn/graphsage (MSE)",
    )
    parser.add_argument("--alpha", type=float,
                        help="override hybrid blend alpha for this run")
    parser.add_argument(
        "--tune-alpha", action="store_true",
        help="sweep hybrid.tuning.grid on a validation slice carved from train; pick best alpha",
    )
    args = parser.parse_args(argv)

    if args.include_ablation and not args.advanced:
        parser.error("--include-ablation requires --advanced")
    if args.graph_only:
        args.graph = True
    if args.graph_only and args.advanced:
        parser.error("--graph-only cannot be combined with --advanced")
    if args.graph_only and args.tune_alpha:
        parser.error("--graph-only cannot be combined with --tune-alpha")
    if args.checkpoint_tag and not args.graph:
        parser.error("--checkpoint-tag requires --graph or --graph-only")
    if args.train_only and not args.graph:
        parser.error("--train-only requires --graph or --graph-only")
    if args.graph_epochs is not None and not args.graph:
        parser.error("--graph-epochs requires --graph or --graph-only")
    if args.graph_num_negatives is not None and not args.graph:
        parser.error("--graph-num-negatives requires --graph or --graph-only")
    if args.graph_weight_decay is not None and not args.graph:
        parser.error("--graph-weight-decay requires --graph or --graph-only")
    if args.graph_feature_set is not None and not args.graph:
        parser.error("--graph-feature-set requires --graph or --graph-only")

    from src.evaluation._audit_shared import (
        processed_dataset_key,
        requested_split_protocol,
        resolve_split_protocol,
    )

    config = load_config(args.config)
    if args.graph_epochs is not None:
        config.setdefault("graph", {})["epochs"] = args.graph_epochs
    if args.graph_num_negatives is not None:
        config.setdefault("graph", {})["num_negatives"] = args.graph_num_negatives
    if args.graph_weight_decay is not None:
        config.setdefault("graph", {})["weight_decay"] = args.graph_weight_decay
    split_config = {"split_protocol": requested_split_protocol(config)}
    split_protocol = resolve_split_protocol(
        config["processed_dir"], args.dataset, split_config
    )
    artifact_dataset = processed_dataset_key(args.dataset, split_protocol)
    train, test, metadata = _load_processed(config["processed_dir"], artifact_dataset)
    if not args.quiet:
        print(
            f"[{args.dataset}] loaded train={len(train):,}, test={len(test):,}, "
            f"metadata={len(metadata):,}",
            flush=True,
        )

    mc = config.get("models", {})
    embedder = build_embedder(config)
    if not args.quiet:
        print(f"[{args.dataset}] embedder device: {embedder.device}", flush=True)
        if not args.graph_only:
            print(
                f"[{args.dataset}] collaborative filtering device: cpu "
                "(scikit-surprise SVD/KNN)",
                flush=True,
            )

    chosen_alpha = args.alpha
    if args.tune_alpha:
        from src.evaluation.tune import tune_alpha as _tune
        from src.models.content_based import ContentBasedRecommender
        from src.models.cf import SVDRecommender
        from src.models.weighted_hybrid import WeightedHybrid

        tuning = config["hybrid"].get("tuning", {})
        cache_dir = Path(config["processed_dir"]) / artifact_dataset / "embeddings"

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
            progress=not args.quiet,
        )
        chosen_alpha = result.best_alpha
        if not args.quiet:
            print(f"[{args.dataset}] tuned alpha={chosen_alpha} from scores={result.scores}", flush=True)

    models = build_models(
        config, artifact_dataset, embedder,
        no_knn=args.no_knn,
        advanced=args.advanced,
        alpha=chosen_alpha,
        progress=not args.quiet,
        include_ablation=args.include_ablation,
        graph=args.graph,
        graph_only=args.graph_only,
        graph_feature_set=args.graph_feature_set,
    )
    if args.only_model:
        if args.only_model not in models:
            parser.error(
                f"--only-model must be one of: {', '.join(sorted(models))}"
            )
        models = {args.only_model: models[args.only_model]}
    if args.graph and not args.quiet:
        graph_devices = {
            name: str(getattr(model, "device", "n/a"))
            for name, model in models.items()
            if name in {"lightgcn", "graphsage", "graphsage_bpr"}
        }
        print(f"[{args.dataset}] graph model devices: {graph_devices}", flush=True)

    out_dir = Path(config["processed_dir"]) / artifact_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_filename = (
        f"metrics_{args.checkpoint_tag}.json"
        if args.checkpoint_tag
        else "metrics.json"
    )
    metrics_path = out_dir / metrics_filename

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
        checkpoint_dir=Path(config["processed_dir"]) / artifact_dataset / "graph_checkpoints"
        if args.graph else None,
        metrics_path=metrics_path,
        checkpoint_tag=args.checkpoint_tag,
        train_only=args.train_only,
        split_protocol=split_protocol,
    )

    if not args.train_only:
        metrics_path.write_text(table.to_json(orient="records", indent=2))
        if not args.quiet:
            print(f"[{args.dataset}] wrote final metrics.json", flush=True)
    print(table.to_string(index=False))
    if not args.train_only:
        print(f"\nMetrics -> {metrics_path}")


if __name__ == "__main__":
    main()

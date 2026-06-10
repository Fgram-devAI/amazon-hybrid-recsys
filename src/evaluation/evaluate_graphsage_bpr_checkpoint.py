"""Evaluate a stored GraphSAGE-BPR checkpoint without retraining."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from src.data.config import load_config
from src.evaluation.evaluate import _load_processed
from src.evaluation.evaluate_graphsage_checkpoint import evaluate_fitted_graphsage
from src.models.embedding import build_embedder
from src.models.graphsage_bpr import GraphSAGEBPRRecommender


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a saved GraphSAGE-BPR checkpoint.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", help="defaults to graph_checkpoints/graphsage_bpr.pt")
    parser.add_argument("--output", help="defaults to metrics_graphsage_bpr_checkpoint.json")
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    train, test, metadata = _load_processed(config["processed_dir"], args.dataset)
    processed = Path(config["processed_dir"]) / args.dataset
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        processed / "graph_checkpoints" / "graphsage_bpr.pt"
    )
    output = Path(args.output) if args.output else (
        processed / "metrics_graphsage_bpr_checkpoint.json"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    progress = not args.quiet
    mc = config.get("models", {})
    gc = config.get("graph", {})
    af = config.get("advanced_features", {})
    af_dir = processed / "advanced_features"

    embedder = build_embedder(config)
    model = GraphSAGEBPRRecommender(
        embedder=embedder,
        generic_roots=af.get("generic_category_roots", []),
        max_vocab=int(af.get("category_vocab_max", 256)),
        min_doc_freq=int(af.get("category_min_doc_freq", 5)),
        hidden_dim=int(gc.get("embedding_dim", 64)),
        n_layers=int(gc.get("n_layers", 2)),
        epochs=int(gc.get("epochs", 10)),
        lr=float(gc.get("lr", 0.005)),
        num_negatives=int(gc.get("num_negatives", 1)),
        batch_size=int(gc.get("batch_size", 1024)),
        seed=int(gc.get("seed", 42)),
        device=str(gc.get("device", "auto")),
        cache_dir=af_dir / "title_desc_embeddings",
        review_features_dir=af_dir,
        validation_fraction=float(gc.get("validation_fraction", 0.1)),
        progress=progress,
    )

    start = perf_counter()
    if progress:
        print(f"[{args.dataset}] preparing GraphSAGE-BPR inference state", flush=True)
    model.prepare_for_checkpoint(train, metadata)
    model.load_checkpoint(checkpoint)
    if model._final_embeddings is None:
        model.cache_final_embeddings()
    if progress:
        print(
            f"[{args.dataset}] loaded checkpoint in {perf_counter() - start:.1f}s -> {checkpoint}",
            flush=True,
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
    )
    table.loc[:, "model"] = "graphsage_bpr_checkpoint"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table.to_json(orient="records", indent=2))
    print(table.to_string(index=False))
    print(f"\nMetrics -> {output}")


if __name__ == "__main__":
    main()

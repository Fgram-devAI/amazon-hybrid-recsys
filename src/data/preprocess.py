"""End-to-end preprocessing: raw .jsonl.gz -> clean train/test/metadata + EDA.

Reads the active dataset from config, builds the interaction matrix (dedup +
k-core + per-user split), prepares item metadata, writes Parquet/JSON artifacts
under data/processed/<dataset_key>/, and returns the EDA summary.
"""

import json
from pathlib import Path

from .config import active_dataset_category, dataset_k_core, load_config
from .eda import summarize_eda
from .interactions import (
    apply_k_core,
    deduplicate_interactions,
    load_interactions,
    split_leave_last_out,
    split_per_user,
)
from .load import read_jsonl_gz
from .metadata import prepare_metadata
from .sources import raw_paths, resolve_existing

_MISSING_FLAGS = ["price_missing", "average_rating_missing", "rating_number_missing"]


def _test_items_not_in_train(train, test):
    """Count held-out items unseen in train; useful for later CF evaluation handling."""
    return int(len(set(test["parent_asin"]) - set(train["parent_asin"])))


def _split_dir_suffix(split_protocol: str) -> str:
    """Return the output directory suffix for a given split protocol.

    The default 80/20 protocol writes to the plain ``<dataset_key>/`` directory
    (no suffix) so that existing downstream code keeps working unchanged.
    Every other protocol gets its own sibling directory to avoid clobbering
    the baseline artifacts.
    """
    return "" if split_protocol == "per_user_chronological_80_20" else f"__{split_protocol}"


def preprocess_dataset(config, *, limit=None):
    """Run the full pipeline for the active dataset; return the EDA summary dict."""
    category = active_dataset_category(config)
    dataset_key = config["active_dataset"]
    review_path, meta_path = raw_paths(config["raw_dir"], category)
    review_path = resolve_existing(review_path)
    meta_path = resolve_existing(meta_path)
    pp = config["preprocessing"]
    split_protocol = pp.get("split_protocol", "per_user_chronological_80_20")
    dedup_policy = pp.get("dedup_policy", "latest")

    interactions, raw_count = load_interactions(read_jsonl_gz(review_path, limit=limit))
    valid_count = len(interactions)
    deduped = deduplicate_interactions(interactions, policy=dedup_policy)
    k_core = dataset_k_core(config)
    kcore = apply_k_core(deduped, k_core)

    validation = None
    if split_protocol == "leave_last_out":
        train, validation, test = split_leave_last_out(kcore)
    else:
        train, test = split_per_user(kcore, pp["test_size"], pp["random_seed"])

    metadata = prepare_metadata(read_jsonl_gz(meta_path, limit=limit))
    keep_items = set(kcore["parent_asin"].unique())
    metadata = metadata[metadata["parent_asin"].isin(keep_items)].reset_index(drop=True)

    summary = summarize_eda(
        raw_count, valid_count, deduped, kcore, train, test, pp["min_rating_relevant"]
    )
    summary["k_core_applied"] = k_core
    summary["split_protocol"] = split_protocol
    summary["dedup_policy"] = dedup_policy
    summary["test_items_not_in_train"] = _test_items_not_in_train(train, test)
    summary["metadata_items"] = len(metadata)
    summary["metadata_missingness"] = {
        col: float(metadata[col].mean()) for col in _MISSING_FLAGS if col in metadata
    }
    # Make the train/test totals interpretable when the split mode produces a
    # validation set. summarize_eda only sees train+test, so without this field
    # a leave_last_out summary would silently undercount the cohort.
    if validation is not None:
        summary["validation_interactions"] = int(len(validation))
        summary["train_test_counts_exclude_validation"] = True

    out_dir = Path(config["processed_dir"]) / f"{dataset_key}{_split_dir_suffix(split_protocol)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    kcore.to_parquet(out_dir / "interactions.parquet", index=False)
    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    if validation is not None:
        validation.to_parquet(out_dir / "validation.parquet", index=False)
    metadata.to_parquet(out_dir / "metadata.parquet", index=False)
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    """CLI: preprocess the active dataset defined in config/config.yaml."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess an Amazon Reviews dataset.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", help="override active_dataset (e.g. digital_music)")
    parser.add_argument("--limit", type=int, help="cap records (for quick dev runs)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.dataset:
        config["active_dataset"] = args.dataset

    print(f"Preprocessing '{config['active_dataset']}' ...")
    summary = preprocess_dataset(config, limit=args.limit)
    print(json.dumps(summary, indent=2))
    print(f"Artifacts -> {config['processed_dir']}/{config['active_dataset']}/")


if __name__ == "__main__":
    main()

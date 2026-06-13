"""Shared helpers for the evaluation-audit branch.

resolve_split_protocol: read split_protocol from eda_summary.json, fall back to
config. compute_checkpoint_audit_metrics: assemble the full audit dict from a
list of per-user (ranked, relevant) pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.metrics import aggregate_metric_bundle, compute_user_metric_bundle

DEFAULT_SPLIT_PROTOCOL = "per_user_chronological_80_20"


def processed_dataset_key(dataset_key: str, split_protocol: str) -> str:
    """Return the processed artifact directory key for a split protocol."""
    if split_protocol == DEFAULT_SPLIT_PROTOCOL:
        return dataset_key
    suffix = f"__{split_protocol}"
    return dataset_key if dataset_key.endswith(suffix) else f"{dataset_key}{suffix}"


def resolve_split_protocol(processed_dir, dataset_key, eval_config) -> str:
    fallback = eval_config.get("split_protocol", DEFAULT_SPLIT_PROTOCOL)
    candidates = [
        Path(processed_dir) / processed_dataset_key(dataset_key, fallback) / "eda_summary.json",
        Path(processed_dir) / dataset_key / "eda_summary.json",
    ]
    for summary_path in dict.fromkeys(candidates):
        try:
            if not summary_path.exists():
                continue
            data = json.loads(summary_path.read_text())
            if "split_protocol" in data:
                return str(data["split_protocol"])
        except (OSError, json.JSONDecodeError):
            pass
    return fallback


def compute_checkpoint_audit_metrics(per_user_data, k, split_protocol) -> dict:
    """Assemble the audit metrics dict from a list of {'ranked': [...], 'relevant': set(...)}.

    Returns the aggregated bundle with split_protocol attached. Used by the
    three checkpoint evaluators in Task 7.
    """
    bundles = []
    for entry in per_user_data:
        bundle = compute_user_metric_bundle(entry["ranked"], entry["relevant"], k)
        if bundle is not None:
            bundles.append(bundle)
    aggregated = aggregate_metric_bundle(bundles, k=k)
    aggregated["split_protocol"] = split_protocol
    return aggregated

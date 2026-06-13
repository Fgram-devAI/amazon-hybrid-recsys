"""Shared helpers for the evaluation-audit branch.

resolve_split_protocol: read split_protocol from eda_summary.json, fall back to
config. compute_checkpoint_audit_metrics: assemble the full audit dict from a
list of per-user (ranked, relevant) pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.metrics import aggregate_metric_bundle, compute_user_metric_bundle


def resolve_split_protocol(processed_dir, dataset_key, eval_config) -> str:
    summary_path = Path(processed_dir) / dataset_key / "eda_summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text())
            if "split_protocol" in data:
                return str(data["split_protocol"])
        except (OSError, json.JSONDecodeError):
            pass
    return eval_config.get("split_protocol", "per_user_chronological_80_20")


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

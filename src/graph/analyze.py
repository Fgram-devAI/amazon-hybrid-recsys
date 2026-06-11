"""Read-only graph analysis orchestrator.

Loads ``<processed_dir>/train.parquet`` (train-only — leakage rule), builds
the bipartite graph + item-item projection, computes structural EDA, runs
the configured community / spectral methods, computes category alignment from
``metadata.parquet`` when category labels can be derived (or from
``item_labels.parquet`` in tiny tests/manual overrides), and writes one JSON report to
``<processed_dir>/graph_analysis/report.json``.

CLI: ``./.venv/bin/python -m src.graph.analyze --dataset <key>``

All generated artifacts (report.json, figures, projections) stay under
``data/processed/<dataset>/graph_analysis/`` and are gitignored.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.config import load_config
from src.graph.build import build_train_bipartite_graph
from src.graph.communities import (
    CommunityResult,
    compute_alignment,
    run_girvan_newman,
    run_leiden,
    run_louvain,
    run_spectral,
)
from src.graph.eda import compute_bipartite_eda, compute_projection_eda
from src.graph.projection import project_item_item

_LOG = logging.getLogger(__name__)


def _normalise_categories(value: object) -> list[str]:
    """Return a normalized category list from metadata values."""
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(value)] if str(value).strip() else []


def _load_labels(processed_dir: Path, generic_roots: list[str] | None = None) -> dict[str, str]:
    """Return {item_id: category_label} from item_labels.parquet or metadata.parquet.

    ``item_labels.parquet`` is supported for tiny tests/manual overrides. Real
    dataset runs derive labels from ``metadata.parquet`` by taking the first
    non-generic category after removing configured generic roots such as
    ``Video Games`` / ``Movies & TV`` / ``Digital Music``. If no labels can be
    derived, return an empty dict and downstream alignment is reported as None.
    """
    path = processed_dir / "item_labels.parquet"
    if not path.exists():
        meta_path = processed_dir / "metadata.parquet"
        if not meta_path.exists():
            return {}
        meta = pd.read_parquet(meta_path)
        roots = set(generic_roots or [])
        labels: dict[str, str] = {}
        for row in meta.itertuples(index=False):
            item = str(getattr(row, "parent_asin"))
            cats = _normalise_categories(getattr(row, "categories", None))
            filtered = [c for c in cats if c not in roots]
            if filtered:
                labels[item] = filtered[-1]
        return labels
    df = pd.read_parquet(path)
    return dict(zip(df["parent_asin"], df["category"]))


def _serialize_result(
    result: CommunityResult | None,
    item_to_label: dict[str, str],
) -> dict[str, Any] | None:
    if result is None:
        return None
    payload = asdict(result)
    # Sets are not JSON-serializable.
    payload["communities"] = [sorted(c) for c in result.communities]
    payload["n_communities"] = len(result.communities)
    payload["sizes"] = sorted(
        (len(c) for c in result.communities), reverse=True
    )
    payload["alignment"] = (
        compute_alignment(result.communities, item_to_label) if item_to_label else None
    )
    return payload


def _run_girvan_newman_louvain_subgraph(
    item_graph: Any,
    louvain_result: CommunityResult,
    max_nodes: int,
    weight: str,
) -> CommunityResult | None:
    """Run Girvan-Newman on top-degree nodes from the largest Louvain community."""
    if not louvain_result.communities:
        return None

    largest = max(louvain_result.communities, key=len)
    if not largest:
        return None

    ranked_nodes = sorted(largest, key=lambda node: item_graph.degree(node), reverse=True)
    selected = ranked_nodes[:max_nodes]
    subgraph = item_graph.subgraph(selected).copy()
    result = run_girvan_newman(subgraph, max_nodes=max_nodes, weight=weight)
    return replace(
        result,
        method="girvan_newman_louvain_subgraph",
        extras={
            **result.extras,
            "source": "largest_louvain_community_top_degree",
            "source_louvain_community_size": len(largest),
            "n_nodes_selected": len(selected),
        },
    )


def run_graph_analysis(
    processed_dir: Path,
    dataset: str,
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    output_name: str = "report.json",
) -> dict[str, Any]:
    """Run the full read-only analysis pipeline; write a JSON report."""
    if Path(output_name).name != output_name or not output_name.endswith(".json"):
        raise ValueError("output_name must be a JSON filename without directories")

    knobs = dict(config["graph_analysis"])
    if overrides:
        knobs.update({k: v for k, v in overrides.items() if v is not None})

    _LOG.info("loading train split from %s", processed_dir / "train.parquet")
    train = pd.read_parquet(processed_dir / "train.parquet")
    _LOG.info("building train-only bipartite graph (%d interactions)", len(train))
    bg = build_train_bipartite_graph(train)
    _LOG.info(
        "projecting item-item graph (min_shared_users=%s, top_n_items=%s)",
        knobs["min_shared_users"],
        knobs.get("top_n_items"),
    )
    item_graph = project_item_item(
        bg,
        min_shared_users=int(knobs["min_shared_users"]),
        top_n_items=knobs.get("top_n_items"),
    )
    _LOG.info(
        "projection built: nodes=%d, edges=%d",
        item_graph.number_of_nodes(),
        item_graph.number_of_edges(),
    )

    weight = knobs.get("projection_weight", "weight_jaccard")
    seed = int(knobs.get("random_state", 42))
    _LOG.info("loading category labels for alignment")
    labels = _load_labels(
        processed_dir,
        generic_roots=config.get("advanced_features", {}).get("generic_category_roots", []),
    )
    _LOG.info("loaded %d item labels", len(labels))

    communities: dict[str, dict[str, Any] | None] = {}
    _LOG.info("running Louvain community detection")
    louvain_result = run_louvain(item_graph, weight=weight, seed=seed)
    communities["louvain"] = _serialize_result(louvain_result, labels)
    _LOG.info("running Leiden community detection (optional)")
    communities["leiden"] = _serialize_result(
        run_leiden(item_graph, weight=weight, seed=seed), labels
    )
    for k in knobs.get("spectral_k_values", []):
        try:
            _LOG.info("running spectral clustering (k=%s)", k)
            communities[f"spectral_k={k}"] = _serialize_result(
                run_spectral(item_graph, k=int(k), weight=weight, random_state=seed),
                labels,
            )
        except ValueError as exc:
            _LOG.warning("Skipping spectral_k=%s: %s", k, exc)
            communities[f"spectral_k={k}"] = None

    try:
        _LOG.info("running Girvan-Newman small-subgraph baseline")
        communities["girvan_newman"] = _serialize_result(
            run_girvan_newman(
                item_graph,
                max_nodes=int(knobs.get("girvan_newman_max_nodes", 500)),
                weight=weight,
            ),
            labels,
        )
    except ValueError as exc:
        _LOG.warning(
            "Skipping Girvan-Newman on full graph (%s); pass a subgraph for the report.",
            exc,
        )
        communities["girvan_newman"] = None

    _LOG.info("running Girvan-Newman on largest Louvain community subgraph")
    communities["girvan_newman_louvain_subgraph"] = _serialize_result(
        _run_girvan_newman_louvain_subgraph(
            item_graph,
            louvain_result,
            max_nodes=int(knobs.get("girvan_newman_max_nodes", 500)),
            weight=weight,
        ),
        labels,
    )

    report: dict[str, Any] = {
        "dataset": dataset,
        "graph_analysis_config": knobs,
        "bipartite_eda": compute_bipartite_eda(bg),
        "projection_eda": compute_projection_eda(item_graph),
        "communities": communities,
    }

    # Ensure the in-memory report is JSON-round-trip clean so the equality
    # check (on_disk == report) holds even if upstream EDA returns numpy scalars.
    report = json.loads(json.dumps(report, default=str))

    out_dir = processed_dir / "graph_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / output_name).write_text(json.dumps(report, indent=2))
    _LOG.info("wrote %s", out_dir / output_name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train-only graph analysis.")
    parser.add_argument(
        "--dataset",
        help="Dataset key from config/config.yaml (defaults to active_dataset).",
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config YAML."
    )
    parser.add_argument(
        "--min-shared-users",
        type=int,
        help="Override graph_analysis.min_shared_users for this run.",
    )
    parser.add_argument(
        "--top-n-items",
        type=int,
        help="Override graph_analysis.top_n_items for this run.",
    )
    parser.add_argument(
        "--spectral-k-values",
        help="Comma-separated override for graph_analysis.spectral_k_values, e.g. 10,25,50.",
    )
    parser.add_argument(
        "--output-name",
        default="report.json",
        help="JSON filename under data/processed/<dataset>/graph_analysis/.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    dataset = args.dataset or config["active_dataset"]
    processed_dir = Path(config["processed_dir"]) / dataset
    if not processed_dir.exists():
        raise SystemExit(f"processed dir not found: {processed_dir}")

    spectral_k_values = None
    if args.spectral_k_values:
        spectral_k_values = [
            int(part.strip()) for part in args.spectral_k_values.split(",") if part.strip()
        ]
    overrides = {
        "min_shared_users": args.min_shared_users,
        "top_n_items": args.top_n_items,
        "spectral_k_values": spectral_k_values,
    }
    report = run_graph_analysis(
        processed_dir,
        dataset,
        config,
        overrides=overrides,
        output_name=args.output_name,
    )
    _LOG.info(
        "completed graph analysis: output=%s (%d items, %d edges)",
        args.output_name,
        report["projection_eda"]["n_nodes"],
        report["projection_eda"]["n_edges"],
    )


if __name__ == "__main__":
    main()

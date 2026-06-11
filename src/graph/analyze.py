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
from dataclasses import asdict
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


def run_graph_analysis(
    processed_dir: Path,
    dataset: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the full read-only analysis pipeline; write a JSON report."""
    train = pd.read_parquet(processed_dir / "train.parquet")
    knobs = config["graph_analysis"]
    bg = build_train_bipartite_graph(train)
    item_graph = project_item_item(
        bg,
        min_shared_users=int(knobs["min_shared_users"]),
        top_n_items=knobs.get("top_n_items"),
    )

    weight = knobs.get("projection_weight", "weight_jaccard")
    seed = int(knobs.get("random_state", 42))
    labels = _load_labels(
        processed_dir,
        generic_roots=config.get("advanced_features", {}).get("generic_category_roots", []),
    )

    communities: dict[str, dict[str, Any] | None] = {}
    communities["louvain"] = _serialize_result(
        run_louvain(item_graph, weight=weight, seed=seed), labels
    )
    communities["leiden"] = _serialize_result(
        run_leiden(item_graph, weight=weight, seed=seed), labels
    )
    for k in knobs.get("spectral_k_values", []):
        try:
            communities[f"spectral_k={k}"] = _serialize_result(
                run_spectral(item_graph, k=int(k), weight=weight, random_state=seed),
                labels,
            )
        except ValueError as exc:
            _LOG.warning("Skipping spectral_k=%s: %s", k, exc)
            communities[f"spectral_k={k}"] = None

    try:
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
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    dataset = args.dataset or config["active_dataset"]
    processed_dir = Path(config["processed_dir"]) / dataset
    if not processed_dir.exists():
        raise SystemExit(f"processed dir not found: {processed_dir}")

    report = run_graph_analysis(processed_dir, dataset, config)
    _LOG.info(
        "wrote graph_analysis/report.json (%d items, %d edges)",
        report["projection_eda"]["n_nodes"],
        report["projection_eda"]["n_edges"],
    )


if __name__ == "__main__":
    main()

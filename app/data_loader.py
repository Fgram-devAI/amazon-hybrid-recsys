"""Dashboard data loading layer.

Prefers ``data/processed/<dataset>/`` artifacts when present, otherwise reads the
committed curated bundle under ``app/assets/demo/``. Each non-EDA artifact falls
back to the demo bundle independently and emits a note describing the fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DEMO_DIR = _PACKAGE_DIR / "assets" / "demo"
DEFAULT_PROCESSED_DIR = Path("data/processed")


@dataclass
class DashboardData:
    dataset: str
    mode: str
    eda_summary: dict[str, Any]
    model_metrics: dict[str, Any]
    graph_analysis: dict[str, Any] | None
    graph_subgraph_3d: dict[str, Any] | None
    sample_items: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dashboard_data(
    dataset: str = "video_games",
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    demo_dir: Path = DEFAULT_DEMO_DIR,
) -> DashboardData:
    local_dir = processed_dir / dataset
    has_local_eda = local_dir.is_dir() and (local_dir / "eda_summary.json").is_file()
    notes: list[str] = []

    if has_local_eda:
        mode = "local"
        eda = _read_json(local_dir / "eda_summary.json")

        local_metrics_path = local_dir / "model_metrics_summary.json"
        if local_metrics_path.is_file():
            metrics = _read_json(local_metrics_path)
        else:
            metrics = _read_json(demo_dir / "model_metrics_summary.json")
            notes.append(
                "model_metrics: using bundled demo summary "
                "(no local model_metrics_summary.json found)."
            )

        local_graph_path = local_dir / "graph_analysis_summary.json"
        demo_graph_path = demo_dir / "graph_analysis_summary.json"
        if local_graph_path.is_file():
            graph = _read_json(local_graph_path)
        elif demo_graph_path.is_file():
            graph = _read_json(demo_graph_path)
            notes.append("graph_analysis: using bundled demo summary.")
        else:
            graph = None
            notes.append("graph_analysis: no summary available.")

        local_graph_3d_path = local_dir / "graph_subgraph_3d.json"
        demo_graph_3d_path = demo_dir / "graph_subgraph_3d.json"
        if local_graph_3d_path.is_file():
            graph_3d = _read_json(local_graph_3d_path)
        elif demo_graph_3d_path.is_file():
            graph_3d = _read_json(demo_graph_3d_path)
            notes.append("graph_subgraph_3d: using bundled capped demo graph.")
        else:
            graph_3d = None
            notes.append("graph_subgraph_3d: no capped graph sample available.")

        local_items_path = local_dir / "sample_items.json"
        if local_items_path.is_file():
            sample_items = _read_json(local_items_path)
        else:
            sample_items = _read_json(demo_dir / "sample_items.json")
            notes.append("sample_items: using bundled demo items.")
    else:
        mode = "demo"
        eda = _read_json(demo_dir / f"{dataset}_eda_summary.json")
        metrics = _read_json(demo_dir / "model_metrics_summary.json")
        demo_graph_path = demo_dir / "graph_analysis_summary.json"
        graph = _read_json(demo_graph_path) if demo_graph_path.is_file() else None
        demo_graph_3d_path = demo_dir / "graph_subgraph_3d.json"
        graph_3d = _read_json(demo_graph_3d_path) if demo_graph_3d_path.is_file() else None
        sample_items = _read_json(demo_dir / "sample_items.json")

    assert isinstance(eda, dict)
    assert isinstance(metrics, dict)
    assert isinstance(sample_items, list)

    return DashboardData(
        dataset=dataset,
        mode=mode,
        eda_summary=eda,
        model_metrics=metrics,
        graph_analysis=graph,
        graph_subgraph_3d=graph_3d,
        sample_items=sample_items,
        notes=notes,
    )

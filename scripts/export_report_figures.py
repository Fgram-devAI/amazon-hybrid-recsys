"""Export report-ready figures from the Streamlit dashboard artifacts.

The Streamlit app renders most charts from pandas DataFrames, so this script
loads the same artifacts and writes static PNG/PDF figures for Overleaf.

Example:
    ./.venv/bin/python scripts/export_report_figures.py --dataset video_games
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from app import charts
from app.data_loader import DEFAULT_DEMO_DIR, DEFAULT_PROCESSED_DIR, load_dashboard_data


DEFAULT_OUTPUT_DIR = Path("docs/report/figures")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _save(fig: plt.Figure, output_dir: Path, name: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(f"wrote {path}")
    plt.close(fig)


def _bar(
    df: pd.DataFrame,
    *,
    x: str,
    y: str | list[str],
    title: str,
    ylabel: str,
    rotate: int = 0,
    figsize: tuple[float, float] = (8.0, 4.2),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    df.plot(kind="bar", x=x, y=y, ax=ax, rot=rotate)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _plot_preprocessing_funnel(eda: dict[str, Any], output_dir: Path, formats: list[str]) -> None:
    df = charts.preprocessing_funnel(eda)
    if df.empty:
        return
    fig = _bar(
        df,
        x="step",
        y="count",
        title="Video Games preprocessing funnel",
        ylabel="Interactions",
        rotate=20,
    )
    _save(fig, output_dir, "preprocessing_funnel", formats)


def _plot_rating_distribution(eda: dict[str, Any], output_dir: Path, formats: list[str]) -> None:
    hist = eda.get("rating_hist_after")
    if not isinstance(hist, dict):
        return
    df = charts.rating_histogram(hist)
    fig = _bar(
        df,
        x="rating",
        y="count",
        title="Rating distribution after k-core",
        ylabel="Ratings",
        figsize=(7.0, 4.0),
    )
    _save(fig, output_dir, "rating_distribution", formats)


def _table(metrics: dict[str, Any], name: str, label_col: str = "model") -> pd.DataFrame:
    rows = metrics.get("tables", {}).get(name, [])
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    return charts.metrics_table(rows, label_col=label_col)


def _plot_advanced_metrics(metrics: dict[str, Any], output_dir: Path, formats: list[str]) -> None:
    df = _table(metrics, "advanced")
    if df.empty:
        return
    if {"Model", "RMSE", "MAE"}.issubset(df.columns):
        fig = _bar(
            df,
            x="Model",
            y=["RMSE", "MAE"],
            title="Rating prediction metrics",
            ylabel="Error",
            rotate=25,
            figsize=(9.5, 4.4),
        )
        _save(fig, output_dir, "model_error_metrics", formats)
    rank_cols = [col for col in ["P@10", "R@10", "F1@10"] if col in df.columns]
    if rank_cols:
        fig = _bar(
            df,
            x="Model",
            y=rank_cols,
            title="Sampled-candidate ranking metrics",
            ylabel="Score",
            rotate=25,
            figsize=(9.5, 4.4),
        )
        _save(fig, output_dir, "model_ranking_metrics", formats)


def _plot_graph_metrics(metrics: dict[str, Any], output_dir: Path, formats: list[str]) -> None:
    df = _table(metrics, "graph")
    if df.empty or "F1@10" not in df.columns:
        return
    plot_df = df[["Model", "F1@10"]].copy()
    fig = _bar(
        plot_df,
        x="Model",
        y="F1@10",
        title="Graph checkpoint F1@10",
        ylabel="F1@10",
        rotate=30,
        figsize=(10.5, 4.2),
    )
    _save(fig, output_dir, "graph_checkpoint_f1", formats)


def _plot_graph_analysis(graph: dict[str, Any] | None, output_dir: Path, formats: list[str]) -> None:
    if not graph:
        return
    rows = graph.get("projections", [])
    if not isinstance(rows, list) or not rows:
        return
    df = pd.DataFrame(rows)
    if {"projection", "items", "edges", "largest_cc"}.issubset(df.columns):
        fig = _bar(
            df,
            x="projection",
            y=["items", "edges", "largest_cc"],
            title="Item-item projection scale",
            ylabel="Count",
            rotate=25,
            figsize=(9.5, 4.4),
        )
        _save(fig, output_dir, "graph_projection_scale", formats)

    alignment_cols = [
        col for col in ["louvain_purity", "louvain_nmi", "spectral_k50_nmi"] if col in df.columns
    ]
    if alignment_cols and "projection" in df.columns:
        fig, ax = plt.subplots(figsize=(9.5, 4.4))
        df.plot(x="projection", y=alignment_cols, marker="o", ax=ax)
        ax.set_title("Community/category alignment")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        _save(fig, output_dir, "community_alignment", formats)


def _export_3d_graph(
    payload: dict[str, Any] | None,
    output_dir: Path,
    formats: list[str],
) -> None:
    if not payload:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = charts.graph_subgraph_3d_figure(payload)
    html_path = output_dir / "graph_subgraph_3d.html"
    fig.write_html(html_path)
    print(f"wrote {html_path}")
    for fmt in formats:
        path = output_dir / f"graph_subgraph_3d.{fmt}"
        try:
            fig.write_image(path)
        except Exception as exc:
            print(f"skipped {path}: Plotly static export unavailable ({exc})")
            continue
        print(f"wrote {path}")


def _write_manifest(output_dir: Path, dataset: str, mode: str, notes: list[str]) -> None:
    payload = {
        "dataset": dataset,
        "mode": mode,
        "notes": notes,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export static report figures from app artifacts.")
    parser.add_argument("--dataset", default="video_games")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated matplotlib formats for static figures (default: png,pdf).",
    )
    parser.add_argument(
        "--skip-3d",
        action="store_true",
        help="Skip Plotly 3D graph export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    data = load_dashboard_data(
        args.dataset,
        processed_dir=args.processed_dir,
        demo_dir=args.demo_dir,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_preprocessing_funnel(data.eda_summary, output_dir, formats)
    _plot_rating_distribution(data.eda_summary, output_dir, formats)
    _plot_advanced_metrics(data.model_metrics, output_dir, formats)
    _plot_graph_metrics(data.model_metrics, output_dir, formats)
    _plot_graph_analysis(data.graph_analysis, output_dir, formats)
    if not args.skip_3d:
        _export_3d_graph(data.graph_subgraph_3d, output_dir, formats)
    _write_manifest(output_dir, data.dataset, data.mode, data.notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

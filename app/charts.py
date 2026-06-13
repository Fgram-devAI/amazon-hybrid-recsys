"""Reusable pandas DataFrame helpers for the Streamlit dashboard.

Pure data shaping — no Streamlit imports. Streamlit consumes these via
``st.dataframe``, ``st.bar_chart``, and one Plotly 3D graph sample.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

METRIC_COLUMNS = ["rmse", "mae", "p_at_10", "r_at_10", "f1_at_10"]
GRAPH_3D_NODE_CAP = 300
GRAPH_3D_EDGE_CAP = 2_000
METRIC_LABELS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "p_at_10": "P@10",
    "r_at_10": "R@10",
    "f1_at_10": "F1@10",
}


def metrics_table(rows: list[dict[str, Any]], label_col: str = "model") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keep = [label_col] + [c for c in METRIC_COLUMNS if c in df.columns]
    df = df[keep].copy()
    rename = {label_col: label_col.replace("_", " ").title(), **METRIC_LABELS}
    return df.rename(columns=rename)


def rating_histogram(hist: dict[str, int]) -> pd.DataFrame:
    rows = [{"rating": float(k), "count": int(v)} for k, v in hist.items()]
    return pd.DataFrame(rows).sort_values("rating").reset_index(drop=True)


def preprocessing_funnel(eda: dict[str, Any]) -> pd.DataFrame:
    k = eda.get("k_core_applied", "k")
    steps = [
        ("raw_reviews", "Raw reviews"),
        ("after_drop_invalid", "After drop invalid"),
        ("after_dedup", "After dedup"),
        ("after_kcore", f"After {k}-core"),
    ]
    rows = [{"step": label, "count": int(eda[key])} for key, label in steps if key in eda]
    return pd.DataFrame(rows)


def graph_subgraph_3d_figure(
    payload: dict[str, Any],
    *,
    node_cap: int = GRAPH_3D_NODE_CAP,
    edge_cap: int = GRAPH_3D_EDGE_CAP,
) -> go.Figure:
    raw_nodes = payload.get("nodes", [])
    raw_edges = payload.get("edges", [])
    nodes = sorted(
        raw_nodes,
        key=lambda node: float(node.get("degree", 0)),
        reverse=True,
    )[:node_cap]
    node_by_id = {node["id"]: node for node in nodes}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    kept_edges = 0
    for edge in raw_edges:
        if kept_edges >= edge_cap:
            break
        source = node_by_id.get(edge.get("source"))
        target = node_by_id.get(edge.get("target"))
        if not source or not target:
            continue
        edge_x.extend([source["x"], target["x"], None])
        edge_y.extend([source["y"], target["y"], None])
        edge_z.extend([source["z"], target["z"], None])
        kept_edges += 1

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line={"width": 1, "color": "rgba(120,120,120,0.35)"},
        hoverinfo="none",
        name="co-rating links",
    )
    node_trace = go.Scatter3d(
        x=[node["x"] for node in nodes],
        y=[node["y"] for node in nodes],
        z=[node["z"] for node in nodes],
        mode="markers",
        marker={
            "size": [max(4, min(14, float(node.get("degree", 1)) ** 0.5)) for node in nodes],
            "color": [node.get("community", 0) for node in nodes],
            "colorscale": "Viridis",
            "opacity": 0.9,
        },
        text=[
            f"{node.get('label', node.get('id'))}<br>{node.get('category', '-')}"
            for node in nodes
        ],
        hoverinfo="text",
        name="items",
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=620,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        title=_graph_cap_title(
            len(raw_nodes),
            len(raw_edges),
            len(nodes),
            kept_edges,
            node_cap,
            edge_cap,
        ),
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
        showlegend=False,
    )
    return fig


def _graph_cap_title(
    raw_node_count: int,
    raw_edge_count: int,
    shown_node_count: int,
    shown_edge_count: int,
    node_cap: int,
    edge_cap: int,
) -> str:
    if raw_node_count <= node_cap and raw_edge_count <= edge_cap:
        return "Largest Louvain community sample"
    return (
        "Largest Louvain community sample "
        f"(showing {shown_node_count}/{raw_node_count} nodes, "
        f"{shown_edge_count}/{raw_edge_count} edges)"
    )

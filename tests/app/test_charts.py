"""Tests for app.charts: pure DataFrame builders."""
from __future__ import annotations

from app.charts import (
    GRAPH_3D_EDGE_CAP,
    GRAPH_3D_NODE_CAP,
    METRIC_COLUMNS,
    graph_subgraph_3d_figure,
    metrics_table,
    preprocessing_funnel,
    rating_histogram,
)


def test_metrics_table_keeps_label_first_and_drops_missing_metric_columns() -> None:
    rows = [
        {"model": "svd", "rmse": 1.13, "mae": 0.82, "p_at_10": 0.03},
        {"model": "popularity", "rmse": 1.23, "mae": 0.91, "p_at_10": 0.08},
    ]
    df = metrics_table(rows, label_col="model")
    assert list(df.columns)[0].lower() == "model"
    assert "R@10" not in df.columns
    assert "F1@10" not in df.columns
    assert "RMSE" in df.columns
    assert df.iloc[0]["RMSE"] == 1.13


def test_metrics_table_supports_feature_set_label() -> None:
    rows = [
        {
            "feature_set": "full",
            "rmse": 1.25,
            "mae": 0.98,
            "p_at_10": 0.03,
            "r_at_10": 0.21,
            "f1_at_10": 0.05,
        },
    ]
    df = metrics_table(rows, label_col="feature_set")
    assert "feature" in df.columns[0].lower()
    assert set(df.columns) >= {"RMSE", "MAE", "P@10", "R@10", "F1@10"}


def test_rating_histogram_returns_sorted_two_column_frame() -> None:
    hist = {"5.0": 100, "1.0": 5, "3.0": 20, "2.0": 7, "4.0": 30}
    df = rating_histogram(hist)
    assert list(df.columns) == ["rating", "count"]
    assert list(df["rating"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert df.loc[df["rating"] == 5.0, "count"].iloc[0] == 100


def test_preprocessing_funnel_orders_steps_and_uses_k_label() -> None:
    eda = {
        "raw_reviews": 100,
        "after_drop_invalid": 95,
        "after_dedup": 90,
        "after_kcore": 80,
        "k_core_applied": 5,
    }
    df = preprocessing_funnel(eda)
    assert list(df["count"]) == [100, 95, 90, 80]
    assert df.iloc[-1]["step"] == "After 5-core"


def test_metric_columns_constant_has_all_five_keys() -> None:
    assert METRIC_COLUMNS == ["rmse", "mae", "p_at_10", "r_at_10", "f1_at_10"]


def test_graph_subgraph_3d_figure_handles_small_payload() -> None:
    payload = {
        "nodes": [
            {
                "id": "A",
                "label": "Alpha",
                "category": "RPG",
                "community": 1,
                "degree": 3,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "id": "B",
                "label": "Beta",
                "category": "RPG",
                "community": 1,
                "degree": 2,
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
            },
        ],
        "edges": [{"source": "A", "target": "B", "weight": 0.5}],
    }
    fig = graph_subgraph_3d_figure(payload)
    assert len(list(fig.data)) == 2


def test_graph_subgraph_3d_figure_caps_large_payload() -> None:
    nodes = [
        {
            "id": f"N{i}",
            "label": f"Node {i}",
            "category": "Games",
            "community": 1,
            "degree": i,
            "x": float(i),
            "y": float(i % 10),
            "z": float(i % 7),
        }
        for i in range(GRAPH_3D_NODE_CAP + 50)
    ]
    edges = [
        {
            "source": f"N{i % GRAPH_3D_NODE_CAP}",
            "target": f"N{(i + 1) % GRAPH_3D_NODE_CAP}",
            "weight": 0.1,
        }
        for i in range(GRAPH_3D_EDGE_CAP + 500)
    ]

    fig = graph_subgraph_3d_figure({"nodes": nodes, "edges": edges})
    figure_json = fig.to_plotly_json()
    edge_trace, node_trace = figure_json["data"]

    assert len(node_trace["x"]) == GRAPH_3D_NODE_CAP
    assert len(edge_trace["x"]) <= GRAPH_3D_EDGE_CAP * 3
    assert "showing" in str(fig.layout.title.text)

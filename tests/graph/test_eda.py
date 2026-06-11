"""Structural EDA on tiny synthetic graphs with hand-checkable values."""

from __future__ import annotations

import pytest

from src.graph.build import build_train_bipartite_graph
from src.graph.eda import compute_bipartite_eda, compute_projection_eda
from src.graph.projection import project_item_item


def test_bipartite_eda_reports_counts_density_and_components(coratings_toy) -> None:
    bg = build_train_bipartite_graph(coratings_toy)
    eda = compute_bipartite_eda(bg)

    assert eda["n_users"] == 5
    assert eda["n_items"] == 3
    assert eda["n_edges"] == 10
    # Density = edges / (n_users * n_items) = 10 / 15
    assert eda["density"] == pytest.approx(10 / 15)
    assert eda["n_connected_components"] == 1
    assert eda["largest_component_size"] == 8
    # User-side degree summary.
    assert eda["user_degree"]["min"] == 1
    assert eda["user_degree"]["max"] == 3
    # Item-side degree summary.
    assert eda["item_degree"]["min"] == 2
    assert eda["item_degree"]["max"] == 4


def test_projection_eda_includes_both_weight_distributions(coratings_toy) -> None:
    bg = build_train_bipartite_graph(coratings_toy)
    item_graph = project_item_item(bg, min_shared_users=1)
    eda = compute_projection_eda(item_graph)

    assert eda["n_nodes"] == 3
    assert eda["n_edges"] == item_graph.number_of_edges()
    assert eda["density"] == pytest.approx(
        2 * item_graph.number_of_edges() / (3 * 2)
    )
    assert "weight_count" in eda["weights"]
    assert "weight_jaccard" in eda["weights"]
    assert eda["weights"]["weight_count"]["max"] == 4
    assert eda["weights"]["weight_jaccard"]["max"] == pytest.approx(1.0)
    # Clustering coefficient reported (defined; may be 0 on this toy).
    assert "clustering_coefficient_mean" in eda


def test_projection_eda_handles_empty_edge_set(coratings_toy) -> None:
    bg = build_train_bipartite_graph(coratings_toy)
    item_graph = project_item_item(bg, min_shared_users=10)
    eda = compute_projection_eda(item_graph)
    assert eda["n_edges"] == 0
    assert eda["weights"]["weight_count"]["count"] == 0
    assert eda["weights"]["weight_jaccard"]["count"] == 0

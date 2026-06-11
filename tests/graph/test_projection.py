"""Item-item projection must store BOTH weight_count and weight_jaccard."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from src.graph.build import build_train_bipartite_graph
from src.graph.projection import project_item_item, to_sparse_adjacency


def test_projection_stores_both_weights(coratings_toy) -> None:
    bg = build_train_bipartite_graph(coratings_toy)
    item_graph = project_item_item(bg, min_shared_users=1)

    # Edge (i1, i2): 4 shared users out of 4 union users -> jaccard = 1.0
    assert item_graph["i1"]["i2"]["weight_count"] == 4
    assert item_graph["i1"]["i2"]["weight_jaccard"] == pytest.approx(1.0)

    # Edge (i1, i3): 1 shared user (u1) out of union {u1..u5} = 5 -> 0.2
    assert item_graph["i1"]["i3"]["weight_count"] == 1
    assert item_graph["i1"]["i3"]["weight_jaccard"] == pytest.approx(0.2)


def test_min_shared_users_filters_thin_edges(coratings_toy) -> None:
    bg = build_train_bipartite_graph(coratings_toy)
    item_graph = project_item_item(bg, min_shared_users=2)
    # Only (i1, i2) survives count >= 2.
    assert set(item_graph.edges()) == {("i1", "i2")}


def test_top_n_items_cap_by_train_degree(coratings_toy) -> None:
    bg = build_train_bipartite_graph(coratings_toy)
    # Cap to top 2 items by train degree -> drops i3 (lowest degree).
    item_graph = project_item_item(bg, min_shared_users=1, top_n_items=2)
    assert set(item_graph.nodes()) == {"i1", "i2"}


def test_jaccard_symmetry_and_range(coratings_toy) -> None:
    bg = build_train_bipartite_graph(coratings_toy)
    item_graph = project_item_item(bg, min_shared_users=1)
    for u, v, data in item_graph.edges(data=True):
        j = data["weight_jaccard"]
        assert 0.0 < j <= 1.0
        # Undirected: same edge under reversed lookup.
        assert math.isclose(item_graph[v][u]["weight_jaccard"], j)


def test_to_sparse_adjacency_returns_csr_with_chosen_weight(coratings_toy) -> None:
    bg = build_train_bipartite_graph(coratings_toy)
    item_graph = project_item_item(bg, min_shared_users=1)
    adj, idx_to_item = to_sparse_adjacency(item_graph, weight="weight_jaccard")
    n = item_graph.number_of_nodes()
    assert adj.shape == (n, n)
    # Symmetry.
    diff = (adj - adj.T).tocoo()
    assert diff.nnz == 0
    # The chosen weight is what came out.
    i1 = idx_to_item.index("i1")
    i2 = idx_to_item.index("i2")
    assert adj[i1, i2] == pytest.approx(1.0)


def test_unweighted_isolated_items_kept_when_below_threshold(coratings_toy) -> None:
    """Items with zero qualifying edges still appear as isolated nodes."""
    bg = build_train_bipartite_graph(coratings_toy)
    item_graph = project_item_item(bg, min_shared_users=10)  # impossibly high
    assert isinstance(item_graph, nx.Graph)
    assert set(item_graph.nodes()) == {"i1", "i2", "i3"}
    assert item_graph.number_of_edges() == 0

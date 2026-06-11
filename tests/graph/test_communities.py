"""Community detection + category alignment on tiny planted graphs."""

from __future__ import annotations

import networkx as nx
import pytest

from src.graph.build import build_train_bipartite_graph
from src.graph.communities import (
    CommunityResult,
    compute_alignment,
    run_girvan_newman,
    run_louvain,
    run_spectral,
)
from src.graph.projection import project_item_item


def test_run_louvain_recovers_planted_two_communities(two_clique_train) -> None:
    bg = build_train_bipartite_graph(two_clique_train)
    item_graph = project_item_item(bg, min_shared_users=2)
    result = run_louvain(item_graph, weight="weight_jaccard", seed=42)

    assert isinstance(result, CommunityResult)
    # Every item appears in exactly one community.
    flat = [n for community in result.communities for n in community]
    assert sorted(flat) == sorted(item_graph.nodes())
    assert len(flat) == len(set(flat))
    # Two planted communities recovered.
    assert len(result.communities) == 2
    cluster_a = {f"iA{i}" for i in (1, 2, 3)}
    cluster_b = {f"iB{i}" for i in (1, 2, 3)}
    assert {frozenset(c) for c in result.communities} == {
        frozenset(cluster_a),
        frozenset(cluster_b),
    }
    assert result.modularity is not None
    assert result.modularity > 0.0
    assert result.method == "louvain"


def test_compute_alignment_purity_and_nmi_perfect_match() -> None:
    g = nx.Graph()
    g.add_nodes_from(["a", "b", "c", "d"])
    partition = [{"a", "b"}, {"c", "d"}]
    labels = {"a": "x", "b": "x", "c": "y", "d": "y"}
    align = compute_alignment(partition, labels)
    assert align["purity"] == pytest.approx(1.0)
    assert align["nmi"] == pytest.approx(1.0)
    assert align["n_labeled_items"] == 4


def test_compute_alignment_ignores_items_without_labels() -> None:
    partition = [{"a", "b", "c"}, {"d", "e"}]
    labels = {"a": "x", "b": "x", "d": "y"}  # c and e unlabeled
    align = compute_alignment(partition, labels)
    assert align["n_labeled_items"] == 3
    assert 0.0 < align["purity"] <= 1.0


def test_compute_alignment_low_for_random_partition() -> None:
    partition = [{"a", "c"}, {"b", "d"}]
    labels = {"a": "x", "b": "x", "c": "y", "d": "y"}
    align = compute_alignment(partition, labels)
    # Partition crosses the label boundary -> NMI is 0.
    assert align["purity"] == pytest.approx(0.5)
    assert align["nmi"] == pytest.approx(0.0, abs=1e-9)


def test_run_spectral_recovers_planted_communities(two_clique_train) -> None:
    bg = build_train_bipartite_graph(two_clique_train)
    # Use min_shared_users=1 to include the bridge edge so both clusters are connected
    item_graph = project_item_item(bg, min_shared_users=1)
    result = run_spectral(item_graph, k=2, weight="weight_jaccard", random_state=42)

    assert result.method == "spectral_k=2"
    flat = [n for c in result.communities for n in c]
    # With the bridge edge, both clusters are in one connected component.
    assert sorted(flat) == sorted(item_graph.nodes())
    cluster_a = {f"iA{i}" for i in (1, 2, 3)}
    cluster_b = {f"iB{i}" for i in (1, 2, 3)}
    assert {frozenset(c) for c in result.communities} == {
        frozenset(cluster_a),
        frozenset(cluster_b),
    }


def test_run_girvan_newman_returns_split_on_two_clique_subgraph(
    two_clique_train,
) -> None:
    bg = build_train_bipartite_graph(two_clique_train)
    # Use min_shared_users=1 to include the bridge edge
    item_graph = project_item_item(bg, min_shared_users=1)
    result = run_girvan_newman(item_graph, max_nodes=500)

    assert result.method == "girvan_newman"
    flat = [n for c in result.communities for n in c]
    assert sorted(flat) == sorted(item_graph.nodes())
    assert len(result.communities) >= 2
    cluster_a = {f"iA{i}" for i in (1, 2, 3)}
    cluster_b = {f"iB{i}" for i in (1, 2, 3)}
    # The bridge edge is the first to break -> first split returns the planted
    # cliques exactly.
    assert {frozenset(c) for c in result.communities[:2]} == {
        frozenset(cluster_a),
        frozenset(cluster_b),
    }


def test_run_girvan_newman_refuses_subgraph_above_cap() -> None:
    g = nx.path_graph(10)
    with pytest.raises(ValueError, match="exceeds girvan_newman_max_nodes"):
        run_girvan_newman(g, max_nodes=5)

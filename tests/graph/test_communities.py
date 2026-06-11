"""Community detection + category alignment on tiny planted graphs."""

from __future__ import annotations

import networkx as nx
import pytest

from src.graph.build import build_train_bipartite_graph
from src.graph.communities import (
    CommunityResult,
    compute_alignment,
    run_louvain,
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

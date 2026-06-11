"""Community detection + spectral clustering on the item-item projection.

Methods (declared by ``CommunityResult.method``):
- ``louvain``: NetworkX built-in louvain_communities (replaces stale
  python-louvain).
- ``leiden``: leidenalg + python-igraph if installed; otherwise this module's
  ``run_leiden`` returns ``None`` and logs a warning. Tests must NOT require
  leidenalg.
- ``spectral_k=<K>``: scikit-learn SpectralClustering on the sparse Jaccard
  precomputed affinity, restricted to the largest connected component.
- ``girvan_newman``: NetworkX edge-betweenness splits on a tractable subgraph
  only (caller MUST pass a small subgraph; module enforces a node cap).

Category alignment: purity + normalized mutual information against filtered
category labels derived from ``metadata.parquet``. Reuse the same generic-root
filtering convention as advanced-models when possible; if no labels can be
derived, skip alignment cleanly and set alignment to ``None``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

from src.graph.projection import to_sparse_adjacency

try:
    import pyamg  # noqa: F401  # pyright: ignore[reportMissingImports]

    _HAS_PYAMG = True
except ImportError:  # pragma: no cover
    _HAS_PYAMG = False

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommunityResult:
    """Result from a community detection algorithm.

    Attributes:
        method: Algorithm name (e.g. 'louvain', 'leiden', 'spectral_k=5').
        communities: List of communities, each a set of node IDs.
        modularity: Modularity score if available, else None.
        extras: Algorithm-specific metadata (e.g. weight, seed).
    """

    method: str
    communities: list[set[str]]
    modularity: float | None
    extras: dict[str, Any]


def _modularity(item_graph: nx.Graph, communities: list[set[str]], weight: str) -> float | None:
    """Compute modularity for a partition, handling edge cases gracefully."""
    try:
        return float(nx.community.modularity(item_graph, communities, weight=weight))
    except (ZeroDivisionError, nx.NetworkXError):
        return None


def run_louvain(
    item_graph: nx.Graph,
    weight: str = "weight_jaccard",
    seed: int = 42,
) -> CommunityResult:
    """NetworkX built-in Louvain on the item-item projection.

    Args:
        item_graph: Weighted undirected graph (e.g. from project_item_item).
        weight: Edge attribute to use (default 'weight_jaccard').
        seed: Random seed for reproducibility.

    Returns:
        CommunityResult with communities as sets of item node IDs.
    """
    communities = [
        set(c)
        for c in nx.community.louvain_communities(
            item_graph, weight=weight, seed=seed
        )
    ]
    return CommunityResult(
        method="louvain",
        communities=communities,
        modularity=_modularity(item_graph, communities, weight),
        extras={"weight": weight, "seed": seed},
    )


def compute_alignment(
    communities: list[set[str]],
    item_to_label: Mapping[str, str],
) -> dict[str, Any]:
    """Purity + normalized mutual information vs filtered category labels.

    Items without a label are ignored. Returns ``purity`` in [0, 1],
    ``nmi`` in [0, 1], and ``n_labeled_items`` for diagnostics.

    Args:
        communities: List of communities, each a set of node IDs.
        item_to_label: Mapping from node ID to category label string.

    Returns:
        Dict with keys 'purity', 'nmi', 'n_labeled_items'.
    """
    cluster_ids: list[int] = []
    label_ids: list[int] = []
    label_to_int: dict[str, int] = {}
    correct = 0
    total = 0

    for cluster_idx, community in enumerate(communities):
        labels_here: list[str] = []
        for node in community:
            label = item_to_label.get(node)
            if label is None:
                continue
            labels_here.append(label)
            cluster_ids.append(cluster_idx)
            if label not in label_to_int:
                label_to_int[label] = len(label_to_int)
            label_ids.append(label_to_int[label])

        if not labels_here:
            continue
        # Majority label in this community.
        counts: dict[str, int] = {}
        for label in labels_here:
            counts[label] = counts.get(label, 0) + 1
        correct += max(counts.values())
        total += len(labels_here)

    purity = (correct / total) if total else 0.0
    if total:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nmi = float(
                normalized_mutual_info_score(
                    np.asarray(label_ids), np.asarray(cluster_ids)
                )
            )
    else:
        nmi = 0.0

    return {"purity": float(purity), "nmi": nmi, "n_labeled_items": total}


def _largest_component_subgraph(item_graph: nx.Graph) -> tuple[nx.Graph, list[str]]:
    """Extract the largest connected component as a subgraph.

    Returns (subgraph, node_list).
    """
    if item_graph.number_of_nodes() == 0:
        return item_graph, []
    components = list(nx.connected_components(item_graph))
    largest = max(components, key=len)
    sub = item_graph.subgraph(largest).copy()
    return sub, list(sub.nodes())


def run_spectral(
    item_graph: nx.Graph,
    k: int,
    weight: str = "weight_jaccard",
    random_state: int = 42,
) -> CommunityResult:
    """Spectral clustering on the largest connected component.

    Uses scikit-learn ``SpectralClustering`` with a sparse precomputed
    affinity (Jaccard adjacency). Prefers ``eigen_solver='amg'`` when
    ``pyamg`` is installed (faster + more stable on big sparse graphs);
    falls back to ``'arpack'`` otherwise.

    Args:
        item_graph: Weighted undirected graph (e.g. from project_item_item).
        k: Number of clusters to find (>= 2).
        weight: Edge attribute to use (default 'weight_jaccard').
        random_state: Random seed for reproducibility.

    Returns:
        CommunityResult with communities as sets of item node IDs.

    Raises:
        ValueError: If k < 2 or largest component has fewer than k nodes.
    """
    if k < 2:
        raise ValueError("k must be >= 2 for spectral clustering")
    sub, nodes = _largest_component_subgraph(item_graph)
    if len(nodes) < k:
        raise ValueError(
            f"largest component has {len(nodes)} nodes, less than k={k}"
        )
    adj, idx_to_item = to_sparse_adjacency(sub, weight=weight)
    eigen_solver = "amg" if _HAS_PYAMG else "arpack"
    model = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        eigen_solver=eigen_solver,
        assign_labels="cluster_qr",
        random_state=random_state,
    )
    labels = model.fit_predict(sp.csr_matrix(adj))

    communities_dict: dict[int, set[str]] = {}
    for node, label in zip(idx_to_item, labels):
        communities_dict.setdefault(int(label), set()).add(node)
    communities = list(communities_dict.values())

    return CommunityResult(
        method=f"spectral_k={k}",
        communities=communities,
        modularity=_modularity(sub, communities, weight),
        extras={
            "k": k,
            "weight": weight,
            "eigen_solver": eigen_solver,
            "n_nodes_used": len(nodes),
            "random_state": random_state,
        },
    )


def run_girvan_newman(
    item_graph: nx.Graph,
    max_nodes: int = 500,
    weight: str = "weight_jaccard",
) -> CommunityResult:
    """Illustrative Girvan-Newman on a tractable subgraph.

    Refuses to run when the input exceeds ``max_nodes`` — Girvan-Newman is
    O(m * n^2) and is documented as a small-scale comparison only. The
    caller is expected to pre-restrict to a top-degree subgraph or a single
    large community.

    Args:
        item_graph: Weighted undirected graph (e.g. from project_item_item).
        max_nodes: Maximum allowed number of nodes; raises ValueError if exceeded.
        weight: Edge attribute to use (default 'weight_jaccard').

    Returns:
        CommunityResult with communities as sets of item node IDs (first split).

    Raises:
        ValueError: If item_graph has more than max_nodes nodes.
    """
    n = item_graph.number_of_nodes()
    if n > max_nodes:
        raise ValueError(
            f"input has {n} nodes, exceeds girvan_newman_max_nodes={max_nodes}"
        )
    iterator = nx.community.girvan_newman(item_graph)
    # Take the first split (the most informative on small inputs).
    first_split = next(islice(iterator, 1), None)
    if first_split is None:
        communities = [set(item_graph.nodes())]
    else:
        communities = [set(c) for c in first_split]
    return CommunityResult(
        method="girvan_newman",
        communities=communities,
        modularity=_modularity(item_graph, communities, weight),
        extras={"max_nodes": max_nodes, "n_nodes_used": n},
    )


def run_leiden(
    item_graph: nx.Graph,
    weight: str = "weight_jaccard",
    seed: int = 42,
) -> CommunityResult | None:
    """Leiden community detection if ``leidenalg`` + ``igraph`` are installed.

    Returns ``None`` and logs a warning when the optional dependencies are
    missing — the rest of the analysis pipeline still runs (Louvain +
    Spectral are required and always available).
    """
    try:
        import igraph as ig  # pyright: ignore[reportMissingImports]
        import leidenalg  # pyright: ignore[reportMissingImports]
    except ImportError:
        _LOG.warning(
            "leidenalg / python-igraph not installed; skipping Leiden. "
            "Install with: pip install leidenalg python-igraph"
        )
        return None

    ig_graph = ig.Graph.from_networkx(item_graph)
    # NetworkX edge attribute names survive the conversion; map by name.
    weights = [float(item_graph[u][v].get(weight, 0.0)) for u, v in item_graph.edges()]
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        seed=seed,
    )
    node_names = ig_graph.vs["_nx_name"]
    communities: list[set[str]] = []
    for member_indices in partition:
        communities.append({node_names[i] for i in member_indices})

    return CommunityResult(
        method="leiden",
        communities=communities,
        modularity=_modularity(item_graph, communities, weight),
        extras={"weight": weight, "seed": seed},
    )

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
from typing import Any

import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

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

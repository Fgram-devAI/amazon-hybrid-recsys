"""Weighted item-item co-rating projection of the train bipartite graph.

Every edge carries BOTH weights so analysis (EDA / Louvain / Leiden / Spectral)
can pick its preferred view without re-projecting:

    weight_count   = number of users who co-rated the two items
    weight_jaccard = |co-raters| / |union of raters|

Default for community / spectral algorithms downstream: ``weight_jaccard``
(reduces popularity bias).
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
from scipy import sparse

from src.graph.build import BipartiteTrainGraph


def project_item_item(
    bipartite: BipartiteTrainGraph,
    min_shared_users: int = 3,
    top_n_items: int | None = None,
) -> nx.Graph:
    """Project the train bipartite graph onto items, weighted.

    ``min_shared_users`` controls densification: an edge is added only when
    items share at least this many users.

    ``top_n_items`` optionally restricts the projection to the top-N items by
    training degree (safety cap for local runs); ``None`` keeps every item.
    """
    if min_shared_users < 1:
        raise ValueError("min_shared_users must be >= 1")

    g = bipartite.graph
    item_nodes: list[str] = list(bipartite.idx_to_item)

    if top_n_items is not None and top_n_items < len(item_nodes):
        item_nodes = sorted(
            item_nodes, key=lambda it: g.degree(it), reverse=True
        )[:top_n_items]

    item_set = set(item_nodes)
    # Per-item rater set (users connected to that item in the bipartite graph).
    raters: dict[str, set[str]] = {
        it: set(g.neighbors(it)) for it in item_nodes
    }

    item_graph: nx.Graph = nx.Graph()
    item_graph.add_nodes_from(item_nodes)

    # Two items can only share a user if both appear in some user's neighbour
    # list; iterating user-by-user is O(sum d_u^2) which is tractable for
    # synthetic + real item projections under min_shared_users >= 3.
    pair_counts: dict[tuple[str, str], int] = {}
    for user, attrs in g.nodes(data=True):
        if attrs.get("bipartite") != 0:
            continue
        user_items = [it for it in g.neighbors(user) if it in item_set]
        user_items.sort()
        for a_idx, a in enumerate(user_items):
            for b in user_items[a_idx + 1 :]:
                key = (a, b)
                pair_counts[key] = pair_counts.get(key, 0) + 1

    for (a, b), count in pair_counts.items():
        if count < min_shared_users:
            continue
        union = len(raters[a] | raters[b])
        jaccard = count / union if union else 0.0
        item_graph.add_edge(
            a, b, weight_count=int(count), weight_jaccard=float(jaccard)
        )

    return item_graph


def to_sparse_adjacency(
    item_graph: nx.Graph,
    weight: str = "weight_jaccard",
    nodelist: Iterable[str] | None = None,
) -> tuple[sparse.csr_matrix, list[str]]:
    """Return a sparse CSR adjacency matrix using the chosen edge weight.

    Returns ``(adjacency, idx_to_item)``; ``idx_to_item[i]`` is the item id at
    matrix row/column ``i``. Used as the precomputed affinity for sklearn
    ``SpectralClustering`` and for scipy graph utilities.
    """
    nodes: list[str] = list(nodelist) if nodelist is not None else list(item_graph.nodes())
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for u, v, attrs in item_graph.edges(data=True):
        w = float(attrs.get(weight, 0.0))
        if w == 0.0:
            continue
        i, j = index[u], index[v]
        rows.append(i)
        cols.append(j)
        data.append(w)
        rows.append(j)
        cols.append(i)
        data.append(w)
    adj = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
        shape=(n, n),
    )
    return adj, nodes

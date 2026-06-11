"""Structural EDA for the train bipartite graph and the item-item projection.

Returns plain JSON-serializable dicts that the analyse CLI then writes to
``data/processed/<dataset>/graph_analysis/report.json``. Optional figure
helpers (degree histogram, weight histograms) are kept here too — they
write PNGs to the same directory and are deliberately small.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from src.graph.build import BipartiteTrainGraph


def _summarize(values: list[float] | list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    # If input values are all ints, keep min/max/median as ints
    is_integer = all(isinstance(v, (int, np.integer)) for v in values)
    return {
        "count": int(arr.size),
        "min": int(arr.min()) if is_integer else float(arr.min()),
        "max": int(arr.max()) if is_integer else float(arr.max()),
        "mean": float(arr.mean()),
        "median": int(np.median(arr)) if is_integer else float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
    }


def compute_bipartite_eda(bg: BipartiteTrainGraph) -> dict[str, Any]:
    """Structural metrics for the bipartite user-item graph."""
    g = bg.graph
    users = [n for n, d in g.nodes(data=True) if d.get("bipartite") == 0]
    items = [n for n, d in g.nodes(data=True) if d.get("bipartite") == 1]
    n_users, n_items = len(users), len(items)
    n_edges = g.number_of_edges()
    max_edges = n_users * n_items
    density = (n_edges / max_edges) if max_edges else 0.0

    user_degrees = [g.degree(u) for u in users]
    item_degrees = [g.degree(it) for it in items]

    components = list(nx.connected_components(g))
    component_sizes = [len(c) for c in components]

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_edges": n_edges,
        "density": density,
        "n_connected_components": len(components),
        "largest_component_size": max(component_sizes) if component_sizes else 0,
        "component_size_distribution": _summarize(component_sizes),
        "user_degree": _summarize(user_degrees),
        "item_degree": _summarize(item_degrees),
    }


def compute_projection_eda(item_graph: nx.Graph) -> dict[str, Any]:
    """Structural metrics for the item-item co-rating projection."""
    n_nodes = item_graph.number_of_nodes()
    n_edges = item_graph.number_of_edges()
    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 0
    density = (n_edges / max_edges) if max_edges else 0.0

    degrees = [d for _, d in item_graph.degree()]
    components = list(nx.connected_components(item_graph))
    component_sizes = [len(c) for c in components]
    if n_edges:
        weight_counts = [int(a["weight_count"]) for _, _, a in item_graph.edges(data=True)]
        weight_jaccs = [float(a["weight_jaccard"]) for _, _, a in item_graph.edges(data=True)]
        clustering = nx.average_clustering(item_graph, weight="weight_jaccard")
    else:
        weight_counts, weight_jaccs, clustering = [], [], 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "n_connected_components": len(components),
        "largest_component_size": max(component_sizes) if component_sizes else 0,
        "component_size_distribution": _summarize(component_sizes),
        "degree": _summarize(degrees),
        "clustering_coefficient_mean": float(clustering),
        "weights": {
            "weight_count": _summarize(weight_counts),
            "weight_jaccard": _summarize(weight_jaccs),
        },
    }

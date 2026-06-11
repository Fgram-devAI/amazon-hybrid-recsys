"""Train-only bipartite user-item graph construction.

Test interactions never enter the graph or any tensor returned here. Edge views:
    * observed (all train ratings)    -> for GraphSAGE edge regression
    * positive (rating >= threshold)  -> for LightGCN/BPR

Node indexing convention (matches torch_geometric.nn.models.LightGCN):
    * users occupy indices [0, num_users)
    * items occupy indices [num_users, num_users + num_items)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class BipartiteGraph:
    user_index: dict[str, int]
    item_index: dict[str, int]
    user_offset: int
    item_offset: int
    num_nodes: int

    # observed (all train ratings) view
    propagation_edge_index: torch.Tensor      # (2, 2 * n_train_edges) undirected, unified node space
    edge_label_index: torch.Tensor            # (2, n_train_edges) user -> item, unified node space
    edge_label_rating: torch.Tensor           # (n_train_edges,) float ratings aligned to edge_label_index

    # positive-only (rating >= threshold) view
    positive_propagation_edge_index: torch.Tensor   # (2, 2 * n_positive_edges) undirected
    positive_edge_label_index: torch.Tensor         # (2, n_positive_edges) user -> item


def build_graph(train: pd.DataFrame, *, min_rating_positive: float) -> BipartiteGraph:
    """Build the train-only bipartite graph. test rows must NOT be passed in."""
    users = sorted(train["user_id"].astype(str).unique().tolist())
    items = sorted(train["parent_asin"].astype(str).unique().tolist())
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {it: i for i, it in enumerate(items)}
    user_offset = 0
    item_offset = len(users)
    num_nodes = len(users) + len(items)

    u_local = train["user_id"].map(user_index).to_numpy(dtype=np.int64)
    i_local = train["parent_asin"].map(item_index).to_numpy(dtype=np.int64)
    i_global = i_local + item_offset
    ratings = train["rating"].to_numpy(dtype=np.float32)

    edge_label_index = torch.from_numpy(np.vstack([u_local, i_global]))
    edge_label_rating = torch.from_numpy(ratings)

    # undirected propagation edges = both directions
    prop_src = np.concatenate([u_local, i_global])
    prop_dst = np.concatenate([i_global, u_local])
    propagation_edge_index = torch.from_numpy(np.vstack([prop_src, prop_dst]))

    pos_mask = ratings >= float(min_rating_positive)
    pos_u = u_local[pos_mask]
    pos_i = i_global[pos_mask]
    positive_edge_label_index = torch.from_numpy(np.vstack([pos_u, pos_i]))
    pos_prop_src = np.concatenate([pos_u, pos_i])
    pos_prop_dst = np.concatenate([pos_i, pos_u])
    positive_propagation_edge_index = torch.from_numpy(np.vstack([pos_prop_src, pos_prop_dst]))

    return BipartiteGraph(
        user_index=user_index,
        item_index=item_index,
        user_offset=user_offset,
        item_offset=item_offset,
        num_nodes=num_nodes,
        propagation_edge_index=propagation_edge_index,
        edge_label_index=edge_label_index,
        edge_label_rating=edge_label_rating,
        positive_propagation_edge_index=positive_propagation_edge_index,
        positive_edge_label_index=positive_edge_label_index,
    )

"""GraphSAGE recommender: SAGEConv stack over the bipartite graph + MLP edge head.

The edge-regression MLP takes [user_node_embedding ; item_node_embedding] and
predicts rating. The rating is NEVER an input feature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.features.node_features import (
    build_item_node_features,
    build_user_node_features,
)
from src.graph.build import BipartiteGraph, build_graph
from src.models.graph_base import GraphRecommender


class _SAGEEdgeModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(n_layers):
            layers.append(SAGEConv(dim, hidden_dim))
            dim = hidden_dim
        self.convs = nn.ModuleList(layers)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return h

    def predict_edge(
        self, h: torch.Tensor, edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        u = h[edge_label_index[0]]
        v = h[edge_label_index[1]]
        return self.head(torch.cat([u, v], dim=-1)).squeeze(-1)


class GraphSAGERecommender(GraphRecommender):
    def __init__(
        self,
        *,
        embedder: object,
        generic_roots: list[str],
        max_vocab: int,
        min_doc_freq: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        epochs: int = 20,
        lr: float = 0.005,
        batch_size: int = 1024,
        seed: int = 42,
        device: str = "auto",
        cache_dir: object = None,
        review_features_dir: object = None,
        progress: bool = False,
    ) -> None:
        super().__init__(device=device)
        self.embedder = embedder
        self.generic_roots = list(generic_roots)
        self.max_vocab = int(max_vocab)
        self.min_doc_freq = int(min_doc_freq)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.cache_dir = cache_dir
        self.review_features_dir = review_features_dir
        self.progress = bool(progress)
        self._graph: BipartiteGraph | None = None
        self._model: _SAGEEdgeModel | None = None
        self._x: torch.Tensor | None = None

    def fit(self, train: pd.DataFrame, metadata: object = None) -> "GraphSAGERecommender":
        if metadata is None:
            raise ValueError("GraphSAGERecommender.fit requires metadata")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._fit_means(train)
        if self.progress:
            print(
                f"[graphsage] fit start: rows={len(train):,}, device={self.device}, "
                f"epochs={self.epochs}, hidden_dim={self.hidden_dim}, layers={self.n_layers}",
                flush=True,
            )

        self._graph = build_graph(train, min_rating_positive=4.0)
        if self.progress:
            print(
                f"[graphsage] graph: users={len(self._graph.user_index):,}, "
                f"items={len(self._graph.item_index):,}, "
                f"observed_edges={self._graph.edge_label_index.shape[1]:,}",
                flush=True,
            )

        # filter metadata to items that appear in the graph
        item_ids_in_graph = list(self._graph.item_index.keys())
        meta_df: pd.DataFrame = metadata  # type: ignore[assignment]
        meta_filtered = meta_df[meta_df["parent_asin"].isin(item_ids_in_graph)].copy()
        if self.progress:
            print(
                f"[graphsage] building item features: metadata_items={len(meta_filtered):,}, "
                f"cache_dir={self.cache_dir}",
                flush=True,
            )

        item_feats, item_ids = build_item_node_features(
            meta_filtered,
            embedder=self.embedder,  # type: ignore[arg-type]
            generic_roots=self.generic_roots,
            max_vocab=self.max_vocab,
            min_doc_freq=self.min_doc_freq,
            cache_dir=Path(str(self.cache_dir)) if self.cache_dir is not None else None,
            review_features_dir=Path(str(self.review_features_dir))
                if self.review_features_dir is not None else None,
            use_item_sentiment=True,
        )
        if self.progress:
            print(
                f"[graphsage] item features: shape={item_feats.shape}, "
                f"aligned_items={len(item_ids):,}",
                flush=True,
            )

        # align item-features row order to graph.item_index
        item_pos = {iid: pos for pos, iid in enumerate(item_ids)}
        item_order = np.asarray(
            [item_pos[iid] for iid in self._graph.item_index.keys()
             if iid in item_pos],
            dtype=np.int64,
        )
        item_feats = item_feats[item_order]
        item_dim = item_feats.shape[1]

        user_ids = list(self._graph.user_index.keys())
        user_feats, _ = build_user_node_features(
            train,
            user_ids=user_ids,
            review_features_dir=Path(str(self.review_features_dir))
                if self.review_features_dir is not None else None,
        )
        if self.progress:
            print(f"[graphsage] user features: shape={user_feats.shape}", flush=True)

        # project user features into item-feature dim with a learned linear later;
        # here we just zero-pad / truncate so the unified x has a single dim.
        if user_feats.shape[1] < item_dim:
            pad = np.zeros((user_feats.shape[0], item_dim - user_feats.shape[1]), dtype=np.float32)
            user_feats = np.hstack([user_feats, pad])
        elif user_feats.shape[1] > item_dim:
            user_feats = user_feats[:, :item_dim]

        x = np.vstack([user_feats, item_feats]).astype(np.float32)
        self._x = torch.from_numpy(x).to(self.device)
        if self.progress:
            print(f"[graphsage] node feature matrix: shape={self._x.shape}", flush=True)

        self._model = _SAGEEdgeModel(item_dim, self.hidden_dim, self.n_layers).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        edge_index = self._graph.propagation_edge_index.to(self.device)
        edge_label_index = self._graph.edge_label_index.to(self.device)
        edge_label_rating = self._graph.edge_label_rating.to(self.device)

        self._model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(edge_label_index.shape[1])
            epoch_loss = 0.0
            n_batches = int(np.ceil(perm.shape[0] / self.batch_size))
            for start in range(0, perm.shape[0], self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_edges = edge_label_index[:, idx]
                batch_y = edge_label_rating[idx]
                h = self._model.encode(self._x, edge_index)
                preds = self._model.predict_edge(h, batch_edges)
                loss = F.mse_loss(preds, batch_y)
                epoch_loss += float(loss.detach().cpu())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if self.progress:
                print(
                    f"[graphsage] epoch {epoch + 1}/{self.epochs}: "
                    f"batches={n_batches:,}, mean_mse_loss={epoch_loss / max(n_batches, 1):.4f}",
                    flush=True,
                )
        self._model.eval()
        return self

    def predict(self, user_id: str, parent_asin: str) -> float:
        if self._model is None or self._graph is None or self._x is None:
            return self._clip(self._fallback(user_id, parent_asin))
        uid = self._graph.user_index.get(user_id)
        iid = self._graph.item_index.get(parent_asin)
        if uid is None or iid is None:
            return self._clip(self._fallback(user_id, parent_asin))
        edge_index = self._graph.propagation_edge_index.to(self.device)
        edge_label = torch.tensor(
            [[uid], [iid + self._graph.item_offset]],
            dtype=torch.long, device=self.device,
        )
        with torch.no_grad():
            h = self._model.encode(self._x, edge_index)
            pred = self._model.predict_edge(h, edge_label).item()
        return self._clip(pred)

    def state_dict(self) -> dict[str, object]:
        assert self._model is not None
        return {"model": self._model.state_dict()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        assert self._model is not None
        self._model.load_state_dict(state["model"])  # type: ignore[arg-type]

"""LightGCN recommender wrapping torch_geometric.nn.models.LightGCN.

Native objective: BPR for top-K ranking. predict() returns a calibrated rating
computed by least-squares-fitting (beta, intercept) on a train-derived validation
slice (NEVER on test). Calibrated rating is clipped to [1, 5].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import LightGCN

from src.evaluation.tune import split_validation
from src.graph.build import BipartiteGraph, build_graph
from src.models.graph_base import GraphRecommender


class LightGCNRecommender(GraphRecommender):
    def __init__(
        self,
        *,
        embedding_dim: int = 64,
        n_layers: int = 3,
        epochs: int = 20,
        lr: float = 0.001,
        num_negatives: int = 1,
        batch_size: int = 1024,
        seed: int = 42,
        device: str = "auto",
        min_rating_positive: float = 4.0,
        validation_fraction: float = 0.1,
    ) -> None:
        super().__init__(device=device)
        self.embedding_dim = int(embedding_dim)
        self.n_layers = int(n_layers)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.num_negatives = int(num_negatives)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.min_rating_positive = float(min_rating_positive)
        self.validation_fraction = float(validation_fraction)
        self._graph: BipartiteGraph | None = None
        self._model: LightGCN | None = None
        self._calibration_beta: float | None = None
        self._calibration_intercept: float | None = None

    def fit(self, train: pd.DataFrame, metadata: object = None) -> "LightGCNRecommender":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._fit_means(train)

        # train -> (train_only, val) for calibration
        train_only, val = split_validation(
            train, validation_fraction=self.validation_fraction, seed=self.seed,
        )
        if val.empty:
            train_only, val = train, train.head(min(64, len(train)))

        # Build a graph over train_only first to calibrate, then refit on full train
        cal_graph = build_graph(train_only, min_rating_positive=self.min_rating_positive)
        cal_model = self._train_bpr(cal_graph)
        self._calibration_beta, self._calibration_intercept = self._fit_calibration(
            cal_model, cal_graph, val,
        )

        # Final fit on full train
        self._graph = build_graph(train, min_rating_positive=self.min_rating_positive)
        self._model = self._train_bpr(self._graph)
        return self

    def _train_bpr(self, graph: BipartiteGraph) -> LightGCN:
        model = LightGCN(
            num_nodes=graph.num_nodes,
            embedding_dim=self.embedding_dim,
            num_layers=self.n_layers,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        pos_edges = graph.positive_edge_label_index.to(self.device)
        if pos_edges.shape[1] == 0:
            return model.eval()
        prop = graph.positive_propagation_edge_index.to(self.device)
        n_items = len(graph.item_index)
        item_offset = graph.item_offset
        rng = np.random.default_rng(self.seed)

        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(pos_edges.shape[1])
            for start in range(0, perm.shape[0], self.batch_size):
                idx = perm[start:start + self.batch_size]
                u = pos_edges[0, idx]
                pos_i = pos_edges[1, idx]
                # sampled negatives in item index space, shifted to unified node space
                neg_local = rng.integers(0, n_items, size=u.shape[0])
                neg_i = torch.from_numpy(neg_local).to(self.device) + item_offset
                pos_score = model(prop, edge_label_index=torch.stack([u, pos_i]))
                neg_score = model(prop, edge_label_index=torch.stack([u, neg_i]))
                loss = F.softplus(neg_score - pos_score).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.eval()

    def _fit_calibration(
        self,
        model: LightGCN,
        graph: BipartiteGraph,
        val: pd.DataFrame,
    ) -> tuple[float, float]:
        if val.empty:
            return 0.0, float(self.global_mean_)
        u_local: list[int] = []
        i_global: list[int] = []
        ratings: list[float] = []
        for _, row in val.iterrows():
            uid = graph.user_index.get(str(row["user_id"]))
            iid = graph.item_index.get(str(row["parent_asin"]))
            if uid is None or iid is None:
                continue
            u_local.append(uid)
            i_global.append(iid + graph.item_offset)
            ratings.append(float(row["rating"]))
        if not u_local:
            return 0.0, float(self.global_mean_)
        u_t = torch.tensor(u_local, dtype=torch.long, device=self.device)
        i_t = torch.tensor(i_global, dtype=torch.long, device=self.device)
        prop = graph.positive_propagation_edge_index.to(self.device)
        with torch.no_grad():
            raw = model(prop, edge_label_index=torch.stack([u_t, i_t])).cpu().numpy()
        y = np.asarray(ratings, dtype=np.float32)
        if raw.std() == 0:
            return 0.0, float(y.mean())
        # simple linear: rating ~ beta * raw + intercept
        beta, intercept = np.polyfit(raw, y, 1)
        return float(beta), float(intercept)

    def predict(self, user_id: str, parent_asin: str) -> float:
        if self._model is None or self._graph is None:
            return self._clip(self._fallback(user_id, parent_asin))
        uid = self._graph.user_index.get(user_id)
        iid = self._graph.item_index.get(parent_asin)
        if uid is None or iid is None:
            return self._clip(self._fallback(user_id, parent_asin))
        u_t = torch.tensor([uid], dtype=torch.long, device=self.device)
        i_t = torch.tensor(
            [iid + self._graph.item_offset], dtype=torch.long, device=self.device,
        )
        prop = self._graph.positive_propagation_edge_index.to(self.device)
        with torch.no_grad():
            raw = self._model(prop, edge_label_index=torch.stack([u_t, i_t])).item()
        beta = self._calibration_beta if self._calibration_beta is not None else 0.0
        intercept = (
            self._calibration_intercept
            if self._calibration_intercept is not None
            else self.global_mean_
        )
        return self._clip(beta * raw + intercept)

    def state_dict(self) -> dict[str, object]:
        assert self._model is not None
        return {
            "model": self._model.state_dict(),
            "calibration_beta": self._calibration_beta,
            "calibration_intercept": self._calibration_intercept,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        assert self._model is not None
        self._model.load_state_dict(state["model"])  # type: ignore[arg-type]
        self._calibration_beta = state.get("calibration_beta")  # type: ignore[assignment]
        self._calibration_intercept = state.get("calibration_intercept")  # type: ignore[assignment]
